// Behaviour summary report (PDF).
//
// EthoReport.build(pose, clusters, { jsPDF }) turns an etho-pose dataset and an etho-clusters export
// into a jsPDF document: the same analysis as the Jupyter notebook template (analysis_notebook.py),
// drawn as vector figures with a short text description of each result. It reads only the two
// exported objects, so it can also run outside the viewer (e.g. Node with the jspdf package).
// build_viewer.py inlines this file into the viewer page; the page loads jsPDF from a CDN on demand.
//
// EthoReport.analyze(pose, clusters) returns the numbers alone (per-frame measures, time budget,
// bouts, transitions, PCA summary, per-state medians and z-scores, average poses).
(function (root) {
  'use strict';

  // ---------- numeric helpers (NaN-skipping, numpy conventions) ----------
  const finite = (a) => { const o = []; for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) o.push(a[i]); return o; };
  const sortedFinite = (a) => Float64Array.from(finite(a)).sort();
  function pctSorted(s, q) {           // numpy.percentile, linear interpolation
    if (!s.length) return NaN;
    const p = ((s.length - 1) * q) / 100, lo = Math.floor(p), hi = Math.min(lo + 1, s.length - 1);
    return s[lo] + (s[hi] - s[lo]) * (p - lo);
  }
  const percentile = (a, q) => pctSorted(sortedFinite(a), q);
  const median = (a) => percentile(a, 50);
  function meanOf(a) { let s = 0, n = 0; for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) { s += a[i]; n++; } return n ? s / n : NaN; }
  function stdOf(a) {                  // ddof = 1, like pandas
    const m = meanOf(a); let s = 0, n = 0;
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) { s += (a[i] - m) ** 2; n++; }
    return n > 1 ? Math.sqrt(s / (n - 1)) : NaN;
  }
  function gradient(x) {               // numpy.gradient, unit spacing, first-order edges
    const n = x.length, g = new Float64Array(n);
    if (n < 2) return g;
    g[0] = x[1] - x[0]; g[n - 1] = x[n - 1] - x[n - 2];
    for (let i = 1; i < n - 1; i++) g[i] = (x[i + 1] - x[i - 1]) / 2;
    return g;
  }
  function unwrap(a) {                 // numpy.unwrap
    const out = Float64Array.from(a);
    let corr = 0;
    for (let i = 1; i < a.length; i++) {
      const dd = a[i] - a[i - 1];
      if (Number.isFinite(dd)) {
        let m = ((((dd + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
        if (m === -Math.PI && dd > 0) m = Math.PI;
        if (Math.abs(dd) >= Math.PI) corr += m - dd;
      }
      out[i] = a[i] + corr;
    }
    return out;
  }
  function boxStats(values) {          // matplotlib boxplot: quartiles, whiskers at 1.5 IQR, fliers
    const s = sortedFinite(values);
    if (!s.length) return null;
    const q1 = pctSorted(s, 25), med = pctSorted(s, 50), q3 = pctSorted(s, 75), iqr = q3 - q1;
    let lo = q3, hi = q1;
    const fliers = [];
    for (const v of s) {
      if (v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr) fliers.push(v);
      else { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    return { q1, med, q3, lo: Math.min(lo, q1), hi: Math.max(hi, q3), fliers, n: s.length };
  }
  const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  const rowSum = (r) => r.reduce((x, y) => x + y, 0);

  // ---------- analysis: a port of the notebook template ----------
  function analyze(pose, cl) {
    if (!pose || pose.format !== 'etho-pose') throw new Error('not an etho-pose dataset');
    if (!cl || cl.format !== 'etho-clusters') throw new Error('not an etho-clusters export');
    const src = cl.source, clip = pose.clips[src.clip.index];
    const fps = Number(clip.fps), nF = Number(clip.frame_count);
    const model = cl.model, states = model.states, K = states.length;
    const COLORS = states.map((s) => s.color), STATE_NAMES = states.map((s) => `S${s.id}`);
    const animals = cl.animals, A = animals.length;
    const units = pose.space.units || 'm', TO_DISPLAY = units === 'm' ? 100 : 1, LEN = units === 'm' ? 'cm' : units;
    const dims = Number(pose.space.dims);
    const skelOf = {};
    pose.animals.forEach((a) => { skelOf[a.id] = pose.skeletons[a.skeleton]; });
    const skel0 = skelOf[animals[0].id];
    const jointNames = skel0.joints.map((j) => j.name), bones = skel0.bones || [], J = jointNames.length;
    const axisInfo = cl.features.egocentric_axis || {};
    const ANT = jointNames.includes(axisInfo.anterior) ? jointNames.indexOf(axisInfo.anterior) : J - 1;
    const POST = jointNames.includes(axisInfo.posterior) ? jointNames.indexOf(axisInfo.posterior) : 0;
    const noseName = ['snout', 'nose', 'head'].find((n) => jointNames.includes(n));
    const NOSE = noseName ? jointNames.indexOf(noseName) : ANT;

    // viewer convention: x, height, z (2D x, y lies on the floor)
    const up = pose.space.up || '+Y', sgn = up[0] === '-' ? -1 : 1;
    const toYUp = (p) => {
      if (dims === 2) return [p[0], 0, p[1]];
      if (up[1] === 'Y') return [p[0], sgn * p[1], sgn * p[2]];
      if (up[1] === 'Z') return [p[0], sgn * p[2], -sgn * p[1]];
      return [sgn * p[1], sgn * p[0], -p[2]];
    };
    // frames x joints x 3 (Y up, display units), gaps linearly interpolated (edges held)
    function trackArray(aid) {
      const raw = clip.tracks[aid].positions, Ja = skelOf[aid].joints.length;
      const tmp = new Float64Array(nF * Ja * dims).fill(NaN);
      for (let f = 0; f < nF; f++) {
        const fr = raw[f];
        if (!fr) continue;
        for (let j = 0; j < Ja; j++) { const q = fr[j]; if (q) for (let c = 0; c < dims; c++) tmp[(f * Ja + j) * dims + c] = q[c] == null ? NaN : q[c]; }
      }
      for (let j = 0; j < Ja; j++) for (let c = 0; c < dims; c++) {
        const at = (f) => (f * Ja + j) * dims + c, known = [];
        for (let f = 0; f < nF; f++) if (Number.isFinite(tmp[at(f)])) known.push(f);
        if (!known.length || known.length === nF) continue;
        let k = 0;
        for (let f = 0; f < nF; f++) {
          if (Number.isFinite(tmp[at(f)])) continue;
          while (k < known.length - 1 && known[k + 1] < f) k++;
          if (f < known[0]) tmp[at(f)] = tmp[at(known[0])];
          else if (f > known[known.length - 1]) tmp[at(f)] = tmp[at(known[known.length - 1])];
          else { const f0 = known[k], f1 = known[k + 1]; tmp[at(f)] = tmp[at(f0)] + ((tmp[at(f1)] - tmp[at(f0)]) * (f - f0)) / (f1 - f0); }
        }
      }
      const out = new Float64Array(nF * Ja * 3);
      for (let i = 0; i < nF * Ja; i++) {
        const p = toYUp(Array.from(tmp.subarray(i * dims, i * dims + dims)));
        for (let c = 0; c < 3; c++) out[i * 3 + c] = p[c] * TO_DISPLAY;
      }
      return out;
    }
    const P = animals.map((a) => trackArray(a.id));
    const LABELS = animals.map((a) => Int32Array.from(a.labels));          // 1-based
    const at = (p, f, j, c) => p[(f * J + j) * 3 + c];
    const centroids = P.map((p) => {
      const cen = new Float64Array(nF * 3);
      for (let f = 0; f < nF; f++) for (let c = 0; c < 3; c++) { let s = 0; for (let j = 0; j < J; j++) s += at(p, f, j, c); cen[f * 3 + c] = s / J; }
      return cen;
    });

    // ---- per-frame measures (one row per animal per frame, like the notebook's df)
    const MEASURES = ['speed', 'accel', 'turn_rate'].concat(dims === 3 ? ['centroid_height', 'max_height'] : [], ['body_length'],
      A > 1 ? ['nearest_distance', 'nose_nose'] : []);
    const UNITS = { speed: `${LEN}/s`, accel: `${LEN}/s²`, turn_rate: '°/s', centroid_height: LEN, max_height: LEN, body_length: LEN, nearest_distance: LEN, nose_nose: LEN };
    const N = A * nF, col = {};
    MEASURES.forEach((m) => { col[m] = new Float64Array(N); });
    const rowAnimal = new Int32Array(N), rowState = new Int32Array(N);
    for (let a = 0; a < A; a++) {
      const p = P[a], cen = centroids[a], o = a * nF;
      const vx = gradient(Array.from({ length: nF }, (_, f) => cen[f * 3])).map((v) => v * fps);
      const vz = gradient(Array.from({ length: nF }, (_, f) => cen[f * 3 + 2])).map((v) => v * fps);
      const ax = gradient(vx).map((v) => v * fps), az = gradient(vz).map((v) => v * fps);
      const heading = unwrap(Array.from({ length: nF }, (_, f) => Math.atan2(at(p, f, ANT, 2) - at(p, f, POST, 2), at(p, f, ANT, 0) - at(p, f, POST, 0))));
      const turn = gradient(heading);
      for (let f = 0; f < nF; f++) {
        const r = o + f;
        rowAnimal[r] = a; rowState[r] = LABELS[a][f];
        col.speed[r] = Math.hypot(vx[f], vz[f]);
        col.accel[r] = Math.hypot(ax[f], az[f]);
        col.turn_rate[r] = Math.abs((turn[f] * fps * 180) / Math.PI);
        col.body_length[r] = Math.hypot(at(p, f, ANT, 0) - at(p, f, POST, 0), at(p, f, ANT, 1) - at(p, f, POST, 1), at(p, f, ANT, 2) - at(p, f, POST, 2));
        if (dims === 3) {
          col.centroid_height[r] = cen[f * 3 + 1];
          let mh = -Infinity; for (let j = 0; j < J; j++) mh = Math.max(mh, at(p, f, j, 1));
          col.max_height[r] = mh;
        }
        if (A > 1) {
          let dmin = Infinity, nmin = Infinity;
          for (let b = 0; b < A; b++) {
            if (b === a) continue;
            const q = P[b], cq = centroids[b];
            dmin = Math.min(dmin, Math.hypot(cq[f * 3] - cen[f * 3], cq[f * 3 + 2] - cen[f * 3 + 2]));
            nmin = Math.min(nmin, Math.hypot(at(q, f, NOSE, 0) - at(p, f, NOSE, 0), at(q, f, NOSE, 1) - at(p, f, NOSE, 1), at(q, f, NOSE, 2) - at(p, f, NOSE, 2)));
          }
          col.nearest_distance[r] = dmin; col.nose_nose[r] = nmin;
        }
      }
    }
    const byState = (m) => Array.from({ length: K }, (_, c) => { const o = []; for (let r = 0; r < N; r++) if (rowState[r] === c + 1) o.push(col[m][r]); return o; });

    // ---- time budget
    const counts = new Array(K).fill(0), perAnimal = animals.map(() => new Array(K).fill(0));
    for (let r = 0; r < N; r++) { const c = rowState[r] - 1; if (c >= 0 && c < K) { counts[c]++; perAnimal[rowAnimal[r]][c]++; } }
    const share = counts.map((n) => n / N);
    const perAnimalShare = perAnimal.map((row) => { const s = rowSum(row) || 1; return row.map((n) => n / s); });

    // ---- bouts
    const bouts = [];
    animals.forEach((a, ai) => a.bouts.forEach((b) => bouts.push({ animal: ai, state: b.state, start: b.start, end: b.end, duration_s: b.duration_s })));
    const boutDur = Array.from({ length: K }, (_, c) => bouts.filter((b) => b.state === c + 1).map((b) => b.duration_s));
    const boutTable = boutDur.map((d) => (d.length ? {
      bouts: d.length, mean_s: meanOf(d), median_s: median(d), p90_s: percentile(d, 90), total_s: rowSum(d),
    } : null));

    // ---- transitions
    const T = Array.from({ length: K }, () => new Array(K).fill(0));
    LABELS.forEach((lab) => { for (let i = 0; i + 1 < lab.length; i++) if (lab[i + 1] !== lab[i]) T[lab[i] - 1][lab[i + 1] - 1]++; });
    const Tn = T.map((row) => { const s = rowSum(row); return row.map((v) => (s > 0 ? v / s : 0)); });
    const M = model.transition_matrix;
    const stay = M.map((row, i) => row[i]);
    const expectedBout = stay.map((p) => 1 / Math.max(1 - p, 1e-9) / fps);
    const nextState = Tn.map((row) => (rowSum(row) ? argmax(row) : -1));

    // ---- feature space
    const feat = cl.features, red = cl.reduction;
    const evr = red.explained_variance_ratio || [];
    const L = red.loadings || [], names = feat.feature_names || [];
    const npc = red.method === 'pca' ? red.n_components : L.length;
    let nodeImportance = [], topLoadings = [];
    if (L.length && names.length) {
      const imp = new Map();
      names.forEach((nm, j) => {
        let v = 0; for (let i = 0; i < L.length; i++) v += L[i][j] * L[i][j] * (evr[i] || 0);
        const parts = nm.split('/'), key = parts[0] === 'own' ? parts[1] : `${parts[1]} (other)`;
        imp.set(key, (imp.get(key) || 0) + v);
      });
      const tot = [...imp.values()].reduce((x, y) => x + y, 0) || 1;
      nodeImportance = [...imp.entries()].map(([k, v]) => ({ node: k, pct: (v / tot) * 100 })).sort((x, y) => y.pct - x.pct);
      for (let i = 0; i < Math.min(3, L.length); i++) {
        const order = names.map((_, j) => j).sort((x, y) => Math.abs(L[i][y]) - Math.abs(L[i][x])).slice(0, 6);
        topLoadings.push({ pc: i + 1, evr: evr[i], items: order.map((j) => ({ name: names[j], w: L[i][j] })) });
      }
    }

    // ---- PC space
    const emb = animals.map((a) => a.embedding.map((r) => r.map(Number)));
    const embDims = emb[0] && emb[0][0] ? emb[0][0].length : 0;
    const allE = [].concat(...emb), allL = [].concat(...LABELS.map((l) => Array.from(l)));
    const hasFlow = states.every((s) => s.A != null) && embDims >= 2;

    // ---- per-state medians and z-scores against all frames
    const allMed = {}, allStd = {};
    MEASURES.forEach((m) => { allMed[m] = median(col[m]); allStd[m] = stdOf(col[m]); });
    const med = [], z = [];
    for (let c = 0; c < K; c++) {
      med.push({}); z.push({});
      MEASURES.forEach((m) => {
        const v = []; for (let r = 0; r < N; r++) if (rowState[r] === c + 1) v.push(col[m][r]);
        med[c][m] = v.length ? median(v) : NaN;
        z[c][m] = allStd[m] > 0 ? (med[c][m] - allMed[m]) / allStd[m] : NaN;
      });
    }

    // ---- average egocentric pose per state: forward, lateral, height
    const avgPose = Array.from({ length: K }, () => ({ sum: new Float64Array(J * 3), n: 0 }));
    const egoAbs = [], egoH = [];
    for (let a = 0; a < A; a++) {
      const p = P[a], cen = centroids[a];
      for (let f = 0; f < nF; f++) {
        const th = Math.atan2(at(p, f, ANT, 2) - at(p, f, POST, 2), at(p, f, ANT, 0) - at(p, f, POST, 0));
        const cs = Math.cos(th), sn = Math.sin(th), c = LABELS[a][f] - 1, acc = avgPose[c];
        for (let j = 0; j < J; j++) {
          const dx = at(p, f, j, 0) - cen[f * 3], dz = at(p, f, j, 2) - cen[f * 3 + 2];
          const fw = dx * cs + dz * sn, lat = -dx * sn + dz * cs, h = at(p, f, j, 1);
          if (acc) { acc.sum[j * 3] += fw; acc.sum[j * 3 + 1] += lat; acc.sum[j * 3 + 2] += h; }
          egoAbs.push(Math.abs(fw), Math.abs(lat)); egoH.push(h);
        }
        if (acc) acc.n++;
      }
    }
    const meanPose = avgPose.map((acc) => (acc.n ? Array.from({ length: J }, (_, j) => [0, 1, 2].map((q) => acc.sum[j * 3 + q] / acc.n)) : null));
    const poseLim = percentile(egoAbs, 99) * 1.1, heightLim = dims === 3 ? percentile(egoH, 99.5) * 1.1 : 1;

    return {
      pose, cl, clip, fps, nF, K, A, J, dims, units, LEN, TO_DISPLAY, COLORS, STATE_NAMES, animals, states, model, feat, red,
      jointNames, bones, ANT, POST, NOSE, axisInfo, MEASURES, UNITS, col, rowAnimal, rowState, N, byState,
      share, perAnimalShare, bouts, boutDur, boutTable, T, Tn, M, stay, expectedBout, nextState,
      evr, npc, nodeImportance, topLoadings, emb, embDims, allE, allL, hasFlow, allMed, med, z, meanPose, poseLim, heightLim,
    };
  }

  // ---------- text ----------
  const MEASURE_NAMES = { accel: 'acceleration', nose_nose: 'nose-to-nose distance' };
  const pretty = (m) => MEASURE_NAMES[m] || m.replace(/_/g, ' ');
  const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
  const fmt = (v, d = 2) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(d)) : '-');
  const listJoin = (a) => (a.length <= 1 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);
  // The standard PDF fonts only carry Latin-1: map the few other symbols we use, drop the rest.
  const latin1 = (s) => String(s).replace(/[–—−]/g, '-').replace(/→/g, '->').replace(/≈/g, '~').replace(/≥/g, '>=').replace(/≤/g, '<=')
    .replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/…/g, '...').replace(/[^\x00-\xff]/g, '?');

  function stateSummary(R) {
    const order = R.share.map((_, c) => c).sort((x, y) => R.share[y] - R.share[x]);
    return order.map((c) => {
      const hi = R.MEASURES.filter((m) => R.z[c][m] >= 0.5).map(pretty), lo = R.MEASURES.filter((m) => R.z[c][m] <= -0.5).map(pretty);
      const desc = [];
      if (hi.length) desc.push(`higher ${listJoin(hi)}`);
      if (lo.length) desc.push(`lower ${listJoin(lo)}`);
      const bt = R.boutTable[c], nxt = R.nextState[c];
      return {
        c,
        text: `${pct(R.share[c])} of time` + (bt ? `, ${bt.bouts} bout${bt.bouts === 1 ? '' : 's'}, median ${bt.median_s.toFixed(2)} s` : '') +
          (desc.length ? `; ${desc.join(' and ')} than average` : '; close to average on every measure') +
          (nxt >= 0 ? `; usually followed by ${R.STATE_NAMES[nxt]}` : '') + '.',
      };
    });
  }

  // ---------- drawing primitives ----------
  const INK = [31, 35, 40], MUTED = [107, 114, 128], RULE = [205, 209, 214], ACCENT = [143, 181, 168], LIGHT = [200, 200, 200];
  function rgb(c) {
    if (Array.isArray(c)) return c;
    const s = String(c || '').trim();
    let m = /^#([0-9a-f]{6})$/i.exec(s);
    if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
    m = /^#([0-9a-f]{3})$/i.exec(s);
    if (m) return [0, 1, 2].map((i) => parseInt(m[1][i] + m[1][i], 16));
    m = /^rgba?\(([^)]+)\)$/i.exec(s);
    if (m) return m[1].split(',').slice(0, 3).map((v) => Math.round(parseFloat(v)));
    return [128, 128, 128];
  }
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const RDBU = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'].map(rgb);
  const rdbu = (t) => { t = Math.max(0, Math.min(1, t)) * (RDBU.length - 1); const i = Math.min(Math.floor(t), RDBU.length - 2); return mix(RDBU[i], RDBU[i + 1], t - i); };
  const greys = (t) => { const v = Math.round(255 - 255 * Math.max(0, Math.min(1, t))); return [v, v, v]; };
  const lum = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];

  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) { const d = Math.abs(lo) || 1; lo -= d / 2; hi += d / 2; }
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), r = raw / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  const tickLabel = (v, ticks) => {
    const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(v) || 1;
    const d = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
    const a = Math.abs(v);
    return a >= 1e5 ? `${+(v / 1000).toFixed(0)}k` : v.toFixed(d);
  };
  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of (Math.log10(hi / lo) < 2 ? [1, 2, 5] : [1])) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
    }
    return out;
  }

  function Pdf(jsPDF, opts) {
    const doc = new jsPDF({ unit: 'pt', format: 'letter', compress: true });
    const W = doc.internal.pageSize.getWidth(), H = doc.internal.pageSize.getHeight(), M = 54, CW = W - 2 * M;
    const g = { doc, W, H, M, CW, y: M };
    const alphaCache = new Map();
    g.alpha = (a) => {
      if (!doc.GState) return;
      if (!alphaCache.has(a)) alphaCache.set(a, new doc.GState({ opacity: a, 'stroke-opacity': a }));
      doc.setGState(alphaCache.get(a));
    };
    g.font = (size, style = 'normal', color = INK) => { doc.setFont('helvetica', style); doc.setFontSize(size); doc.setTextColor(...color); };
    g.text = (s, x, y, o) => doc.text(latin1(s), x, y, o);
    g.width = (s) => doc.getTextWidth(latin1(s));
    g.fill = (c) => doc.setFillColor(...rgb(c));
    g.stroke = (c, w = 0.6) => { doc.setDrawColor(...rgb(c)); doc.setLineWidth(w); };
    g.newPage = () => { doc.addPage(); g.y = M; };
    g.ensure = (h) => { if (g.y + h > H - M) g.newPage(); };
    g.space = (h) => { g.y += h; };
    // vertical text (rotated 90 degrees, reading upwards) centred on (x, yMid)
    g.vtext = (s, x, yMid) => g.text(s, x, yMid + g.width(s) / 2, { angle: 90 });
    g.polyline = (pts) => {
      if (pts.length < 2) return;
      doc.lines(pts.slice(1).map((p, i) => [p[0] - pts[i][0], p[1] - pts[i][1]]), pts[0][0], pts[0][1], [1, 1], 'S', false);
    };
    g.opts = opts;
    return g;
  }

  // flowing text --------------------------------------------------------------
  function heading(g, text, level = 1) {
    const size = level === 1 ? 14 : 11;
    g.ensure(level === 1 ? 160 : 110);
    if (level === 1 && g.y > g.M + 1) g.space(10);
    g.font(size, 'bold');
    g.text(text, g.M, g.y + size * 0.8);
    g.y += size * 1.25;
    if (level === 1) { g.stroke(RULE, 0.6); g.doc.line(g.M, g.y, g.M + g.CW, g.y); g.y += 7; } else g.y += 2;
  }
  function paragraph(g, text, { size = 9.5, color = INK, style = 'normal', indent = 0, after = 6, width } = {}) {
    g.font(size, style, color);
    const lines = g.doc.splitTextToSize(latin1(text), (width || g.CW) - indent), lh = size * 1.32;
    lines.forEach((ln) => { g.ensure(lh); g.doc.text(ln, g.M + indent, g.y + size * 0.8); g.y += lh; });
    g.y += after;
  }
  const note = (g, text) => paragraph(g, text, { size: 8.5, color: MUTED, after: 8 });
  function bullets(g, items, { size = 9.5 } = {}) {
    items.forEach((it) => {
      g.font(size); const lines = g.doc.splitTextToSize(latin1(it), g.CW - 12), lh = size * 1.32;
      g.ensure(lh * Math.min(lines.length, 2));
      g.fill(ACCENT); g.doc.circle(g.M + 3, g.y + size * 0.5, 1.6, 'F');
      g.font(size);
      lines.forEach((ln) => { g.ensure(lh); g.doc.text(ln, g.M + 12, g.y + size * 0.8); g.y += lh; });
      g.y += 2;
    });
    g.y += 4;
  }
  // a figure: reserve h points (new page if needed) and draw into the box
  function figure(g, h, draw) { g.ensure(h); draw({ x: g.M, y: g.y, w: g.CW, h }); g.y += h + 6; }

  // tables: columns [{label, w (fraction), align}], rows of cells (string or {text, color, bold})
  function table(g, columns, rows, { size = 8.5, swatchCol = -1 } = {}) {
    const lh = size * 1.6, total = columns.reduce((s, c) => s + c.w, 0), xs = [];
    let acc = g.M;
    columns.forEach((c) => { xs.push(acc); acc += (c.w / total) * g.CW; });
    const cellX = (i, align) => (align === 'right' ? xs[i] + (columns[i].w / total) * g.CW - 4 : xs[i] + 4);
    const header = () => {
      g.font(size, 'bold', MUTED);
      columns.forEach((c, i) => g.text(c.label, cellX(i, c.align), g.y + lh * 0.68, { align: c.align === 'right' ? 'right' : 'left' }));
      g.y += lh; g.stroke(RULE, 0.6); g.doc.line(g.M, g.y, g.M + g.CW, g.y);
    };
    g.ensure(lh * Math.min(rows.length + 1, 4)); header();
    rows.forEach((row) => {
      if (g.y + lh > g.H - g.M) { g.newPage(); header(); }
      row.forEach((cell, i) => {
        const c = typeof cell === 'object' && cell ? cell : { text: cell };
        let x = cellX(i, columns[i].align);
        if (i === swatchCol && c.color) { g.fill(c.color); g.doc.rect(x, g.y + lh * 0.3, 7, 7, 'F'); x += 11; }
        g.font(size, c.bold ? 'bold' : 'normal', c.muted ? MUTED : INK);
        const maxW = (columns[i].w / total) * g.CW - 8 - (i === swatchCol ? 11 : 0);
        let t = latin1(c.text == null ? '' : c.text);
        if (g.doc.getTextWidth(t) > maxW) {      // truncate with an ellipsis
          let base = t;
          while (base.length > 1 && g.doc.getTextWidth(`${base}...`) > maxW) base = base.slice(0, -1);
          t = `${base}...`;
        }
        g.doc.text(t, x, g.y + lh * 0.68, { align: columns[i].align === 'right' ? 'right' : 'left' });
      });
      g.y += lh; g.stroke([234, 236, 239], 0.4); g.doc.line(g.M, g.y, g.M + g.CW, g.y);
    });
    g.y += 10;
  }

  // axes ------------------------------------------------------------------------
  // box {x,y,w,h}; o: xlim, ylim, xlog, ylog, title, titleColor, xlabel, ylabel, xcats (category names),
  // ycats, noXTicks, noYTicks, padL, padB, rotateX
  function axes(g, box, o) {
    const doc = g.doc, padT = o.title ? 14 : 4, padR = o.padR != null ? o.padR : 6;
    let rot = 0;
    if (o.rotateX && o.xcats) { g.font(7); rot = Math.max(...o.xcats.map((nm) => g.width(nm))) - 6; }
    const padB = o.padB != null ? o.padB : (o.noXTicks ? 4 : 14) + (o.xlabel ? 11 : 0) + rot;
    const padL = o.padL != null ? o.padL : (o.noYTicks ? 4 : 30) + (o.ylabel ? 10 : 0);
    const px = box.x + padL, py = box.y + padT, pw = Math.max(10, box.w - padL - padR), ph = Math.max(10, box.h - padT - padB);
    const tx = o.xlog ? Math.log10 : (v) => v, ty = o.ylog ? Math.log10 : (v) => v;
    const [x0, x1] = o.xlim, [y0, y1] = o.ylim;
    const X = (v) => px + ((tx(v) - tx(x0)) / (tx(x1) - tx(x0) || 1)) * pw;
    const Y = (v) => py + ph - ((ty(v) - ty(y0)) / (ty(y1) - ty(y0) || 1)) * ph;
    const ax = { X, Y, px, py, pw, ph, box };
    ax.frame = () => {
      g.stroke([90, 90, 90], 0.6);
      doc.line(px, py + ph, px + pw, py + ph);
      doc.line(px, py, px, py + ph);
      g.font(7, 'normal', MUTED);
      if (!o.noXTicks) {
        if (o.xcats) {
          o.xcats.forEach((nm, i) => {
            const x = X(i + 0.5);
            if (o.rotateX) g.text(nm, x + 2.5, py + ph + 4 + g.width(nm), { angle: 90 });
            else g.text(nm, x, py + ph + 9, { align: 'center' });
          });
        } else {
          const ticks = o.xticks || (o.xlog ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.round(pw / 55))));
          ticks.filter((v) => v >= Math.min(x0, x1) - 1e-9 && v <= Math.max(x0, x1) + 1e-9).forEach((v) => {
            g.stroke([90, 90, 90], 0.5); doc.line(X(v), py + ph, X(v), py + ph + 2.5);
            g.text(o.xfmt ? o.xfmt(v) : tickLabel(v, ticks), X(v), py + ph + 9, { align: 'center' });
          });
        }
      }
      if (!o.noYTicks) {
        if (o.ycats) o.ycats.forEach((nm, i) => g.text(nm, px - 3, Y(i + 0.5) + 2.4, { align: 'right' }));
        else {
          const ticks = o.yticks || (o.ylog ? logTicks(y0, y1) : niceTicks(y0, y1, Math.max(3, Math.round(ph / 30))));
          ticks.filter((v) => v >= Math.min(y0, y1) - 1e-9 && v <= Math.max(y0, y1) + 1e-9).forEach((v) => {
            g.stroke([90, 90, 90], 0.5); doc.line(px - 2.5, Y(v), px, Y(v));
            g.text(o.yfmt ? o.yfmt(v) : tickLabel(v, ticks), px - 4, Y(v) + 2.4, { align: 'right' });
          });
        }
      }
      g.font(7.5, 'normal', INK);
      if (o.xlabel) g.text(o.xlabel, px + pw / 2, box.y + box.h - 2, { align: 'center' });
      if (o.ylabel) g.vtext(o.ylabel, box.x + 7, py + ph / 2);
      if (o.title) { g.font(8.5, 'bold', o.titleColor ? rgb(o.titleColor) : INK); g.text(o.title, px + pw / 2, box.y + 9, { align: 'center' }); }
    };
    ax.clip = (fn) => { doc.saveGraphicsState(); doc.rect(px, py, pw, ph, null); doc.clip(); doc.discardPath(); fn(); doc.restoreGraphicsState(); };
    return ax;
  }

  function legendRow(g, x, y, names, colors, { size = 7.5, maxW } = {}) {
    let cx = x, cy = y;
    g.font(size);
    names.forEach((nm, i) => {
      const w = 10 + g.width(nm) + 10;
      if (maxW && cx + w > x + maxW) { cx = x; cy += size * 1.6; }
      g.fill(colors[i]); g.doc.rect(cx, cy - size * 0.7, 7, 7, 'F');
      g.font(size); g.text(nm, cx + 10, cy);
      cx += w;
    });
    return cy;
  }

  function colorbar(g, x, y, w, h, cmap, lo, hi, label, ticks) {
    const n = 40;
    for (let i = 0; i < n; i++) { g.fill(cmap(1 - i / (n - 1))); g.doc.rect(x, y + (i * h) / n, w, h / n + 0.3, 'F'); }
    g.stroke([90, 90, 90], 0.4); g.doc.rect(x, y, w, h, 'S');
    g.font(6.5, 'normal', MUTED);
    (ticks || [lo, (lo + hi) / 2, hi]).forEach((v) => { const yy = y + h - ((v - lo) / (hi - lo)) * h; g.text(+v.toFixed(2) === 0 ? '0' : String(+v.toFixed(2)), x + w + 2, yy + 2.2); });
    if (label) { g.font(6.5, 'normal', INK); g.vtext(label, x + w + 22, y + h / 2); }
  }

  function boxplotPanel(g, box, R, groups, o) {
    const K = groups.length, stats = groups.map(boxStats);
    let lo = Infinity, hi = -Infinity;
    stats.forEach((s) => { if (!s) return; const v = o.fliers ? [s.lo, s.hi, ...s.fliers] : [s.lo, s.hi]; v.forEach((q) => { lo = Math.min(lo, q); hi = Math.max(hi, q); }); });
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    let ylim;
    if (o.ylog) { lo = Math.max(lo, 1e-3); ylim = [lo / 1.4, hi * 1.4]; } else { const pad = (hi - lo) * 0.06 || 1; ylim = [lo - pad, hi + pad]; }
    const ax = axes(g, box, { xlim: [0, K], ylim, ylog: o.ylog, xcats: R.STATE_NAMES, rotateX: K > 8, title: o.title, ylabel: o.ylabel, padL: o.padL });
    const bw = Math.min(18, (ax.pw / K) * 0.6), doc = g.doc;
    stats.forEach((s, c) => {
      if (!s) return;
      const xm = ax.X(c + 0.5), col = rgb(R.COLORS[c]);
      g.stroke([60, 60, 60], 0.5);
      doc.line(xm, ax.Y(s.lo), xm, ax.Y(s.q1)); doc.line(xm, ax.Y(s.q3), xm, ax.Y(s.hi));
      doc.line(xm - bw / 4, ax.Y(s.lo), xm + bw / 4, ax.Y(s.lo)); doc.line(xm - bw / 4, ax.Y(s.hi), xm + bw / 4, ax.Y(s.hi));
      g.fill(mix(col, [255, 255, 255], 0.2));
      doc.rect(xm - bw / 2, ax.Y(s.q3), bw, Math.max(0.6, ax.Y(s.q1) - ax.Y(s.q3)), 'FD');
      g.stroke([0, 0, 0], 1); doc.line(xm - bw / 2, ax.Y(s.med), xm + bw / 2, ax.Y(s.med));
      if (o.fliers) { g.stroke([90, 90, 90], 0.4); s.fliers.forEach((v) => doc.circle(xm, ax.Y(v), 1.3, 'S')); }
    });
    ax.frame();
    return ax;
  }

  function heatmapPanel(g, box, mat, o) {
    const rows = mat.length, cols = mat[0].length;
    const ax = axes(g, box, { xlim: [0, cols], ylim: [rows, 0], xcats: o.xcats, ycats: o.ycats, rotateX: o.rotateX, title: o.title, xlabel: o.xlabel, ylabel: o.ylabel, padL: o.padL, padR: o.padR });
    // y axis runs downwards: Y(i) is the top of row i
    const Ycat = (i) => ax.py + (i / rows) * ax.ph;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = mat[i][j], c = Number.isFinite(v) ? o.cmap(v) : [235, 235, 235];
      g.fill(c); g.doc.rect(ax.X(j), Ycat(i), ax.pw / cols + 0.2, ax.ph / rows + 0.2, 'F');
      const label = o.annot && o.annot(v, i, j);
      if (label) {
        g.font(Math.min(7, (ax.pw / cols) / 3.2), 'normal', lum(c) < 120 ? [255, 255, 255] : INK);
        g.text(label, ax.X(j + 0.5), Ycat(i + 0.5) + 2.3, { align: 'center' });
      }
    }
    ax.frame();
    return ax;
  }

  // ---------- report ----------
  function build(pose, cl, options = {}) {
    const jsPDF = options.jsPDF || (root.jspdf && root.jspdf.jsPDF);
    if (!jsPDF) throw new Error('jsPDF is not loaded');
    const R = analyze(pose, cl);
    const g = Pdf(jsPDF, options), doc = g.doc;
    const { K, A, fps, nF, LEN, STATE_NAMES: SN, COLORS } = R;
    const dsName = (pose.meta && pose.meta.name) || R.cl.source.dataset || 'dataset';
    const clipName = R.clip.name || `clip ${R.cl.source.clip.index}`;
    const summary = stateSummary(R);
    const order = summary.map((s) => s.c);
    const top = order[0], rare = order[order.length - 1];
    const anyStd = (m) => R.MEASURES.includes(m);

    // ===== title page =====
    g.font(20, 'bold'); g.text('Behaviour summary', g.M, g.y + 18); g.y += 28;
    g.font(12, 'normal', MUTED); g.text(`${dsName}${pose.clips.length > 1 ? `  ·  clip "${clipName}"` : ''}`, g.M, g.y + 10); g.y += 18;
    g.font(8.5, 'normal', MUTED);
    g.text(`Generated ${options.date || new Date().toISOString().slice(0, 10)} from the Pose Dataset Viewer's clustering export${R.cl.created ? ` (${R.cl.created.slice(0, 16).replace('T', ' ')} UTC)` : ''}.`, g.M, g.y + 8);
    g.y += 20;
    if (pose.meta && pose.meta.description) paragraph(g, pose.meta.description, { size: 9, color: MUTED, after: 8 });

    heading(g, 'At a glance');
    const glance = [
      `${nF} frames at ${+fps.toFixed(3)} fps (${(nF / fps).toFixed(1)} s) of ${A} animal${A > 1 ? 's' : ''} with ${R.J} body nodes each (${R.dims}D).`,
      `A switching linear dynamical model split the behaviour into ${K} states` +
        (R.model.k_auto != null ? (R.model.k_auto === R.model.k ? ', the number BIC favoured.' : `; BIC favoured ${R.model.k_auto}, ${K} was chosen in the viewer.`) : '.'),
      `${SN[top]} is the most common state (${pct(R.share[top])} of frames); ${SN[rare]} the rarest (${pct(R.share[rare])}).`,
    ];
    const longest = R.boutTable.map((b, c) => (b ? [b.median_s, c] : [-1, c])).sort((x, y) => y[0] - x[0])[0];
    glance.push(`${R.bouts.length} bouts in total, ${(R.bouts.length / A / (nF / fps / 60)).toFixed(1)} per animal per minute; ${SN[longest[1]]} has the longest typical bout (median ${longest[0].toFixed(2)} s).`);
    if (A > 1 && anyStd('nearest_distance')) {
      const close = R.med.map((m, c) => [m.nearest_distance, c]).filter((v) => Number.isFinite(v[0])).sort((x, y) => x[0] - y[0])[0];
      if (close) glance.push(`Animals are closest in ${SN[close[1]]} (median ${fmt(close[0], 1)} ${LEN} between centroids, against ${fmt(R.allMed.nearest_distance, 1)} ${LEN} overall).`);
    }
    if (R.evr.length && R.red.method === 'pca') {
      const cum = R.red.cumulative_variance ? R.red.cumulative_variance[R.npc - 1] : R.evr.slice(0, R.npc).reduce((x, y) => x + y, 0);
      glance.push(`${R.npc} principal components carry ${pct(cum)} of the feature variance` + (R.nodeImportance.length ? `; ${listJoin(R.nodeImportance.slice(0, 3).map((n) => n.node))} weigh most in them.` : '.'));
    }
    bullets(g, glance);

    heading(g, 'What was clustered', 2);
    const feat = R.feat, red = R.red, hp = R.model.hyperparameters || {};
    const info = [
      ['Feature set', feat.mode], ['Features per sample', String((feat.feature_names || []).length)],
      ['Removed nodes', (feat.removed_nodes || []).join(', ') || 'none'],
      ['Egocentric axis', `${R.axisInfo.posterior} -> ${R.axisInfo.anterior}`],
    ];
    if (feat.mode === 'social') {
      if (feat.align_others) info.push(['Other animals rotated onto the body axis (older option)', 'yes']);
      const k = feat.proximity_kernel;
      const side = k && k.lateral_scale != null && k.lateral_scale < 1 ? k.lateral_scale : 1;
      info.push(['Proximity kernel', !k ? 'off' : side < 1
        ? `elliptical Gaussian field: d0 = ${k.d0_body_lengths} body lengths (${(k.d0 * R.TO_DISPLAY).toFixed(1)} ${LEN}) ahead and behind, ${+(side * k.d0_body_lengths).toFixed(2)} to the sides${k.dims ? `, ${k.dims}D` : ''}`
        : `${k.dims ? `isotropic ${k.dims}D ` : ''}Gaussian on centroid-to-centroid distance, d0 = ${k.d0_body_lengths} body lengths (${(k.d0 * R.TO_DISPLAY).toFixed(1)} ${LEN})`]);
    }
    if (feat.mode === 'social' && feat.others_scale != null && feat.others_scale !== 1) info.push(['Others vs own', feat.others_balance ? `balanced 50/50 in contact (others x${feat.others_scale})` : `other animals weighted x${feat.others_scale} relative to the own pose`]);
    const pp = R.cl.preprocessing;
    if (pp) info.push(['Pose clean-up', [pp.swaps ? 'L/R swaps' : null, pp.outliers ? `outliers (w ${pp.outliers.window}, t ${pp.outliers.threshold})` : null,
      `${pp.fill === 'body' ? 'body-relative' : pp.fill} fill`, pp.smooth ? `${pp.smooth.method === 'gaussian' ? 'Gaussian' : 'median'} ${pp.smooth.size} f` : null].filter(Boolean).join(' · ')]);
    info.push(['Reduction', red.method === 'pca' ? `PCA, ${red.n_components} PCs = ${pct(red.cumulative_variance[red.n_components - 1])} of variance` : `none (model on ${red.n_components} raw features)`]);
    info.push(['Model', `switching linear dynamics, k = ${R.model.k} (best by BIC: k = ${R.model.k_auto})`]);
    if (hp.k_min != null) info.push(['k range searched', `${hp.k_min}-${hp.k_max}`]);
    if (hp.stickiness != null) info.push(['Stickiness', String(hp.stickiness)]);
    if (hp.max_em_iterations != null) info.push(['EM iterations (maximum)', String(hp.max_em_iterations)]);
    const opt = hp.optimized;
    if (opt) info.push(['Chosen by Optimize rsLDS', opt.one_frame_bouts != null
      ? `${opt.searched === 'proximity field' ? `field reach ${opt.reach_body_lengths}, side ${opt.side_reach}` : `stickiness ${opt.stickiness}, ${opt.iterations} iter.`}: ${opt.one_frame_bouts} one-frame bouts, best of ${opt.settings_tried}`
      : `${opt.stable_states} stable states (mean stability ${opt.stability}), best of ${opt.settings_tried} settings`]);
    if (R.model.log_likelihood_per_step != null) info.push(['Log-likelihood per step', R.model.log_likelihood_per_step.toFixed(3)]);
    table(g, [{ label: 'Setting', w: 0.42 }, { label: 'Value', w: 0.58 }], info);

    const ksel = R.model.k_selection || [];
    if (ksel.length > 1) {
      heading(g, 'Model selection', 2);
      paragraph(g, 'Every k in the searched range was fit; BIC (lower is better) trades the fit against the number of parameters. The circled point is the k used for everything in this report.');
      figure(g, 150, (b) => {
        const ks = ksel.map((s) => s.k), bic = ksel.map((s) => s.bic);
        const lo = Math.min(...bic), hi = Math.max(...bic), pad = (hi - lo) * 0.08 || 1;
        const ax = axes(g, { x: b.x, y: b.y, w: b.w * 0.62, h: b.h }, { xlim: [ks[0] - 0.5, ks[ks.length - 1] + 0.5], ylim: [lo - pad, hi + pad], xticks: ks, xlabel: 'k (number of states)', ylabel: 'BIC', padL: 50 });
        g.stroke(ACCENT, 1.2); g.polyline(ks.map((k, i) => [ax.X(k), ax.Y(bic[i])]));
        g.fill(ACCENT); ks.forEach((k, i) => doc.circle(ax.X(k), ax.Y(bic[i]), 2, 'F'));
        const i = ks.indexOf(R.model.k);
        if (i >= 0) { g.stroke([0, 0, 0], 1.2); doc.circle(ax.X(ks[i]), ax.Y(bic[i]), 4.5, 'S'); }
        ax.frame();
      });
    }

    // ===== ethogram =====
    g.newPage();
    heading(g, '1. Ethogram');
    const switches = R.animals.map((a) => a.bouts.length - 1);
    paragraph(g, `The state of each animal over time. Each colour is one state; ${A > 1 ? 'the animals switch' : 'the animal switches'} state ` +
      `${(switches.reduce((x, y) => x + y, 0) / A / (nF / fps / 60)).toFixed(1)} times per minute on average.`);
    figure(g, 18 * A + 50, (b) => {
      const ax = axes(g, { x: b.x, y: b.y + 22, w: b.w, h: b.h - 22 }, { xlim: [0, nF / fps], ylim: [A, 0], xlabel: 'time (s)', noYTicks: true, padL: 58 });
      R.animals.forEach((an, a) => {
        an.bouts.forEach((bt) => {
          g.fill(COLORS[bt.state - 1]);
          doc.rect(ax.X(bt.start / fps), ax.py + (a / A) * ax.ph + 1, Math.max(0.25, ax.X((bt.end + 1) / fps) - ax.X(bt.start / fps)), ax.ph / A - 2, 'F');
        });
        g.font(7.5); g.text(an.name, ax.px - 4, ax.py + ((a + 0.5) / A) * ax.ph + 2.5, { align: 'right' });
      });
      ax.frame();
      legendRow(g, ax.px, b.y + 10, SN, COLORS, { maxW: ax.pw });
    });

    // ===== time budget =====
    heading(g, '2. Time budget');
    let budgetText = `Share of frames in each state, overall and per animal. ${SN[top]} takes ${pct(R.share[top])} of the time` +
      (order.length > 1 ? ` and ${SN[order[1]]} ${pct(R.share[order[1]])}` : '') + '.';
    if (A > 1) {
      const diffs = R.share.map((_, c) => { const v = R.perAnimalShare.map((r) => r[c]); return [Math.max(...v) - Math.min(...v), c, v]; }).sort((x, y) => y[0] - x[0]);
      const [d, c, v] = diffs[0];
      if (d >= 0.05) {
        const hiA = v.indexOf(Math.max(...v)), loA = v.indexOf(Math.min(...v));
        budgetText += ` The animals differ most in ${SN[c]}: ${R.animals[hiA].name} spends ${pct(v[hiA])} of its time there, ${R.animals[loA].name} ${pct(v[loA])}.`;
      } else budgetText += ' The animals divide their time between the states in much the same way (no state differs by 5 points or more).';
    }
    paragraph(g, budgetText);
    figure(g, 46 + Math.max(110, 14 * A), (b) => {
      const w0 = b.w * 0.4;
      const ax = axes(g, { x: b.x, y: b.y, w: w0, h: b.h }, { xlim: [0, K], ylim: [0, Math.max(...R.share) * 100 * 1.12], xcats: SN, rotateX: K > 8, ylabel: '% of frames', title: 'Overall' });
      R.share.forEach((s, c) => { g.fill(COLORS[c]); const bw = (ax.pw / K) * 0.7; doc.rect(ax.X(c + 0.5) - bw / 2, ax.Y(s * 100), bw, ax.Y(0) - ax.Y(s * 100), 'F'); });
      ax.frame();
      const nameW = Math.max(...R.animals.map((a) => { g.font(7); return g.width(a.name); })) + 10;
      const ax2 = axes(g, { x: b.x + w0 + 16, y: b.y, w: b.w - w0 - 16, h: b.h }, { xlim: [0, 100], ylim: [A, 0], ycats: R.animals.map((a) => a.name), xlabel: '% of frames', title: 'Per animal', padL: nameW });
      R.perAnimalShare.forEach((row, a) => {
        let left = 0;
        const y0 = ax2.py + (a / A) * ax2.ph, hh = ax2.ph / A;
        row.forEach((s, c) => { g.fill(COLORS[c]); doc.rect(ax2.X(left * 100), y0 + hh * 0.15, ax2.X((left + s) * 100) - ax2.X(left * 100), hh * 0.7, 'F'); left += s; });
      });
      ax2.frame();
    });
    if (A > 1) {
      table(g, [{ label: 'Animal', w: 1.6 }, ...SN.map((n) => ({ label: n, w: 1, align: 'right' }))],
        R.animals.map((a, i) => [a.name, ...R.perAnimalShare[i].map((s) => `${(s * 100).toFixed(1)}%`)]));
    }

    // ===== bouts =====
    heading(g, '3. Bouts');
    const allDur = R.bouts.map((b) => b.duration_s);
    const shortest = R.boutTable.map((b, c) => (b ? [b.median_s, c] : [Infinity, c])).sort((x, y) => x[0] - y[0])[0];
    const frame = 1 / fps;
    const flicker = R.boutTable.map((b, c) => (b && b.median_s <= 1.5 * frame ? SN[c] : null)).filter(Boolean);
    paragraph(g, `A bout is an unbroken run of one state in one animal's sequence. Across all ${R.bouts.length} bouts the median lasts ${median(allDur).toFixed(2)} s; ` +
      `${SN[longest[1]]} has the longest bouts (median ${longest[0].toFixed(2)} s) and ${SN[shortest[1]]} the shortest (median ${shortest[0].toFixed(2)} s).` +
      (flicker.length ? ` ${listJoin(flicker)} mostly last${flicker.length === 1 ? 's' : ''} a single frame: ${flicker.length === 1 ? 'a transitional state' : 'transitional states'} rather than sustained behaviour.` : ''));
    figure(g, 170, (b) => boxplotPanel(g, { x: b.x, y: b.y, w: b.w * 0.75, h: b.h }, R, R.boutDur, { ylog: true, fliers: true, ylabel: 'bout duration (s, log scale)', title: 'Bout durations by state', padL: 44 }));
    table(g, [{ label: 'State', w: 1 }, { label: 'Bouts', w: 1, align: 'right' }, { label: 'Mean (s)', w: 1, align: 'right' }, { label: 'Median (s)', w: 1, align: 'right' },
      { label: '90th pct (s)', w: 1, align: 'right' }, { label: 'Total (s)', w: 1, align: 'right' }],
    R.boutTable.map((bt, c) => (bt ? [{ text: SN[c], color: COLORS[c] }, String(bt.bouts), bt.mean_s.toFixed(2), bt.median_s.toFixed(2), bt.p90_s.toFixed(2), bt.total_s.toFixed(1)] : null)).filter(Boolean),
    { swatchCol: 0 });

    // ===== transitions =====
    g.ensure(380);
    heading(g, '4. Transitions');
    const pairs = [];
    R.Tn.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v > 0) pairs.push([v, i, j]); }));
    pairs.sort((x, y) => y[0] - x[0]);
    const sticky = R.stay.map((p, c) => [p, c]).sort((x, y) => y[0] - x[0]);
    paragraph(g, 'Left: what follows each bout (rows = from, columns = to; each row sums to 1). Right: the model\'s frame-to-frame transition matrix; its diagonal is the probability of staying in a state for one more frame, so the expected bout lasts about 1 / (1 - p) frames.');
    if (pairs.length) {
      paragraph(g, `The most reliable sequences are ${listJoin(pairs.slice(0, 3).map(([v, i, j]) => `${SN[i]} -> ${SN[j]} (${pct(v, 0)} of ${SN[i]} bouts)`))}. ` +
        `${SN[sticky[0][1]]} is the most persistent state in the model (stay probability ${sticky[0][0].toFixed(3)}, about ${R.expectedBout[sticky[0][1]].toFixed(2)} s per bout).`);
    }
    const hmSize = Math.min(g.CW / 2 - 10, 230);
    figure(g, hmSize, (b) => {
      const annot = (v) => (K <= 10 && v >= 0.05 ? v.toFixed(2) : '');
      [[R.Tn, 'Bout-to-bout transitions'], [R.M, 'Model transition matrix (per frame)']].forEach(([mat, title], k) => {
        const mx = Math.max(1e-9, ...mat.map((r) => Math.max(...r)));
        const x = b.x + k * (b.w / 2);
        heatmapPanel(g, { x, y: b.y, w: b.w / 2 - 34, h: b.h }, mat, { cmap: (v) => greys((v / mx) * 0.92), xcats: SN, ycats: SN, rotateX: K > 6, title, xlabel: 'to', ylabel: 'from', annot, padL: 34 });
        colorbar(g, x + b.w / 2 - 30, b.y + 14, 6, b.h - 46, (t) => greys(t * 0.92), 0, mx, null);
      });
    });
    table(g, [{ label: 'State', w: 1 }, { label: 'Stay probability', w: 1.3, align: 'right' }, { label: 'Expected bout (s)', w: 1.3, align: 'right' }, { label: 'Most common next state', w: 1.6, align: 'right' }],
      SN.map((nm, c) => [{ text: nm, color: COLORS[c] }, R.stay[c].toFixed(3), R.expectedBout[c].toFixed(2), R.nextState[c] >= 0 ? SN[R.nextState[c]] : '-']), { swatchCol: 0 });

    // ===== feature space =====
    if (R.evr.length) {
      g.ensure(330);
      heading(g, '5. The feature space');
      let fsText = 'How much variance each principal component carries, and which body parts drive the kept components. Node importance sums each node\'s squared loadings over its axes and the kept PCs, weighted by each PC\'s share of the variance.';
      if (R.nodeImportance.length) {
        const own = R.nodeImportance.filter((n) => !/\(other\)$/.test(n.node)).reduce((s, n) => s + n.pct, 0);
        fsText += ` PC1 alone explains ${pct(R.evr[0])}. The leading nodes are ${listJoin(R.nodeImportance.slice(0, 3).map((n) => `${n.node} (${n.pct.toFixed(0)}%)`))}.`;
        if (feat.mode === 'social') fsText += ` The animal's own body accounts for ${own.toFixed(0)}% of the weighted loading and the other animal${A > 2 ? 's' : ''} for ${(100 - own).toFixed(0)}%.`;
      }
      paragraph(g, fsText);
      figure(g, 180, (b) => {
        const shown = Math.min(R.evr.length, Math.max(R.npc + 3, 8)), w0 = b.w * 0.45;
        let cum = 0; const cums = R.evr.slice(0, shown).map((v) => (cum += v) * 100);
        const ax = axes(g, { x: b.x, y: b.y, w: w0, h: b.h }, { xlim: [0.4, shown + 0.6], ylim: [0, 105], xticks: Array.from({ length: shown }, (_, i) => i + 1), xlabel: 'PC', ylabel: '% variance', title: 'Variance explained (kept PCs in green)' });
        R.evr.slice(0, shown).forEach((v, i) => { g.fill(i < R.npc ? ACCENT : LIGHT); doc.rect(ax.X(i + 1) - 0.35 * (ax.pw / shown), ax.Y(v * 100), 0.7 * (ax.pw / shown), ax.Y(0) - ax.Y(v * 100), 'F'); });
        g.stroke([0, 0, 0], 0.9); g.polyline(cums.map((v, i) => [ax.X(i + 1), ax.Y(v)]));
        g.fill([0, 0, 0]); cums.forEach((v, i) => doc.circle(ax.X(i + 1), ax.Y(v), 1.5, 'F'));
        ax.frame();
        if (R.nodeImportance.length) {
          const items = R.nodeImportance.slice(0, 15), n = items.length;
          g.font(7); const nameW = Math.max(...items.map((it) => g.width(it.node))) + 8;
          const ax2 = axes(g, { x: b.x + w0 + 20, y: b.y, w: b.w - w0 - 20, h: b.h }, { xlim: [0, Math.max(...items.map((i) => i.pct)) * 1.1], ylim: [n, 0], ycats: items.map((i) => i.node), xlabel: '% of weighted loading', title: 'Node importance in the kept PCs', padL: nameW });
          items.forEach((it, i) => { g.fill(/\(other\)$/.test(it.node) ? mix(ACCENT, [70, 90, 140], 0.5) : ACCENT); doc.rect(ax2.X(0), ax2.py + (i / n) * ax2.ph + 1, ax2.X(it.pct) - ax2.X(0), ax2.ph / n - 2, 'F'); });
          ax2.frame();
        }
      });
      if (R.topLoadings.length) {
        paragraph(g, 'Strongest loadings of the first principal components (feature: weight). A feature is animal/node/axis in the animal\'s own frame; "other" features describe the other animal.', { size: 8.5, color: MUTED });
        const rows = Array.from({ length: 6 }, (_, r) => R.topLoadings.map((t) => (t.items[r] ? `${t.items[r].name} (${t.items[r].w >= 0 ? '+' : ''}${t.items[r].w.toFixed(2)})` : '')));
        table(g, R.topLoadings.map((t) => ({ label: `PC${t.pc} (${(t.evr * 100).toFixed(0)}%)`, w: 1 })), rows, { size: 7.5 });
      }
    }

    // ===== PC space =====
    if (R.embDims >= 1) {
      g.ensure(400);
      heading(g, '6. States in PC space');
      paragraph(g, 'Every frame in the first two principal components, coloured by state; then one panel per state with its flow field (the step its linear dynamics predicts from each point, other PCs held at the state\'s mean) and its 15 longest bouts as trajectories. The circled cross marks the field\'s fixed point, where its trajectories converge or circle.');
      const E = R.allE, Lb = R.allL, nPts = E.length, stride = Math.max(1, Math.ceil(nPts / 6000));
      const ex = (i) => E[i][0], ey = (i) => (R.embDims > 1 ? E[i][1] : 0);
      const xs = E.map((e) => e[0]), ys = E.map((e) => (R.embDims > 1 ? e[1] : 0));
      let x0 = percentile(xs, 0.5), x1 = percentile(xs, 99.5), y0 = percentile(ys, 0.5), y1 = percentile(ys, 99.5);
      const pxp = 0.08 * (x1 - x0 || 1), pyp = 0.08 * (y1 - y0 || 1);
      x0 -= pxp; x1 += pxp; y0 -= pyp; y1 += pyp;
      const ev = (i) => (R.evr[i] != null ? ` (${(R.evr[i] * 100).toFixed(0)}%)` : '');
      figure(g, 250, (b) => {
        const ax = axes(g, { x: b.x, y: b.y, w: b.w * 0.62, h: b.h }, { xlim: [x0, x1], ylim: [y0, y1], xlabel: `PC1${ev(0)}`, ylabel: R.embDims > 1 ? `PC2${ev(1)}` : '', title: 'Frames in PC space' });
        ax.clip(() => {
          g.alpha(0.7);
          for (let c = 0; c < K; c++) {
            g.fill(COLORS[c]);
            for (let i = 0; i < nPts; i += stride) if (Lb[i] === c + 1) doc.circle(ax.X(ex(i)), ax.Y(ey(i)), 1.1, 'F');
          }
          g.alpha(1);
        });
        ax.frame();
        let ly = b.y + 24;
        g.font(7.5);
        SN.forEach((nm, c) => { g.fill(COLORS[c]); doc.rect(b.x + b.w * 0.66, ly - 6, 7, 7, 'F'); g.font(7.5); g.text(`${nm}  ${pct(R.share[c], 0)}`, b.x + b.w * 0.66 + 11, ly); ly += 12; });
        if (stride > 1) { g.font(6.5, 'normal', MUTED); g.text(`every ${stride}th frame shown`, b.x + b.w * 0.66, ly + 4); }
      });

      if (R.embDims >= 2) {
        const cols = K > 4 ? 3 : 2, rowsN = Math.ceil(K / cols), pw = (g.CW - (cols - 1) * 8) / cols, ph = pw * 0.82;
        const m = R.hasFlow ? R.states[0].A.length : 0, gx = 11, gy = 9;
        const longestOf = (c) => R.bouts.filter((bt) => bt.state === c + 1 && bt.end > bt.start).sort((p, q) => (q.end - q.start) - (p.end - p.start)).slice(0, 15);
        const bgStride = Math.max(1, Math.ceil(nPts / 3000));
        for (let r = 0; r < rowsN; r++) {
          figure(g, ph, (b) => {
            for (let k = 0; k < cols; k++) {
              const c = r * cols + k;
              if (c >= K) break;
              const ax = axes(g, { x: b.x + k * (pw + 8), y: b.y, w: pw, h: ph }, { xlim: [x0, x1], ylim: [y0, y1], noXTicks: true, noYTicks: true, title: `${SN[c]} · ${pct(R.share[c], 0)} of time`, titleColor: COLORS[c] });
              ax.clip(() => {
                g.alpha(0.35); g.fill(LIGHT);
                for (let i = 0; i < nPts; i += bgStride) doc.rect(ax.X(ex(i)) - 0.55, ax.Y(ey(i)) - 0.55, 1.1, 1.1, 'F');
                g.alpha(1);
                const members = []; for (let i = 0; i < nPts; i++) if (Lb[i] === c + 1) members.push(i);
                const st = R.states[c];
                if (R.hasFlow && members.length && m >= 2) {
                  const mu = Array.from({ length: m }, (_, q) => members.reduce((s, i) => s + (E[i][q] || 0), 0) / members.length);
                  const vecs = [];
                  for (let iy = 0; iy < gy; iy++) for (let ix = 0; ix < gx; ix++) {
                    const z = mu.slice(); z[0] = x0 + ((x1 - x0) * ix) / (gx - 1); z[1] = y0 + ((y1 - y0) * iy) / (gy - 1);
                    const v = [0, 1].map((q) => st.A[q].reduce((s, a, j) => s + a * z[j], 0) + st.b[q] - z[q]);
                    vecs.push([z[0], z[1], ax.X(z[0] + v[0]) - ax.X(z[0]), ax.Y(z[1] + v[1]) - ax.Y(z[1])]);
                  }
                  // arrows scaled so the longest spans about one grid cell (matplotlib quiver autoscale)
                  const cell = Math.min(ax.pw / gx, ax.ph / gy), lmax = Math.max(1e-9, ...vecs.map((v) => Math.hypot(v[2], v[3]))), sc = (0.9 * cell) / lmax;
                  g.stroke([119, 119, 119], 0.5); g.fill([119, 119, 119]);
                  vecs.forEach(([zx, zy, dx, dy]) => {
                    const X0 = ax.X(zx), Y0 = ax.Y(zy), X1 = X0 + dx * sc, Y1 = Y0 + dy * sc, len = Math.hypot(X1 - X0, Y1 - Y0);
                    if (len < 0.3) return;
                    doc.line(X0, Y0, X1, Y1);
                    const ux = (X1 - X0) / len, uy = (Y1 - Y0) / len, hl = Math.min(2.6, len * 0.45);
                    doc.triangle(X1, Y1, X1 - ux * hl - uy * hl * 0.5, Y1 - uy * hl + ux * hl * 0.5, X1 - ux * hl + uy * hl * 0.5, Y1 - uy * hl - ux * hl * 0.5, 'F');
                  });
                }
                g.stroke(COLORS[c], 0.9);
                longestOf(c).forEach((bt) => {
                  const e = R.emb[bt.animal].slice(bt.start, bt.end + 1);
                  g.polyline(e.map((p) => [ax.X(p[0]), ax.Y(p[1])]));
                  g.fill(COLORS[c]); doc.circle(ax.X(e[0][0]), ax.Y(e[0][1]), 1.3, 'F');
                });
                const fp = st.fixed_point;
                if (R.hasFlow && fp && fp[0] >= x0 && fp[0] <= x1 && fp[1] >= y0 && fp[1] <= y1) {
                  const X = ax.X(fp[0]), Y = ax.Y(fp[1]);
                  g.stroke([0, 0, 0], 0.9); doc.circle(X, Y, 3.6, 'S');
                  doc.line(X - 2.5, Y - 2.5, X + 2.5, Y + 2.5); doc.line(X - 2.5, Y + 2.5, X + 2.5, Y - 2.5);
                }
              });
              g.stroke(RULE, 0.5); doc.rect(ax.px, ax.py, ax.pw, ax.ph, 'S');
              g.font(8.5, 'bold', rgb(COLORS[c])); g.text(`${SN[c]} · ${pct(R.share[c], 0)} of time`, ax.px + ax.pw / 2, ax.box.y + 9, { align: 'center' });
            }
          });
        }
        if (!R.hasFlow) note(g, 'The model was fit on the raw features, so its flow fields are not in PC space: only the bouts are drawn.');
      }
    }

    // ===== behaviour per state =====
    g.ensure(330);
    heading(g, '7. Behaviour in each state');
    paragraph(g, `Per-frame measures for every animal: centroid speed and acceleration on the floor, turning rate of the body axis` +
      (R.dims === 3 ? ', centroid and maximum height' : '') + ', body length (anterior-posterior distance)' +
      (A > 1 ? ' and, with several animals, the distance to the nearest other animal and the nose-to-nose distance' : '') +
      '. Derivatives use central differences. Boxes span the quartiles, whiskers 1.5 x IQR; outliers are not drawn.');
    const extremes = R.MEASURES.map((m) => {
      const v = R.med.map((x, c) => [x[m], c]).filter((q) => Number.isFinite(q[0])).sort((p, q) => q[0] - p[0]);
      return v.length > 1 ? `${pretty(m)}: highest in ${SN[v[0][1]]} (${fmt(v[0][0])} ${R.UNITS[m]}), lowest in ${SN[v[v.length - 1][1]]} (${fmt(v[v.length - 1][0])})` : null;
    }).filter(Boolean);
    bullets(g, extremes, { size: 8.5 });
    const nm = R.MEASURES.length, mcols = 3, mrows = Math.ceil(nm / mcols), mw = (g.CW - (mcols - 1) * 10) / mcols;
    for (let r = 0; r < mrows; r++) {
      figure(g, 120, (b) => {
        for (let k = 0; k < mcols; k++) {
          const i = r * mcols + k;
          if (i >= nm) break;
          const m = R.MEASURES[i];
          boxplotPanel(g, { x: b.x + k * (mw + 10), y: b.y, w: mw, h: b.h }, R, R.byState(m), { title: `${pretty(m)} (${R.UNITS[m]})` });
        }
      });
    }
    g.ensure(140 + 16 * K);
    heading(g, 'How each state differs', 2);
    paragraph(g, 'The median of each measure per state, as a z-score against all frames: (state median - overall median) / overall standard deviation. Red is above average, blue below.');
    g.font(7); const rotH = Math.max(...R.MEASURES.map((m) => g.width(pretty(m))));
    figure(g, 4 + 16 * K + 8 + rotH, (b) => {
      const mat = R.z.map((row) => R.MEASURES.map((m) => row[m]));
      const w = Math.min(b.w - 60, 50 + nm * 52);
      heatmapPanel(g, { x: b.x, y: b.y, w, h: b.h }, mat, {
        cmap: (v) => rdbu((v + 2) / 4), xcats: R.MEASURES.map(pretty), ycats: SN, rotateX: true, padL: 30,
        annot: (v) => (!Number.isFinite(v) ? '' : Math.abs(v) < 0.05 ? '0.0' : `${v > 0 ? '+' : ''}${v.toFixed(1)}`),
      });
      colorbar(g, b.x + w + 8, b.y + 4, 7, 16 * K, (t) => rdbu(t), -2, 2, 'z', [-2, -1, 0, 1, 2]);
    });
    paragraph(g, 'Median of each measure per state:', { size: 8.5, color: MUTED, after: 2 });
    table(g, [{ label: 'Measure', w: 2.6 }, ...SN.map((nm) => ({ label: nm, w: 1, align: 'right' })), { label: 'all', w: 1, align: 'right' }],
      R.MEASURES.map((m) => [`${pretty(m)} (${R.UNITS[m]})`, ...R.med.map((row) => fmt(row[m])), { text: fmt(R.allMed[m]), bold: true }]), { size: K > 10 ? 6.5 : 7.5 });

    // ===== average pose =====
    g.ensure(200);
    heading(g, '8. Average pose in each state');
    paragraph(g, `Every frame expressed in the animal's own frame (centroid at the origin, body axis pointing right) and averaged per state. ` +
      `Top row: seen from above${R.dims === 3 ? '; bottom row: seen from the side' : ''}.`);
    {
      const perRow = Math.min(K, 6), s = (g.CW - (perRow - 1) * 6) / perRow, lim = R.poseLim || 1, hl = R.heightLim || 1;
      const sideH = R.dims === 3 ? Math.min(s, Math.max(24, (s * (1.1 * hl)) / (2 * lim))) : 0;
      for (let r0 = 0; r0 < K; r0 += perRow) {
        figure(g, 14 + s + (sideH ? sideH + 4 : 0), (b) => {
          for (let k = 0; k < perRow && r0 + k < K; k++) {
            const c = r0 + k, mp = R.meanPose[c], x = b.x + k * (s + 6), col = rgb(COLORS[c]);
            g.font(8.5, 'bold', col); g.text(SN[c], x + s / 2, b.y + 9, { align: 'center' });
            g.stroke([234, 236, 239], 0.5); doc.rect(x, b.y + 14, s, s, 'S');
            if (sideH) doc.rect(x, b.y + 18 + s, s, sideH, 'S');
            if (!mp) continue;
            const views = [[(p) => x + ((p[0] + lim) / (2 * lim)) * s, (p) => b.y + 14 + s - ((p[1] + lim) / (2 * lim)) * s]];
            if (sideH) views.push([(p) => x + ((p[0] + lim) / (2 * lim)) * s, (p) => b.y + 18 + s + sideH - ((p[2] + 0.1 * hl) / (1.1 * hl)) * sideH]);
            views.forEach(([fx, fy]) => {
              g.stroke(col, 1.4); doc.setLineCap('round');
              R.bones.forEach(([i, j]) => { if (mp[i] && mp[j]) doc.line(fx(mp[i]), fy(mp[i]), fx(mp[j]), fy(mp[j])); });
              g.fill(col); mp.forEach((p) => doc.circle(fx(p), fy(p), 1.2, 'F'));
              doc.setLineCap('butt');
            });
          }
        });
      }
      g.font(7, 'normal', MUTED);
      g.text(`Each box spans ${(2 * lim).toFixed(1)} ${LEN} along the body axis.`, g.M, g.y + 6); g.y += 16;
    }

    // ===== summary =====
    g.ensure(160);
    heading(g, '9. Summary');
    paragraph(g, 'A plain-language description of each state, most common first: its share of time, typical bout length, what usually follows it, and the measures where its median is well above or below the median over all frames (|z| >= 0.5).');
    summary.forEach(({ c, text }) => {
      g.font(9.5); const lines = doc.splitTextToSize(latin1(text), g.CW - 34), lh = 9.5 * 1.32;
      g.ensure(lh * lines.length + 6);
      g.fill(COLORS[c]); doc.rect(g.M, g.y + 1.5, 8, 8, 'F');
      g.font(9.5, 'bold'); g.text(SN[c], g.M + 12, g.y + 9.5 * 0.8);
      g.font(9.5); lines.forEach((ln, i) => { doc.text(ln, g.M + 34, g.y + 9.5 * 0.8 + i * lh); });
      g.y += lh * lines.length + 6;
    });
    note(g, 'All numbers are recomputed from the exported pose data and clustering output (etho-pose and etho-clusters files); missing pose samples are linearly interpolated. Lengths are in ' + LEN + '.');

    // footers
    const n = doc.getNumberOfPages();
    for (let i = 1; i <= n; i++) {
      doc.setPage(i);
      g.font(7, 'normal', MUTED);
      g.text(`Behaviour summary · ${dsName}${pose.clips.length > 1 ? ` · ${clipName}` : ''}`, g.M, g.H - 30);
      g.text(`${i} / ${n}`, g.W - g.M, g.H - 30, { align: 'right' });
    }
    return doc;
  }

  root.EthoReport = { analyze, build };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.EthoReport;
})(typeof globalThis !== 'undefined' ? globalThis : this);
