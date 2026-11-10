/*
 * Pose clean-up for etho-pose datasets (the viewer's Clean tab).
 *
 *   PoseClean.clean(dataset, options) -> { dataset, report }
 *
 * Works on a copy; every clip and animal is cleaned joint by joint in four steps:
 *   1. left/right swaps   for each X_L / X_R joint pair, a frame is swapped back when BOTH joints
 *                         sit on the wrong side of the body axis (posterior -> anterior node, on the
 *                         floor) by more than a quarter of their usual offset. Which side is "left"
 *                         is learned from the clip (the side each joint is on in most frames), so
 *                         the handedness of the coordinate system does not matter, and one outlier
 *                         cannot trigger a swap on its own.
 *   2. outliers           Hampel filter: a sample whose distance from the rolling median of its own
 *                         track (window frames, centred) exceeds threshold x the robust scale of
 *                         those distances (1.4826 x their median over the clip) is marked missing.
 *   3. fill gaps          every missing sample is filled: linear, cubic (Hermite with finite-
 *                         difference slopes), or body-relative: the body-axis nodes are filled
 *                         linearly, every other joint's offset from the posterior node is expressed
 *                         in the animal's own frame (rotated by its heading on the floor), that offset
 *                         is filled linearly, and the joint is placed from the frame's own body. This
 *                         follows the animal through turns and fast moves during long gaps. Before the
 *                         first / after the last sample: held flat.
 *   4. smoothing          none, moving median (window frames) or Gaussian (sigma frames), per
 *                         coordinate; windows are truncated at the ends of the clip.
 * Steps 1, 2 and 4 can be switched off; with everything off the result equals the viewer's own
 * gap filling. api/etho.py (clean_pose) implements the same steps, so a model trained on cleaned
 * data is applied to new data cleaned the same way.
 */
(function (root) {
  'use strict';

  const DEFAULTS = { swaps: true, outliers: true, window: 7, threshold: 4, fill: 'linear', smooth: 'gaussian', size: 1 };
  const ANT = ['neck', 'chest', 'spine_mid', 'head', 'snout', 'nose'], POST = ['pelvis', 'tail_base', 'hips'];

  const medianOf = (v) => {
    const s = v.slice().sort((a, b) => a - b), n = s.length;
    return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
  };

  // floor-plane coordinates of a raw sample (the axes that are not "up")
  function floorAxes(space) {
    if (space.dims === 2) return [0, 1];
    const up = (space.up || '+Y')[1];
    return up === 'Y' ? [0, 2] : up === 'Z' ? [0, 1] : [1, 2];
  }

  // ---- 1. left/right swaps
  function fixSwaps(P, nF, J, dims, names, space) {
    const at = (f, j, c) => P[(f * J + j) * dims + c];
    const ant = ANT.map((n) => names.indexOf(n)).find((i) => i >= 0), post = POST.map((n) => names.indexOf(n)).find((i) => i >= 0);
    const fixed = {};
    if (ant === undefined || post === undefined || ant === post) return fixed;
    const [ax, az] = floorAxes(space);
    const lat = (f, j) => {
      const ux = at(f, ant, ax) - at(f, post, ax), uz = at(f, ant, az) - at(f, post, az), n = Math.hypot(ux, uz);
      return n > 0 ? (ux * (at(f, j, az) - at(f, post, az)) - uz * (at(f, j, ax) - at(f, post, ax))) / n : NaN;
    };
    names.forEach((nm, i) => {
      const m = /^(.*)_L$/.exec(nm);
      const k = m ? names.indexOf(`${m[1]}_R`) : -1;
      if (k < 0) return;
      const li = new Float64Array(nF).fill(NaN), lk = new Float64Array(nF).fill(NaN);
      for (let f = 0; f < nF; f++) { li[f] = lat(f, i); lk[f] = lat(f, k); }
      const vi = [], vk = [];
      for (let f = 0; f < nF; f++) if (Number.isFinite(li[f]) && Number.isFinite(lk[f])) { vi.push(li[f]); vk.push(lk[f]); }
      if (vi.length < 10) return;
      const si = Math.sign(medianOf(vi)), sk = Math.sign(medianOf(vk));
      if (!si || si === sk) return;                               // no consistent sides: leave the pair alone
      const mi = 0.25 * medianOf(vi.map(Math.abs)), mk = 0.25 * medianOf(vk.map(Math.abs));
      let n = 0;
      for (let f = 0; f < nF; f++) {
        if (!(li[f] * si < -mi && lk[f] * sk < -mk)) continue;   // both on the wrong side
        for (let c = 0; c < dims; c++) {
          const a = (f * J + i) * dims + c, b = (f * J + k) * dims + c, t = P[a];
          P[a] = P[b]; P[b] = t;
        }
        n++;
      }
      if (n) fixed[`${nm}/${names[k]}`] = n;
    });
    return fixed;
  }

  // ---- 2. outliers (Hampel on the distance from the rolling median)
  function removeOutliers(P, nF, J, dims, win, thr) {
    const h = Math.max(1, Math.floor(win / 2)), removed = new Int32Array(J);
    for (let j = 0; j < J; j++) {
      const r = new Float64Array(nF).fill(NaN);
      for (let f = 0; f < nF; f++) {
        if (!Number.isFinite(P[(f * J + j) * dims])) continue;
        let d2 = 0, ok = true;
        for (let c = 0; c < dims && ok; c++) {
          const w = [];
          for (let g = Math.max(0, f - h); g <= Math.min(nF - 1, f + h); g++) { const v = P[(g * J + j) * dims + c]; if (Number.isFinite(v)) w.push(v); }
          if (w.length < 3) ok = false;
          else d2 += (P[(f * J + j) * dims + c] - medianOf(w)) ** 2;
        }
        if (ok) r[f] = Math.sqrt(d2);
      }
      const rs = [];
      for (let f = 0; f < nF; f++) if (Number.isFinite(r[f])) rs.push(r[f]);
      const scale = 1.4826 * medianOf(rs);
      if (!(scale > 0)) continue;
      for (let f = 0; f < nF; f++) {
        if (r[f] > thr * scale) { for (let c = 0; c < dims; c++) P[(f * J + j) * dims + c] = NaN; removed[j]++; }
      }
    }
    return removed;
  }

  // ---- 3. fill gaps in one series (NaN = missing); returns false when nothing is known
  function fillSeries(v, cubic) {
    const n = v.length, known = [];
    for (let i = 0; i < n; i++) if (Number.isFinite(v[i])) known.push(i);
    if (!known.length) return false;
    for (let i = 0; i < known[0]; i++) v[i] = v[known[0]];
    for (let i = known[known.length - 1] + 1; i < n; i++) v[i] = v[known[known.length - 1]];
    for (let q = 0; q + 1 < known.length; q++) {
      const i0 = known[q], i1 = known[q + 1], h = i1 - i0;
      if (h < 2) continue;
      const p0 = v[i0], p1 = v[i1];
      let m0 = (p1 - p0) / h, m1 = m0;                                         // secant slopes
      if (cubic) {
        if (q > 0) m0 = (p1 - v[known[q - 1]]) / (i1 - known[q - 1]);
        if (q + 2 < known.length) m1 = (v[known[q + 2]] - p0) / (known[q + 2] - i0);
      }
      for (let i = i0 + 1; i < i1; i++) {
        const t = (i - i0) / h;
        if (!cubic) { v[i] = p0 + (p1 - p0) * t; continue; }
        const t2 = t * t, t3 = t2 * t;
        v[i] = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * h * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * h * m1;
      }
    }
    return true;
  }

  // body-relative filling of every joint of one animal (anchors linear, the rest via body-frame offsets)
  function fillBody(P, nF, J, dims, names, space) {
    const ant = ANT.map((n) => names.indexOf(n)).find((i) => i >= 0), post = POST.map((n) => names.indexOf(n)).find((i) => i >= 0);
    if (ant === undefined || post === undefined || ant === post) return null;            // no body axis: caller fills linearly
    const [ax, az] = floorAxes(space), others = [];
    for (let c = 0; c < dims; c++) if (c !== ax && c !== az) others.push(c);               // the up axis (3D)
    const series = new Float64Array(nF), empty = new Set();
    const fillJoint = (j) => {
      for (let c = 0; c < dims; c++) {
        for (let f = 0; f < nF; f++) series[f] = P[(f * J + j) * dims + c];
        if (!fillSeries(series, false)) { empty.add(j); return; }
        for (let f = 0; f < nF; f++) P[(f * J + j) * dims + c] = series[f];
      }
    };
    fillJoint(post); fillJoint(ant);
    if (empty.has(post) || empty.has(ant)) return null;
    const cs = new Float64Array(nF), sn = new Float64Array(nF);
    for (let f = 0; f < nF; f++) {
      const ux = P[(f * J + ant) * dims + ax] - P[(f * J + post) * dims + ax], uz = P[(f * J + ant) * dims + az] - P[(f * J + post) * dims + az];
      const n = Math.hypot(ux, uz) || 1;
      cs[f] = ux / n; sn[f] = uz / n;
    }
    const fw = new Float64Array(nF), lat = new Float64Array(nF), ups = others.map(() => new Float64Array(nF));
    for (let j = 0; j < J; j++) {
      if (j === post || j === ant) continue;
      const miss = [];
      for (let f = 0; f < nF; f++) {
        const o = (f * J + j) * dims, b = (f * J + post) * dims;
        if (!Number.isFinite(P[o])) { fw[f] = lat[f] = NaN; ups.forEach((u) => { u[f] = NaN; }); miss.push(f); continue; }
        const dx = P[o + ax] - P[b + ax], dz = P[o + az] - P[b + az];
        fw[f] = dx * cs[f] + dz * sn[f]; lat[f] = -dx * sn[f] + dz * cs[f];
        others.forEach((c, q) => { ups[q][f] = P[o + c] - P[b + c]; });
      }
      if (!miss.length) continue;
      if (!fillSeries(fw, false)) { empty.add(j); continue; }
      fillSeries(lat, false); ups.forEach((u) => fillSeries(u, false));
      miss.forEach((f) => {
        const o = (f * J + j) * dims, b = (f * J + post) * dims;
        P[o + ax] = P[b + ax] + fw[f] * cs[f] - lat[f] * sn[f];
        P[o + az] = P[b + az] + fw[f] * sn[f] + lat[f] * cs[f];
        others.forEach((c, q) => { P[o + c] = P[b + c] + ups[q][f]; });
      });
    }
    return empty;
  }

  // ---- 4. smoothing of one complete series
  function smoothSeries(v, method, size) {
    const n = v.length, out = new Float64Array(n);
    if (method === 'median') {
      const h = Math.max(1, Math.floor(size / 2));
      for (let i = 0; i < n; i++) {
        const w = [];
        for (let g = Math.max(0, i - h); g <= Math.min(n - 1, i + h); g++) w.push(v[g]);
        out[i] = medianOf(w);
      }
    } else {
      const R = Math.max(1, Math.ceil(3 * size)), k = [];
      for (let d = -R; d <= R; d++) k.push(Math.exp(-0.5 * (d / size) ** 2));
      for (let i = 0; i < n; i++) {
        let s = 0, ws = 0;
        for (let d = -R; d <= R; d++) { const g = i + d; if (g >= 0 && g < n) { s += k[d + R] * v[g]; ws += k[d + R]; } }
        out[i] = s / ws;
      }
    }
    v.set(out);
  }

  function clean(ds, options = {}) {
    const o = { ...DEFAULTS, ...options };
    const out = JSON.parse(JSON.stringify(ds));
    const dims = out.space.dims, report = { swaps: {}, outliers: 0, missing: 0, samples: 0, byJoint: {} };
    out.clips.forEach((clip) => {
      const nF = clip.frame_count;
      out.animals.forEach((an) => {
        const tr = clip.tracks[an.id];
        if (!tr) return;
        const names = out.skeletons[an.skeleton].joints.map((j) => j.name), J = names.length;
        const P = new Float64Array(nF * J * dims).fill(NaN);
        for (let f = 0; f < nF; f++) {
          const fr = tr.positions[f];
          if (!fr) continue;
          for (let j = 0; j < J; j++) if (fr[j]) for (let c = 0; c < dims; c++) P[(f * J + j) * dims + c] = fr[j][c];
        }
        for (let i = 0; i < P.length; i += dims) { report.samples++; if (!Number.isFinite(P[i])) report.missing++; }
        if (o.swaps) {
          const fx = fixSwaps(P, nF, J, dims, names, out.space);
          Object.entries(fx).forEach(([pair, n]) => { report.swaps[pair] = (report.swaps[pair] || 0) + n; });
        }
        if (o.outliers) {
          const rm = removeOutliers(P, nF, J, dims, o.window, o.threshold);
          rm.forEach((n, j) => { report.outliers += n; if (n) report.byJoint[names[j]] = (report.byJoint[names[j]] || 0) + n; });
        }
        const series = new Float64Array(nF);
        let empty = o.fill === 'body' ? fillBody(P, nF, J, dims, names, out.space) : null;
        const bodyFilled = !!empty;
        if (!empty) empty = new Set();
        for (let j = 0; j < J; j++) {
          if (empty.has(j)) continue;
          for (let c = 0; c < dims; c++) {
            for (let f = 0; f < nF; f++) series[f] = P[(f * J + j) * dims + c];
            if (!bodyFilled && !fillSeries(series, o.fill === 'cubic')) { empty.add(j); continue; }
            if (o.smooth === 'median' || o.smooth === 'gaussian') smoothSeries(series, o.smooth, o.size);
            for (let f = 0; f < nF; f++) P[(f * J + j) * dims + c] = series[f];
          }
        }
        tr.positions = Array.from({ length: nF }, (_, f) => names.map((_, j) => (empty.has(j) ? null
          : Array.from({ length: dims }, (_, c) => P[(f * J + j) * dims + c]))));
      });
    });
    const settings = { swaps: !!o.swaps, outliers: o.outliers ? { window: o.window, threshold: o.threshold } : null, fill: o.fill,
      smooth: o.smooth === 'median' || o.smooth === 'gaussian' ? { method: o.smooth, size: o.size } : null };
    out.meta = { ...(out.meta || {}), cleaned: settings };
    return { dataset: out, report, settings };
  }

  const api = { DEFAULTS, clean };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PoseClean = api;
})(typeof window !== 'undefined' ? window : globalThis);
