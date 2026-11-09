/*
 * etho-pose: a standard container for pose data over time (spec: viewer/POSE_FORMAT.md).
 *
 * Handles anything from a single animal's 2D centroid to groups of animals with 3D skeletons,
 * with zero or more videos. Exposes `PoseFormat` (browser global, or module.exports under Node):
 *
 *   PoseFormat.load(textOrObject, { name })  -> standard dataset (converts legacy files, validates)
 *   PoseFormat.validate(dataset)             -> array of error strings (empty when valid)
 *   PoseFormat.fromLegacy(object)            -> standard dataset from a pose-testset /
 *                                               quadruped-skeleton-testset file
 */
(function (root) {
  'use strict';

  const FORMAT = 'etho-pose';
  const VERSION = '1.0';

  const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const slug = (s) => String(s).trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'x';

  // ---------- validation ----------
  function validate(ds) {
    const errs = [];
    const err = (m) => { if (errs.length < 50) errs.push(m); };
    if (!isObj(ds)) return ['The file is not a JSON object.'];
    if (ds.format !== FORMAT) err(`format must be "${FORMAT}".`);
    if (typeof ds.version !== 'string' || ds.version.split('.')[0] !== VERSION.split('.')[0]) {
      err(`version must be "${VERSION.split('.')[0]}.x" (got ${JSON.stringify(ds.version)}).`);
    }

    const space = ds.space;
    if (!isObj(space)) err('space is missing.');
    const dims = isObj(space) ? space.dims : undefined;
    if (dims !== 2 && dims !== 3) err('space.dims must be 2 or 3.');
    if (isObj(space) && space.up !== undefined && !/^[+-][XYZ]$/.test(space.up)) err('space.up must be one of "+Y", "+Z", ... .');

    // skeletons
    const skels = isObj(ds.skeletons) ? ds.skeletons : null;
    if (!skels || !Object.keys(skels).length) err('skeletons must define at least one skeleton.');
    const jointCount = {};
    Object.entries(skels || {}).forEach(([id, sk]) => {
      const where = `skeletons.${id}`;
      if (!isObj(sk) || !Array.isArray(sk.joints) || !sk.joints.length) { err(`${where}.joints must be a non-empty list.`); return; }
      const n = sk.joints.length, names = new Set();
      sk.joints.forEach((j, i) => {
        if (!isObj(j) || typeof j.name !== 'string' || !j.name) err(`${where}.joints[${i}] needs a name.`);
        else if (names.has(j.name)) err(`${where}: joint name "${j.name}" is repeated.`);
        else names.add(j.name);
        if (isObj(j) && j.parent !== undefined && j.parent !== null && !(Number.isInteger(j.parent) && j.parent >= 0 && j.parent < n)) {
          err(`${where}.joints[${i}].parent must be a joint index or null.`);
        }
      });
      if (sk.bones !== undefined && !Array.isArray(sk.bones)) err(`${where}.bones must be a list.`);
      (Array.isArray(sk.bones) ? sk.bones : []).forEach((b, i) => {
        if (!Array.isArray(b) || b.length !== 2 || !b.every((v) => Number.isInteger(v) && v >= 0 && v < n)) {
          err(`${where}.bones[${i}] must be [parentIndex, childIndex].`);
        }
      });
      jointCount[id] = n;
    });

    // animals
    const animals = Array.isArray(ds.animals) ? ds.animals : null;
    if (!animals || !animals.length) err('animals must list at least one animal.');
    const animalSkel = {};
    (animals || []).forEach((a, i) => {
      if (!isObj(a) || typeof a.id !== 'string' || !a.id) { err(`animals[${i}] needs an id.`); return; }
      if (animalSkel[a.id] !== undefined) err(`animal id "${a.id}" is repeated.`);
      if (!(a.skeleton in jointCount)) err(`animals[${i}].skeleton "${a.skeleton}" is not defined in skeletons.`);
      animalSkel[a.id] = a.skeleton;
    });

    // videos (optional, may be empty)
    const videoIds = new Set();
    if (ds.videos !== undefined && !Array.isArray(ds.videos)) err('videos must be a list (it may be empty).');
    (Array.isArray(ds.videos) ? ds.videos : []).forEach((v, i) => {
      if (!isObj(v) || typeof v.id !== 'string' || !v.id) { err(`videos[${i}] needs an id.`); return; }
      if (videoIds.has(v.id)) err(`video id "${v.id}" is repeated.`);
      videoIds.add(v.id);
      if (typeof v.uri !== 'string' || !v.uri) err(`videos[${i}].uri is missing.`);
      if (v.fps !== undefined && !(isNum(v.fps) && v.fps > 0)) err(`videos[${i}].fps must be a positive number.`);
    });

    // clips
    const clips = Array.isArray(ds.clips) ? ds.clips : null;
    if (!clips || !clips.length) err('clips must contain at least one clip.');
    (clips || []).forEach((c, ci) => {
      const where = `clips[${ci}]${isObj(c) && c.name ? ` ("${c.name}")` : ''}`;
      if (!isObj(c)) { err(`${where} is not an object.`); return; }
      if (!(isNum(c.fps) && c.fps > 0)) err(`${where}.fps must be a positive number.`);
      if (!(Number.isInteger(c.frame_count) && c.frame_count > 0)) err(`${where}.frame_count must be a positive integer.`);
      const n = c.frame_count;
      if (c.timestamps !== undefined && !(Array.isArray(c.timestamps) && c.timestamps.length === n && c.timestamps.every(isNum))) {
        err(`${where}.timestamps must hold frame_count numbers.`);
      }
      if (c.root_motion_per_loop !== undefined && !(Array.isArray(c.root_motion_per_loop) && c.root_motion_per_loop.length === dims && c.root_motion_per_loop.every(isNum))) {
        err(`${where}.root_motion_per_loop must hold ${dims} numbers.`);
      }
      if (!isObj(c.tracks) || !Object.keys(c.tracks).length) { err(`${where}.tracks must hold at least one animal's track.`); return; }
      Object.entries(c.tracks).forEach(([aid, tr]) => {
        const tw = `${where}.tracks.${aid}`;
        if (!(aid in animalSkel)) { err(`${tw}: "${aid}" is not in animals.`); return; }
        const J = jointCount[animalSkel[aid]];
        if (!isObj(tr) || !Array.isArray(tr.positions)) { err(`${tw}.positions is missing.`); return; }
        if (tr.positions.length !== n) err(`${tw}.positions has ${tr.positions.length} frames, frame_count is ${n}.`);
        for (let f = 0; f < tr.positions.length; f++) {
          const fr = tr.positions[f];
          if (fr === null) continue;  // whole frame missing
          if (!Array.isArray(fr) || fr.length !== J) { err(`${tw}.positions[${f}] must hold ${J} joints (or be null).`); break; }
          const bad = fr.findIndex((p) => p !== null && !(Array.isArray(p) && p.length === dims && p.every(isNum)));
          if (bad >= 0) { err(`${tw}.positions[${f}][${bad}] must be ${dims} numbers or null.`); break; }
        }
        if (tr.confidence !== undefined) {
          const ok = Array.isArray(tr.confidence) && tr.confidence.length === n &&
            tr.confidence.every((fr) => fr === null || (Array.isArray(fr) && fr.length === J && fr.every((v) => v === null || isNum(v))));
          if (!ok) err(`${tw}.confidence must be [frame][joint] numbers (or null).`);
        }
      });
      if (c.events !== undefined && !Array.isArray(c.events)) err(`${where}.events must be a list.`);
      (Array.isArray(c.events) ? c.events : []).forEach((e, ei) => {
        const ew = `${where}.events[${ei}]`;
        if (!isObj(e) || typeof e.name !== 'string' || !e.name) { err(`${ew} needs a name.`); return; }
        if (!(Array.isArray(e.active) && e.active.length === n)) err(`${ew}.active must hold frame_count values.`);
        if (e.animal !== undefined && e.animal !== null && !(e.animal in animalSkel)) err(`${ew}.animal "${e.animal}" is not in animals.`);
      });
      if (c.videos !== undefined && !Array.isArray(c.videos)) err(`${where}.videos must be a list.`);
      (Array.isArray(c.videos) ? c.videos : []).forEach((v, vi) => {
        if (!isObj(v) || !videoIds.has(v.id)) err(`${where}.videos[${vi}].id must name an entry of videos.`);
        else if (v.frame_offset !== undefined && !Number.isInteger(v.frame_offset)) err(`${where}.videos[${vi}].frame_offset must be an integer.`);
      });
    });
    return errs;
  }

  // ---------- legacy converters ----------
  // pose-testset 1.x (mouse_*.json) and quadruped-skeleton-testset 1.x (skeleton_dataset.json):
  // one skeleton shared by every animal; positions are [frame][joint][coord], or
  // [frame][animal][joint][coord] when an "animals" list is present.
  function fromLegacy(src) {
    if (!isObj(src.skeleton) || !Array.isArray(src.skeleton.joints) || !Array.isArray(src.clips) || !src.clips.length) {
      throw new Error('This legacy file has no skeleton or clips.');
    }
    const firstPos = src.clips[0].positions;
    const multi = Array.isArray(src.animals) && src.animals.length > 0;
    const sample = multi ? firstPos[0][0][0] : firstPos[0][0];
    const dims = src.dimensions || (Array.isArray(sample) ? sample.length : 3);
    const species = src.species || (src.format === 'quadruped-skeleton-testset' ? 'quadruped' : undefined);

    const skId = slug(species || 'skeleton');
    const sk = src.skeleton;
    const skeleton = {
      joints: sk.joints.map((j) => ({ name: j.name, parent: Number.isInteger(j.parent) && j.parent >= 0 ? j.parent : null })),
      bones: (sk.bones || []).map((b) => [b[0], b[1]]),
    };
    if (Array.isArray(sk.rest_pose)) skeleton.rest_pose = sk.rest_pose;
    if (Array.isArray(sk.bone_lengths_m)) skeleton.bone_lengths = sk.bone_lengths_m;

    const used = new Set();
    const uniqueId = (base) => { let id = slug(base), k = 2; while (used.has(id)) id = `${slug(base)}-${k++}`; used.add(id); return id; };
    const animals = multi
      ? src.animals.map((a, i) => ({ id: uniqueId(a.name || `animal ${i + 1}`), name: a.name || `animal ${i + 1}`, skeleton: skId }))
      : [{ id: uniqueId(species || 'animal'), name: species || 'animal', skeleton: skId }];
    animals.forEach((a) => { if (species) a.species = species; });

    const known = new Set(['name', 'fps', 'frame_count', 'loop', 'root_motion_per_loop', 'positions', 'events', 'foot_contacts', 'duration_s']);
    const clips = src.clips.map((c, ci) => {
      const tracks = {};
      animals.forEach((a, ai) => {
        tracks[a.id] = { positions: multi ? c.positions.map((fr) => fr[ai]) : c.positions };
      });
      const events = [];
      Object.entries(c.events || {}).forEach(([name, arr]) => events.push({ name, animal: null, active: arr.map((v) => (v ? 1 : 0)) }));
      // per-joint contacts (dog clips) become joint-scoped events of the single animal
      Object.entries(c.foot_contacts || {}).forEach(([joint, arr]) => events.push({
        name: 'contact', animal: animals[0].id, joint, active: arr.map((v) => (v ? 1 : 0)),
      }));
      const extra = {};
      Object.keys(c).forEach((k) => { if (!known.has(k)) extra[k] = c[k]; });
      const out = {
        name: c.name || `clip ${ci + 1}`,
        fps: c.fps,
        frame_count: c.frame_count != null ? c.frame_count : c.positions.length,
        loop: !!c.loop,
        tracks,
        events,
        videos: [],
      };
      if (Array.isArray(c.root_motion_per_loop) && c.root_motion_per_loop.some((v) => v)) out.root_motion_per_loop = c.root_motion_per_loop.slice(0, dims);
      if (Object.keys(extra).length) out.extra = extra;
      return out;
    });

    const cs = src.coordinate_system || {};
    const space = { dims, units: src.units === 'meters' ? 'm' : (src.units || 'm') };
    if (dims === 3) space.up = cs.up || '+Y';
    else space.view = cs.view || 'top-down';
    space.origin = cs.origin || undefined;
    if (space.origin === undefined) delete space.origin;

    const meta = { description: src.description || '', source: `converted from ${src.format} ${src.version || ''}`.trim() };
    if (species) meta.species = species;
    return { format: FORMAT, version: VERSION, meta, space, skeletons: { [skId]: skeleton }, animals, videos: [], clips };
  }

  const LEGACY = new Set(['pose-testset', 'quadruped-skeleton-testset']);

  // ---------- entry point ----------
  function load(input, opts = {}) {
    let obj = input;
    if (typeof input === 'string') {
      try { obj = JSON.parse(input); } catch (e) { throw new Error(`Not valid JSON: ${e.message}`); }
    }
    if (!isObj(obj)) throw new Error('The file is not a JSON object.');
    let ds;
    if (obj.format === FORMAT) ds = obj;
    else if (LEGACY.has(obj.format)) ds = fromLegacy(obj);
    else throw new Error(`Unrecognized format ${JSON.stringify(obj.format)}. Expected "${FORMAT}" or a legacy pose-testset file.`);
    if (!Array.isArray(ds.videos)) ds.videos = [];
    const errs = validate(ds);
    if (errs.length) throw new Error(`Invalid ${FORMAT} data:\n- ${errs.join('\n- ')}`);
    if (!isObj(ds.meta)) ds.meta = {};
    if (!ds.meta.name) ds.meta.name = opts.name || ds.meta.description || 'dataset';
    return ds;
  }

  const api = { FORMAT, VERSION, load, validate, fromLegacy };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PoseFormat = api;
})(typeof window !== 'undefined' ? window : globalThis);
