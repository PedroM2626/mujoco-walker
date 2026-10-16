"""The explicit-contact-pairs question, measured: does the boxer-rl trick buy MJX speed here?

`boxer-rl` keeps its MJX contact buffer small by declaring contacts explicitly (`<pair>`) instead of
leaving them to the automatic contype/conaffinity filter. Whether that helps *this* model is a question
with a number, because MuJoCo's automatic filter already excludes same-body, welded and parent-child
pairs before MJX ever sees them: `walker_ragdoll.xml` has 18 collidable geoms, the filter admits 126 of
their 171 pairs (`candidate_pairs` below), and MJX's static contact buffer is 138 slots - the same 138
whether the pairs are declared or not. So the two cells below separate what the trick could and could
not change:

  * ``full``   - all 126 candidates declared explicitly, every geom's contype/conaffinity zeroed.
    Physics-identical *by construction* if MuJoCo computes the same contacts, which is verified on
    rollouts rather than assumed. Measured result: MJX's buffer stays 138 and nothing gets faster.
  * ``gait``   - the boxer-style reduction: feet/shins/thighs against the floor and against each other,
    plus head/torso/waists/butt against the floor so a fall still stops. Arms drop out of collision
    entirely.
    This is a *different MDP* - the published reward reads contacts (``bad_support_penalty`` counts
    non-foot floor contacts), so the divergence is measured and reported, never hidden.

Verification (MuJoCo-C, native): fixed random-action rollouts from the published reset poses, comparing
per-step contact geom-pair sets and qpos/qvel between the published model and each variant.

Bench (JAX): static contact slots for each variant, then the same batched stepper `bench_mjx.py` uses,
so the numbers sit in the same column as the committed MJX table.

    python bench_mjx_pairs.py --verify                 # native fidelity, any machine with mujoco
    python bench_mjx_pairs.py --bench --sizes 128,512  # needs jax+mjx (WSL for GPU)
"""

import argparse
import json
import os
import time

import numpy as np

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "walker_ragdoll.xml")

SIZE_ELEMENT = '<size nconmax="512" njmax="4096"/>'

# The gait-reduced collision set: what a walker minimally needs. Feet, shins and thighs collide with
# the floor and with each other (so the legs cannot pass through one another mid-stride); head, torso
# and pelvis collide with the floor only (so a fall stops instead of sinking). Everything else - arms
# against anything, torso against legs - is removed, which is exactly the kind of cut boxer-rl makes.
GAIT_FLOOR_GEOMS = ("right_foot", "left_foot", "right_shin1", "left_shin1",
                    "right_thigh1", "left_thigh1", "head", "torso1", "butt", "uwaist", "lwaist")
GAIT_SELF_GEOMS = ("right_foot", "left_foot", "right_shin1", "left_shin1",
                   "right_thigh1", "left_thigh1")


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)


def candidate_pairs(model, xml_text=None):
    """The pairs MuJoCo's automatic filter allows, read off MuJoCo's own filter.

    Rather than reimplementing the exclusion rules (a first attempt did, and got it wrong twice: once
    by excluding floor-vs-torso contacts, which MuJoCo allows because the world body is exempt from the
    parent-child rule, and once by *allowing* left_foot-vs-left_thigh1, which MuJoCo excludes despite
    the bodies not being a direct parent-child pair), every collidable geom is given a 5 m collision
    margin in a scratch copy of the model. With a margin that large every allowed pair generates a
    contact regardless of distance, so the union of contact pairs over a few states *is* the candidate
    set, computed by the same code that runs in training.
    """
    if xml_text is None:
        with open(XML_PATH, encoding="utf-8") as handle:
            xml_text = handle.read()
    puffed = xml_text.replace(
        '<geom conaffinity="1" condim="3" contype="1" density="1000"',
        '<geom conaffinity="1" condim="3" contype="1" density="1000" margin="5"')
    puffed = puffed.replace(
        '<geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor"',
        '<geom condim="3" friction="1 .1 .1" margin="5" material="MatPlane" name="floor"')
    if puffed == xml_text:
        raise RuntimeError("margin rewrite did not apply; walker_ragdoll.xml changed?")
    puffed_model = mujoco.MjModel.from_xml_string(puffed)
    data = mujoco.MjData(puffed_model)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        mujoco.mj_resetData(puffed_model, data)
        data.qpos[:] = puffed_model.qpos0 + rng.normal(scale=0.3, size=puffed_model.nq)
        data.qvel[:] = rng.normal(scale=1.0, size=puffed_model.nv)
        mujoco.mj_forward(puffed_model, data)
        for k in range(data.ncon):
            i, j = int(data.contact[k].geom1), int(data.contact[k].geom2)
            seen.add((min(i, j), max(i, j)))
    return sorted(seen)


def pair_friction(model, i, j):
    """The 5-component friction an automatic contact between geoms i and j gets.

    ``geom_friction`` is stored 3-wide (slide, spin, roll); the automatic path expands each side to
    (slide, slide, spin, roll, roll) and takes the elementwise max, which is what the runtime contact
    shows (floor/head -> [1, 1, 0.1, 0.1, 0.1]). A ``<pair>`` friction attribute with only 3 numbers
    parses as (slide1, slide2, spin) - a different convention - so all five components are written
    out explicitly.
    """
    fi, fj = model.geom_friction[i], model.geom_friction[j]
    expand = lambda f: np.array([f[0], f[0], f[1], f[2], f[2]])
    return np.maximum(expand(fi), expand(fj))


def build_variant_xml(model, pairs):
    """The published XML with `pairs` declared explicitly and the automatic filter disabled.

    Every geom's contype/conaffinity is zeroed (the default class rewrite covers the ones that inherit),
    so the declared pairs are the only collisions. condim=3 matches the geoms; friction is the
    elementwise max the automatic path would compute; solref/solimp are left at the model default,
    which is what the automatic path blends to here because every geom uses the default.
    """
    with open(XML_PATH, encoding="utf-8") as handle:
        text = handle.read()
    text = text.replace(
        '<geom conaffinity="1" condim="3" contype="1" density="1000"',
        '<geom conaffinity="0" condim="3" contype="0" density="1000"')
    text = text.replace(
        '<geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor"',
        '<geom condim="3" contype="0" conaffinity="0" friction="1 .1 .1" material="MatPlane" '
        'name="floor"')
    lines = []
    for i, j in pairs:
        friction = " ".join(f"{v:g}" for v in pair_friction(model, i, j))
        lines.append(f'    <pair geom1="{geom_name(model, i)}" geom2="{geom_name(model, j)}" '
                     f'condim="3" friction="{friction}"/>')
    contact = "<contact>\n" + "\n".join(lines) + "\n  </contact>\n  "
    i = text.index("<tendon>")
    text = text[:i] + contact + text[i:]
    i = text.index("<option")
    text = text[:i] + "  " + SIZE_ELEMENT + "\n  " + text[i:]
    return text


def rollout_contact_trace(model, ctrl_sequence, qpos0, qvel0=None):
    """Per-step (sorted geom-pair tuple set, qpos, qvel) for one fixed control sequence."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos0
    if qvel0 is not None:
        data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)
    trace = []
    for ctrl in ctrl_sequence:
        data.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)
        pairs = tuple(sorted((int(data.contact[k].geom1), int(data.contact[k].geom2))
                             for k in range(data.ncon)))
        trace.append((pairs, data.qpos.copy(), data.qvel.copy()))
    return trace


def compare_traces(a, b):
    """Contact-set equality per step and the largest state divergence over the rollout."""
    assert len(a) == len(b)
    pair_mismatches = sum(1 for (pa, _, _), (pb, _, _) in zip(a, b) if pa != pb)
    first_diverge = None
    max_qpos = max_qvel = 0.0
    for step, ((_, qa, va), (_, qb, vb)) in enumerate(zip(a, b)):
        dq = float(np.abs(qa - qb).max())
        dv = float(np.abs(va - vb).max())
        max_qpos, max_qvel = max(max_qpos, dq), max(max_qvel, dv)
        if first_diverge is None and dq > 1e-6:
            first_diverge = step
    return {"steps": len(a), "contact_set_mismatch_steps": pair_mismatches,
            "first_qpos_divergence_step": first_diverge,
            "max_abs_qpos_diff": max_qpos, "max_abs_qvel_diff": max_qvel}


def verify(variants, episodes=6, steps=300, seed=0):
    """Fixed random-action rollouts from standing and fallen starts, published model vs each variant."""
    published = mujoco.MjModel.from_xml_path(XML_PATH)
    rng = np.random.default_rng(seed)
    report = {}
    for name, xml in variants.items():
        variant = mujoco.MjModel.from_xml_string(xml)
        rows = []
        for ep in range(episodes):
            qpos0 = np.array(published.qpos0, dtype=np.float64)
            qpos0 = qpos0 + rng.normal(scale=0.01, size=published.nq)
            qvel0 = rng.normal(scale=0.05, size=published.nv) if ep % 2 else None
            ctrl = rng.uniform(-0.5, 0.5, size=(steps, published.nu))
            a = rollout_contact_trace(published, ctrl, qpos0, qvel0)
            b = rollout_contact_trace(variant, ctrl, qpos0, qvel0)
            rows.append(compare_traces(a, b))
        report[name] = {
            "episodes": episodes, "steps_per_episode": steps,
            "contact_set_mismatch_steps_total": sum(r["contact_set_mismatch_steps"] for r in rows),
            "episodes_with_any_contact_mismatch": sum(
                1 for r in rows if r["contact_set_mismatch_steps"]),
            "first_divergence_step_min": min(
                (r["first_qpos_divergence_step"] for r in rows
                 if r["first_qpos_divergence_step"] is not None), default=None),
            "max_abs_qpos_diff": max(r["max_abs_qpos_diff"] for r in rows),
            "max_abs_qvel_diff": max(r["max_abs_qvel_diff"] for r in rows),
            "candidate_pairs": len(candidate_pairs(variant)) if variant.npair == 0
            else int(variant.npair),
        }
    return report


def bench(variants, sizes, seconds=6.0, frame_skip=5):
    """Static contact slots + batched step rate per variant, on whatever backend jax picks."""
    import jax
    import mujoco.mjx as mjx
    from bench_mjx import compile_batch_stepper, make_batched_data

    report = {}
    for name, xml in variants.items():
        model = mujoco.MjModel.from_xml_string(xml)
        mjx_model = mjx.put_model(model)
        slots = int(mjx.make_data(mjx_model).ncon)
        cells = {"declared_pairs": int(model.npair), "static_contact_slots": slots,
                 "jax_backend": jax.default_backend()}
        for nworld in sizes:
            rng = np.random.default_rng(0)
            fn = compile_batch_stepper(mjx_model, mjx, frame_skip, unroll=False)
            batch = make_batched_data(mjx_model, mjx, nworld, rng)
            ctrl = np.asarray(rng.uniform(-0.5, 0.5, size=(nworld, model.nu)))
            out = fn(batch, ctrl)
            jax.block_until_ready(out)
            for _ in range(3):
                out = fn(out, ctrl)
            jax.block_until_ready(out)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                for _ in range(20):
                    out = fn(out, ctrl)
                jax.block_until_ready(out)
                n += 20 * nworld
            cells[f"nworld_{nworld}"] = {
                "env_steps_per_s": round(n / (time.perf_counter() - t0), 1),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        report[name] = cells
        print(name, cells, flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sizes", default="128,512")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    published = mujoco.MjModel.from_xml_path(XML_PATH)
    full_pairs = candidate_pairs(published, open(XML_PATH, encoding="utf-8").read())
    print(f"published model: {published.ngeom} geoms, {len(full_pairs)} candidate pairs, "
          f"{published.npair} declared pairs")

    name_of = {gid: geom_name(published, gid) for gid in range(published.ngeom)}
    gait_set = set(GAIT_FLOOR_GEOMS) | set(GAIT_SELF_GEOMS)
    gait_pairs = [(i, j) for i, j in full_pairs
                  if (name_of[i] == "floor" and name_of[j] in GAIT_FLOOR_GEOMS)
                  or (name_of[j] == "floor" and name_of[i] in GAIT_FLOOR_GEOMS)
                  or (name_of[i] in GAIT_SELF_GEOMS and name_of[j] in GAIT_SELF_GEOMS)]
    with open(XML_PATH, encoding="utf-8") as handle:
        xml_text = handle.read()
    i = xml_text.index("<option")
    published_xml = xml_text[:i] + "  " + SIZE_ELEMENT + "\n  " + xml_text[i:]
    variants = {
        "published_automatic": published_xml,
        "explicit_full": build_variant_xml(published, full_pairs),
        "gait_reduced": build_variant_xml(published, gait_pairs),
    }
    print(f"gait-reduced set: {len(gait_pairs)} pairs "
          f"(floor x {len(GAIT_FLOOR_GEOMS)} + self among {len(GAIT_SELF_GEOMS)} leg geoms)")

    result = {"protocol": ("explicit <pair> variants of walker_ragdoll.xml: the full candidate set "
                           "declared verbatim (contype/conaffinity zeroed), and a boxer-style gait-only "
                           "reduction (feet/shins/thighs vs floor and each other, head/torso/butt vs "
                           "floor, arms removed from collision)"),
              "published_candidate_pairs": len(full_pairs)}
    if args.verify:
        result["verify"] = verify(variants)
    if args.bench:
        result["bench"] = bench(variants, [int(s) for s in args.sizes.split(",")], args.seconds)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
        print(f"wrote {args.out}")
    if not args.verify and not args.bench:
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
