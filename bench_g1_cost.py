"""What the boxing project's robot costs on this machine, in the unit a training budget pays in.

`boxer-rl` trains two Unitree G1 humanoids in MJX in a fraction of this repository's wall clock, and
the contact-configuration explanation was measured and rejected (see `bench_mjx_pairs.py`). The other
candidate explanation is the model itself, so this script puts both robots on one axis: **microseconds
per simulated second**, which is what a fixed number of environment steps costs regardless of how each
model chose to slice its timestep. The ragdoll steps at dt=0.002 with frame_skip=5, so one policy
action is 0.01 s of simulated time; G1's MJX scene steps at 0.004. Comparing "steps/s" between them
would compare two different clocks.

Two paths are measured, because the claim is about where each robot is cheap:

  * native MuJoCo-C in this process (`--native`) - the path this project trains on;
  * MJX on the GPU, batched (`--mjx`, run under WSL with the mujoco 3.16 / jax 0.11 venv) - the path
    boxer-rl trains on.

The G1 files are not vendored. They are 36 MB of STL meshes under `external/` (gitignored), fetched
from a pinned Menagerie commit, and the artifact records that commit plus the hash of each XML, so a
cost number names the model it was measured on without the repository carrying the assets.

    python bench_g1_cost.py --native --out benchmarks/g1_native_cost.json
    # and under WSL, in the MJX venv:
    python bench_g1_cost.py --mjx --sizes 64,256 --out benchmarks/g1_mjx_cost.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
RAGDOLL_XML = os.path.join(HERE, "walker_ragdoll.xml")
MENAGERIE = os.path.join(HERE, "external", "mujoco_menagerie")
G1_DIR = os.path.join(MENAGERIE, "unitree_g1")
# The two G1 descriptions Menagerie ships: the native one with mesh colliders, and the MJX one whose
# geoms are all contype/conaffinity 0 with the foot-floor pairs declared by the scene that includes it.
G1_TARGETS = (("g1_native", os.path.join(G1_DIR, "g1.xml")),
              ("g1_mjx_scene", os.path.join(G1_DIR, "scene_mjx.xml")))
CONTROL_STEP_S = 0.01          # one WalkerRagdoll-v0 action: 5 mj_steps at dt 0.002


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def menagerie_commit():
    out = subprocess.run(["git", "-C", MENAGERIE, "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or None


def model_facts(model, xml):
    collidable = sum(1 for g in range(model.ngeom)
                     if model.geom_contype[g] or model.geom_conaffinity[g])
    return {"xml": os.path.relpath(xml, HERE).replace(os.sep, "/"),
            "xml_sha256": sha256(xml), "nbody": int(model.nbody), "nq": int(model.nq),
            "nv": int(model.nv), "nu": int(model.nu), "ngeom": int(model.ngeom),
            "collidable_geoms": int(collidable), "declared_pairs": int(model.npair),
            "nmesh": int(model.nmesh), "timestep": float(model.opt.timestep),
            "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
            "steps_per_control_action": round(CONTROL_STEP_S / float(model.opt.timestep), 2)}


def time_native(model, seconds=4.0, warmup=200):
    """µs per mj_step and per simulated second, on the CPU, in this process."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(warmup):
        mujoco.mj_step(model, data)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(200):
            mujoco.mj_step(model, data)
        n += 200
    dt = time.perf_counter() - t0
    us_per_step = dt / n * 1e6
    return {"mj_step_us": round(us_per_step, 1), "steps_per_s": round(n / dt, 1),
            "per_simulated_second_us": round(us_per_step / model.opt.timestep, 1),
            "control_actions_per_s": round(1.0 / (us_per_step * 1e-6
                                                 * (CONTROL_STEP_S / model.opt.timestep)), 1),
            "steps": n, "seconds": round(dt, 1)}


def time_mjx(model, mjx_model, mjx, nworld, seconds=6.0):
    """Batched MJX stepping on whatever backend jax picked, reported in the same units as native.

    MJX steps one simulation step per call per world, so the comparable quantity is again *simulated
    seconds per wall second*: mj_steps/s divided by the model's own steps-per-second. A world count is
    part of the cell because the batch is the only reason to run MJX at all.
    """
    import jax
    import jax.numpy as jnp
    from bench_mjx import compile_batch_stepper, make_batched_data

    rng = np.random.default_rng(0)
    step = compile_batch_stepper(mjx_model, mjx, frame_skip=1, unroll=False)
    batch = make_batched_data(mjx_model, mjx, nworld, rng)
    ctrl = jnp.asarray(rng.uniform(-0.5, 0.5, size=(nworld, model.nu)), dtype=jnp.float32)
    t_compile = time.perf_counter()
    out = step(batch, ctrl)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t_compile
    # Warm up before the clock starts, exactly as bench_mjx does. The first calls after a compile
    # allocate inside XLA's pool and cost about the same whatever the batch size: measured here at
    # 1.88 s per call for both 64 and 256 worlds, which is an allocator, not a simulator. Without
    # this the cell reports a rate 29x under the published harness for the same model on the same GPU.
    for _ in range(5):
        out = step(out, ctrl)
    jax.block_until_ready(out)
    t_probe = time.perf_counter()
    out = step(out, ctrl)
    jax.block_until_ready(out)
    per_call = max(time.perf_counter() - t_probe, 1e-6)
    block = max(1, int(0.5 / per_call))

    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(block):
            out = step(out, ctrl)
        jax.block_until_ready(out)
        n += block * nworld
    dt = time.perf_counter() - t0
    mj_steps_per_s = n / dt
    return {"nworld": nworld, "backend": jax.default_backend(),
            "compile_s": round(compile_s, 1), "per_call_ms": round(per_call * 1e3, 1),
            "calls_per_block": block,
            "mj_steps_per_s": round(mj_steps_per_s, 1),
            "simulated_seconds_per_wall_second": round(mj_steps_per_s * float(model.opt.timestep), 2),
            "control_actions_per_s": round(mj_steps_per_s * float(model.opt.timestep)
                                           / CONTROL_STEP_S, 2),
            "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--mjx", action="store_true")
    ap.add_argument("--sizes", default="64,256")
    ap.add_argument("--nconmax", type=int, default=512,
                    help="MJX contact capacity handed to the published instrument")
    ap.add_argument("--njmax", type=int, default=4096,
                    help="MJX constraint capacity handed to the published instrument")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--models", default="walker_ragdoll,g1_native,g1_mjx_scene",
                    help="which of the loaded models to time; the older MJX venv cannot put a "
                         "pair-declaring model through the GPU path at all, so this is how the "
                         "ragdoll-only comparison is made")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {"protocol": ("microseconds per mj_step and per simulated second, one process, "
                           "back to back; the ragdoll is this repository's published world and the "
                           "G1 files are Menagerie's, fetched to external/ and not vendored"),
              "control_step_s": CONTROL_STEP_S,
              "menagerie_commit": menagerie_commit(),
              # Recorded because these two facts changed the numbers: the MJX levels moved ~1.3x with
              # the XLA memory fraction (the WSL GPU partition is smaller than the card), and the
              # mujoco/jax pair is a different instrument from the one the published MJX table was
              # taken under - the older venv cannot even put a pair-declaring model on the GPU.
              "instrument": {"mujoco": getattr(mujoco, "__version__", None),
                             "xla_python_client_mem_fraction":
                                 os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
                             "python": subprocess.run([sys.executable, "-V"], capture_output=True,
                                                       text=True).stdout.strip()},
              "models": {}}
    loaded = {"walker_ragdoll": RAGDOLL_XML}
    loaded.update(dict(G1_TARGETS))
    for name, xml in loaded.items():
        if name not in args.models.split(","):
            continue
        if not os.path.exists(xml):
            print(f"{name}: missing ({xml}) - fetch it with the sparse Menagerie clone in the README")
            continue
        model = mujoco.MjModel.from_xml_path(xml)
        cell = model_facts(model, xml)
        if args.native:
            cell["native"] = time_native(model, args.seconds)
        if args.mjx:
            from bench_mjx import build_mjx
            try:
                # The published instrument rather than a second copy of it: build_mjx is what hands
                # MJX concrete <size> capacities, and put_model on a file whose model compiles with
                # nconmax=njmax=-1 measures a different simulator state - which is how this harness
                # landed ~40x under benchmarks/mjx_gpu_wsl.json for the very same ragdoll.
                _, mjx_model, mjx = build_mjx(args.nconmax, args.njmax, False, xml=xml)
            except NotImplementedError as exc:
                # Not a failure to paper over: Menagerie's native G1 description collides cylinders
                # against meshes, and MJX implements no such pair - which is exactly why the project
                # ships a second, primitive-collider file for the GPU path. The cost comparison has to
                # say which model each number belongs to.
                cell["mjx"] = {"unsupported": str(exc)}
            else:
                cell["mjx"] = [time_mjx(model, mjx_model, mjx, int(s), args.seconds)
                               for s in args.sizes.split(",")]
                cell["mjx_instrument"] = (f"bench_mjx.build_mjx nconmax={args.nconmax} "
                                          f"njmax={args.njmax}, one mj_step per vmap'd call")
        result["models"][name] = cell
        nat = cell.get("native", {})
        print(f"{name:16s} nv={cell['nv']:3d} nu={cell['nu']:3d} pairs={cell['declared_pairs']:4d} "
              f"dt={cell['timestep']:.4f} "
              f"mj_step={nat.get('mj_step_us','-'):>7} us  "
              f"per simulated second={nat.get('per_simulated_second_us','-'):>12} us")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
