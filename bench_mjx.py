"""MJX (JAX) throughput benchmark for WalkerRagdoll-v0.

Answers one question: does batching ``walker_ragdoll.xml`` through MJX beat the
process-parallel MuJoCo-C vector env that is already committed
(``envs/parallel_vector_env.py``, ~6.7k env-steps/s at n=32)?

An "env step" here is one policy action = ``frame_skip`` mj_steps at
``timestep=0.002``, exactly as ``WalkerRagdollEnv.step`` does it, and it includes
``mj_rnePostConstraint`` because the impact cost reads ``cfrc_ext``.
Throughput is ``N worlds x actions/s``.

Usage:
    # CPU-only JAX on this machine; one subprocess per batch size so a hopeless
    # XLA compile cannot take the whole sweep down with it.
    python bench_mjx.py --sizes 32,128,256,512,1024 --json benchmarks/mjx_cpu_batch.json

    python bench_mjx.py --sizes 32 --single 32        # one size, in this process
    python bench_mjx.py --mode cpu-ref                # MuJoCo-C physics only
    python bench_mjx.py --x64                         # float64 MJX (parity studies)

Notes on what is being measured:
  * ``jax.default_backend()`` is **cpu** in this venv: jaxlib ships no CUDA wheels
    for native Windows, so every number below is CPU XLA. GPU MJX needs WSL2 or
    the repo Dockerfile.
  * XLA compile time grows with the batch size on CPU (the vmap'd graph is
    unrolled per world), so compile time is measured and reported separately from
    the steady-state rate. ``--compile-deadline`` aborts a size that exceeds it.
  * MJX is dense: every geom pair in the model is collision-tested every step, so
    its cost is almost independent of the state, while MuJoCo-C only pays for
    active contacts. See the printed ``ncon`` slot count.

GETTING THE GPU NUMBERS (WSL2, verified 2026-10-01)
  The repo Dockerfile is ``python:3.11-slim`` with no CUDA base image, so it cannot
  run MJX on the GPU either. WSL2 works:

      wsl.exe -d Ubuntu-24.04
      python3 -m venv ~/venv-mjx
      ~/venv-mjx/bin/pip install -r /mnt/d/mujoco-walker/requirements-mjx.txt
      ~/venv-mjx/bin/pip install "jax-cuda12-plugin[with-cuda]==0.10.2"
      cd /mnt/d/mujoco-walker
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 ~/venv-mjx/bin/python bench_mjx.py --sizes 32,128,512,1024

  ``XLA_PYTHON_CLIENT_MEM_FRACTION`` matters on an 8 GB card: JAX preallocates 75%
  of VRAM by default, the desktop compositor holds some of it, and the allocator
  then retries downwards and spams ``CUDA_ERROR_OUT_OF_MEMORY`` to stderr.
  With that set, ``jax.devices()`` reports ``[CudaDevice(id=0)]`` and this script
  benchmarks the RTX 4070 Laptop through it.

RESULT (do not re-derive; see benchmarks/mjx_cpu_batch.json and mjx_gpu_wsl.json)
  MJX loses. Best MJX-CPU was 32 env-steps/s at nworld=32 against the committed
  MuJoCo-C process path at 6,694; MJX-GPU saturates at ~1,275 env-steps/s as the
  environment is configured (RK4 + cfrc_ext), and only reaches ~4,338 if the model
  is switched to Euler, which changes the dynamics. No MJX env is shipped.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "walker_ragdoll.xml")

# Reference numbers from the committed CPU path, measured 2026-01 on this machine
# (Windows 11, i9-14900HX, .venv: python 3.8.10, mujoco 3.2.3, gymnasium 0.29.1).
# See benchmarks/vec_backend_scaling.json for the full table.  Labelled as
# historical in every JSON record so nobody mistakes them for this run's output.
CPU_REFERENCE = {
    "source": "benchmarks/vec_backend_scaling.json + bench_env.py, measured 2026-10-01",
    "mj_step_per_s_single_thread": 14000.0,
    "env_step_us_single_thread": 698.9,
    "parallel_vec_env_steps_per_s_n32": 6694.0,
    "note": "parallel path is 32 worker processes; ~130us of serial parent-side pickle work per env per vector step is the ceiling",
}

SIZE_ELEMENT = '<size nconmax="{nconmax}" njmax="{njmax}"/>'


def xml_with_sizes(nconmax=512, njmax=4096, integrator=None, xml=None):
    """Compile-ready XML text with the contact/constraint capacities MJX requires (no file is touched).

    MJX needs concrete constraint/contact capacities.  ``MjModel.nconmax`` is a
    read-only property in the Python binding, so the only way to set it is to
    compile the model from XML text that already contains the element.  Both of this
    repo's models compile with ``nconmax = njmax = -1`` from their own files, so a
    caller that skips this function and hands ``mjx.put_model`` the bare model is not
    measuring the same simulator state as the table below - which is how one hand
    -rolled MJX timer came out ~40x under the numbers committed here.

    ``xml`` points at any model file, so a foreign robot is priced by this instrument
    rather than by a second copy of it.

    ``integrator`` rewrites the option for headroom studies only - the committed
    environment is RK4 and every headline number uses it.
    """
    path = xml or XML_PATH
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    sizes = SIZE_ELEMENT.format(nconmax=nconmax, njmax=njmax)
    if integrator:
        # MuJoCo's XML parser is picky about the exact spelling of each keyword
        # ("Euler" parses, "euler"/"EULER" do not), so map from the enum-style
        # names this CLI accepts rather than passing the user's casing through.
        spelling = {"EULER": "Euler", "IMPLICIT": "implicit",
                    "IMPLICITFAST": "implicitfast", "RK4": "RK4"}.get(integrator.upper())
        if spelling is None:
            raise ValueError(f"unknown integrator {integrator!r}")
        text, hits = re.subn(r'integrator="[^"]*"', f'integrator="{spelling}"', text, count=1)
        if not hits:
            raise RuntimeError(f"{os.path.basename(path)} declares no integrator to rewrite")
    if "<size" in text:
        # A model that ships its own capacities gets them rewritten in place: MuJoCo keeps
        # the last <size> in the document, so appending a second one would leave whichever
        # wins dependent on where it sits.
        text, hits = re.subn(r"<size\b[^>]*/>", sizes, text, count=1)
        if not hits:
            raise RuntimeError(f"{os.path.basename(path)} has a <size> element this rewrite cannot reach")
        return text
    if "<option" in text:
        i = text.index("<option")
    else:
        # Foreign scenes often carry their <option> in an include, so the only guaranteed
        # anchor is just inside the root element.
        i = text.index(">", text.index("<mujoco")) + 1
    return text[:i] + "  " + sizes + "\n  " + text[i:]


# --------------------------------------------------------------------------- #
# MuJoCo-C (the committed path), measured in this same process
# --------------------------------------------------------------------------- #
def _sha256(path):
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def cpu_physics_reference(seconds, frame_skip, xml=None):
    """MuJoCo-C cost of the physics inside one env step, in two states.

    MuJoCo-C is sparse: it only pays for *active* contacts, and a ragdoll driven by
    random actions spends most of its time on the floor with many more of them than
    when it stands.  MJX is dense - all 138 geom pairs are solved every step - so its
    CPU cost looks like MuJoCo's worst case *always*.  Both states are reported so
    the comparison cannot be made to look good by accidentally picking the cheap one.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml or XML_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    def measure(settle_actions, reset_every):
        mujoco.mj_resetData(model, data)
        if settle_actions:
            for _ in range(settle_actions):
                data.ctrl[:] = rng.uniform(-0.5, 0.5, size=model.nu)
                mujoco.mj_step(model, data, nstep=frame_skip)
        ncon, nefc = data.ncon, data.nefc

        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for _ in range(200):
                data.ctrl[:] = rng.uniform(-0.5, 0.5, size=model.nu)
                mujoco.mj_step(model, data, nstep=frame_skip)
                # mj_step does not produce cfrc_ext; the impact cost needs the same
                # extra pass WalkerRagdollEnv._step_mujoco_simulation makes.
                mujoco.mj_rnePostConstraint(model, data)
            n += 200
            if reset_every and n % reset_every == 0:
                mujoco.mj_resetData(model, data)
        action = n / (time.perf_counter() - t0)
        return {
            "policy_actions_per_s": action,
            "us_per_policy_action": 1e6 / action,
            "mj_step_equivalent_per_s": action * frame_skip,
            "ncon": int(ncon),
            "nefc": int(nefc),
        }

    standing = measure(settle_actions=0, reset_every=10000)
    fallen = measure(settle_actions=300, reset_every=0)
    return {
        "backend": "MuJoCo C, single thread, in-process",
        "frame_skip": frame_skip,
        "standing": standing,
        "fallen": fallen,
        # Kept for the table printer: the fallen state is the one a training run is in
        # most of the time, and the one MJX is comparable to.
        "mj_step_per_s": standing["mj_step_equivalent_per_s"],
        "env_steps_per_s_n1": standing["policy_actions_per_s"],
        "us_per_policy_action": standing["us_per_policy_action"],
    }


# --------------------------------------------------------------------------- #
# MJX
# --------------------------------------------------------------------------- #
def build_mjx(nconmax, njmax, x64, integrator=None, xml=None):
    import jax
    import mujoco
    import mujoco.mjx as mjx

    if x64:
        jax.config.update("jax_enable_x64", True)
    path = os.path.abspath(xml or XML_PATH)
    text = xml_with_sizes(nconmax, njmax, integrator, xml)
    if "<include" in text:
        # A scene that includes its robot cannot be compiled from a string: the include and the
        # meshdir resolve against the file's own directory, which from_xml_string has no way to
        # know. The rewritten text goes beside the original so the same instrument reads it.
        sibling = os.path.join(os.path.dirname(path), "_mjx_sized_" + os.path.basename(path))
        with open(sibling, "w", encoding="utf-8") as handle:
            handle.write(text)
        model = mujoco.MjModel.from_xml_path(sibling)
    else:
        model = mujoco.MjModel.from_xml_string(text)
    mjx_model = mjx.put_model(model)
    return model, mjx_model, mjx


def make_batched_data(mjx_model, mjx, nworld, rng, qpos_noise=0.01):
    """Tile one mjx.Data into an nworld batch.

    ``ne/nf/nl/nefc/ncon`` are static Python ints in MJX 3.2.7 (the contact set is
    fixed by the model's geom pairs, not by the state), so they must be left alone
    rather than tiled - vmap carries them through as pytree aux data.
    """
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu

    d0 = mjx.make_data(mjx_model)
    qpos = jnp.asarray(d0.qpos) + jnp.array(
        rng.normal(scale=qpos_noise, size=np.shape(d0.qpos)), dtype=jnp.float32
    )
    d0 = d0.replace(qpos=qpos, ctrl=jnp.zeros_like(jnp.asarray(d0.ctrl)))
    d0 = mjx.forward(mjx_model, d0)

    def tile(x):
        if isinstance(x, (int, np.integer, float, np.floating)):
            return x
        a = jnp.asarray(x)
        return jnp.broadcast_to(a[None], (nworld,) + a.shape)

    return jtu.tree_map(tile, d0)


def compile_batch_stepper(mjx_model, mjx, frame_skip, unroll, postconstraint=True):
    """Return jit(vmap(step x frame_skip [+ rnePostConstraint])) over (data, ctrl)."""
    import jax

    def step_world(d, ctrl):
        if unroll:
            for _ in range(frame_skip):
                d = d.replace(ctrl=ctrl)
                d = mjx.step(mjx_model, d)
        else:
            def body(i, dd):
                return mjx.step(mjx_model, dd.replace(ctrl=ctrl))
            d = jax.lax.fori_loop(0, frame_skip, body, d)
        if not postconstraint:
            return d
        # cfrc_ext feeds impact_cost; mjx.step does not compute it, exactly like
        # mj_step, so the extra pass the CPU env makes has to be mirrored here.
        return mjx.rne_postconstraint(mjx_model, d)

    return jax.jit(jax.vmap(step_world, in_axes=(0, 0)))


def mjx_single_step_rate(mjx_model, mjx, seconds):
    """MJX physics rate at nworld=1, frame_skip=1: apples-to-apples with mj_step/s."""
    import jax
    import jax.numpy as jnp

    d0 = mjx.make_data(mjx_model)
    step1 = jax.jit(lambda d: mjx.step(mjx_model, d))
    d = step1(d0)
    jax.block_until_ready(d)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(100):
            d = step1(d)
        jax.block_until_ready(d)
        n += 100
    return n / (time.perf_counter() - t0)


def bench_mjx_batch(nworld, actions, seconds, frame_skip, nconmax, njmax, x64,
                   compile_deadline, unroll, integrator=None, postconstraint=True, xml=None):
    import jax
    import mujoco

    rng = np.random.default_rng(0)
    model, mjx_model, mjx = build_mjx(nconmax, njmax, x64, integrator, xml)
    path = os.path.abspath(xml or XML_PATH)

    # dt/action in the record is the *compiled* model's own timestep, not this repo's 0.002:
    # pricing a model that slices 0.004 under a 0.002 label is the exact unit error this
    # comparison exists to avoid.
    dt_per_action = frame_skip * float(model.opt.timestep)
    rec = {
        "nworld": nworld,
        "xml": os.path.basename(path),
        "xml_sha256": _sha256(path),
        "frame_skip": frame_skip,
        "dt_per_action": dt_per_action,
        "mjx_static_contact_slots": int(mjx.make_data(mjx_model).ncon),
        "mjx_nconmax": int(model.nconmax),
        "mjx_njmax": int(model.njmax),
        "dtype": "float64" if x64 else "float32",
        "compile_mode": "unrolled" if unroll else "fori_loop",
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "postconstraint_for_cfrc_ext": bool(postconstraint),
        "jax_backend": jax.default_backend(),
        "jax_devices": str(jax.devices()),
    }

    fn = compile_batch_stepper(mjx_model, mjx, frame_skip, unroll, postconstraint)
    batch = make_batched_data(mjx_model, mjx, nworld, rng)
    ctrl = jnp_array(rng.uniform(-0.5, 0.5, size=(nworld, model.nu)))

    t0 = time.perf_counter()
    out = fn(batch, ctrl)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    rec["compile_s"] = compile_s
    if compile_s > compile_deadline:
        rec["status"] = f"compile exceeded deadline ({compile_s:.0f}s)"

    # Warm up the steady state (first few calls still settle allocations).
    d = out
    for _ in range(5):
        d = fn(d, ctrl)
    jax.block_until_ready(d)

    # Enough repetitions to cover `seconds`, measured in whole blocks so the
    # Python loop overhead cannot be mistaken for simulator time.
    t_single = time.perf_counter()
    d = fn(d, ctrl)
    jax.block_until_ready(d)
    per_call = max(time.perf_counter() - t_single, 1e-6)
    block = max(1, int(0.5 / per_call))

    elapsed = 0.0
    total = 0
    clock = time.perf_counter()
    while True:
        for _ in range(block):
            d = fn(d, ctrl)
        jax.block_until_ready(d)
        total += block
        elapsed = time.perf_counter() - clock
        if elapsed >= seconds:
            break
        if elapsed > compile_deadline:
            break

    aps = total / elapsed
    rec.update({
        "actions_per_s": aps,
        "env_steps_per_s": aps * nworld,
        "us_per_env_step": 1e6 / aps / nworld,
        "us_per_vector_step": 1e6 / aps,
        "vector_steps_per_s": aps,
        # The two units that survive a change of model: aggregate physics steps, and aggregate
        # simulated seconds advanced per wall-clock second. steps/s alone compares a 0.002 model
        # with a 0.004 one as if the clocks were the same.
        "mj_steps_per_s": aps * nworld * frame_skip,
        "simulated_seconds_per_wall_second": aps * nworld * frame_skip * float(model.opt.timestep),
        "measured_s": elapsed,
        "status": rec.get("status", "ok"),
    })
    rec["mjx_single_step_physics_per_s_n1"] = mjx_single_step_rate(mjx_model, mjx, min(seconds, 3.0))
    return rec


def jnp_array(x):
    import jax.numpy as jnp
    return jnp.asarray(x, dtype=jnp.float32)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_single(args):
    if args.mode == "cpu-ref":
        return cpu_physics_reference(args.seconds, args.frame_skip, args.xml)
    return bench_mjx_batch(
        nworld=args.single, actions=args.actions, seconds=args.seconds,
        frame_skip=args.frame_skip, nconmax=args.nconmax, njmax=args.njmax,
        x64=args.x64, compile_deadline=args.compile_deadline, unroll=args.unroll,
        integrator=args.integrator, postconstraint=not args.no_postconstraint, xml=args.xml,
    )


def run_sweep(args):
    results = []
    if args.mode != "mjx":
        raise SystemExit(f"--mode {args.mode} is handled before the sweep")
    results.append(cpu_physics_reference(args.seconds, args.frame_skip, args.xml))
    for n in args.sizes:
        print(f"[mjx-bench] nworld={n} launching subprocess "
              f"(compile deadline {args.compile_deadline:.0f}s)", flush=True)
        # One result file per size: reusing a single path lets a crashed child
        # leave the previous size's record behind and the parent reads it as this one.
        size_out = args.out or os.path.join(HERE, f"_mjx_scratch_single_{n}.json")
        if os.path.exists(size_out):
            os.remove(size_out)
        cmd = [sys.executable, os.path.abspath(__file__), "--single", str(n),
               "--seconds", str(args.seconds), "--frame-skip", str(args.frame_skip),
               "--nconmax", str(args.nconmax), "--njmax", str(args.njmax),
               "--compile-deadline", str(args.compile_deadline),
               "--out", size_out]
        if args.x64:
            cmd.append("--x64")
        if args.unroll:
            cmd.append("--unroll")
        if args.integrator:
            cmd += ["--integrator", args.integrator]
        if args.no_postconstraint:
            cmd.append("--no-postconstraint")
        if args.xml:
            # absolute: the child runs with cwd=HERE, which is not where a shell's cwd was.
            cmd += ["--xml", os.path.abspath(args.xml)]
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, timeout=args.compile_deadline + args.seconds + 240,
                           cwd=HERE)
        except subprocess.TimeoutExpired:
            results.append({"nworld": n, "status": "TIMEOUT",
                            "note": "XLA compile exceeded the deadline; treat as not viable at this batch size",
                            "wall_s": time.perf_counter() - t0})
            print(f"[mjx-bench] nworld={n} TIMEOUT after {time.perf_counter()-t0:.0f}s", flush=True)
            continue
        except subprocess.CalledProcessError as e:
            results.append({"nworld": n, "status": f"ERROR rc={e.returncode}"})
            print(f"[mjx-bench] nworld={n} FAILED rc={e.returncode}", flush=True)
            continue
        with open(size_out, "r", encoding="utf-8") as f:
            results.append(json.load(f))
        r = results[-1]
        print(f"[mjx-bench] nworld={n}: {r.get('env_steps_per_s', 0):,.0f} env-steps/s "
              f"compile {r.get('compile_s', 0):.1f}s ({r.get('status')})", flush=True)
    return results


def print_table(results, cpu_parallel_n32):
    print()
    print(f"host: {platform.processor() or 'unknown'}   {platform.system()} {platform.release()}")
    print("=" * 96)
    hdr = f"{'path':40s} {'env-steps/s':>12s} {'us/env-step':>12s} {'compile/s':>10s}  status"
    print(hdr)
    print("-" * 96)
    for r in results:
        if r.get("backend", "").startswith("MuJoCo C"):
            st, fa = r["standing"], r["fallen"]
            print(f"{'MuJoCo-C physics only (n=1, 1 thread)':40s} "
                  f"{st['policy_actions_per_s']:>12,.0f} {st['us_per_policy_action']:>12.1f} {'-':>10s}  "
                  f"standing, mj_step {st['mj_step_equivalent_per_s']:,.0f}/s, ncon={st['ncon']}")
            print(f"{'MuJoCo-C physics only (n=1, fallen)':40s} "
                  f"{fa['policy_actions_per_s']:>12,.0f} {fa['us_per_policy_action']:>12.1f} {'-':>10s}  "
                  f"ncon={fa['ncon']} nefc={fa['nefc']}")
            continue
        if "dtype" not in r:
            # TIMEOUT / ERROR records carry no measurement, only a status.
            print(f"{'MJX nworld=' + str(r.get('nworld', '?')):40s} "
                  f"{'-> no result':>12s} {'-':>12s} {'-':>10s}  {r.get('status')}")
            continue
        label = f"MJX {r['dtype']} {r['jax_backend']} nworld={r['nworld']} (skip={r['frame_skip']})"
        print(f"{label:40s} {r['env_steps_per_s']:>12,.0f} {r['us_per_env_step']:>12.1f} "
              f"{r['compile_s']:>10.1f}  {r['status']}")
    print("-" * 96)
    print(f"{'MuJoCo-C ParallelVectorEnv n=32 (committed)':40s} "
          f"{cpu_parallel_n32:>12,.0f} {1e6/cpu_parallel_n32:>12.1f} {'-':>10s}  reference, "
          f"measured 2026-10-01")
    best = [r for r in results if "env_steps_per_s" in r and "nworld" in r]
    if best:
        top = max(best, key=lambda r: r["env_steps_per_s"])
        print(f"\nbest MJX-CPU: {top['env_steps_per_s']:,.0f} env-steps/s at nworld={top['nworld']} "
              f"= {top['env_steps_per_s']/cpu_parallel_n32:.3f}x the committed parallel path")
        print(f"go/no-go bar (3x {cpu_parallel_n32:,.0f}) = {3*cpu_parallel_n32:,.0f} env-steps/s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="32,128,256,512,1024",
                   help="comma-separated world batch sizes for the sweep")
    p.add_argument("--single", type=int, default=None, help="run one size in this process")
    p.add_argument("--mode", choices=["mjx", "cpu-ref"], default="mjx")
    p.add_argument("--seconds", type=float, default=3.0, help="steady-state wall clock per measurement")
    p.add_argument("--actions", type=int, default=0, help="unused; timing is wall-clock bounded")
    p.add_argument("--frame-skip", type=int, default=5, help="mj_steps per policy action")
    p.add_argument("--xml", default=None,
                   help="price this model file instead of walker_ragdoll.xml; set --frame-skip to "
                        "its own control step / timestep (the G1 MJX scene: 0.02 s / 0.004 s = 5)")
    p.add_argument("--nconmax", type=int, default=512)
    p.add_argument("--njmax", type=int, default=4096)
    p.add_argument("--x64", action="store_true", help="enable float64 MJX (slower)")
    p.add_argument("--unroll", action="store_true", help="unroll frame_skip instead of lax.fori_loop")
    p.add_argument("--integrator", default=None, type=lambda s: s.upper(),
                   help="rewrite the model integrator for headroom studies "
                        "(EULER|IMPLICITFAST; default: keep the model's RK4)")
    p.add_argument("--no-postconstraint", action="store_true",
                   help="skip mjx.rne_postconstraint (drops cfrc_ext, so impact_cost is unavailable)")
    p.add_argument("--compile-deadline", type=float, default=420.0,
                   help="abort a batch size whose compile+measurement exceeds this many seconds")
    p.add_argument("--json", default=None, help="write the sweep results here")
    p.add_argument("--out", default=None, help="internal: single-result path for subprocesses")
    p.add_argument("--cpu-parallel-json", default=os.path.join("benchmarks", "vec_backend_scaling.json"),
                   help="committed CPU parallel numbers to compare against")
    args = p.parse_args()

    if args.single is not None:
        rec = run_single(args)
        with open(args.out or "mjx_single.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(json.dumps(rec, indent=2))
        return

    if args.mode == "cpu-ref":
        rec = cpu_physics_reference(args.seconds, args.frame_skip, args.xml)
        print(json.dumps(rec, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
        return

    cpu_parallel = CPU_REFERENCE["parallel_vec_env_steps_per_s_n32"]
    path = os.path.join(HERE, args.cpu_parallel_json)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            row = [r for r in doc.get("full_trainer_stack", []) if r.get("num_envs") == 32]
            if row:
                cpu_parallel = float(row[0]["parallel_env_steps_per_s"])
        except Exception as e:  # noqa: BLE001 - the hardcoded reference still stands
            print(f"[mjx-bench] could not read {path}: {e}")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    args.sizes = sizes
    results = run_sweep(args)
    print_table(results, cpu_parallel)

    if args.json:
        doc = {
            "machine": f"{platform.system()} {platform.release()}, {platform.processor()}",
            "backend": "MJX on JAX CPU (jaxlib has no CUDA wheel for native Windows)",
            "protocol": (f"{os.path.basename(args.xml) if args.xml else 'WalkerRagdoll model'},"
                         f" frame_skip={args.frame_skip}, "
                         f"dt/action={args.frame_skip*0.002:.3f}s, ctrl U(-0.5,0.5), "
                         f"{args.seconds}s steady-state per size, "
                         f"nconmax={args.nconmax} njmax={args.njmax}"),
            "dtype": "float64" if args.x64 else "float32",
            "results": results,
            "cpu_reference": CPU_REFERENCE,
            "cpu_parallel_env_steps_per_s_n32": cpu_parallel,
        }
        out = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
