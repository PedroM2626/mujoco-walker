"""Where does a Dreamer update actually go? Profile the real trainer, do not guess.

Dreamer is the Phase-1 trainer whose wall clock is almost entirely learner: with the update gate
closed it collects 5,000 steps in 37 s (README, learner-side section), so ~96% of a normal run is
gradient work. The world-model KL was already batched over time (2.06x on that op, 1.10x on the
run) and the reason the run barely moved is that the rest of the update is more Python loops over
the same tiny MLPs: `WorldModel.forward` walks 50 timesteps, the imagination rollout walks
`--imag-horizon` more.

Collapsing a recurrence is not available the way it was for the REDQ ensemble - h_t depends on
h_{t-1} - so before proposing anything (per-head stacking, CUDA graph capture, compile) this
measures what each part costs. It runs the real `train_dreamer.train_dreamer()` under
torch.profiler for a short budget and reports self CUDA time and kernel-launch counts attributed to
the Python frames, because on 256-unit nets the launch count is the quantity that matters.

    python bench_dreamer_update.py                     # default: 9k steps, 4 sync envs
    python bench_dreamer_update.py --steps 20000

The budget has to clear the trainer's own update gate (`global_step >= 5000`), so 9k steps profile
roughly 1,000 updates - enough that the per-call numbers below divide cleanly.

Writes benchmarks/dreamer_update_profile.json. Keep MLflow pointed at a throwaway database so a
profile run does not show up as training evidence.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

CHILD = r'''
import json, os, sys, torch
from torch.profiler import ProfilerActivity, profile, record_function
sys.argv = ["train_dreamer.py"] + sys.argv[1:]
import train_dreamer as td

def patched_forward(self, obs_seq, action_seq, done_seq):
    with record_function("world_model_forward"):
        return _orig_forward(self, obs_seq, action_seq, done_seq)

def patched_transition(self, h, z, action):
    with record_function("rssm_transition"):
        return _orig_transition(self, h, z, action)

_orig_forward = td.WorldModel.forward
_orig_transition = td.RSSM.transition
td.WorldModel.forward = patched_forward
td.RSSM.transition = patched_transition

acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU]
with profile(activities=acts) as prof:
    td.train_dreamer()

rows = {}
for e in prof.key_averages():
    if e.count == 0:
        continue
    rows[e.key] = {"count": e.count, "self_cpu_us": round(e.self_cpu_time_total, 1),
                   "self_cuda_us": round(getattr(e, "self_device_time_total", 0.0), 1)}
with open(os.environ["DREAMER_PROFILE_OUT"], "w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=2)
'''


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["profile", "ab", "reproducibility", "scaling",
                                      "checkpoint-cost"], default="profile")
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reps", type=int, default=60,
                   help="timed updates per config in --mode scaling (5 untimed warmup run first)")
    return p.parse_args()


BASE = ["--device", "cuda", "--vec-backend", "sync", "--num-envs", "4", "--batch-size", "16",
        "--seq-len", "50", "--checkpoint-interval", "100000000", "--buffer-size", "20000",
        "--task-phase", "target", "--reset-mode", "upright"]
# Order alternates between reps so a run sharing this box hits both arms about equally.
# Both arms name their mode explicitly: `--update-graph` became the CUDA default, so an arm that
# passes nothing would quietly measure the captured path and report it as eager.
EAGER = ["--no-update-graph"]
GRAPH = ["--update-graph"]
FLOOR = ["--learning-starts", "999999"]
AB_ARMS = [("eager", "e1", EAGER), ("graph", "g1", GRAPH),
           ("floor_eager", "fe1", FLOOR + EAGER),
           ("floor_graph", "fg1", FLOOR + GRAPH),
           ("graph", "g2", GRAPH), ("eager", "e2", EAGER),
           ("floor_graph", "fg2", FLOOR + GRAPH),
           ("floor_eager", "fe2", FLOOR + EAGER)]


def trainer_cmd(run_id, args, extra):
    return ["--run-id", run_id, "--seed", str(args.seed), "--total-timesteps", str(args.steps)] + BASE + extra


def run_trainer(run_id, args, extra):
    """One short real Dreamer run; returns wall seconds and the episodic-return lines it printed."""
    db = "sqlite:///" + os.path.join(tempfile.gettempdir(), f"bench_{run_id}.db").replace(os.sep, "/")
    env = dict(os.environ, MLFLOW_TRACKING_URI=db)
    t0 = time.perf_counter()
    proc = subprocess.run([PY, "-u", "train_dreamer.py"] + trainer_cmd(run_id, args, extra),
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    dt = time.perf_counter() - t0
    returns = [float(re.search(r"episodic_return=([-\d.]+)", l).group(1))
               for l in proc.stdout.splitlines() if "episodic_return=" in l]
    print(f"{run_id:6} {dt:8.1f} s  {args.steps / dt:7.1f} env-steps/s  episodes={len(returns)} "
          f"exit={proc.returncode}", flush=True)
    if proc.returncode:
        print("\n".join(proc.stderr.splitlines()[-10:]), flush=True)
        return None, []
    return round(dt, 2), returns


def clean_runs(prefixes):
    for entry in ("checkpoints", "runs"):
        root = os.path.join(ROOT, entry)
        if not os.path.isdir(root):
            continue
        for d in os.scandir(root):
            if d.is_dir() and any(d.name.startswith(p) for p in prefixes):
                shutil.rmtree(d.path, ignore_errors=True)


def main():
    args = parse_args()
    if args.mode == "ab":
        return ab(args)
    if args.mode == "reproducibility":
        return reproducibility(args)
    if args.mode == "scaling":
        return scaling(args)
    if args.mode == "checkpoint-cost":
        return checkpoint_cost(args)
    return run_profile(args)


SCALING_CONFIGS = [  # (seq_len, batch, imag_horizon) - the shipped one first, then one-axis sweeps
    (50, 16, 15), (50, 16, 1), (50, 16, 4), (50, 16, 30), (6, 16, 15), (25, 16, 15),
    # The batch sweep answers the other half: if the update barely grows with batch, the box is
    # not doing arithmetic, it is paying per launch, and the head count is what to cut.
    (50, 64, 15), (50, 256, 15),
]


def _other_gpu_contexts():
    """How many processes already hold a CUDA context on this GPU - the window the rate came from."""
    try:
        proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError):
        return None
    pids = [int(x) for x in proc.stdout.split() if x.strip().isdigit()]
    return len([p for p in pids if p != os.getpid()])


def _slope(points):
    """Least-squares slope of (x, y) - used here for ms per extra loop step."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / den if den else float("nan")


def scaling(args):
    """What one imagination step and one world-model timestep actually cost, eager and captured.

    Graph capture removed the dispatch, so the question the README leaves open - "the imagination
    rollout and the actor-critic losses are Python loops too, that is where the same technique
    would go next" - needs a number before any code: both loops are recurrences (h_t depends on
    h_{t-1}) and cannot be collapsed the way the REDQ ensemble was. The only fusion left is across
    heads that share an input, so it is worth exactly the fraction of the update those heads
    occupy. This measures that fraction by sweeping one axis at a time.
    """
    import statistics as st
    import torch
    import train_dreamer as td
    from types import SimpleNamespace

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBS, ACT = 49, 17

    def build(L, B, H, graph):
        torch.manual_seed(0)
        mods = [td.WorldModel(OBS, ACT).to(dev), td.DreamerActor(256, 32, ACT).to(dev),
                td.DreamerCritic(256, 32).to(dev)]
        opts = [td.make_adam(m.parameters(), 3e-4, graph) for m in mods]
        torch.manual_seed(1)
        batch = (torch.randn(L, B, OBS, device=dev), torch.randn(L - 1, B, ACT, device=dev),
                 torch.randn(L - 1, B, 1, device=dev),
                 torch.zeros(L - 1, B, 1, device=dev, dtype=torch.bool))
        torch.manual_seed(2)
        idx = torch.randperm(L * B)[:B].to(dev)
        a = SimpleNamespace(seq_len=L, batch_size=B, imag_horizon=H, gamma=0.99, kl_weight=1.0)
        fn = lambda b, i: td.dreamer_update(*mods, *opts, b, a, i,
                                            zero_grad_set_to_none=not graph)
        if graph:
            return td.CapturedDreamerUpdate(fn, batch, idx), batch, idx
        return fn, batch, idx

    def time_arm(L, B, H, graph):
        fn, batch, idx = build(L, B, H, graph)
        for _ in range(5):
            fn(batch, idx)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(batch, idx)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        return {"median_ms": round(st.median(samples), 2), "min_ms": round(min(samples), 2),
                "max_ms": round(max(samples), 2), "reps": args.reps}

    out = {"device": str(dev), "note": "isolated dreamer_update calls at the shipped sizes "
                                      "(seq_len 50, batch 16, imag_horizon 15), one config at a "
                                      "time; marginal costs come from the one-axis sweeps",
           "gpu_other_contexts": _other_gpu_contexts(), "configs": {}}
    print(f"{'seq':>4}{'batch':>7}{'imag':>6}  {'eager ms':>18}  {'captured ms':>18}")
    for L, B, H in SCALING_CONFIGS:
        eager, cap = time_arm(L, B, H, False), time_arm(L, B, H, True)
        out["configs"][f"L{L}_B{B}_H{H}"] = {"seq_len": L, "batch": B, "imag_horizon": H,
                                             "eager": eager, "captured": cap}
        print(f"{L:>4}{B:>7}{H:>6}  {eager['median_ms']:>9} /{eager['min_ms']:<8}"
              f"  {cap['median_ms']:>9} /{cap['min_ms']:<8}")

    base = out["configs"]["L50_B16_H15"]
    out["marginal_ms_per_unit"] = {}
    for arm in ("eager", "captured"):
        pts_h = [(v, out["configs"][f"L50_B16_H{v}"][arm]["median_ms"]) for v in (1, 4, 15, 30)]
        pts_l = [(v, out["configs"][f"L{v}_B16_H15"][arm]["median_ms"]) for v in (6, 25, 50)]
        pts_b = [(v, out["configs"][f"L50_B{v}_H15"][arm]["median_ms"]) for v in (16, 64, 256)]
        out["marginal_ms_per_unit"][f"imag_horizon_{arm}"] = round(_slope(pts_h), 3)
        out["marginal_ms_per_unit"][f"seq_len_{arm}"] = round(_slope(pts_l), 3)
        out[f"batch_scaling_{arm}"] = {str(x): y for x, y in pts_b}
    out["baseline_captured_ms"] = base["captured"]["median_ms"]
    target = os.path.join(ROOT, "benchmarks", "dreamer_update_scaling.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("wrote", os.path.relpath(target, ROOT), "marginal:",
          out.get("marginal_ms_per_unit"))
    return 0


def ab(args):
    """Eager update against captured update, same seed, same budget, with a no-update floor each."""
    if args.steps == 9000:
        args.steps = 5400          # 5,000 collection + 100 updates at the shipped num_envs=4
    results, episodes = {}, {}
    try:
        for label, tag, extra in AB_ARMS:
            results[tag], episodes[tag] = run_trainer(f"bd_{tag}", args, extra)
    finally:
        clean_runs(("bd_",))

    out = os.path.join(ROOT, "benchmarks", "dreamer_update_graph_ab.json")
    n_updates = (args.steps - 5000) // 4          # one update per vector step after the gate
    summary = {}
    for rep in ("1", "2"):
        e, g = results.get(f"e{rep}"), results.get(f"g{rep}")
        fe, fg = results.get(f"fe{rep}"), results.get(f"fg{rep}")
        if None in (e, g, fe, fg):
            continue
        ms_e, ms_g = (e - fe) * 1e3 / n_updates, (g - fg) * 1e3 / n_updates
        s = {"eager_s": e, "graph_s": g, "floor_eager_s": fe, "floor_graph_s": fg,
             "end_to_end_ratio": round(e / g, 2),
             "update_phase_eager_s": round(e - fe, 2), "update_phase_graph_s": round(g - fg, 2),
             "ms_per_update_eager": round(ms_e, 1), "ms_per_update_graph": round(ms_g, 1),
             "update_phase_ratio": round((e - fe) / (g - fg), 2) if g > fg else None}
        if s["update_phase_ratio"] is not None:
            # Projected 1M: the floor arm's own collection rate for the env steps, plus the measured
            # per-update cost for the 248,750 updates that budget implies. Derived arithmetic, not a
            # measurement - quoted as such wherever it appears.
            per_step = fe / args.steps
            updates_1m = (1_000_000 - 5000) // 4
            s["projected_1m"] = {
                "updates": updates_1m,
                "collection_s": round(per_step * 1_000_000, 0),
                "eager_h": round((per_step * 1e6 + updates_1m * ms_e / 1e3) / 3600, 1),
                "graph_h": round((per_step * 1e6 + updates_1m * ms_g / 1e3) / 3600, 1),
                "ratio": round((per_step * 1e6 + updates_1m * ms_e / 1e3)
                               / (per_step * 1e6 + updates_1m * ms_g / 1e3), 2)}
        summary[f"rep {rep}"] = s
        print(f"rep {rep}: total {e:.1f} s eager vs {g:.1f} s graph = {s['end_to_end_ratio']}x; "
              f"{s['ms_per_update_eager']} ms vs {s['ms_per_update_graph']} ms per update = "
              f"{s['update_phase_ratio']}x; " +
              (f"projected 1M {s['projected_1m']['eager_h']} h -> {s['projected_1m']['graph_h']} h"
               f" = {s['projected_1m']['ratio']}x" if "projected_1m" in s else "floor too noisy to subtract"),
              flush=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"runs_s": {k: v for k, v in results.items()}, "summary": summary,
                   "config": {"steps": args.steps, "num_envs": 4, "seq_len": 50, "batch_size": 16,
                              "imag_horizon": 15, "learning_starts": 5000,
                              "updates_per_run": n_updates},
                   "note": "whole short runs of the real trainer; the floor arms are the same "
                           "command with --learning-starts above the budget, so they collect without "
                           "ever updating. Order alternated between reps. The live REDQ 1M run shared "
                           "the box with these. projected_1m is arithmetic on those two rates, not a "
                           "run that was performed."}, handle, indent=2)
    print("wrote", os.path.relpath(out, ROOT))
    return 0


def reproducibility(args):
    """Do two runs of the identical build, identical seed and args, land on the same weights?"""
    args.steps = 5400
    a_ret, b_ret = [], []
    try:
        # EAGER on purpose, and named: the published artifact says "a CUDA-graph-free
        # comparison", and since capture became the CUDA default an unmarked arm would no longer be
        # one.
        ta, a_ret = run_trainer("bdr_a", args, EAGER)
        tb, b_ret = run_trainer("bdr_b", args, EAGER)
    finally:
        pass
    import torch
    paths = {"a": os.path.join(ROOT, "checkpoints", "bdr_a", f"dreamer_ckpt_{args.steps}.pt"),
             "b": os.path.join(ROOT, "checkpoints", "bdr_b", f"dreamer_ckpt_{args.steps}.pt")}
    # weights_only=False because this reads this repo's own checkpoint format (the trainer's resume
    # path uses the same call); it is never pointed at files from outside the repository.
    ck = {k: torch.load(v, map_location="cpu", weights_only=False) for k, v in paths.items()
          if os.path.exists(v)}
    clean_runs(("bdr_",))
    if len(ck) != 2:
        print("checkpoint(s) missing; nothing to compare", ck.keys())
        return 1
    tensors = {k: [t for sd in (ck[k][n] for n in ("model_state_dict", "actor_state_dict",
                                                    "critic_state_dict")) for t in sd.values()]
               for k in ck}
    n = len(tensors["a"])
    worse = max((x - y).abs().max().item() for x, y in zip(tensors["a"], tensors["b"]))
    identical = sum(1 for x, y in zip(tensors["a"], tensors["b"]) if torch.equal(x, y))
    payload = {
        "steps": args.steps, "seed": args.seed, "tensors_compared": n,
        "bit_identical_tensors": identical, "worst_abs_weight_diff": float(f"{worse:.3e}"),
        "wall_s": {"a": ta, "b": tb},
        "episodic_returns": {"a": a_ret, "b": b_ret},
        "returns_sequence_equal": a_ret == b_ret,
        "note": "same build, same --seed, same args, two processes. A CUDA-graph-free comparison: "
                "this measures whether the trainer is reproducible at all, which is a precondition "
                "for calling any Dreamer A/B paired.",
    }
    out = os.path.join(ROOT, "benchmarks", "dreamer_reproducibility.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"{identical}/{n} tensors bit-identical, worst |diff| {worse:.3e}; "
          f"return sequences equal: {payload['returns_sequence_equal']}")
    print("wrote", os.path.relpath(out, ROOT))
    return 0


CHECKPOINT_INTERVALS = (200_000, 100_000, 50_000, 20_000)


def checkpoint_cost(args):
    """What a Dreamer checkpoint costs to write, and what each interval would have saved.

    The 1M-step run that died at 84,456 steps lost everything because the default interval was
    200,000 - larger than the run had reached. Choosing a new default is arithmetic on two measured
    numbers (bytes and milliseconds per save), so they are measured here rather than guessed at.
    """
    import statistics as st
    import torch
    import train_dreamer as td

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = td.WorldModel(49, 17).to(dev)
    actor = td.DreamerActor(256, 32, 17).to(dev)
    critic = td.DreamerCritic(256, 32).to(dev)
    mods = (model, actor, critic)
    opts = [td.make_adam(m.parameters(), 3e-4, False) for m in mods]
    for o in opts:  # materialise Adam's state, as a training run would before the first save
        for p in o.param_groups[0]["params"]:
            p.grad = torch.zeros_like(p)
        o.step()
    state = {"model_state_dict": model.state_dict(), "actor_state_dict": actor.state_dict(),
             "critic_state_dict": critic.state_dict(),
             "model_opt_state_dict": opts[0].state_dict(),
             "actor_opt_state_dict": opts[1].state_dict(),
             "critic_opt_state_dict": opts[2].state_dict(),
             "global_step": 1000, "obs_rms": None, "rng_state": td.get_rng_state()}

    times, size = [], 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dreamer_ckpt_1000.pt")
        for _ in range(5):
            t0 = time.perf_counter()
            torch.save(state, path)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            size = os.path.getsize(path)

    median = st.median(times)
    out = {"device": str(dev), "reps": len(times),
           "checkpoint_bytes": size, "save_ms_median": round(median, 2),
           "save_ms_min": round(min(times), 2),
           "per_1m_steps": {str(iv): {"saves": 1_000_000 // iv,
                                      "disk_mib": round(1_000_000 // iv * size / 2**20, 1),
                                      "write_s": round(1_000_000 // iv * median / 1e3, 2),
                                      # The point of the exercise: how much progress a crash throws away.
                                      "max_progress_lost_steps": iv}
                             for iv in CHECKPOINT_INTERVALS},
           "note": "isolated torch.save of one full Dreamer checkpoint (three networks, three "
                   "Adam states, rng) at the shipped sizes; written by "
                   "python bench_dreamer_update.py --mode checkpoint-cost"}
    target = os.path.join(ROOT, "benchmarks", "dreamer_checkpoint_cost.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"one checkpoint: {size/2**20:.2f} MiB, {median:.1f} ms to write (min {min(times):.1f})")
    for iv, v in out["per_1m_steps"].items():
        print(f"  interval {int(iv):>7}: {v['saves']:>3} saves, {v['disk_mib']:>6.1f} MiB, "
              f"{v['write_s']:>5.2f} s per 1M steps, up to {v['max_progress_lost_steps']} steps lost")
    print("wrote", os.path.relpath(target, ROOT))
    return 0


def run_profile(args):
    out = os.path.join(ROOT, "benchmarks", "dreamer_update_profile.json")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    tmp.close()
    db = "sqlite:///" + os.path.join(tempfile.gettempdir(), "bench_dreamer_profile.db").replace(os.sep, "/")
    trainer_args = ["--run-id", "bre_dreamer_profile", "--seed", str(args.seed), "--device", "cuda",
                    # The eager path, explicitly: this profile exists to count the per-timestep
                    # Python dispatch (14,400 aten calls) that graph capture removes, so a run that
                    # silently captured would report a different thing than the published numbers.
                    "--no-update-graph",
                    "--vec-backend", "sync", "--num-envs", str(args.num_envs),
                    "--total-timesteps", str(args.steps), "--batch-size", "16", "--seq-len", "50",
                    "--checkpoint-interval", "1000000", "--buffer-size", "20000",
                    "--task-phase", "target", "--reset-mode", "upright"]
    env = dict(os.environ, MLFLOW_TRACKING_URI=db, DREAMER_PROFILE_OUT=tmp.name)
    proc = subprocess.run([PY, "-u", "-c", CHILD] + trainer_args, cwd=ROOT, env=env,
                          capture_output=True, text=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode:
        sys.stderr.write(proc.stderr[-3000:])
        return 1
    with open(tmp.name, encoding="utf-8") as handle:
        rows = json.load(handle)

    interesting = ["world_model_forward", "rssm_transition", "aten::linear", "aten::baddmm",
                   "aten::addmm", "aten::t", "aten::cat", "aten::mul", "aten::add", "aten::relu",
                   "aten::elu", "aten::gru", "aten::softplus", "aten::randn_like", "aten::to",
                   "aten::copy_", "aten::_to_copy", "aten::mse_loss", "aten::chunk", "aten::split",
                   "aten::mean", "aten::sum", "aten::stack", "aten::clamp", "aten::tanh",
                   "aten::embedding", "aten::sub", "aten::div", "aten::clamp_min",
                   "aten::kl_divergence", "aten::floor", "aten::neg", "aten::exp", "aten::log",
                   "aten::maximum", "aten::sign", "aten::pow", "aten::rsqrt", "aten::lerp",
                   "aten::threshold", "aten::index_select", "aten::index", "aten::select",
                   "aten::reshape", "aten::view", "aten::expand", "aten::contiguous"]
    picked = {k: rows[k] for k in rows if k in ("world_model_forward", "rssm_transition")}
    launches = sum(v["count"] for k, v in rows.items() if k.startswith("cudaLaunchKernel")
                   or k.startswith("aten::"))
    total_cuda = sum(v["self_cuda_us"] for v in rows.values())
    total_cpu = sum(v["self_cpu_us"] for v in rows.values())
    summary = {
        "profiled_steps": args.steps, "num_envs": args.num_envs, "seed": args.seed,
        "self_cuda_us_total": round(total_cuda, 1), "self_cpu_us_total": round(total_cpu, 1),
        "cpu_over_cuda_ratio": round(total_cpu / total_cuda, 2) if total_cuda else None,
        "aten_call_count": launches,
        "frames": picked,
        "top_ops_by_call_count": dict(sorted(
            ((k, v["count"]) for k, v in rows.items()), key=lambda kv: -kv[1])[:40]),
        "top_ops_by_self_cuda_time": dict(sorted(
            ((k, v["self_cuda_us"]) for k, v in rows.items() if v["self_cuda_us"] > 0),
            key=lambda kv: -kv[1])[:40]),
        "note": "torch.profiler over one short real train_dreamer() run (sync backend, 4 envs, "
                "updates from step 500). Self CPU above self CUDA means the update is launch-bound: "
                "the GPU waits for Python. This is a profile, not an A/B - the live REDQ 1M run "
                "shared the box with it.",
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nself CPU {total_cpu/1e6:.2f} s against self CUDA {total_cuda/1e6:.2f} s "
          f"-> cpu/cuda ratio {summary['cpu_over_cuda_ratio']}, {launches} aten/launch calls")
    print(f"{'frame':26}{'calls':>10}{'self cpu ms':>13}{'self cuda ms':>15}")
    for k, v in picked.items():
        print(f"{k[:26]:26}{v['count']:>10}{v['self_cpu_us']/1e3:>13.1f}{v['self_cuda_us']/1e3:>15.1f}")
    print(f"\n{'op':28}{'calls':>10}{'self cuda ms':>15}")
    for k, count in list(summary["top_ops_by_call_count"].items())[:20]:
        print(f"{k[:28]:28}{count:>10}{rows[k]['self_cuda_us']/1e3:>15.1f}")
    print("wrote", os.path.relpath(out, ROOT))

    for path in (os.path.join(ROOT, "checkpoints", "bre_dreamer_profile"), tmp.name):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    runs = os.path.join(ROOT, "runs")
    if os.path.isdir(runs):
        for entry in os.scandir(runs):
            if entry.is_dir() and entry.name.startswith("bre_dreamer_profile"):
                shutil.rmtree(entry.path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
