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
    p.add_argument("--mode", choices=["profile", "ab", "reproducibility"], default="profile")
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


BASE = ["--device", "cuda", "--vec-backend", "sync", "--num-envs", "4", "--batch-size", "16",
        "--seq-len", "50", "--checkpoint-interval", "100000000", "--buffer-size", "20000",
        "--task-phase", "target", "--reset-mode", "upright"]
# Order alternates between reps so a run sharing this box hits both arms about equally.
AB_ARMS = [("eager", "e1", []), ("graph", "g1", ["--update-graph"]),
           ("floor_eager", "fe1", ["--learning-starts", "999999"]),
           ("floor_graph", "fg1", ["--learning-starts", "999999", "--update-graph"]),
           ("graph", "g2", ["--update-graph"]), ("eager", "e2", []),
           ("floor_graph", "fg2", ["--learning-starts", "999999", "--update-graph"]),
           ("floor_eager", "fe2", ["--learning-starts", "999999"])]


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
    return run_profile(args)


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
        ta, a_ret = run_trainer("bdr_a", args, [])
        tb, b_ret = run_trainer("bdr_b", args, [])
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


def run_profile(args):
    out = os.path.join(ROOT, "benchmarks", "dreamer_update_profile.json")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    tmp.close()
    db = "sqlite:///" + os.path.join(tempfile.gettempdir(), "bench_dreamer_profile.db").replace(os.sep, "/")
    trainer_args = ["--run-id", "bre_dreamer_profile", "--seed", str(args.seed), "--device", "cuda",
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
