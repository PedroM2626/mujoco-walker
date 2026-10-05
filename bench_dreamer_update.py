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
                                      "checkpoint-cost", "heads-ab", "loop-split", "real-rate",
                                      "amortize"],
                   default="profile")
    p.add_argument("--heads-config", default="50x15",
                   help="seq_len x imag_horizon for --mode heads-ab (one config per process)")
    p.add_argument("--json-out", default=None, help="where --mode heads-ab writes its JSON")
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reps", type=int, default=60,
                   help="timed updates per config in --mode scaling, timed iterations in "
                        "--mode loop-split (5 untimed warmup run first)")
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
    if args.mode == "heads-ab":
        return heads_ab(args)
    if args.mode == "loop-split":
        return loop_split(args)
    if args.mode == "real-rate":
        return real_rate(args)
    if args.mode == "amortize":
        return amortize(args)
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
        samples, diverged_at = [], None
        for i in range(args.reps):
            t0 = time.perf_counter()
            out = fn(batch, idx)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
            # A Dreamer update on random weights and random rewards diverges after a few dozen
            # steps (the imagined returns compound against an untrained critic), and a diverged arm
            # keeps launching the same kernels while meaning nothing. The count is reported so the
            # margin prices the graph shape and the reader knows how much of it came from a live
            # model.
            if isinstance(out, dict) and any(float(v) != float(v) for v in out.values()):
                diverged_at = i
                break
        return {"median_ms": round(st.median(samples), 2), "min_ms": round(min(samples), 2),
                "max_ms": round(max(samples), 2), "reps": args.reps,
                "diverged_at_rep": diverged_at}

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


def heads_ab(args):
    """Price the fused imagination heads against the two modules, one config per process.

    Two separate runs of `--mode scaling` disagreed: the per-imagination-step cost fell 21% after
    the fusion while the whole update fell 1.7%, and the eager arm - which the fusion barely
    touches - moved 1.8x between the windows. That is machine state, not effect, so both
    implementations have to be measured in the same process, same minute, alternating reps.

    One process holds two arms and one config, because that is what this box tolerates: with four
    CUDA graphs alive a device-side assert fires inside `binary_cross_entropy`, and once one has
    fired the context is poisoned, so every later error is that same assert reported at an unrelated
    operation. A horizon sweep across configs is not measurable here without shipping a harness that
    lies, so the shipped configuration is what is priced.
    """
    import statistics as st
    import torch
    import train_dreamer as td
    from types import SimpleNamespace

    L, H = (int(x) for x in args.heads_config.lower().split("x"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBS, ACT, B = 49, 17, 16

    def arm(impl):
        torch.manual_seed(0)
        mods = [td.WorldModel(OBS, ACT).to(dev), td.DreamerActor(256, 32, ACT).to(dev),
                td.DreamerCritic(256, 32).to(dev)]
        opts = [td.make_adam(m.parameters(), 3e-4, dev.type == "cuda") for m in mods]
        torch.manual_seed(1)
        batch = (torch.randn(L, B, OBS, device=dev), torch.randn(L - 1, B, ACT, device=dev),
                 torch.randn(L - 1, B, 1, device=dev),
                 torch.zeros(L - 1, B, 1, device=dev, dtype=torch.bool))
        torch.manual_seed(2)
        idx = torch.randperm(L * B)[:B].to(dev)
        a = SimpleNamespace(seq_len=L, batch_size=B, imag_horizon=H, gamma=0.99, kl_weight=1.0,
                            heads_impl=impl)
        fn = lambda b, i: td.dreamer_update(*mods, *opts, b, a, i,
                                            zero_grad_set_to_none=dev.type != "cuda")
        return (td.CapturedDreamerUpdate(fn, batch, idx) if dev.type == "cuda" else fn), batch, idx

    def timed(fn, batch, idx):
        """One update, timed - and refused once the nets have diverged.

        These arms take real optimizer steps on random weights and random rewards, so after a few
        dozen of them the model can walk into a NaN, and the next `binary_cross_entropy` kills the
        process with a device-side assert that then poisons the CUDA context: every later error is
        that assert reported at an unrelated operation. Timing a diverged arm is meaningless anyway,
        so the run stops here with a number instead of a stack trace.
        """
        t0 = time.perf_counter()
        out = fn(batch, idx)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if isinstance(out, dict):
            for key, value in out.items():
                v = float(value)
                if v != v or v in (float("inf"), float("-inf")):
                    raise SystemExit(f"[HEADS-AB] {key} went non-finite after a timed update; the "
                                     "arms stopped training meaningfully, so the remaining samples "
                                     "would be junk. Lower --reps.")
        return dt

    fused_fn, fb, fi = arm("fused")
    sep_fn, sb, si = arm("separate")
    for _ in range(3):
        timed(fused_fn, fb, fi)
        timed(sep_fn, sb, si)
    fused_ms, sep_ms = [], []
    for _ in range(args.reps):          # alternating: drift hits both arms about equally
        fused_ms.append(timed(fused_fn, fb, fi))
        sep_ms.append(timed(sep_fn, sb, si))
    f_med, s_med = st.median(fused_ms), st.median(sep_ms)
    out = {"device": str(dev), "reps": args.reps, "seq_len": L, "imag_horizon": H,
           "gpu_other_contexts": _other_gpu_contexts(),
           "fused_median_ms": round(f_med, 3), "separate_median_ms": round(s_med, 3),
           "separate_minus_fused_ms": round(s_med - f_med, 3),
           "fused_over_separate": round(f_med / s_med, 4),
           "note": "isolated dreamer_update calls, both head implementations in one process, "
                   "reps alternating between arms; one config per process (see the docstring)"}
    target = args.json_out or os.path.join(ROOT, "benchmarks", "dreamer_heads_ab.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"L{L} H{H}: fused {f_med:.3f} ms vs separate {s_med:.3f} ms "
          f"(ratio {f_med / s_med:.4f}) -> {os.path.relpath(target, ROOT)}")
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


def real_rate(args):
    """The shipped trainer's steady-state rate, measured at two budgets so fixed costs cancel.

    Every Dreamer rate published so far came from one short run divided by its own step count, which
    charges two fixed costs to every step: interpreter + CUDA init (~12 s per process) and the
    one-time CUDA graph capture. At the 5,400-step budget used by `--mode ab` there are only 100
    updates, so the capture is most of the measured "update phase", and the floor arm's startup
    inflates the collection term - the two errors push the projected 1M apart by design.

    Running the identical command at two budgets and subtracting removes both: (t2 - t1) is pure
    steady-state work over (n2 - n1) steps, and the intercept is the fixed cost. Two arms, because
    the difference between the with-update and collect-only slopes is what one update costs inside a
    real run rather than in isolation.
    """
    import statistics

    budgets = (args.steps, 2 * args.steps)
    if args.steps == 9000:
        args.steps, budgets = 20000, (20000, 40000)  # 100x the ab budget's update count, so the
        # capture cost is 0.3% of the update phase instead of most of it
    # The smoke that designed this ran the cells in arm order, and the first process of the batch
    # paid a cold-start 9 s larger than the rest, which made the second budget *faster* than the
    # first and the slope negative. So the four cells run in a rotated order, twice, and the slope
    # comes from medians with the two independent estimates reported.
    order = [("captured", "big"), ("collect", "small"), ("collect", "big"), ("captured", "small")]
    cells = {}
    try:
        for rep in (1, 2):
            for label, which in order:
                budget = budgets[1] if which == "big" else budgets[0]
                sub = argparse.Namespace(**vars(args))
                sub.steps = budget
                dt, _ = run_trainer(f"bdr{label}{budget}_{rep}", sub,
                                    GRAPH if label == "captured" else FLOOR + GRAPH)
                if dt is None:
                    print("arm failed; nothing to subtract")
                    return 1
                cells.setdefault((label, budget), []).append(dt)
    finally:
        clean_runs(("bdr",))

    out = {"raw_totals_s": {f"{label}_{budget}": [round(v, 1) for v in vals]
                            for (label, budget), vals in sorted(cells.items())}}
    for label in ("captured", "collect"):
        t1, t2 = statistics.median(cells[(label, budgets[0])]), statistics.median(cells[(label, budgets[1])])
        it1, it2 = budgets[0] / args.num_envs, budgets[1] / args.num_envs
        ms_per_iter = (t2 - t1) / (it2 - it1) * 1e3
        per_rep = []
        for i in range(len(cells[(label, budgets[0])])):
            a, b = cells[(label, budgets[0])][i], cells[(label, budgets[1])][i]
            per_rep.append(round((b - a) / (it2 - it1) * 1e3, 2))
        out[label] = {"ms_per_iteration": round(ms_per_iter, 2),
                      "per_rep_ms_per_iteration": per_rep,
                      "env_steps_per_s": round(1000.0 * args.num_envs / ms_per_iter, 1),
                      "median_totals_s": {str(budgets[0]): round(t1, 1), str(budgets[1]): round(t2, 1)}}
    full, coll = out["captured"]["ms_per_iteration"], out["collect"]["ms_per_iteration"]
    out["update_ms_per_iteration"] = round(full - coll, 2)
    out["update_share_pct"] = round(100.0 * (full - coll) / full, 1)
    # A straight line through the two budgets is not `startup + rate x iterations`: the first
    # learning_starts / num_envs iterations of every run sit below the update gate, so the
    # with-update arm is bent and its naive intercept came out negative (-10.7 s). Solving the two
    # budgets against the gate-aware model instead puts the startup back where a subprocess belongs,
    # and the pair is the only honest read of what capture cost: the with-update intercept also holds
    # the one-time capture, so their difference bounds it.
    gate_iters = 5000.0 / args.num_envs
    for label, per_iter in (("collect", coll), ("captured", full)):
        t1 = out[label]["median_totals_s"][str(budgets[0])]
        it1 = budgets[0] / args.num_envs
        extra = out["update_ms_per_iteration"] * gate_iters / 1e3 if label == "captured" else 0.0
        out[label]["startup_s"] = round(t1 - per_iter * it1 / 1e3 + extra, 1)
    out["startup_s_note"] = ("the collect arm is process startup; the captured arm is startup plus the "
                             "one-time graph capture, so their difference is capture and noise")
    out["hours_per_1m_env_steps"] = round(1e6 / args.num_envs * full / 1000.0 / 3600.0, 2)
    out["config"] = {"num_envs": args.num_envs, "seq_len": 50, "batch_size": 16,
                     "imag_horizon": 15, "learning_starts": 5000, "update_graph": "on (default)",
                     "budgets": list(budgets), "reps_per_cell": 2}
    out["gpu_other_contexts"] = _other_gpu_contexts()
    out["note"] = "steady-state rates from the two-budget subtraction on the real trainer, cells " \
                  "rotated and run twice, medians used; the update is one captured dreamer_update " \
                  "per num_envs env steps. Assumes time linear in steps, which is why the smaller " \
                  "budget already fills the 20,000-transition replay buffer."

    path = os.path.join(ROOT, "benchmarks", "dreamer_real_rate.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    for label in ("captured", "collect"):
        print(f"{label:9} {out[label]['ms_per_iteration']:7.2f} ms/iter  "
              f"{out[label]['env_steps_per_s']:7.1f} env-steps/s  "
              f"startup {out[label]['startup_s']:.1f} s  "
              f"reps {out[label]['per_rep_ms_per_iteration']}")
    print(f"update {out['update_ms_per_iteration']:.2f} ms per iteration = "
          f"{out['update_share_pct']:.1f}% of the loop -> "
          f"{out['hours_per_1m_env_steps']:.2f} h per 1M env steps")
    print("wrote", os.path.relpath(path, ROOT))
    return 0


AMORTIZE_NS = (4, 8, 16)


def amortize(args):
    """What raising `--num-envs` buys in wall clock, and what it costs in gradient steps.

    The captured update is 12.6-12.8 ms of a ~17 ms iteration at the shipped `--num-envs 4`, and the
    trainer does exactly one update per collection step. So the only remaining lever inside the loop
    is arithmetic that does not touch the update: collect more environment steps per gradient step.
    That is a real change of algorithm (update-to-data ratio 1/num_envs), not a free speedup, so it
    is priced on both axes.

    Each `n` is measured at two budgets, as in `--mode real-rate`, so the process startup and the
    one-time capture cancel; `--learning-starts` is lowered to 1000 so the larger batches still
    reach the update gate inside a short budget.
    """
    import statistics

    out = {"gpu_other_contexts": _other_gpu_contexts(), "runs_s": {}, "per_num_envs": {}}
    budgets = (8000, 16000)
    try:
        for n in AMORTIZE_NS:
            per = {}
            for budget in budgets:
                sub = argparse.Namespace(**vars(args))
                sub.steps = budget
                sub.num_envs = n
                # BASE already carries --num-envs 4, and argparse keeps the last occurrence, so the
                # per-arm override has to be appended rather than edited into the shared flag set.
                extra = GRAPH + ["--num-envs", str(n), "--learning-starts", "1000"]
                dt, _ = run_trainer(f"bdm_n{n}_{budget}", sub, extra)
                out["runs_s"][f"n{n}_{budget}"] = dt
                per[budget] = dt
            if None in per.values():
                continue
            it1, it2 = budgets[0] / n, budgets[1] / n
            ms_per_iter = (per[budgets[1]] - per[budgets[0]]) / (it2 - it1) * 1e3
            updates = (budgets[1] - 1000) / n          # one update per iteration past the gate
            out["per_num_envs"][n] = {
                "ms_per_iteration": round(ms_per_iter, 2),
                "env_steps_per_s": round(1000.0 * n / ms_per_iter, 1),
                "gradient_steps_per_1k_env_steps": round(1000.0 / n, 1),
                "hours_per_1m_env_steps": round(1e6 / n * ms_per_iter / 1000.0 / 3600.0, 2),
                "updates_in_window": int(updates),
            }
            print(f"n={n:<3} {out['per_num_envs'][n]['ms_per_iteration']:7.2f} ms/iter  "
                  f"{out['per_num_envs'][n]['env_steps_per_s']:7.1f} env-steps/s  "
                  f"{out['per_num_envs'][n]['gradient_steps_per_1k_env_steps']:6.1f} grad steps per "
                  f"1k env steps  {out['per_num_envs'][n]['hours_per_1m_env_steps']:.2f} h per 1M",
                  flush=True)
    finally:
        clean_runs(("bdm_",))

    base = out["per_num_envs"].get(AMORTIZE_NS[0])
    for n, row in out["per_num_envs"].items():
        row["speedup_vs_shipped_n4"] = round(row["env_steps_per_s"] / base["env_steps_per_s"], 2)
    out["config"] = {"seq_len": 50, "batch_size": 16, "imag_horizon": 15, "learning_starts": 1000,
                     "budgets": list(budgets), "update_graph": "on", "vec_backend": "sync"}
    out["note"] = ("steady-state rates per num_envs from the two-budget subtraction; one captured "
                   "dreamer_update per collection step at every n, so the update-to-data ratio falls "
                   "as 1/num_envs - the wall clock and the gradient count move together and both are "
                   "reported.")
    path = os.path.join(ROOT, "benchmarks", "dreamer_amortize_n.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("wrote", os.path.relpath(path, ROOT))
    return 0


LOOP_SEGMENTS = ("policy pass", "envs.step", "state pass", "rb.add", "rb.sample", "captured update")


def loop_split(args):
    """Price one Dreamer iteration the way the trainer assembles it, in one process and one window.

    The shipped update measures 12.62 ms alone (benchmarks/dreamer_update_scaling.json) and the real
    trainer's collection-only floor measured 29.6 ms per iteration in a *different* window, so the
    two cannot be subtracted. Here the same minutes and the same clocks carry every segment, and the
    question changes from "is the learner the bottleneck" (true when the update ran eager at 293 ms)
    to "what is left now that the update costs 12 ms".

    The pieces come from train_dreamer itself - build_vec_env with the trainer's own flag set, the
    same wrappers, WorldModel/DreamerActor/DreamerCritic, SequenceReplayBuffer, the extracted
    advance_recurrent_state and CapturedDreamerUpdate - so this times the shipped path, not a
    paraphrase of it.
    """
    import numpy as np
    import statistics as st
    import torch
    import train_dreamer as td

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_argv = list(sys.argv)
    sys.argv = (["train_dreamer.py"] + BASE + ["--num-envs", str(args.num_envs), "--seed",
                str(args.seed), "--total-timesteps", str(args.steps), "--update-graph"])
    targs = td.parse_dreamer_args()
    sys.argv = saved_argv

    envs = td.build_vec_env(targs, "bdr_split", capture_video=False)
    envs = td.wrap_normalize_observation(envs)
    envs = td.wrap_transform_observation(envs, lambda o: np.clip(o, -10, 10))
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    torch.manual_seed(0)
    model = td.WorldModel(obs_dim, act_dim).to(dev)
    actor = td.DreamerActor(256, 32, act_dim).to(dev)
    critic = td.DreamerCritic(256, 32).to(dev)
    opts = [td.make_adam(m.parameters(), targs.learning_rate, True) for m in (model, actor, critic)]
    rb = td.SequenceReplayBuffer(targs.buffer_size, targs.num_envs,
                                 envs.single_observation_space.shape,
                                 envs.single_action_space.shape, dev)
    obs = envs.reset()[0]
    h, z = model.rssm.initial_state(targs.num_envs, dev)

    # Fill the buffer past the trainer's own gate so the sampled batch is a real one, and capture
    # against that shape, which is what CapturedDreamerUpdate needs a first batch for anyway.
    while rb.filled <= targs.seq_len + 10:
        warm = np.array([envs.single_action_space.sample() for _ in range(targs.num_envs)])
        next_obs, rew, term, trunc, _ = envs.step(warm)
        h, z = td.advance_recurrent_state(model, h, z, warm, next_obs, term, trunc)
        rb.add(obs, warm, rew, np.logical_or(term, trunc))
        obs = next_obs
    batch = rb.sample(targs.batch_size, targs.seq_len)
    idx = torch.randperm(targs.seq_len * targs.batch_size)[:targs.batch_size]
    captured = td.CapturedDreamerUpdate(
        lambda b, i: td.dreamer_update(model, actor, critic, opts[0], opts[1], opts[2], b, targs, i,
                                       zero_grad_set_to_none=False), batch, idx)

    state = {"h": h, "z": z, "obs": obs}

    def iteration(mark):
        """One collect+update step in the trainer's order, advancing the lane state."""
        with torch.no_grad():  # as the trainer has it, once past the random-action opening
            actions = actor.get_action(state["h"], state["z"], sample=True).cpu().numpy()
        mark()
        next_obs, rew, term, trunc, _ = envs.step(actions)
        mark()
        state["h"], state["z"] = td.advance_recurrent_state(model, state["h"], state["z"], actions,
                                                            next_obs, term, trunc)
        mark()
        rb.add(state["obs"], actions, rew, np.logical_or(term, trunc))
        mark()
        fresh = rb.sample(targs.batch_size, targs.seq_len)
        start = torch.randperm(targs.seq_len * targs.batch_size)[:targs.batch_size]
        mark()
        losses = captured(fresh, start)
        mark()
        state["obs"] = next_obs
        return losses

    def one_rep():
        """The iteration timed segment by segment, then the same iteration timed as a whole.

        The boundary synchronize() calls are what make the attribution honest, and they also make
        the attributed total larger than what the trainer pays, so the second pass - no inner syncs,
        one sync at the end - reports the loop as the trainer actually runs it.
        """
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        marks = []

        def mark():
            sync()
            marks.append(time.perf_counter())

        sync()
        marks.append(time.perf_counter())
        losses = iteration(mark)
        seg = {name: (marks[i + 1] - marks[i]) * 1e3 for i, name in enumerate(LOOP_SEGMENTS)}

        t0 = time.perf_counter()
        iteration(lambda: None)
        sync()
        whole = (time.perf_counter() - t0) * 1e3
        finite = all(float(v) == float(v) for v in losses.values())
        return seg, whole, finite

    for _ in range(5):  # warmup: first replays, cudnn autotune, the buffer window moving
        one_rep()
    segs, totals, diverged_at = [], [], None
    for rep in range(args.reps):
        seg, whole, finite = one_rep()
        segs.append(seg)
        totals.append(whole)
        if not finite and diverged_at is None:
            diverged_at = rep
            break

    med = {name: st.median([s[name] for s in segs]) for name in LOOP_SEGMENTS}
    attributed = sum(med.values())
    whole = st.median(totals)
    collection = attributed - med["captured update"]
    out = {
        "device": str(dev), "num_envs": targs.num_envs, "seq_len": targs.seq_len,
        "batch_size": targs.batch_size, "imag_horizon": targs.imag_horizon,
        "heads_impl": targs.heads_impl, "vec_backend": targs.vec_backend,
        "task_phase": targs.task_phase, "reset_mode": targs.reset_mode,
        "reps": len(segs), "diverged_at_rep": diverged_at,
        "gpu_other_contexts": _other_gpu_contexts(),
        "segments_ms": {k: round(v, 2) for k, v in med.items()},
        "attributed_iteration_ms": round(attributed, 2),
        "whole_loop_ms": round(whole, 2),
        "collection_ms": round(collection, 2),
        "collection_share_pct": round(100.0 * collection / attributed, 1),
        "update_share_pct": round(100.0 * med["captured update"] / attributed, 1),
        "env_steps_per_s_at_this_loop": round(1000.0 * targs.num_envs / whole, 1),
        "hours_per_1m_steps_at_this_loop": round(1e6 / targs.num_envs * whole / 1000.0 / 3600.0, 2),
        "note": "medians over whole collect+update iterations of the shipped loop, rebuilt from "
                "train_dreamer's own pieces with synchronize() at every segment boundary; one "
                "iteration is num_envs env steps plus one update. attributed_iteration_ms is the "
                "sum of the segments, whole_loop_ms is the same iteration without the boundary "
                "syncs, which is what the trainer waits on.",
    }
    print(f"{'segment':24}{'median ms':>11}{'of iteration':>14}")
    for name in LOOP_SEGMENTS:
        print(f"{name:24}{med[name]:>11.2f}{100 * med[name] / attributed:>13.1f}%")
    print(f"{'attributed total':24}{attributed:>11.2f}{100.0:>13.1f}%")
    print(f"{'whole loop, no syncs':24}{whole:>11.2f}  = "
          f"{out['env_steps_per_s_at_this_loop']:.1f} env-steps/s, "
          f"{out['hours_per_1m_steps_at_this_loop']:.2f} h per 1M steps at n={targs.num_envs}")
    path = os.path.join(ROOT, "benchmarks", "dreamer_loop_split.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("wrote", os.path.relpath(path, ROOT))
    envs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
