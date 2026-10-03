"""Paired comparison of two Phase-4 models over the same seeded episodes.

The 50-episode table reports a mean and a std per model, which is enough to rank clear gaps
and not enough for the near ones: with std 650-1100 and 50 episodes, the standard error of a
single mean is 90-150 points, so neighbours inside that band cannot be separated by their
averages. `evaluate_all.py` resets episode *i* with `seed+i` for every model, so the same
episode indices are the same initial states and the two samples are paired. Paired
differences cancel the episode-to-episode variance that both models share, which is the only
way this protocol can say anything about a 12-point gap.

Reads the per-episode arrays written next to the run report, so the test is reproducible from
the artefacts rather than from whatever was in memory when the table was typed.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.join(ROOT, "openai_walker")

# The pairs the 50-episode table could not separate. The last two are the same comparison read
# from each side, because the committed artifact has carried both since the seeding fix and a
# regeneration that drops a measured row is a content change, not a tidy-up.
#
# BCQ - Teacher is the pair the retired single-episode leaderboard asserted without testing: its
# top row claimed BCQ "beat the online teacher", and that is a claim about these two models.
DEFAULT_PAIRS = [("BC", "Teacher (Upper Bound)"), ("BCQ", "BC+SAC (Regularized)"),
                 ("Extra Trees", "BCQ"), ("BC+SAC (Regularized)", "BCQ"),
                 ("BCQ", "Teacher (Upper Bound)")]


def load_episodes(episodes, seed):
    """Merge every per-episode file for this protocol into one {model: array} dict."""
    paths = [os.path.join(HERE, f"final_episodes_{episodes}ep_seed{seed}.json"),
             os.path.join(HERE, f"final_episodes_extratrees_{episodes}ep_seed{seed}.json")]
    found = {p for p in paths if os.path.exists(p)}
    found.update(glob.glob(os.path.join(HERE, f"final_episodes_extra_*_{episodes}ep_seed{seed}.json")))
    if not found:
        raise SystemExit(
            f"No per-episode file for {episodes} episodes / seed {seed}. Re-run the race: "
            f"`cd openai_walker && python evaluate_all.py --episodes {episodes} --seed {seed}`"
        )
    models = {}
    for p in sorted(found):
        data = json.load(open(p, encoding="utf-8"))
        if data.get("seed") != seed or data.get("episodes") != episodes:
            raise SystemExit(f"{p}: declares seed {data.get('seed')} and "
                             f"{data.get('episodes')} episodes, asked for {seed}/{episodes}")
        for name, rewards in data["models"].items():
            if name in models and models[name] != rewards:
                raise SystemExit(f"{name} appears in two files with different values")
            models[name] = rewards
    return models


def paired_test(a, b, label_a, label_b, resamples=20000, rng_seed=0):
    """Paired statistics for two models' per-episode returns."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise SystemExit(f"{label_a} ran {a.size} episodes, {label_b} ran {b.size}; "
                         "paired tests need the same protocol on both")
    d = a - b
    n = d.size
    rng = np.random.default_rng(rng_seed)
    boot = rng.choice(d, size=(resamples, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        wilcoxon = stats.wilcoxon(d, zero_method="zsplit").pvalue
    except ValueError:
        wilcoxon = float("nan")
    nonzero = d[d != 0.0]
    sd = d.std(ddof=1)
    out = {
        "pair": f"{label_a} - {label_b}",
        "episodes": int(n),
        "mean_a": round(float(a.mean()), 2),
        "mean_b": round(float(b.mean()), 2),
        "mean_difference": round(float(d.mean()), 2),
        "std_of_difference": round(float(sd), 2) if sd == sd else None,
        "bootstrap_95ci": [round(float(lo), 2), round(float(hi), 2)],
        "significant_at_95": bool(lo > 0 or hi < 0),
        # A mean can be tied while one model wins most episodes, and the retired leaderboard was
        # one episode apiece, so the win count is what a single draw actually samples from.
        "wins_a": int((d > 0).sum()),
        "wins_b": int((d < 0).sum()),
        "ties": int(n - nonzero.size),
        "mean_only_significant": bool(abs(d.mean()) >
                                      1.96 * float(np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n))),
        "paired_t_p": round(float(stats.ttest_rel(a, b).pvalue), 4),
        "wilcoxon_p": round(float(wilcoxon), 4),
        "cohens_dz": round(float(d.mean() / sd), 3) if sd > 0 else None,
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pair", action="append", nargs=2, metavar=("A", "B"), default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    models = load_episodes(args.episodes, args.seed)
    pairs = [tuple(x) for x in (args.pair or DEFAULT_PAIRS)]
    results = []
    for a_name, b_name in pairs:
        for name in (a_name, b_name):
            if name not in models:
                print(f"Available: {sorted(models)}")
                raise SystemExit(f"no model named {name!r} in this protocol")
        r = paired_test(models[a_name], models[b_name], a_name, b_name)
        results.append(r)
        verdict = ("distinguible" if r["significant_at_95"] else "NOT distinguishable")
        print(f"{r['pair']:38} d={r['mean_difference']:9.2f} "
              f"95% CI [{r['bootstrap_95ci'][0]:8.2f}, {r['bootstrap_95ci'][1]:8.2f}] "
              f"paired-t p={r['paired_t_p']:.4f} Wilcoxon p={r['wilcoxon_p']:.4f} "
              f"dz={r['cohens_dz']} -> {verdict}")
        if r["mean_only_significant"] != r["significant_at_95"]:
            print(f"    note: the unpaired std-based check and the paired check disagree; "
                  f"the paired one is the valid read for shared-episode designs")

    out = args.out or os.path.join(ROOT, "benchmarks",
                                   f"phase4_paired_{args.episodes}ep_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"protocol": f"paired test over evaluate_all.py --episodes {args.episodes} "
                               f"--seed {args.seed} (episode i reset with seed+i for every model)",
                   "resamples": 20000, "results": results}, handle, indent=2)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
