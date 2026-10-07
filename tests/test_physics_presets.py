"""Contract tests for the physics presets added beside the published v9 world.

These assert the *mechanism*, not any measurement: which model fields a preset touches, that the
default preset touches nothing, that the version string a checkpoint records follows the preset
(otherwise an `euler` run would be stamped as the published MDP), and that the flag survives the
two hops that actually matter - pickling into a process-parallel worker, and reaching the eval
script that scores the checkpoint afterwards.

The cost and the drift of the presets are measurements and live in `benchmarks/physics_presets.json`,
produced by `bench_physics_presets.py`; nothing here re-derives them.
"""

import json
import os
import pickle
import re
import sys
import unittest
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import envs.walker_ragdoll_env as env_module
from envs.walker_ragdoll_env import ENV_VERSION, apply_physics_preset, env_version_of
from train_walker import current_env_version, env_common_kwargs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make(preset, **kwargs):
    return gym.make("WalkerRagdoll-v0", physics_preset=preset, **kwargs)


class TestDefaultPresetIsThePublishedWorld(unittest.TestCase):
    """Nothing that a committed number depends on may move because this feature exists."""

    def setUp(self):
        self.env = _make("v9")

    def tearDown(self):
        self.env.close()

    def reference_model(self):
        return mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))

    def test_the_model_is_untouched(self):
        model, ref = self.env.unwrapped.model, self.reference_model()
        self.assertEqual(model.opt.integrator, ref.opt.integrator)
        self.assertEqual(model.opt.timestep, ref.opt.timestep)
        # Compared against a freshly compiled model rather than against assumed constants: some
        # geoms are inert in the XML by design (the target marker carries contype=0/conaffinity=0),
        # and a test that asserted them all at 1 would fail while nothing had actually changed.
        self.assertTrue((model.geom_contype == ref.geom_contype).all())
        self.assertTrue((model.geom_conaffinity == ref.geom_conaffinity).all())

    def test_the_recorded_version_is_the_module_constant(self):
        self.assertEqual(env_version_of(self.env), ENV_VERSION)
        self.assertEqual(current_env_version(SimpleNamespace(physics_preset="v9")), ENV_VERSION)

    def test_observation_contract_is_unchanged(self):
        self.assertEqual(self.env.observation_space.shape, (46,))
        with _make("v9", task_phase="target") as target_env:
            self.assertEqual(target_env.observation_space.shape, (49,))


class TestPresetsTouchWhatTheyClaim(unittest.TestCase):
    def reference_model(self):
        return mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))

    def test_euler_changes_only_the_integrator(self):
        ref = self.reference_model()
        with _make("euler") as env:
            model = env.unwrapped.model
            self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
            self.assertEqual(model.opt.timestep, ref.opt.timestep)
            self.assertTrue((model.geom_contype == ref.geom_contype).all())
            self.assertTrue((model.geom_conaffinity == ref.geom_conaffinity).all())

    def test_fast_drops_self_collision_and_keeps_the_floor(self):
        ref = self.reference_model()
        with _make("fast") as env:
            model = env.unwrapped.model
            floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker_geom")
            self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
            self.assertGreaterEqual(floor, 0, "the floor geom vanished")
            self.assertEqual(model.geom_contype[floor], 1)
            for gid in range(model.ngeom):
                if gid == floor:
                    continue
                inert_by_design = (ref.geom_contype[gid] == 0 and ref.geom_conaffinity[gid] == 0)
                # The pair rule is contype(a) & conaffinity(b) | contype(b) & conaffinity(a).
                # Body geoms initiate nothing, so body-vs-body never forms; the floor still
                # initiates against their affinity, so floor-vs-body still forms. A geom the
                # model already made inert must stay inert, or the preset has recruited a
                # decorative marker into the collision set.
                self.assertEqual(model.geom_contype[gid], 0, f"geom {gid} still initiates")
                self.assertEqual(model.geom_conaffinity[gid], 0 if inert_by_design else 1,
                                 f"geom {gid} affinity wrong")
            self.assertEqual(model.geom_conaffinity[marker], 0, "the target marker became solid")

    def test_floor_contacts_still_happen_in_fast(self):
        """The standing gate counts foot-vs-floor, so a preset that lost them would break it silently."""
        with _make("fast", task_phase="target") as env:
            env.reset(seed=3)
            feet_ever = False
            for _ in range(120):
                _, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
                _, foot = env.unwrapped.floor_contact_counts
                feet_ever = feet_ever or foot > 0
            self.assertTrue(feet_ever, "no foot-floor contact in 120 steps: the fast preset would "
                                       "make the standing gate unreachable")

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            _make("cheaper")

    def test_apply_helper_rejects_the_unknown_too(self):
        model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))
        with self.assertRaises(ValueError):
            apply_physics_preset(model, "nope")


class TestVersionTravelsWithThePreset(unittest.TestCase):
    def test_each_preset_records_its_own_version(self):
        for preset in env_module.PHYSICS_PRESETS:
            with _make(preset) as env:
                want = (ENV_VERSION if preset == "v9" else f"{ENV_VERSION}_{preset}")
                self.assertEqual(env_version_of(env), want)

    def test_env_version_of_survives_wrappers_and_missing_attrs(self):
        self.assertEqual(env_version_of(None), ENV_VERSION)
        with _make("euler") as env:
            self.assertEqual(env_version_of(env.unwrapped), f"{ENV_VERSION}_euler")

    def test_the_preset_survives_pickling_into_a_worker(self):
        # The parallel backend pickles the spec, and EzPickle is what rebuilds the env in the
        # child: if the preset were not part of the recorded args, half the vector would silently
        # train in the published world.
        with _make("fast", task_phase="target") as env:
            clone = pickle.loads(pickle.dumps(env.unwrapped))
            try:
                self.assertEqual(clone._physics_preset, "fast")
                self.assertEqual(env_version_of(clone), f"{ENV_VERSION}_fast")
                self.assertEqual(clone.model.opt.integrator,
                                 mujoco.mjtIntegrator.mjINT_EULER)
            finally:
                clone.close()

    def test_the_trainers_pass_it_to_every_sub_env(self):
        args = SimpleNamespace(reset_mode="mixed", fixed_reset_probability=0.25,
                               upright_reset_probability=0.15, fallen_velocity_scale=0.35,
                               task_phase="target", target_forward_velocity=1.2,
                               physics_preset="fast")
        kwargs = env_common_kwargs(args)
        self.assertEqual(kwargs["physics_preset"], "fast",
                         "env_common_kwargs dropped the preset, so the trainers would build v9 "
                         "sub-envs while recording a fast version string")


class TestEveryTrainerCanActuallyAskForAPreset(unittest.TestCase):
    """The flag has to be declared by every trainer that builds sub-envs, not just by one.

    `env_common_kwargs` reads the attribute tolerantly (a harness that assembles its own args
    namespace should get the published world, not an AttributeError), and that tolerance is exactly
    what makes the failure silent: a trainer whose parser never declares `--physics-preset` accepts
    nothing, collects in v9, and records `..._v9` in its checkpoints while a user believes they asked
    for a cheaper world. Three of the four trainers were in that state when the preset landed.
    """

    TRAINERS = ("train_walker.py", "train_dreamer.py", "train_redq.py", "train_ars.py")

    def test_the_flag_is_declared(self):
        missing = []
        for name in self.TRAINERS:
            with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
                src = handle.read()
            if '--physics-preset' not in src:
                missing.append(name)
        self.assertEqual(missing, [], f"these trainers accept no preset: {missing}")

    def test_the_ones_that_reuse_the_shared_builder_are_the_risky_ones(self):
        # train_dreamer/train_redq go through build_vec_env, so a missing declaration is invisible;
        # train_ars passes kwargs to make_env by hand, where a missing one is a TypeError instead.
        for name in ("train_dreamer.py", "train_redq.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
                self.assertIn("build_vec_env(", handle.read(),
                              f"{name} no longer builds envs through the shared function; the "
                              "declaration check above may no longer cover it")
        with open(os.path.join(ROOT, "train_ars.py"), encoding="utf-8") as handle:
            self.assertIn("physics_preset=args.physics_preset", handle.read(),
                          "train_ars builds its env by hand and would silently drop the preset")


class TestWallClockProvenanceSurvivesTheLogs(unittest.TestCase):
    """Every training duration the artifacts record as a literal has to still match its log.

    A run's wall clock is the one number this repository cannot recompute: the checkpoints say what
    was collected, not how long it took, and `mlruns.db` - which is gitignored and holds no SAC run at
    all, because `train_walker.py` defines the mlflow helpers and never calls them - cannot be queried
    for it either. So the launcher logs are committed as the primary source
    (`chainNN_evidence.log`) and this test re-derives the spans from them. Without it the recorded
    seconds are a claim that decays the moment nobody can check it.
    """

    TMP_SPAN = re.compile(r"^(\d\d):(\d\d):(\d\d) START (.+)$")
    DONE_SPAN = re.compile(r"^(\d\d):(\d\d):(\d\d) DONE  (.+?) \(exit (\d)\)$")

    def spans(self, chain):
        """Every (label -> [(start, end, exit)]) pair in one launcher log, retries included."""
        path = os.path.join(ROOT, f"chain{chain}_evidence.log")
        self.assertTrue(os.path.exists(path), f"{path} is gone; the durations it backs are "
                                              "unverifiable claims now")
        starts, done = {}, []
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            m = self.TMP_SPAN.match(line)
            if m:
                starts[m.group(4).strip()] = (m.group(1), m.group(2), m.group(3))
                continue
            m = self.DONE_SPAN.match(line)
            if m:
                done.append((m.group(4).strip(), (m.group(1), m.group(2), m.group(3)),
                             int(m.group(5))))
        for label, end, code in done:
            yield label, starts.get(label), end, code

    def find_span(self, chain, label, start):
        for got_label, got_start, got_end, code in self.spans(chain):
            if got_label == label and code == 0 and got_start and tuple(start) == tuple(got_start):
                return got_end
        self.fail(f"chain{chain}: no completed span for {label!r} starting at {':'.join(start)}")

    @staticmethod
    def seconds_between(start, end):
        to_s = lambda t: int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
        delta = to_s(end) - to_s(start)
        return delta + 86400 if delta < 0 else delta

    def parse_window(self, window):
        """'2026-10-05 13:46:54 -> 18:58:31' and '... -> 2026-10-06 01:36:35' to clock pairs.

        A span that crosses midnight names the second date too, so each side is parsed separately.
        """
        m = re.search(r"(\d\d):(\d\d):(\d\d) -> (?:\d{4}-\d\d-\d\d )?(\d\d):(\d\d):(\d\d)", window)
        self.assertIsNotNone(m, f"cannot read a span out of {window!r}")
        return (m.group(1), m.group(2), m.group(3)), (m.group(4), m.group(5), m.group(6))

    def test_the_from_scratch_40m_run_matches_chain14(self):
        art = json.load(open(os.path.join(ROOT, "benchmarks",
                                          "target_learning_curve_from_scratch_paired.json"),
                             encoding="utf-8"))
        run = art["from_scratch_run"]
        start, end = self.parse_window(run["wall_clock"])
        self.assertEqual(end, self.find_span(14, "train SAC on target from scratch, 40M steps",
                                             start))
        self.assertEqual(run["wall_clock_seconds"], self.seconds_between(start, end))

    def test_the_5m_preset_pair_matches_chain18(self):
        art = json.load(open(os.path.join(ROOT, "benchmarks",
                                          "physics_presets_screen5m_paired.json"),
                             encoding="utf-8"))
        labels = {"v9": "screen SAC 5M in v9", "fast": "screen SAC 5M in fast"}
        for preset, arm in art["arms"].items():
            start, end = self.parse_window(arm["wall_clock_window"])
            self.assertEqual(end, self.find_span(18, labels[preset], start))
            self.assertEqual(arm["wall_clock_seconds"], self.seconds_between(start, end))

    def test_the_trainer_pair_matches_chains_19_and_20(self):
        art = json.load(open(os.path.join(ROOT, "benchmarks", "trainer_pair_ppo_sac.json"),
                             encoding="utf-8"))
        spec = {"ppo_v9": (19, "PPO 1M in v9 (n=32)"),
                "ppo_fast": (19, "PPO 1M in fast (n=32)"),
                "sac_v9": (20, "SAC 1M in v9 (n=32, the PPO config)")}
        for key, (chain, label) in spec.items():
            seconds = art["arms"][key]["seconds"]
            log_end = None
            for got_label, got_start, got_end, code in self.spans(chain):
                if got_label == label and code == 0 and got_start:
                    log_end, log_start = got_end, got_start
            self.assertIsNotNone(log_end, f"{label} never completed in chain{chain}")
            self.assertEqual(seconds, self.seconds_between(log_start, log_end),
                             f"{key}: the artifact records {seconds} s, chain{chain} says "
                             f"{self.seconds_between(log_start, log_end)} s")

    def test_the_dreamer_preset_pair_matches_chain21(self):
        art = json.load(open(os.path.join(ROOT, "benchmarks", "physics_presets_dreamer_pair.json"),
                             encoding="utf-8"))
        labels = {"v9": "Dreamer 250k in v9", "fast": "Dreamer 250k in fast"}
        for preset, arm in art["arms"].items():
            start, end = self.parse_window(arm["window"])
            self.assertEqual(end, self.find_span(21, labels[preset], start))
            self.assertEqual(arm["seconds"], self.seconds_between(start, end))

    def test_the_1m_triple_matches_chain22(self):
        art = json.load(open(os.path.join(ROOT, "benchmarks", "physics_presets_sac1m_triple.json"),
                             encoding="utf-8"))
        labels = {"v9": "SAC 1M in v9", "euler": "SAC 1M in euler", "fast": "SAC 1M in fast"}
        for preset, arm in art["arms"].items():
            start, end = self.parse_window(arm["window"])
            self.assertEqual(end, self.find_span(22, labels[preset], start))
            self.assertEqual(arm["train_seconds"], self.seconds_between(start, end))

    def test_the_second_draw_matches_chain23(self):
        """Two worlds at a second seed, and the scorer's own clock, both from one launcher log."""
        art = json.load(open(os.path.join(ROOT, "benchmarks",
                                          "physics_presets_screen5m_draws.json"), encoding="utf-8"))
        arms = art["draws"]["seed8"]["arms"]
        labels = {"v9": "second draw SAC 5M in v9", "fast": "second draw SAC 5M in fast"}
        for preset, arm in arms.items():
            start, end = self.parse_window(arm["wall_clock_window"])
            self.assertEqual(end, self.find_span(23, labels[preset], start))
            self.assertEqual(arm["wall_clock_seconds"], self.seconds_between(start, end))
        # The scoring spans carry no window in the artifact, so they are matched by label alone.
        scored = {label: self.seconds_between(start, end)
                  for label, start, end, code in self.spans(23)
                  if code == 0 and start and label.startswith("score the second draw")}
        self.assertEqual(scored["score the second draw in v9"], arms["v9"]["scoring_seconds"])
        self.assertEqual(scored["score the second draw in fast"], arms["fast"]["scoring_seconds"])

    def test_the_utd_pair_matches_chain24(self):
        """The 2.707x price of the update is only meaningful if both arms shared a window.

        So this checks the adjacency, not just the two spans: the ratio-4 arm has to start at the
        minute the ratio-1 arm's scoring finished, with nothing else claimed in between.
        """
        art = json.load(open(os.path.join(ROOT, "benchmarks", "utd_dose_pair.json"),
                             encoding="utf-8"))
        arms = art["arms"]
        labels = {"n32_r1": "SAC 5M n=32 utd=1", "n32_r4": "SAC 5M n=32 utd=4"}
        for name, arm in arms.items():
            start, end = self.parse_window(arm["wall_clock_window"])
            self.assertEqual(end, self.find_span(24, labels[name], start))
            self.assertEqual(arm["wall_clock_seconds"], self.seconds_between(start, end))
        scored = {label: (start, end) for label, start, end, code in self.spans(24)
                  if code == 0 and start and label.startswith("score SAC 5M")}
        for name in arms:
            label = f"score SAC 5M n=32 utd={arms[name]['utd_ratio']}"
            self.assertEqual(self.seconds_between(*scored[label]),
                             arms[name]["scoring_seconds"], f"{name}: scoring span disagrees")
        r1, r4 = arms["n32_r1"], arms["n32_r4"]
        self.assertEqual(self.parse_window(r4["wall_clock_window"])[0],
                         scored["score SAC 5M n=32 utd=1"][1],
                         "the ratio-4 arm did not start right after the ratio-1 arm was scored, so "
                         "their ratio is a cross-window comparison and the README must say so")
        # The smoke arm that validated --utd-ratio before the long runs: it completed, and it is the
        # only thing in this chain that ran the new loop on CUDA before the two arms did.
        smoke = [self.seconds_between(s, e) for label, s, e, code in self.spans(24)
                 if label == "SAC utd smoke, 4096 steps at ratio 3" and code == 0 and s]
        self.assertTrue(smoke and smoke[0] < 120,
                        "the ratio-3 smoke arm is missing from chain24 or never finished quickly")

    def test_the_ppo_10m_run_matches_chain25(self):
        """One training span and one scoring span, both from chain25's log."""
        art = json.load(open(os.path.join(ROOT, "benchmarks", "ppo_10m_vs_sac.json"),
                             encoding="utf-8"))
        ppo = art["ppo_10m"]
        start, end = self.parse_window(ppo["wall_clock_window"])
        self.assertEqual(end, self.find_span(25, "PPO 10M target n=32 v9", start))
        self.assertEqual(ppo["wall_clock_seconds"], self.seconds_between(start, end))
        scored = {label: self.seconds_between(s, e) for label, s, e, code in self.spans(25)
                  if code == 0 and s and label == "score the PPO 10M curve"}
        self.assertEqual(scored["score the PPO 10M curve"], ppo["scoring_seconds"])
        # The step count the log's completion line would have to agree with is not in the log, but the
        # rollout-boundary arithmetic is: 10M requested cannot be reached, and the artifact must name
        # the number that was actually collected.
        self.assertLess(ppo["collected_steps"], 10000000)
        self.assertEqual(ppo["collected_steps"] % (2048 * 32), 0)
        self.assertIn("PPO", art["protocol"])

    def test_the_dose_second_draw_matches_chain26(self):
        """The replication of 2.7x is only a replication if each draw shared its own window."""
        art = json.load(open(os.path.join(ROOT, "benchmarks", "utd_dose_draws.json"),
                             encoding="utf-8"))
        labels = {("seed7", "1"): "SAC 5M n=32 utd=1", ("seed7", "4"): "SAC 5M n=32 utd=4",
                  ("seed8", "1"): "SAC 5M n=32 utd=1 seed 8",
                  ("seed8", "4"): "SAC 5M n=32 utd=4 seed 8"}
        chains = {"seed7": 24, "seed8": 26}
        for tag, draw in art["draws"].items():
            for ratio, arm in draw["arms"].items():
                start, end = self.parse_window(arm["wall_clock_window"])
                self.assertEqual(end, self.find_span(chains[tag], labels[(tag, ratio)], start))
                self.assertEqual(arm["wall_clock_seconds"], self.seconds_between(start, end))
            scored = {label: (s, e) for label, s, e, code in self.spans(chains[tag])
                      if code == 0 and s and label.startswith("score SAC 5M")}
            for ratio in ("1", "4"):
                want = (f"score SAC 5M n=32 utd={ratio}"
                        + ("" if tag == "seed7" else " seed 8"))
                self.assertEqual(self.seconds_between(*scored[want]),
                                 draw["arms"][ratio]["scoring_seconds"],
                                 f"{tag}/{ratio}: the scorer's span disagrees")
            # Each draw is one window: the ratio-4 arm starts at the minute the ratio-1 arm's scoring
            # finished. Without that, 2.707x and 2.698x would be two cross-window comparisons.
            r1_end = scored[f"score SAC 5M n=32 utd=1"
                            + ("" if tag == "seed7" else " seed 8")][1]
            self.assertEqual(self.parse_window(draw["arms"]["4"]["wall_clock_window"])[0], r1_end,
                             f"{tag}: the two dose arms of this draw did not run back to back")
        # The cross-draw check the paragraph leans on: the two cost ratios are independent because the
        # two windows are - the seed-8 arms ran after midnight, the seed-7 arms the evening before.
        self.assertNotEqual(art["draws"]["seed7"]["arms"]["1"]["wall_clock_window"],
                            art["draws"]["seed8"]["arms"]["1"]["wall_clock_window"])

    def test_the_euler_5m_draws_match_chains_27_and_28(self):
        """Two worlds' clocks from two nights, each checked against its own launcher log."""
        art = json.load(open(os.path.join(ROOT, "benchmarks",
                                          "physics_presets_screen5m_euler_draws.json"),
                             encoding="utf-8"))
        spec = {"seed7": (27, "SAC 5M in euler", "score SAC 5M in euler"),
                "seed8": (28, "SAC 5M in euler seed 8", "score SAC 5M in euler seed 8")}
        for tag, (chain, train_label, score_label) in spec.items():
            arm = art["draws"][tag]
            start, end = self.parse_window(arm["wall_clock_window"])
            self.assertEqual(end, self.find_span(chain, train_label, start),
                             f"{tag}: chain{chain} has no completed {train_label!r} at that start")
            self.assertEqual(arm["wall_clock_seconds"], self.seconds_between(start, end))
            span = [(s, e) for label, s, e, code in self.spans(chain)
                    if label == score_label and code == 0 and s]
            self.assertEqual(len(span), 1, f"{tag}: the scoring span is not in chain{chain}")
            self.assertEqual(self.seconds_between(*span[0]), arm["scoring_seconds"])
        # The v9 clocks the ratios are built from are chain18/chain23's, already gated above; assert
        # the artifact is still quoting those same numbers rather than a re-typed pair.
        draws = json.load(open(os.path.join(ROOT, "benchmarks",
                                            "physics_presets_screen5m_draws.json"), encoding="utf-8"))
        for tag in ("seed7", "seed8"):
            self.assertEqual(art["draws"][tag]["v9_same_draw_seconds"],
                             draws["draws"][tag]["arms"]["v9"]["wall_clock_seconds"])

    def test_the_ppo_second_draw_matches_chain29(self):
        """The 2.28x spread between two windows of one recipe is only a fact if both clocks are."""
        art = json.load(open(os.path.join(ROOT, "benchmarks", "ppo_10m_draws.json"),
                             encoding="utf-8"))
        spec = {"seed7": (25, "PPO 10M target n=32 v9", "score the PPO 10M curve"),
                "seed8": (29, "PPO 10M target n=32 v9 seed 8", "score the PPO 10M curve seed 8")}
        for tag, (chain, train_label, score_label) in spec.items():
            arm = art["draws"][tag]
            start, end = self.parse_window(arm["wall_clock_window"])
            self.assertEqual(end, self.find_span(chain, train_label, start),
                             f"{tag}: chain{chain} has no completed {train_label!r} at that start")
            self.assertEqual(arm["wall_clock_seconds"], self.seconds_between(start, end))
            span = [(s, e) for label, s, e, code in self.spans(chain)
                    if label == score_label and code == 0 and s]
            self.assertEqual(len(span), 1, f"{tag}: the scoring span is not in chain{chain}")
            self.assertEqual(self.seconds_between(*span[0]), arm["scoring_seconds"])
        self.assertEqual(art["same_recipe_two_windows"]["spread_ratio"],
                         round(art["draws"]["seed8"]["wall_clock_seconds"]
                               / art["draws"]["seed7"]["wall_clock_seconds"], 3))

    def test_the_gail_retrain_matches_chain13(self):
        art = json.load(open(os.path.join(ROOT, "benchmarks", "phase4_gail_retrain.json"),
                             encoding="utf-8"))
        run = art["arms"]["fresh_retrain"]["training_run"]
        seconds = run["wall_clock_seconds"]
        # This one duration names its source as the mlflow record, not the launcher: the run window
        # starts when the trainer opens the run and ends when it closes it, while the launcher span
        # also carries the interpreter start and the process exit. Two honest measurements of one
        # run, so the contract here is self-consistency plus agreement to within the startup.
        start, end = self.parse_window(run["wall_clock_source"])
        self.assertEqual(seconds, self.seconds_between(start, end),
                         "the recorded seconds do not follow from the recorded window")
        log_start = log_end = None
        for label, got_start, got_end, code in self.spans(13):
            if label.startswith("train GAIL") and code == 0:
                log_start, log_end = got_start, got_end
        self.assertIsNotNone(log_end, "no completed GAIL training span in chain13")
        launcher = self.seconds_between(log_start, log_end)
        self.assertLessEqual(abs(launcher - seconds), 30,
                             f"the launcher saw {launcher} s and the artifact records {seconds} s; "
                             "more than half a minute apart is not process startup, it is drift")


class TestEvalRefusesToMixWorlds(unittest.TestCase):
    """Source-level guards for eval_phase1.py, which has to stay loadable against old revisions."""

    def setUp(self):
        with open(os.path.join(ROOT, "eval_phase1.py"), encoding="utf-8") as handle:
            self.src = handle.read()

    def test_the_preset_is_only_passed_when_it_is_not_the_default(self):
        # `--env-commit` aliases an older env revision in as the module, and that revision has no
        # physics_preset parameter: an unconditional kwarg would break every re-score to date.
        self.assertIsNotNone(re.search(r'preset_kwargs = \{\} if physics_preset == "v9"', self.src),
                             "eval_phase1.py no longer withholds the preset kwarg for v9, which "
                             "breaks --env-commit re-scores")

    def test_env_commit_and_a_preset_cannot_be_combined(self):
        self.assertIn("aliases an older env revision that has no --physics-preset", self.src,
                      "the guard against scoring a preset run against an aliased revision is gone")

    def test_the_recorded_artifact_names_the_world_it_scored_in(self):
        self.assertIn('"scored_in_version"', self.src,
                      "the artifact stopped recording which world the episodes ran in, which is "
                      "the mistake the device field was added to fix")


if __name__ == "__main__":
    unittest.main()
