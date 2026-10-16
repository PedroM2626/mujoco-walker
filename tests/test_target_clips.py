"""Contract tests for the clip tool and the marker it has to put on screen.

Two things are pinned here. The first is the env change the clips depend on: `target_marker` is a mocap
body, so writing `model.body_pos` alone left it parked 10 m underground and no frame could show the goal.
That fix is display-only, and "display-only" is a claim with a control attached to it - the same seed run
twice, once with the marker moved, has to produce byte-identical observations and rewards. Without that
parity test the fix would be an unverified touch on the published MDP.

The second is the tool's own machinery: the arrival label, the per-geom lowest point (the bounding-sphere
radius is the bug it replaces), the legibility check with a positive control, and the camera that keeps
both agent and goal in shot. The measured clips themselves live in `benchmarks/target_clips_*.json`;
nothing here re-renders an episode.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import envs.walker_ragdoll_env  # noqa: E402,F401
from make_target_clips import (BELOW_FLOOR_TOL_M, arm_depth, arm_floor_contacts, classify,
                               frame_is_legible, geom_lowest_point, goal_camera, marker_in_frame,
                               overlay, render_episode)


def _env(phase="target", **kwargs):
    return gym.make("WalkerRagdoll-v0", task_phase=phase, reset_mode="mixed", **kwargs)


def _telemetry(min_dist=3.0, upright=None, reached=None, healthy=True, vel=0.5):
    reached = min_dist <= 0.45 if reached is None else reached
    upright = (reached and min_dist <= 0.45) if upright is None else upright
    return {"min_target_distance": min_dist, "reached_target": bool(reached),
            "min_target_distance_upright": min_dist if upright else None,
            "reached_target_upright": bool(upright), "mean_x_velocity": vel,
            "steps": 1000, "falls": 0, "return": 0.0,
            "arm_floor_contact_steps": 0, "arm_below_floor_steps": 0,
            "arm_max_depth_below_floor_m": 0.0}


class TestTheMarkerIsWhereTheGoalIs(unittest.TestCase):
    """The fix the clips exist because of, and the control that says it changes nothing else."""

    def test_target_phase_puts_the_marker_on_the_sampled_goal(self):
        env = _env()
        try:
            _, info = env.reset(seed=14)
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            pos = env.unwrapped.data.xpos[env.unwrapped._target_body_id]
            target = env.unwrapped._target_xy
            self.assertAlmostEqual(float(pos[0]), float(target[0]), places=9)
            self.assertAlmostEqual(float(pos[1]), float(target[1]), places=9)
            self.assertAlmostEqual(float(pos[2]), 0.05, places=9)
            self.assertLess(abs(float(pos[0]) - float(env.unwrapped.data.qpos[0])), 6.0)
        finally:
            env.close()

    def test_a_non_target_phase_parks_it_below_the_floor(self):
        env = _env("recovery")
        try:
            env.reset(seed=3)
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            pos = env.unwrapped.data.xpos[env.unwrapped._target_body_id]
            self.assertAlmostEqual(float(pos[2]), -10.0, places=9)
        finally:
            env.close()

    def test_the_marker_is_a_mocap_body_that_collides_with_nothing(self):
        # Why moving it cannot change the dynamics: prescribed pose, no contact pair, not part of any
        # observation or reward term.
        env = _env()
        try:
            u = env.unwrapped
            self.assertGreaterEqual(int(u.model.body_mocapid[u._target_body_id]), 0,
                                    "the body is not mocap, so mocap_pos would not move it")
            gid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker_geom")
            self.assertEqual(int(u.model.geom_contype[gid]), 0)
            self.assertEqual(int(u.model.geom_conaffinity[gid]), 0)
        finally:
            env.close()

    def test_moving_the_marker_leaves_the_observation_and_reward_trace_untouched(self):
        """The parity control: identical seed, identical actions, one env with the goal moved."""
        a_env, b_env = _env(), _env()
        try:
            obs_a, _ = a_env.reset(seed=11)
            obs_b, _ = b_env.reset(seed=11)
            # Same goal for the physics, different painted position.
            b_env.unwrapped._set_marker_xy(float(b_env.unwrapped._target_xy[0]) + 1.7,
                                           float(b_env.unwrapped._target_xy[1]) - 0.9)
            mujoco.mj_forward(b_env.unwrapped.model, b_env.unwrapped.data)
            moved = b_env.unwrapped.data.xpos[b_env.unwrapped._target_body_id]
            self.assertGreater(abs(float(moved[0]) - float(a_env.unwrapped._target_xy[0])), 1.0,
                               "the marker did not move, so this control proves nothing")
            rng = np.random.default_rng(0)
            for step in range(60):
                action = rng.uniform(-0.5, 0.5, size=17)
                obs_a, r_a, t_a, tr_a, i_a = a_env.step(action)
                obs_b, r_b, t_b, tr_b, i_b = b_env.step(action)
                # Not bit-exact: the control calls mj_forward once on one side to move the marker, and
                # an extra forward pass reorders float accumulation by ~1e-15. Physics is identical at
                # any precision a reward could notice.
                np.testing.assert_allclose(np.asarray(obs_a), np.asarray(obs_b), atol=1e-12)
                self.assertAlmostEqual(float(r_a), float(r_b), places=12,
                                       msg=f"reward diverged at step {step}")
                self.assertEqual(t_a, t_b)
                self.assertAlmostEqual(float(i_a["target_distance"]),
                                       float(i_b["target_distance"]), places=12)
        finally:
            a_env.close()
            b_env.close()


class TestArrivalLabels(unittest.TestCase):
    """The label on the filename is the scorer's rule, not a guess from what the frame looks like."""

    RADIUS = 0.45

    def test_distance_arrival_and_upright_arrival_are_different_labels(self):
        fell_in = _telemetry(min_dist=0.30, reached=True, upright=False)
        walked_in = _telemetry(min_dist=0.30, reached=True, upright=True)
        self.assertEqual(classify(fell_in, self.RADIUS, True), "reached_by_falling")
        self.assertEqual(classify(walked_in, self.RADIUS, True), "reached_upright")

    def test_near_miss_dawdling_and_stall_split_on_distance_then_health(self):
        self.assertEqual(classify(_telemetry(min_dist=0.85), self.RADIUS, True), "near_miss")
        self.assertEqual(classify(_telemetry(min_dist=2.6), self.RADIUS, True), "dawdling")
        self.assertEqual(classify(_telemetry(min_dist=2.6), self.RADIUS, False), "stalled")

    def test_the_boundary_is_the_scorers_radius_not_a_rounded_one(self):
        just_in = _telemetry(min_dist=0.45, reached=True, upright=True)
        self.assertEqual(classify(just_in, self.RADIUS, True), "reached_upright")
        # 0.90 m is exactly 2 radii: still a near miss, 0.91 is not.
        self.assertEqual(classify(_telemetry(min_dist=0.90), self.RADIUS, True), "near_miss")
        self.assertEqual(classify(_telemetry(min_dist=0.91), self.RADIUS, True), "dawdling")


class TestGeomLowestPoint(unittest.TestCase):
    """The metric behind "the arm went through the floor", and the bound it replaced."""

    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def gid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def test_a_sphere_is_its_radius_below_its_centre(self):
        gid = self.gid("right_hand")
        self.assertEqual(int(self.model.geom_type[gid]), int(mujoco.mjtGeom.mjGEOM_SPHERE))
        z = float(self.data.geom_xpos[gid][2])
        self.assertAlmostEqual(geom_lowest_point(self.model, self.data, gid),
                               z - float(self.model.geom_size[gid][0]), places=12)

    def test_a_capsule_uses_its_axis_not_its_bounding_sphere(self):
        gid = self.gid("right_uarm1")
        self.assertEqual(int(self.model.geom_type[gid]), int(mujoco.mjtGeom.mjGEOM_CAPSULE))
        axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
        want = (float(self.data.geom_xpos[gid][2]) - float(self.model.geom_size[gid][0])
                - float(self.model.geom_size[gid][1]) * abs(float(axis[2])))
        got = geom_lowest_point(self.model, self.data, gid)
        self.assertAlmostEqual(got, want, places=12)
        # The bug this replaces: rbound is the bounding sphere, far larger than the limb's thickness.
        naive = float(self.data.geom_xpos[gid][2]) - float(self.model.geom_rbound[gid])
        self.assertLess(naive, got - 0.05,
                        "the naive bound no longer over-reports depth, so the tolerance next to it "
                        "is measuring something else")

    def test_depth_is_zero_while_nothing_is_under_the_floor(self):
        gids = [self.gid("right_uarm1"), self.gid("right_hand")]
        self.assertEqual(arm_depth(self.model, self.data, gids), 0.0)
        self.assertLess(arm_depth(self.model, self.data, []), 1e-12)


class TestArmFloorContacts(unittest.TestCase):
    def test_the_contact_is_found_by_geom_pair_and_nothing_else(self):
        model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "walker_ragdoll.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        arm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_hand")
        leg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        data.ncon = 1
        data.contact[0].geom[0] = floor
        data.contact[0].geom[1] = arm
        self.assertTrue(arm_floor_contacts(data, [arm], floor))
        self.assertFalse(arm_floor_contacts(data, [leg], floor),
                        "a foot-on-floor contact is not an arm touching the floor")
        data.contact[0].geom[0] = leg
        data.contact[0].geom[1] = arm
        self.assertFalse(arm_floor_contacts(data, [arm], floor),
                        "two arm geoms touching each other are not the floor")
        data.ncon = 0
        self.assertFalse(arm_floor_contacts(data, [arm], floor))

    def test_the_tolerance_separates_resting_from_sinking(self):
        # Published resting arms penetrate by solver softness (millimetres); the reduced world put
        # them half a metre down. The threshold has to sit between the two, and the committed
        # artifacts say where it does.
        self.assertLess(BELOW_FLOOR_TOL_M, 0.1)
        self.assertGreater(BELOW_FLOOR_TOL_M, 0.005)


class TestFrameChecks(unittest.TestCase):
    """A frame that shows nothing must fail, and a frame that shows something must pass."""

    def _scene(self):
        # Checker floor, blue-grey sky, a tan blob for the robot and a red dot for the marker.
        f = np.zeros((200, 300, 3), dtype=np.uint8)
        f[:100] = (70, 85, 95)
        f[100:, ::20] = (10, 10, 10)
        f[100:, ::20, :] = (200, 160, 110)
        f[80:110, 140:170] = (205, 150, 100)
        f[150:165, 240:255] = (230, 30, 30)
        return f

    def test_positive_control_a_scene_with_a_robot_and_a_marker_passes(self):
        scene = self._scene()
        ok, why = frame_is_legible(scene)
        self.assertTrue(ok, why)
        self.assertGreaterEqual(marker_in_frame(scene), 40, "the marker is not being counted")

    def test_a_blank_or_all_robot_frame_fails(self):
        ok, _ = frame_is_legible(np.full((200, 300, 3), 90, dtype=np.uint8))
        self.assertFalse(ok, "a flat frame passed the legibility check")
        ok, _ = frame_is_legible(np.full((200, 300, 3), (205, 150, 100), dtype=np.uint8))
        self.assertFalse(ok, "a frame that is entirely robot colour passed")
        self.assertEqual(marker_in_frame(np.full((60, 60, 3), 40, dtype=np.uint8)), 0)

    def test_overlay_puts_text_on_the_frame_without_losing_the_scene(self):
        scene = self._scene()
        out = overlay(scene, ["phase=target  world=published", "seed=14  goal distance 3.00 m"],
                      banner="NOT the published MDP")
        self.assertEqual(out.shape, scene.shape)
        self.assertLess(int(out[:52].mean()), int(scene[:52].mean()),
                        "the text strip did not darken the top of the frame")
        self.assertGreater(int(out[:52, :, :].max()), 200, "no light text pixels were drawn")
        red = (out[52:90, :, 0].astype(int) > 200) & (out[52:90, :, 1].astype(int) < 120)
        self.assertTrue(red.any(), "the banner did not appear")


class TestGoalCamera(unittest.TestCase):
    def test_it_looks_across_the_goal_line_and_scales_with_distance(self):
        near = goal_camera(np.array([0.0, 0.0]), np.array([3.0, 0.0]), 3.0)
        self.assertAlmostEqual(near.azimuth, 90.0, places=6)
        self.assertAlmostEqual(near.lookat[0], 1.5, places=6)
        far = goal_camera(np.array([0.0, 0.0]), np.array([5.0, 0.0]), 5.0)
        self.assertGreater(far.distance, near.distance)
        # A goal behind the agent flips the camera to the other side rather than losing it.
        back = goal_camera(np.array([0.0, 0.0]), np.array([-3.0, 0.0]), 3.0)
        self.assertAlmostEqual((back.azimuth - near.azimuth) % 360.0, 180.0, places=6)

    def test_the_distance_has_a_floor_so_close_episodes_stay_framed(self):
        cam = goal_camera(np.array([0.0, 0.0]), np.array([0.2, 0.0]), 0.2)
        self.assertGreaterEqual(cam.distance, 6.0)


class TestTheRenderPipelineRunsEndToEnd(unittest.TestCase):
    """The act of rendering, in the suite, with a scripted policy and no checkpoint.

    The committed clip sidecars cannot be re-derived here - they need the 4M-step actor - so what
    would otherwise rot quietly is the pipeline that writes them: environment under an alternative
    XML, goal-framed camera, overlay, mp4 writer, filmstrip, telemetry. A broken writer or a GL
    regression would then surface only the next time someone tried to make a clip. This runs the
    whole path for a few dozen steps against a zero-action policy, which is enough to touch every
    stage and costs about a second.
    """

    def _render(self, tmp, steps=24, world="published"):
        from make_target_clips import variant_xml_path

        path, tag = variant_xml_path(world, tmp)
        tele = render_episode(lambda obs: np.zeros(17), path, 14, steps, 50, tmp, tag,
                              f"{world}_smoke", 0.45,
                              {"reset_mode": "mixed", "task_phase": "target"})
        return tele, path

    def test_a_short_episode_produces_video_frames_and_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tele, _ = self._render(tmp)
            self.assertEqual(tele["steps"], 24)
            self.assertEqual(tele["frames"], 24)
            self.assertIn(tele["outcome"], {"reached_upright", "reached_by_falling", "near_miss",
                                            "dawdling", "stalled"})
            for key in ("min_target_distance", "reached_target", "arm_floor_contact_steps",
                        "arm_max_depth_below_floor_m", "mean_x_velocity", "video", "frame_check"):
                self.assertIn(key, tele)
            stem = os.path.join(tmp, tele["video"][:-4])
            self.assertGreater(os.path.getsize(stem + ".mp4"), 10_000,
                               "the mp4 is a stub - the writer is not encoding")
            for suffix in ("start", "mid", "end"):
                self.assertTrue(os.path.exists(f"{stem}_{suffix}.png"), f"no {suffix} frame")
            self.assertTrue(os.path.exists(f"{stem}_filmstrip.png"))
            self.assertIn("robot-coloured pixel ratio", tele["frame_check"])
            self.assertGreater(int(tele["frame_check"].split("marker pixels")[1]), 0,
                               "the goal was not in the first frame")

    def test_the_reduced_world_is_reachable_through_the_same_pipeline(self):
        # The variant XML has to load through gymnasium's `xml_file`, which is how the published
        # contrast was produced; a relative path silently resolves against gymnasium's assets
        # directory, and that is the failure this guards.
        with tempfile.TemporaryDirectory() as tmp:
            tele, path = self._render(tmp, steps=12, world="gait_reduced")
            self.assertTrue(os.path.isabs(path))
            self.assertEqual(tele["arm_floor_contact_steps"], 0,
                             "an arm touched the floor in the reduced world, so the pair set changed")


class TestTheCommittedClipsSayWhatTheParagraphClaims(unittest.TestCase):
    """The published contrast: contact where the pair exists, none where it does not."""

    FILES = {"published": os.path.join(ROOT, "benchmarks", "target_clips_published.json"),
             "gait_reduced": os.path.join(ROOT, "benchmarks", "target_clips_gait_reduced.json")}

    def test_both_worlds_exist_and_share_one_checkpoint(self):
        arts = {k: json.load(open(p, encoding="utf-8")) for k, p in self.FILES.items()}
        self.assertEqual({a["model"] for a in arts.values()},
                         {arts["published"]["model"]},
                         "the two worlds are no longer the same policy, so the contrast is not a "
                         "contact-set comparison")
        self.assertEqual({a["world"] for a in arts.values()}, set(self.FILES))

    def test_the_reduced_world_never_touches_the_floor_with_an_arm(self):
        art = json.load(open(self.FILES["gait_reduced"], encoding="utf-8"))
        for ep in art["episodes"]:
            self.assertEqual(ep["arm_floor_contact_steps"], 0,
                             f"seed {ep['seed']}: an arm contacted the floor, so the pair set "
                             "changed - the clip no longer shows what the paragraph says")
            self.assertGreater(ep["arm_max_depth_below_floor_m"], 0.3,
                               f"seed {ep['seed']}: nothing sank through the floor")

    def test_the_published_world_rests_on_the_floor_instead(self):
        art = json.load(open(self.FILES["published"], encoding="utf-8"))
        self.assertTrue(all(ep["arm_floor_contact_steps"] > 0 for ep in art["episodes"]))
        self.assertLess(max(ep["arm_max_depth_below_floor_m"] for ep in art["episodes"]), 0.1)

    def test_every_clip_carries_its_own_evidence(self):
        for world, path in self.FILES.items():
            art = json.load(open(path, encoding="utf-8"))
            for ep in art["episodes"]:
                self.assertIn(ep["outcome"], {"reached_upright", "reached_by_falling",
                                              "near_miss", "dawdling", "stalled"})
                self.assertTrue(ep["video"].startswith(f"{world}_"),
                                f"{world}: {ep['video']} is not named after its world")
                self.assertIn("robot-coloured pixel ratio", ep["frame_check"])
                self.assertIn("marker pixels", ep["frame_check"])
                self.assertGreater(int(ep["frame_check"].split("marker pixels")[1]), 0,
                                   "a clip was written without the goal visible in its first frame")


if __name__ == "__main__":
    unittest.main()
