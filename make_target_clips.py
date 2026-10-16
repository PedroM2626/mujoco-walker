"""Render target-phase episodes as labelled clips, from a checkpoint, in any contact world.

The Phase-1 target task had clips before - four episodes of the 4M-step SAC checkpoint, with the
outcome written into the filename - but the tool that drew them lived in a temporary directory, and so
did the videos. That made the most human-legible evidence in the repository the least reproducible
part of it: nothing committed could say what was on screen, and a cleaned-up temp folder would have
taken the phase, the distance and the goal down with it. This script is the versioned tool. It reuses
`eval_phase1.build_policy`, so the agent in the clip is the same deterministic policy the scored rows
come from, and it writes a JSON sidecar next to the videos so every claim made about a clip can be
checked against a committed artifact.

What goes on screen is deliberate. A frame of a ragdoll near a red sphere does not say whether the
episode counted as an arrival, so each frame carries the phase, the live distance to the goal, the
goal bearing, the standing state and the checkpoint, and the outcome is decided from the same
telemetry the scorer uses (`eval_phase1.score`): distance-only arrival and the stricter
arrival-standing are different labels, because a fall forward can land inside the radius and score
nothing.

    python make_target_clips.py --model checkpoints/sac_target_from_scratch_40m/sac_ckpt_4000000.pt \
        --seeds 14,25,11,29 --out-dir videos
    python make_target_clips.py --world gait_reduced --seeds 14 --out-dir videos

`--world` rebuilds the published XML with the contact pairs of `bench_mjx_pairs.py` and points the
environment at it. That is how the reduced set gets shown for what it is: the same weights walk in a
world where an arm that falls on the floor goes through it, and the script measures exactly that
(`arm_below_floor_steps`, `arm_max_depth_below_floor_m`) instead of asking the viewer to trust a frame.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import envs.walker_ragdoll_env  # noqa: E402,F401  (registers WalkerRagdoll-v0)
from envs import normalize_compat  # noqa: E402,F401  (pickle shim for obs_rms)
from eval_phase1 import EPISODE_STEPS, build_policy  # noqa: E402

FRAME_SIZE = (960, 540)
# The arm bodies whose geoms the gait-reduced contact set takes out of collision.
ARM_BODIES = ("right_upper_arm", "right_lower_arm", "left_upper_arm", "left_lower_arm")
# A resting arm penetrates the floor by solver softness only; centimetres means it went through.
BELOW_FLOOR_TOL_M = 0.02


def classify(telemetry, radius, healthy_at_end):
    """One label per episode, from the same two arrival rules the scorer publishes.

    Distance-only arrival and arrival-standing are kept apart on purpose: a forward fall that lands
    inside the radius is not a walk to the target, and a clip that calls it "reached" is the mistake
    this repository already made once (see docs/lab-notes.md).
    """
    reached = bool(telemetry["reached_target"])
    upright = bool(telemetry["reached_target_upright"])
    if reached and upright:
        return "reached_upright"
    if reached:
        return "reached_by_falling"
    if telemetry["min_target_distance"] <= 2.0 * radius:
        return "near_miss"
    return "dawdling" if healthy_at_end else "stalled"


def geom_lowest_point(model, data, gid):
    """z of the lowest point of a geom, per type.

    `geom_rbound` is the bounding-*sphere* radius, which for the arm capsules is the half-length plus
    the radius (0.167 m of slack on a 0.04 m thick limb). Subtracting it from the centre is how a
    resting arm looked 0.15 m underground, so the sphere and capsule cases are computed from
    `geom_size` instead; anything else falls back to the bound and says so through the caller's
    tolerance.
    """
    z = float(data.geom_xpos[gid][2])
    gtype = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - float(size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
        return z - float(size[0]) - float(size[1]) * abs(float(axis[2]))
    return z - float(model.geom_rbound[gid])


def arm_floor_contacts(data, arm_geom_ids, floor_geom_id):
    """True when an arm geom is in contact with the floor - the pair set is what decides this."""
    arm = set(arm_geom_ids)
    for k in range(data.ncon):
        c = data.contact[k].geom
        if (int(c[0]) == floor_geom_id and int(c[1]) in arm) or \
           (int(c[1]) == floor_geom_id and int(c[0]) in arm):
            return True
    return False


def arm_depth(model, data, arm_geom_ids):
    """How far below the floor plane the lowest arm surface is, in metres (0.0 if none is)."""
    return max((max(0.0, -geom_lowest_point(model, data, gid)) for gid in arm_geom_ids),
               default=0.0)


def goal_camera(robot_xy, target_xy, distance):
    """A free camera that puts the agent on one side and the goal on the other.

    The XML's `track` camera follows the torso from behind along its own axis, which frames the agent
    and loses the marker whenever the goal is off to the side. Looking perpendicular to the
    robot-to-goal line keeps both in shot for the whole episode, which is the point of the clip.
    """
    cam = mujoco.MjvCamera()
    cam.lookat[0] = 0.5 * (float(robot_xy[0]) + float(target_xy[0]))
    cam.lookat[1] = 0.5 * (float(robot_xy[1]) + float(target_xy[1]))
    cam.lookat[2] = 0.8
    bearing = math.degrees(math.atan2(float(target_xy[1]) - float(robot_xy[1]),
                                      float(target_xy[0]) - float(robot_xy[0])))
    cam.azimuth = bearing + 90.0
    cam.elevation = -14.0
    cam.distance = max(6.0, 1.35 * float(distance))
    return cam


def overlay(frame, lines, banner=None):
    """Write the telemetry onto the frame: a dark strip on top, text in it, banner in red."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=20)
    big = ImageFont.load_default(size=24)
    top = 26 * len(lines) + 10
    draw.rectangle([0, 0, img.width, top], fill=(12, 12, 12))
    for i, line in enumerate(lines):
        draw.text((10, 6 + 26 * i), line, fill=(235, 235, 235), font=font)
    if banner:
        draw.text((10, top + 6), banner, fill=(255, 70, 70), font=big)
    return np.asarray(img)


def frame_is_legible(frame, robot_pixel_ratio_range=(0.002, 0.5)):
    """Reject a frame that cannot show anything: flat colour, or all robot, all floor.

    A shape check passes on a black image and on a frame of the wrong scene, so the test here is on
    content: enough spread in the pixels that something is drawn, and a non-trivial amount of the
    robot's tan geom colour without the whole frame being that colour.
    """
    a = np.asarray(frame).astype(np.int16)
    if a.std() < 8.0:
        return False, "flat frame"
    # The ragdoll geoms use the tan `geom` material; the floor is a green/black checker and the sky a
    # blue-grey gradient, so "reddish and greenish in balance" is the robot being present.
    tan = ((a[:, :, 0] > 120) & (a[:, :, 0] - a[:, :, 2] > 25)).mean()
    if not (robot_pixel_ratio_range[0] <= tan <= robot_pixel_ratio_range[1]):
        return False, f"robot-coloured pixel ratio {tan:.4f} outside {robot_pixel_ratio_range}"
    return True, f"robot-coloured pixel ratio {tan:.4f}"


def marker_in_frame(renderer_frame, min_red_pixels=40):
    """Count the target marker's red pixels, so 'the goal is on screen' is a measurement."""
    a = np.asarray(renderer_frame).astype(np.int16)
    red = (a[:, :, 0] > 170) & (a[:, :, 1] < 120) & (a[:, :, 2] < 120)
    return int(red.sum())


def variant_xml_path(world, out_dir):
    """Write the chosen contact world to an XML file the environment can load."""
    if world == "published":
        return os.path.join(HERE, "walker_ragdoll.xml"), "v9 published (automatic filter)"
    import bench_mjx_pairs as pairs

    model = mujoco.MjModel.from_xml_path(pairs.XML_PATH)
    with open(pairs.XML_PATH, encoding="utf-8") as handle:
        text = handle.read()
    full = pairs.candidate_pairs(model, text)
    name_of = {gid: pairs.geom_name(model, gid) for gid in range(model.ngeom)}
    if world == "explicit_full":
        chosen = full
        tag = "explicit <pair> for every filtered candidate - same MDP, verified"
    elif world == "gait_reduced":
        chosen = [(i, j) for i, j in full
                  if (name_of[i] == "floor" and name_of[j] in pairs.GAIT_FLOOR_GEOMS)
                  or (name_of[j] == "floor" and name_of[i] in pairs.GAIT_FLOOR_GEOMS)
                  or (name_of[i] in pairs.GAIT_SELF_GEOMS and name_of[j] in pairs.GAIT_SELF_GEOMS)]
        tag = ("gait-reduced <pair> set - ARMS OUT OF COLLISION, this is NOT the published MDP")
    else:
        raise SystemExit(f"unknown --world {world!r}")
    os.makedirs(out_dir, exist_ok=True)
    # Gymnasium resolves a relative `xml_file` against its own assets directory, so the variant has to
    # reach the environment as an absolute path.
    path = os.path.abspath(os.path.join(out_dir, f"walker_ragdoll_{world}.xml"))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(pairs.build_variant_xml(model, chosen))
    return path, f"{tag} ({len(chosen)} declared pairs)"


def render_episode(policy, path, seed, steps, fps, out_dir, world_tag, model_step, radius,
                   env_kwargs, banner=None):
    """One episode to an mp4 plus start/mid/end frames, with the telemetry it was judged by."""
    env = gym.make("WalkerRagdoll-v0", xml_file=path, **env_kwargs)
    u = env.unwrapped
    u.model.vis.global_.offwidth, u.model.vis.global_.offheight = FRAME_SIZE
    arm_geoms = [int(gid) for gid in range(u.model.ngeom)
                 if int(u.model.geom_bodyid[gid]) in
                 {mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES
                  if mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0}]
    floor_geom = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    renderer = mujoco.Renderer(u.model, height=FRAME_SIZE[1], width=FRAME_SIZE[0])

    obs, info = env.reset(seed=seed)
    policy_reset = getattr(policy, "reset", None)
    if policy_reset:
        policy_reset()
    frames, tele_rows = [], []
    dist = float(info.get("target_distance", np.inf))
    min_dist = dist
    min_dist_upright = float("inf")
    vel_sum = 0.0
    total = 0.0
    below_floor_steps = 0
    arm_contact_steps = 0
    max_depth = 0.0
    n_steps = 0
    was_healthy = u.is_healthy
    falls = 0
    writer = None
    tag = None
    try:
        for i in range(steps):
            action = np.asarray(policy(obs), dtype=np.float64).reshape(-1)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            total += float(reward)
            dist = float(info.get("target_distance", np.inf))
            min_dist = min(min_dist, dist)
            vel_sum += float(info.get("x_velocity", 0.0))
            upright_now = float(u.upright_factor)
            standing_now = float(u.data.qpos[2]) > 1.0 and upright_now > 0.7
            if dist <= radius and standing_now:
                min_dist_upright = min(min_dist_upright, dist)
            is_healthy = u.is_healthy
            if was_healthy and not is_healthy:
                falls += 1
            was_healthy = is_healthy
            depth = arm_depth(u.model, u.data, arm_geoms)
            max_depth = max(max_depth, depth)
            touched = arm_floor_contacts(u.data, arm_geoms, floor_geom)
            arm_contact_steps += int(touched)
            below_floor_steps += int(depth > BELOW_FLOOR_TOL_M)

            cam = goal_camera(u.data.qpos[:2], u._target_xy, dist)
            renderer.update_scene(u.data, camera=cam)
            frame = renderer.render()
            if i == 0:
                ok, why = frame_is_legible(frame)
                tag = f"{why}; marker pixels {marker_in_frame(frame)}"
                if not ok:
                    raise RuntimeError(f"first frame is not legible: {why}")
            lines = [
                f"phase=target  world={world_tag}",
                f"seed={seed}  step {i + 1}/{steps}  goal distance {dist:.2f} m  "
                f"min so far {min_dist:.2f} m",
                f"torso z {float(u.data.qpos[2]):.2f} m  upright {upright_now:.2f}  "
                f"x-vel {float(info.get('x_velocity', 0.0)):+.2f} m/s  falls {falls}",
                f"arm-floor contact {'yes' if touched else 'no '}  "
                f"arm below floor {depth:+.3f} m",
            ]
            frames.append(overlay(frame, lines, banner))
            if writer is None:
                import imageio.v2 as imageio2
                os.makedirs(out_dir, exist_ok=True)
                writer = imageio2.get_writer(os.path.join(
                    out_dir, f"{model_step}s_seed{seed}.mp4"), fps=fps, codec="libx264",
                    pixelformat="yuv420p", macro_block_size=1)
            writer.append_data(frames[-1])
            if terminated:
                break
    finally:
        if writer:
            writer.close()
        renderer.close()
        env.close()

    tele = {"steps": n_steps,
            "min_target_distance": round(min_dist, 3),
            "reached_target": bool(min_dist <= radius),
            "min_target_distance_upright": (round(min_dist_upright, 3)
                                            if np.isfinite(min_dist_upright) else None),
            "reached_target_upright": bool(min_dist_upright <= radius),
            "mean_x_velocity": round(vel_sum / max(n_steps, 1), 4),
            "falls": falls, "return": round(total, 2),
            "arm_floor_contact_steps": arm_contact_steps,
            "arm_below_floor_steps": below_floor_steps,
            "arm_max_depth_below_floor_m": round(max_depth, 4)}
    name = classify(tele, radius, was_healthy)
    stem = os.path.join(out_dir, f"{model_step}s_seed{seed}_{name}")
    mp4 = os.path.join(out_dir, f"{model_step}s_seed{seed}.mp4")
    os.replace(mp4, stem + ".mp4")
    for label, idx in (("start", 0), ("mid", len(frames) // 2), ("end", len(frames) - 1)):
        Image.fromarray(frames[idx]).save(f"{stem}_{label}.png")
    strip_h = 200
    strip_w = int(FRAME_SIZE[0] * strip_h / FRAME_SIZE[1])
    picks = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    strip = Image.new("RGB", (strip_w * len(picks), strip_h + 24), (12, 12, 12))
    draw = ImageDraw.Draw(strip)
    for k, idx in enumerate(picks):
        strip.paste(Image.fromarray(frames[idx]).resize((strip_w, strip_h)),
                    (k * strip_w, 24))
        draw.text((k * strip_w + 4, 4), f"step {int(tele['steps'] * idx / max(len(frames) - 1, 1))}",
                  fill=(235, 235, 235), font=ImageFont.load_default(size=16))
    strip.save(f"{stem}_filmstrip.png")
    tele.update({"seed": seed, "outcome": name, "frames": len(frames),
                 "frame_check": tag, "video": os.path.basename(stem + ".mp4")})
    return tele


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint path (any Phase-1 actor)")
    ap.add_argument("--seeds", default="14,25,11,29")
    ap.add_argument("--world", default="published",
                    choices=("published", "explicit_full", "gait_reduced"))
    ap.add_argument("--steps", type=int, default=EPISODE_STEPS)
    ap.add_argument("--fps", type=int, default=100, help="100 fps is real time at dt 0.01 s")
    ap.add_argument("--out-dir", default="videos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    label, phase, width, policy, policy_reset, rinfo = build_policy(args.model, args.device)
    policy.reset = policy_reset
    ck_step = os.path.basename(args.model).rsplit("_", 1)[-1].split(".")[0]
    env_kwargs = {"reset_mode": "mixed", "task_phase": phase or "target",
                  **(rinfo["reward_kwargs"] or {})}
    if rinfo["target_curriculum"]:
        env_kwargs["target_curriculum"] = True
    path, world_tag = variant_xml_path(args.world, args.out_dir)
    banner = None if args.world == "published" else (
        "NOT the published MDP: arms removed from collision" if args.world == "gait_reduced"
        else "explicit <pair> set, verified same MDP")
    radius = 0.45
    print(f"[CLIP] model={args.model} algo={label} phase={phase} obs={width} "
          f"world={world_tag} curriculum={rinfo['target_curriculum']}")
    t0 = time.perf_counter()
    episodes = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        tele = render_episode(policy, path, seed, args.steps, args.fps, args.out_dir,
                              args.world, f"{args.world}_{ck_step}", radius, env_kwargs, banner)
        episodes.append(tele)
        print(f"[EP {seed}] {tele['outcome']}  min_dist={tele['min_target_distance']} m  "
              f"upright_arrival={tele['reached_target_upright']}  "
              f"arm-floor contact={tele['arm_floor_contact_steps']} steps  "
              f"arm below floor={tele['arm_below_floor_steps']} steps "
              f"(max {tele['arm_max_depth_below_floor_m']} m)  "
              f"[{time.perf_counter() - t0:.0f}s]")

    out_json = args.out_json or os.path.join("benchmarks", f"target_clips_{args.world}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump({"protocol": f"make_target_clips.py: {label} actor, deterministic, "
                              f"task_phase={phase}, reset_mode=mixed, {args.steps} env steps "
                              f"max at {args.fps} fps, world={world_tag}",
                   "model": args.model, "algo": label, "task_phase": phase,
                   "checkpoint_env_version": rinfo["env_version"],
                   "reward_source": rinfo["reward_source"],
                   "value_shaping_recorded_not_applied": rinfo["value_shaping"],
                   "target_curriculum": rinfo["target_curriculum"],
                   "world": args.world, "world_tag": world_tag, "xml": path,
                   "target_radius": radius, "fps": args.fps,
                   "below_floor_tolerance_m": BELOW_FLOOR_TOL_M,
                   "seconds": round(time.perf_counter() - t0, 1),
                   "episodes": episodes}, handle, indent=1)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
