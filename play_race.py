"""Corrida multi-agente do ragdoll customizado (Fases 1-3, WalkerRagdoll-v0).

NÃO confundir com `openai_walker/play_race.py`, que é a corrida offline da
Fase 4 no `Walker2d-v5` padronizado. Este script compara checkpoints
SAC/PPO/TD3/ARS lado a lado no mesmo XML (`build_race_xml`).
Uso: python play_race.py --checkpoints a.pt b.pt --names A B [--headless]
"""
from envs import normalize_compat  # noqa: F401  pickle shim for checkpoint obs_rms; import before torch.load
import os
import sys
import argparse
import random
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco

# Adjust import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_walker import SACAgent, PPOAgent, TD3Agent, adapt_obs_rms

# Distinct colors for each agent's visual geoms to make them easy to identify
AGENT_COLORS = [
    [0.85, 0.15, 0.15, 1.0],  # 0: Vibrant Red
    [0.15, 0.85, 0.15, 1.0],  # 1: Vibrant Green
    [0.15, 0.15, 0.85, 1.0],  # 2: Vibrant Blue
    [0.85, 0.75, 0.15, 1.0],  # 3: Gold/Yellow
    [0.75, 0.15, 0.85, 1.0],  # 4: Magenta/Purple
    [0.15, 0.85, 0.85, 1.0],  # 5: Cyan/Teal
]


class LegacyPPOAgent(nn.Module):
    """Fallback PPOAgent definition for legacy checkpoints with Tanh sequential layers."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_deterministic_action(self, obs):
        return self.actor_mean(obs)


class ARSAgentWrapper(nn.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = weights
        self.bias = bias
    def get_deterministic_action(self, obs_t):
        obs_np = obs_t.cpu().numpy()[0]
        action = np.tanh(np.dot(self.weights, obs_np) + self.bias)
        return torch.as_tensor(action, dtype=torch.float32, device=obs_t.device).unsqueeze(0)


class DreamerAgentWrapper(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()
        from train_dreamer import WorldModel, DreamerActor
        self.model = WorldModel(obs_dim, action_dim).to(device)
        self.actor = DreamerActor(hidden_dim=256, stochastic_dim=32, action_dim=action_dim).to(device)
        self.device = device
        self.reset_state()
        
    def reset_state(self):
        self.h = torch.zeros(1, 256, device=self.device)
        self.z = torch.zeros(1, 32, device=self.device)
        self.last_action = torch.zeros(1, 17, device=self.device)
        
    def get_deterministic_action(self, obs_t):
        with torch.no_grad():
            self.h, _, _, _ = self.model.rssm.transition(self.h, self.z, self.last_action)
            embed = self.model.encoder(obs_t)
            self.z, _, _ = self.model.rssm.posterior(self.h, embed)
            action = self.actor.get_action(self.h, self.z, sample=False)
            self.last_action = action
        return action



class HardcodedRaceAgent:
    def __init__(self, rec_agent, tgt_agent):
        self.rec_agent = rec_agent
        self.tgt_agent = tgt_agent
        
    def get_action(self, obs_t, deterministic=True):
        # obs_t shape [1, 49]
        z = obs_t[0, 0].item()
        upright = obs_t[0, 45].item()
        
        if z < 1.1 or upright < 0.8:
            return self.rec_agent.get_action(obs_t[:, :46], deterministic=deterministic)
        else:
            return self.tgt_agent.get_action(obs_t, deterministic=deterministic)

class MoERaceAgent:
    def __init__(self, rec_agent, tgt_agent, moe_gate):
        self.rec_agent = rec_agent
        self.tgt_agent = tgt_agent
        self.moe_gate = moe_gate
        
    def get_action(self, obs_t, deterministic=True):
        g = self.moe_gate(obs_t[:, :46]).item()
        a_rec, _, _ = self.rec_agent.get_action(obs_t[:, :46], deterministic=deterministic)
        a_tgt, _, _ = self.tgt_agent.get_action(obs_t, deterministic=deterministic)
        action = g * a_rec + (1.0 - g) * a_tgt
        return action, None, None

def load_agent(ckpt_path, device):
    """Loads agent policy, detects obs_dim and handles legacy architecture formats."""
    print(f"[LOAD] Loading checkpoint from {ckpt_path} ...")
    
    if ckpt_path == "hardcoded":
        rec_agent = SACAgent(46, gym.spaces.Box(-1.0, 1.0, shape=(17,))).to(device)
        rec_agent.load_state_dict(torch.load("checkpoints/walker_recovery_v1/sac_ckpt_20000000.pt", map_location=device, weights_only=False).get("actor_state_dict"))
        rec_agent.eval()

        tgt_ckpt = torch.load("checkpoints/walker_target_v1/sac_ckpt_40000000.pt", map_location=device, weights_only=False)
        tgt_agent = SACAgent(49, gym.spaces.Box(-1.0, 1.0, shape=(17,))).to(device)
        tgt_agent.load_state_dict(tgt_ckpt.get("actor_state_dict"))
        tgt_agent.eval()
        
        agent = HardcodedRaceAgent(rec_agent, tgt_agent)
        return agent, "sac", 49, tgt_ckpt.get("obs_rms"), "target", 0.0

    if ckpt_path == "moe":
        from train_moe_gate import MoEGate
        rec_agent = SACAgent(46, gym.spaces.Box(-1.0, 1.0, shape=(17,))).to(device)
        rec_agent.load_state_dict(torch.load("checkpoints/walker_recovery_v1/sac_ckpt_20000000.pt", map_location=device, weights_only=False).get("actor_state_dict"))
        rec_agent.eval()

        tgt_ckpt = torch.load("checkpoints/walker_target_v1/sac_ckpt_40000000.pt", map_location=device, weights_only=False)
        tgt_agent = SACAgent(49, gym.spaces.Box(-1.0, 1.0, shape=(17,))).to(device)
        tgt_agent.load_state_dict(tgt_ckpt.get("actor_state_dict"))
        tgt_agent.eval()
        
        gate = MoEGate().to(device)
        gate.load_state_dict(torch.load("moe_gate.pt", map_location=device))
        gate.eval()
        
        agent = MoERaceAgent(rec_agent, tgt_agent, gate)
        return agent, "sac", 49, tgt_ckpt.get("obs_rms"), "target", 0.0
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Identify algorithm
    algo = checkpoint.get("algo", "sac")
    
    # Extract actor state dict keys
    actor_state = None
    if "actor_state_dict" in checkpoint:
        actor_state = checkpoint["actor_state_dict"]
    elif "agent_state_dict" in checkpoint:
        actor_state = checkpoint["agent_state_dict"]
    else:
        actor_state = checkpoint

    # Check if this uses legacy PPO sequential structure
    is_legacy_ppo = any(k.startswith("actor_mean.") for k in actor_state.keys())

    # Detect obs_dim dynamically
    obs_dim = None
    for key, weight in actor_state.items():
        if "backbone.0.weight" in key or "actor_mean.0.weight" in key or "critic.0.weight" in key:
            obs_dim = weight.shape[1]
            break
            
    if obs_dim is None:
        obs_dim = 49  # Default target phase dimensions
        print(f"[WARN] Could not detect obs_dim. Defaulting to {obs_dim}.")
    else:
        print(f"[LOAD] Detected obs_dim: {obs_dim}")

    action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,))
    action_dim = 17

    if is_legacy_ppo:
        print("[LOAD] Detected legacy PPO sequential architecture.")
        agent = LegacyPPOAgent(obs_dim, action_dim).to(device)
        agent.load_state_dict(actor_state)
    elif algo in {"sac", "sac_actor"} or algo == "redq":
        agent = SACAgent(obs_dim, action_space).to(device)
        agent.load_state_dict(actor_state)
    elif algo == "ppo":
        agent = PPOAgent(obs_dim, action_dim).to(device)
        agent.load_state_dict(actor_state)
    elif algo == "td3":
        agent = TD3Agent(obs_dim, action_space).to(device)
        agent.load_state_dict(actor_state)
    elif algo == "ars":
        weights = checkpoint["weights"]
        bias = checkpoint["bias"]
        agent = ARSAgentWrapper(weights, bias).to(device)
    elif algo == "dreamer":
        agent = DreamerAgentWrapper(obs_dim, action_dim, device)
        if "model_state_dict" in checkpoint:
            agent.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            agent.model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            agent.model.rssm.load_state_dict(checkpoint["rssm_state_dict"])
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    agent.eval()

    obs_rms = checkpoint.get("obs_rms", None)
    task_phase = checkpoint.get("task_phase", "recovery")
    target_forward_velocity = checkpoint.get("target_forward_velocity", 0.8)

    return agent, algo, obs_dim, obs_rms, task_phase, target_forward_velocity


def build_race_xml(base_xml_path, num_agents, target_x, lane_distance=2.0):
    """
    Parses the base MuJoCo XML and duplicates the walker body, tendons,
    and actuators into parallel lanes with custom visual colors.
    """
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    # Find elements
    worldbody = root.find("worldbody")
    tendon = root.find("tendon")
    actuator = root.find("actuator")

    # Locate and extract the original torso body template
    original_torso = None
    for body in worldbody.findall("body"):
        if body.get("name") == "torso":
            original_torso = body
            worldbody.remove(body)
            break

    if original_torso is None:
        raise ValueError("Could not find body name='torso' in base XML.")

    # Remove the single target marker if present (we will spawn lane-specific targets)
    for body in worldbody.findall("body"):
        if body.get("name") == "target_marker":
            worldbody.remove(body)

    # Save original tendons and actuators templates
    original_tendons = []
    if tendon is not None:
        original_tendons = list(tendon.findall("fixed"))
        for t in original_tendons:
            tendon.remove(t)

    original_actuators = []
    if actuator is not None:
        original_actuators = list(actuator.findall("motor"))
        for act in original_actuators:
            actuator.remove(act)

    # For each agent, duplicate torso, tendons, and actuators
    for i in range(num_agents):
        y_offset = (i - (num_agents - 1) / 2) * lane_distance
        agent_color = AGENT_COLORS[i % len(AGENT_COLORS)]
        color_str = f"{agent_color[0]} {agent_color[1]} {agent_color[2]} {agent_color[3]}"

        # 1. Spawn a target marker for this lane
        target_marker = ET.Element("body", {
            "name": f"agent{i}_target_marker",
            "pos": f"{target_x} {y_offset} 0.05"
        })
        ET.SubElement(target_marker, "geom", {
            "name": f"agent{i}_target_marker_geom",
            "type": "sphere",
            "size": "0.15",
            "rgba": color_str,
            "contype": "0",
            "conaffinity": "0"
        })
        worldbody.append(target_marker)

        # 2. Duplicate torso
        torso_copy = ET.fromstring(ET.tostring(original_torso))
        torso_copy.set("pos", f"0.0 {y_offset} 1.4")

        # Recursively rename joint, geom, site, camera elements inside the torso body
        def rename_recursive(node):
            name = node.get("name")
            if name:
                node.set("name", f"agent{i}_{name}")
            
            # Apply walker visual colors to geoms to make competitors distinct
            if node.tag == "geom" and "floor" not in node.get("name", ""):
                # Override rgba
                node.set("rgba", color_str)
                # Remove material reference so color override works reliably
                if "material" in node.attrib:
                    del node.attrib["material"]

            # Rename joints reference inside free or hinge joint tags
            if node.tag == "joint" and name == "root":
                node.set("name", f"agent{i}_root")

            for child in node:
                rename_recursive(child)

        rename_recursive(torso_copy)
        worldbody.append(torso_copy)

        # 3. Duplicate fixed tendons
        for tend in original_tendons:
            tend_copy = ET.fromstring(ET.tostring(tend))
            tend_copy.set("name", f"agent{i}_{tend_copy.get('name')}")
            for joint_el in tend_copy.findall("joint"):
                joint_name = joint_el.get("joint")
                joint_el.set("joint", f"agent{i}_{joint_name}")
            tendon.append(tend_copy)

        # 4. Duplicate motor actuators
        for motor in original_actuators:
            motor_copy = ET.fromstring(ET.tostring(motor))
            motor_copy.set("name", f"agent{i}_{motor_copy.get('name')}")
            joint_name = motor_copy.get("joint")
            motor_copy.set("joint", f"agent{i}_{joint_name}")
            actuator.append(motor_copy)

    temp_xml_path = os.path.join(os.path.dirname(base_xml_path), "temp_race_env.xml")
    tree.write(temp_xml_path)
    return temp_xml_path


def get_agent_observation(model, data, agent_idx, target_x, target_y_initial, obs_dim):
    """
    Extracts the individual observation vector for agent_idx from global MuJoCo simulation data.
    """
    prefix = f"agent{agent_idx}_"
    root_name = f"{prefix}root"
    
    # 1. Joint address pointers
    root_id = model.joint(root_name).id
    root_qposadr = model.jnt_qposadr[root_id]
    root_dofadr = model.jnt_dofadr[root_id]
    
    # Hinge joints in canonical order
    hinge_names = [
        "abdomen_z", "abdomen_y", "abdomen_x",
        "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
        "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
        "right_shoulder1", "right_shoulder2", "right_elbow",
        "left_shoulder1", "left_shoulder2", "left_elbow"
    ]
    
    # 2. Extract position elements
    # root height + root quaternion
    z = data.qpos[root_qposadr + 2]
    quat = data.qpos[root_qposadr + 3: root_qposadr + 7]
    
    hinges_pos = []
    for hname in hinge_names:
        h_id = model.joint(f"{prefix}{hname}").id
        h_adr = model.jnt_qposadr[h_id]
        hinges_pos.append(data.qpos[h_adr])
        
    position = np.array([z] + list(quat) + hinges_pos, dtype=np.float64)
    
    # 3. Extract velocity elements
    # 6-DOF root velocity
    free_vel = data.qvel[root_dofadr: root_dofadr + 6]
    
    hinges_vel = []
    for hname in hinge_names:
        h_id = model.joint(f"{prefix}{hname}").id
        h_dof = model.jnt_dofadr[h_id]
        hinges_vel.append(data.qvel[h_dof])
        
    velocity = np.array(list(free_vel) + hinges_vel, dtype=np.float64)
    
    # 4. Extract Torso Upright Factor
    torso_body_id = model.body(f"{prefix}torso").id
    xmat = data.xmat[torso_body_id]  # 9-element rotation matrix
    upright = np.array([xmat[8]], dtype=np.float64)
    
    # 5. Combine and append target context if model expects it
    if obs_dim >= 49:
        agent_xy = data.qpos[root_qposadr: root_qposadr + 2]
        target_xy = np.array([target_x, target_y_initial])
        rel_xy = target_xy - agent_xy
        distance = np.linalg.norm(rel_xy)
        # Scale/clip target observation to stay within training distribution (2.0 to 5.0 meters)
        max_dist = 5.0
        if distance > max_dist:
            rel_xy = rel_xy * (max_dist / distance)
            distance = max_dist
        target_obs = np.array([rel_xy[0], rel_xy[1], distance], dtype=np.float64)
        return np.concatenate([position, velocity, upright, target_obs])
        
    return np.concatenate([position, velocity, upright])


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent AI Race in MuJoCo (SAC vs PPO vs TD3)")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint file paths or directories")
    parser.add_argument("--names", nargs="+", default=[], help="Display names corresponding to checkpoints")
    parser.add_argument("--target-x", type=float, default=15.0, help="Target X distance for the race finish line")
    parser.add_argument("--headless", action="store_true", help="Run in console headless mode without open window")
    parser.add_argument("--max-steps", type=int, default=2500, help="Maximum simulation steps")
    parser.add_argument("--lane-distance", type=float, default=1.8, help="Lateral space between lanes")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Compute device (auto = cuda if available, else cpu)")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RACE] Running race evaluation on device: {device}")

    # Resolve checkpoints and load agents
    agents = []
    agent_names = []
    
    for idx, path in enumerate(args.checkpoints):
        # Resolve folders to latest checkpoints
        if os.path.isdir(path):
            candidates = []
            for prefix in ("sac_ckpt_", "ppo_ckpt_", "td3_ckpt_"):
                for name in os.listdir(path):
                    if name.startswith(prefix) and name.endswith(".pt"):
                        try:
                            step = int(name[len(prefix):-3])
                            candidates.append((step, os.path.join(path, name)))
                        except ValueError:
                            pass
            if candidates:
                resolved_path = max(candidates)[1]
            else:
                raise FileNotFoundError(f"No checkpoint found in directory: {path}")
        else:
            resolved_path = path

        agent, algo, obs_dim, obs_rms, phase, tf_vel = load_agent(resolved_path, device)
        agents.append({
            "model": agent,
            "algo": algo,
            "obs_dim": obs_dim,
            "obs_rms": obs_rms,
            "phase": phase,
            "target_forward_velocity": tf_vel
        })
        
        # Determine agent display name
        if idx < len(args.names):
            name = args.names[idx]
        else:
            name = f"Agent {idx} ({algo.upper()})"
        agent_names.append(name)

    num_agents = len(agents)
    print(f"[RACE] Total competitors: {num_agents}")
    for idx, name in enumerate(agent_names):
        print(f"  - Competitor {idx}: {name} [{agents[idx]['algo'].upper()}]")

    # Auto-reset: no recovery neural network needed - we teleport fallen agents back upright
    print("[RACE] Auto-reset enabled: fallen agents will be teleported upright at current position.")

    # 1. Build and load cloned XML
    base_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walker_ragdoll.xml")
    race_xml_path = build_race_xml(base_xml, num_agents, args.target_x, args.lane_distance)
    print(f"[RACE] Built race XML: {race_xml_path}")

    # Initialize MuJoCo simulator
    model = mujoco.MjModel.from_xml_path(race_xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Competitor status tracking
    agent_finished = [False] * num_agents
    finish_times = [None] * num_agents
    max_distances = [0.0] * num_agents
    respawn_counts = [0] * num_agents      # how many times each agent auto-reset
    respawn_cooldown = [0] * num_agents    # steps to wait before detecting another fall
    RESPAWN_COOLDOWN_STEPS = 30            # ~0.3s grace period after respawn
    lane_y_coords = []

    # Record initial lane Y coords
    for i in range(num_agents):
        y_val = (i - (num_agents - 1) / 2) * args.lane_distance
        lane_y_coords.append(y_val)

    # Cache initial qpos offsets per joint type (used for upright respawn)
    # We will read the model's default qpos as the "standing pose" template
    standing_qpos_template = data.qpos.copy()

    # Setup rendering
    viewer = None
    if not args.headless:
        # Passive viewer window
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        print("[RACE] Passive viewer window launched.")

    step_idx = 0
    start_wall_time = time.time()
    
    # Reset any recurrent agents at the beginning of the race
    for agent_info in agents:
        if hasattr(agent_info["model"], "reset_state"):
            agent_info["model"].reset_state()
            
    try:
        while step_idx < args.max_steps:
            step_start = time.time()
            
            # Predict action and apply control for each agent
            for i in range(num_agents):
                agent_info = agents[i]
                
                # Extract observations
                obs = get_agent_observation(
                    model, data, i, args.target_x, lane_y_coords[i], agent_info["obs_dim"]
                )
                
                # Update max distance
                root_id = model.joint(f"agent{i}_root").id
                root_qposadr = model.jnt_qposadr[root_id]
                x_pos = data.qpos[root_qposadr]
                max_distances[i] = max(max_distances[i], x_pos)
                
                # Check finish line
                if x_pos >= args.target_x and not agent_finished[i]:
                    agent_finished[i] = True
                    finish_times[i] = step_idx * model.opt.timestep * 5  # multiply by frameskip (5)
                    print(f"\n[FINISH] {agent_names[i]} crossed the finish line at step {step_idx} (sim time: {finish_times[i]:.2f}s)!")

                # --- AUTO-RESET: teleport fallen agents back upright ---
                z_raw = obs[0]
                upright_raw = obs[45] if len(obs) >= 46 else 1.0
                has_fallen = False # Disabled so agents can use their recovery networks naturally

                if has_fallen and respawn_cooldown[i] == 0 and not agent_finished[i]:
                    # Teleport agent to standing pose at current X, lane Y
                    root_id = model.joint(f"agent{i}_root").id
                    root_qposadr = model.jnt_qposadr[root_id]
                    n_qpos = 22  # humanoid root(7) + joints(15) per agent
                    agent_qpos_slice = slice(root_qposadr, root_qposadr + n_qpos)
                    agent_qvel_start = model.jnt_dofadr[root_id]
                    n_qvel = 21  # 6 root DoF + 15 joint DoF
                    agent_qvel_slice = slice(agent_qvel_start, agent_qvel_start + n_qvel)

                    # Teleport agent to standing pose at the starting line (X = 0.0)
                    data.qpos[agent_qpos_slice] = standing_qpos_template[agent_qpos_slice]
                    data.qpos[root_qposadr]     = 0.0                 # reset X progress to 0
                    data.qpos[root_qposadr + 1] = lane_y_coords[i]   # restore lane Y
                    data.qpos[root_qposadr + 2] = 1.35               # upright Z height
                    # identity quaternion = [1, 0, 0, 0]
                    data.qpos[root_qposadr + 3] = 1.0
                    data.qpos[root_qposadr + 4] = 0.0
                    data.qpos[root_qposadr + 5] = 0.0
                    data.qpos[root_qposadr + 6] = 0.0
                    # Zero out all velocities
                    data.qvel[agent_qvel_slice] = 0.0
                    mujoco.mj_forward(model, data)

                    respawn_counts[i] += 1
                    respawn_cooldown[i] = RESPAWN_COOLDOWN_STEPS
                    print(f"\n[RESPAWN] {agent_names[i]} respawned at starting line (respawn #{respawn_counts[i]})")
                    if hasattr(agent_info["model"], "reset_state"):
                        agent_info["model"].reset_state()

                if respawn_cooldown[i] > 0:
                    respawn_cooldown[i] -= 1

                # --- COMPUTE ACTION (always from target policy) ---
                if agent_info["obs_rms"] is not None:
                    mean = agent_info["obs_rms"].mean
                    var = agent_info["obs_rms"].var
                    obs_norm = (obs - mean) / np.sqrt(var + 1e-8)
                    obs_norm = np.clip(obs_norm, -10.0, 10.0)
                else:
                    obs_norm = obs

                obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    if agent_info["algo"] in ["sac", "sac_actor", "redq"]:
                        action_t, _, _ = agent_info["model"].get_action(obs_t, deterministic=True)
                        action = action_t.cpu().numpy()[0]
                    elif agent_info["algo"] in ["ppo", "ars", "dreamer"]:
                        action = agent_info["model"].get_deterministic_action(obs_t).cpu().numpy()[0]
                    elif agent_info["algo"] == "td3":
                        action = agent_info["model"](obs_t).cpu().numpy()[0]

                # Map actions to actuator indices
                actuator_names = [
                    "abdomen_y", "abdomen_z", "abdomen_x",
                    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
                    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
                    "right_shoulder1", "right_shoulder2", "right_elbow",
                    "left_shoulder1", "left_shoulder2", "left_elbow"
                ]
                for act_idx, act_name in enumerate(actuator_names):
                    a_id = model.actuator(f"agent{i}_{act_name}").id
                    data.ctrl[a_id] = action[act_idx]

            # Step simulation by frame_skip (5)
            for _ in range(5):
                mujoco.mj_step(model, data)

            # Camera tracks the leading walker
            if viewer is not None:
                leading_agent = np.argmax(max_distances)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = model.body(f"agent{leading_agent}_torso").id
                viewer.sync()

            # Periodic console race standing updates
            if step_idx % 200 == 0:
                sys.stdout.write(f"\rStep {step_idx:04d}/{args.max_steps:04d} -> " + " | ".join(
                    f"{agent_names[i]}: {data.qpos[model.jnt_qposadr[model.joint(f'agent{i}_root').id]]:.2f}m" for i in range(num_agents)
                ))
                sys.stdout.flush()

            # End race early if all finished
            if all(agent_finished):
                break

            step_idx += 1
            
            # Match real-world wall clock time to look natural in interactive GUI
            if viewer is not None:
                elapsed = time.time() - step_start
                sim_dt = model.opt.timestep * 5
                if elapsed < sim_dt:
                    time.sleep(sim_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[RACE] Race simulation aborted by user.")
    finally:
        if viewer is not None:
            viewer.close()

    # Clean up temp environment XML file
    if os.path.exists(race_xml_path):
        try:
            os.remove(race_xml_path)
        except OSError:
            pass

    # Print Final Standings Table
    print("\n\n" + "=" * 70)
    print("                        FINAL RACE RESULTS")
    print("=" * 70)
    print(f"{'Pos':<5}{'Agent':<28}{'Finish Time':<14}{'Max Distance':<16}{'Respawns':<10}")
    print("-" * 70)

    standings = []
    for i in range(num_agents):
        time_score = finish_times[i] if finish_times[i] is not None else float("inf")
        standings.append((time_score, -max_distances[i], i))
    standings.sort()

    for rank, (time_val, neg_dist, idx) in enumerate(standings):
        name = agent_names[idx]
        t_str = f"{time_val:.3f}s" if time_val != float("inf") else "DNF"
        dist_str = f"{-neg_dist:.2f}m"
        respawn_str = str(respawn_counts[idx])
        print(f"{rank + 1:<5}{name:<28}{t_str:<14}{dist_str:<16}{respawn_str:<10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
