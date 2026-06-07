import mujoco
import mujoco.viewer
import time
import torch
import numpy as np
import pickle
import os
import sys
import xml.etree.ElementTree as ET
from collections import deque

from train import WalkerTeacherNet
from evaluate_all import PolicyNet

def build_race_xml(base_xml, num_agents, lane_distance):
    tree = ET.parse(base_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    tendon = root.find("tendon")
    
    walker_body = None
    for body in worldbody.findall("body"):
        if body.get("name") == "torso":
            walker_body = body
            worldbody.remove(body)
            break
            
    mocap_body = None
    for body in worldbody.findall("body"):
        if body.get("name") == "target_marker":
            mocap_body = body
            worldbody.remove(body)
            break
            
    base_actuators = list(actuator)
    for act in base_actuators:
        actuator.remove(act)
        
    original_tendons = []
    if tendon is not None:
        original_tendons = list(tendon.findall("fixed"))
        for t in original_tendons:
            tendon.remove(t)
        
    AGENT_COLORS = [
        [0.85, 0.15, 0.15, 1.0], [0.15, 0.85, 0.15, 1.0], [0.15, 0.15, 0.85, 1.0],
        [0.85, 0.75, 0.15, 1.0], [0.75, 0.15, 0.85, 1.0], [0.15, 0.85, 0.85, 1.0]
    ]
    
    for i in range(num_agents):
        y_offset = (i - (num_agents - 1) / 2) * lane_distance
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        color_str = f"{color[0]} {color[1]} {color[2]} {color[3]}"
        
        target_marker = ET.Element("body", {"name": f"agent{i}_target_marker", "mocap": "true", "pos": f"5.0 {y_offset} 0.05"})
        ET.SubElement(target_marker, "geom", {"name": f"agent{i}_target_marker_geom", "type": "sphere", "size": "0.1", "rgba": f"{color[0]} {color[1]} {color[2]} 0.5", "contype": "0", "conaffinity": "0"})
        worldbody.append(target_marker)
        
        clone = ET.fromstring(ET.tostring(walker_body))
        clone.set("name", f"agent{i}_torso")
        pos = clone.get("pos", "0 0 1.3").split()
        clone.set("pos", f"{pos[0]} {float(pos[1]) + y_offset} {pos[2]}")
        
        for elem in clone.iter():
            name = elem.get("name")
            if name:
                elem.set("name", f"agent{i}_{name}")
            if elem.tag == "geom" and elem.get("rgba") is None:
                elem.set("rgba", color_str)
                
        worldbody.append(clone)
        
        for tend in original_tendons:
            tend_copy = ET.fromstring(ET.tostring(tend))
            tend_copy.set("name", f"agent{i}_{tend_copy.get('name')}")
            for joint_el in tend_copy.findall("joint"):
                joint_name = joint_el.get("joint")
                joint_el.set("joint", f"agent{i}_{joint_name}")
            tendon.append(tend_copy)
        
        for act in base_actuators:
            act_clone = ET.fromstring(ET.tostring(act))
            act_clone.set("name", f"agent{i}_{act.get('name')}")
            act_clone.set("joint", f"agent{i}_{act.get('joint')}")
            actuator.append(act_clone)
            
    temp_path = "temp_race_offline.xml"
    tree.write(temp_path)
    return temp_path

def get_agent_observation(model, data, agent_idx, mocap_id):
    prefix = f"agent{agent_idx}_"
    root_id = model.joint(f"{prefix}root").id
    qposadr = model.jnt_qposadr[root_id]
    dofadr = model.jnt_dofadr[root_id]
    
    tx = data.mocap_pos[mocap_id, 0]
    ty = data.mocap_pos[mocap_id, 1]
    
    qpos_slice = data.qpos[qposadr: qposadr + 24]
    qvel_slice = data.qvel[dofadr: dofadr + 23]
    
    rel_tx_global = tx - qpos_slice[0]
    rel_ty_global = ty - qpos_slice[1]
    
    qw, qx, qy, qz = qpos_slice[3], qpos_slice[4], qpos_slice[5], qpos_slice[6]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    
    rel_tx = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
    rel_ty = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
    
    return np.concatenate(([rel_tx, rel_ty], qpos_slice[2:], qvel_slice))

def apply_agent_action(model, data, agent_idx, action):
    # Apply action to correct actuators
    start_idx = agent_idx * 17
    data.ctrl[start_idx: start_idx + 17] = action

def main():
    device = torch.device("cpu")
    
    agents = []
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. BC
    with open("scaler.pkl", "rb") as f: scaler_bc = pickle.load(f)
    bc_model = WalkerTeacherNet(188, 17)
    bc_model.load_state_dict(torch.load("teacher_model.pt", weights_only=True, map_location=device))
    agents.append({"name": "BC", "model": bc_model, "scaler": scaler_bc})
    
    # 2. IQL
    with open("scaler_iql.pkl", "rb") as f: scaler_iql = pickle.load(f)
    iql_model = PolicyNet(188, 17)
    iql_model.load_state_dict(torch.load("iql_model.pt", weights_only=True, map_location=device))
    agents.append({"name": "IQL", "model": iql_model, "scaler": scaler_iql})
    
    # 3. CQL
    with open("scaler_cql.pkl", "rb") as f: scaler_cql = pickle.load(f)
    cql_model = PolicyNet(188, 17)
    cql_model.load_state_dict(torch.load("cql_model.pt", weights_only=True, map_location=device))
    agents.append({"name": "CQL", "model": cql_model, "scaler": scaler_cql})
    
    # 4. BC+SAC
    bc_sac_model = PolicyNet(188, 17)
    bc_sac_model.load_state_dict(torch.load("bc_sac_model.pt", weights_only=True, map_location=device))
    agents.append({"name": "BC+SAC", "model": bc_sac_model, "scaler": scaler_bc})
    
    # 5. IQL+SAC
    iql_sac_model = PolicyNet(188, 17)
    iql_sac_model.load_state_dict(torch.load("iql_sac_model.pt", weights_only=True, map_location=device))
    agents.append({"name": "IQL+SAC", "model": iql_sac_model, "scaler": scaler_bc})
    
    # 6. CQL+SAC
    cql_sac_model = PolicyNet(188, 17)
    cql_sac_model.load_state_dict(torch.load("cql_sac_model.pt", weights_only=True, map_location=device))
    agents.append({"name": "CQL+SAC", "model": cql_sac_model, "scaler": scaler_bc})
    
    num_agents = len(agents)
    for a in agents:
        a["model"].eval()
        
    xml_path = build_race_xml("walker_ragdoll.xml", num_agents, 1.8)
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    mocap_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}_target_marker") for i in range(num_agents)]
    mocap_idxs = [m.body_mocapid[mid] for mid in mocap_ids]
    
    history_len = 3
    state_histories = [deque(maxlen=history_len + 1) for _ in range(num_agents)]
    last_actions = [np.zeros(17) for _ in range(num_agents)]
    
    # Initial targets
    for i in range(num_agents):
        d.mocap_pos[mocap_idxs[i], 0] = 5.0
        
    mujoco.mj_forward(m, d)
    
    print("Launching Race with 6 Offline Models!")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        step = 0
        while viewer.is_running():
            steps_to_run = int((time.time() - start_time - d.time) / m.opt.timestep)
            if steps_to_run > 10:
                steps_to_run = 10
                start_time = time.time() - d.time
                
            for _ in range(steps_to_run):
                for i in range(num_agents):
                    x_np = get_agent_observation(m, d, i, mocap_idxs[i])
                    state_histories[i].append(x_np)
                    while len(state_histories[i]) < history_len + 1:
                        state_histories[i].append(x_np)
                        
                    stacked = np.concatenate(list(reversed(state_histories[i])))
                    
                    scaler = agents[i]["scaler"]
                    scaled = (stacked - scaler.mean_) / scaler.scale_
                    x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        if hasattr(agents[i]["model"], "get_action"):
                            ctrl = agents[i]["model"].get_action(x_tensor, deterministic=True).numpy()[0]
                        else:
                            ctrl = agents[i]["model"](x_tensor).squeeze(0).numpy()
                            # Do not inject noise if we want purely deterministic comparison
                            
                        ctrl = np.clip(ctrl, -1.0, 1.0)
                        
                    apply_agent_action(m, d, i, ctrl)
                    
                    # Update target if reached
                    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}_torso")
                    tx, ty = d.xpos[torso_id, 0], d.xpos[torso_id, 1]
                    dist = np.sqrt((tx - d.mocap_pos[mocap_idxs[i], 0])**2 + (ty - d.mocap_pos[mocap_idxs[i], 1])**2)
                    if dist < 0.5:
                        d.mocap_pos[mocap_idxs[i], 0] = tx + np.random.uniform(2.0, 4.0)
                        d.mocap_pos[mocap_idxs[i], 1] = ty + np.random.uniform(-1.0, 1.0)
                        
                mujoco.mj_step(m, d)
                step += 1
                
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
