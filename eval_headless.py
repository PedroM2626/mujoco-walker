import mujoco
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import os
from mujoco_mpc_walker.train import WalkerTeacherNet
from collections import deque

# Load model (47 inputs)
net = WalkerTeacherNet(47, 17)
net.load_state_dict(torch.load('mujoco_mpc_walker/teacher_model.pt', weights_only=True))
net.eval()

# Load scaler
with open('mujoco_mpc_walker/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
scaler_mean = scaler.mean_
scaler_scale = scaler.scale_

# Load mujoco
m = mujoco.MjModel.from_xml_path('walker_ragdoll.xml')
d = mujoco.MjData(m)

target_mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
mocapid = m.body_mocapid[target_mocap_id]
torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

# Initial pos
d.mocap_pos[mocapid, 0] = 2.0
d.mocap_pos[mocapid, 1] = 0.0

mujoco.mj_forward(m, d)

torso_x_hist = []
torso_y_hist = []
target_x_hist = []
target_y_hist = []
z_hist = []

steps = 1500

for i in range(steps):
    tx = d.mocap_pos[mocapid, 0]
    ty = d.mocap_pos[mocapid, 1]
    
    torso_x = d.xpos[torso_id, 0]
    torso_y = d.xpos[torso_id, 1]
    torso_z = d.xpos[torso_id, 2]
    
    torso_x_hist.append(torso_x)
    torso_y_hist.append(torso_y)
    target_x_hist.append(tx)
    target_y_hist.append(ty)
    z_hist.append(torso_z)
    
    dist = np.sqrt((torso_x - tx)**2 + (torso_y - ty)**2)
    if dist < 0.5:
        d.mocap_pos[mocapid, 0] = torso_x + np.random.uniform(-2.5, 2.5)
        d.mocap_pos[mocapid, 1] = torso_y + np.random.uniform(-2.5, 2.5)
        
    qpos = d.qpos.copy()
    qvel = d.qvel.copy()
    
    rel_tx_global = tx - qpos[0]
    rel_ty_global = ty - qpos[1]
    
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    
    rel_tx = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
    rel_ty = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
    
    x_np = np.concatenate(([rel_tx, rel_ty], qpos[2:], qvel))
    
    # Normalize single frame
    x_np_scaled = (x_np - scaler_mean) / scaler_scale
    state_tensor = torch.tensor(x_np_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action = net(state_tensor).numpy()[0]
        # Injetar pequeno ruído Gaussiano para quebrar o congelamento multimodal
        action += np.random.normal(0, 0.15, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
    
    d.ctrl[:] = action
    mujoco.mj_step(m, d)

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(torso_x_hist, torso_y_hist, label='Robot Trajectory', color='blue')
plt.scatter(target_x_hist, target_y_hist, label='Target Positions', color='red', marker='x')
plt.title('Top-Down View of Movement')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(z_hist, label='Torso Height (Z)', color='green')
plt.axhline(y=0.5, color='r', linestyle='--', label='Fall Threshold (0.5m)')
plt.title('Robot Balance over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
out_path = 'trajectory.png'
plt.savefig(out_path)
print("Saved visualization to relative path:", out_path)

# Try to also save a duplicate to the brain folder for artifact visualization
brain_dir = 'C:/Users/pedro/.gemini/antigravity/brain/f1f4b7f3-9795-462b-9db0-b699c9cd1e89'
if os.path.exists(brain_dir):
    try:
        plt.savefig(os.path.join(brain_dir, 'trajectory.png'))
        print("Also saved visualization to brain dir:", os.path.join(brain_dir, 'trajectory.png'))
    except Exception as e:
        print("Could not save to brain dir:", e)
