import mujoco
import mujoco.viewer
import time
import torch
import numpy as np
import pickle

# Import the network architecture
from train import WalkerTeacherNet
from collections import deque

def main():
    model_path = "teacher_model.pt"
    xml_path = "../walker_ragdoll.xml"  # Original physics model
    
    print("Loading MuJoCo Model...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # Identify mocap ID
    target_mocap_name = "target_marker"
    target_mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_mocap_name)
    mocapid = m.body_mocapid[target_mocap_id]
    
    # We need to match the input dimension used during training
    # (2 for rel_tx/ty) + (m.nq - 2 for dropped qpos_0 and qpos_1) + (m.nv for velocities)
    input_dim = 2 + (m.nq - 2) + m.nv
    output_dim = m.nu
    
    print("Loading Trained PyTorch Model...")
    # Carregar modelo (agora com 47 * 4 = 188 inputs devido ao Frame Stacking)
    net = WalkerTeacherNet(188, 17)
    try:
        net.load_state_dict(torch.load('teacher_model.pt', weights_only=True))
        net.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    print("Loading Normalization Scaler...")
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_
    except FileNotFoundError:
        print("Error: scaler.pkl not found! Please run train.py first.")
        return

    # Mover a bolinha um pouco para frente inicialmente
    d.mocap_pos[mocapid, 0] = 2.0
    d.mocap_pos[mocapid, 1] = 0.0

    # Estado para EMA e Histórico
    last_action = np.zeros(m.nu)
    history_len = 3
    state_history = deque(maxlen=history_len + 1)

    print("Starting Interactive Viewer...")
    print("Instruções:")
    print(" - Dê um duplo-clique na bolinha vermelha para selecioná-la.")
    print(" - Segure CTRL e arraste com o Botão Direito para movê-la.")
    print(" - A Rede Neural comandará o Walker para tentar segui-la!")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            # Quantos passos precisamos dar para alcançar o tempo real?
            steps_to_run = int((time.time() - start_time - d.time) / m.opt.timestep)
            
            # Trava de Segurança (Evita a Espiral da Morte travando a tela)
            if steps_to_run > 10:
                steps_to_run = 10
                start_time = time.time() - d.time # Atrasa o relógio real para acompanhar a física
                
            for _ in range(steps_to_run):
                # Extract current state
                tx = d.mocap_pos[mocapid, 0]
                ty = d.mocap_pos[mocapid, 1]
                qpos = d.qpos.copy()
                qvel = d.qvel.copy()
                
                # Translation and Rotation Invariance (First-Person View)
                rel_tx_global = tx - qpos[0]
                rel_ty_global = ty - qpos[1]
                
                qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                rel_tx = rel_tx_global * np.cos(yaw) + rel_ty_global * np.sin(yaw)
                rel_ty = -rel_tx_global * np.sin(yaw) + rel_ty_global * np.cos(yaw)
                
                # Normalização manual nativa ultrarrápida (Ignora a lentidão do Scikit-Learn)
                x_np = np.concatenate(([rel_tx, rel_ty], qpos[2:], qvel))
                state_history.append(x_np)
                # Preenchemos com cópias se for o começo
                while len(state_history) < history_len + 1:
                    state_history.append(x_np)
                
                # Reverte para ficar igual ao treino: [t, t-1, t-2, t-3]
                stacked_state = np.concatenate(list(reversed(state_history)))
                # Normalize stacked frame (188 dims)
                stacked_state_scaled = (stacked_state - scaler_mean) / scaler_scale
                x_tensor = torch.tensor(stacked_state_scaled, dtype=torch.float32).unsqueeze(0)
                
                # Predict control with Neural Network
                with torch.no_grad():
                    ctrl_pred = net(x_tensor).squeeze(0).numpy()
                    # Injetar pequeno ruído Gaussiano para quebrar o congelamento multimodal
                    ctrl_pred += np.random.normal(0, 0.15, size=ctrl_pred.shape)
                    
                    # Filtro EMA para não dar socos nos motores e evitar OOD states
                    ctrl_pred = 0.8 * last_action + 0.2 * ctrl_pred
                    ctrl_pred = np.clip(ctrl_pred, -1.0, 1.0)
                    last_action = ctrl_pred.copy()
                
                # Apply control
                d.ctrl[:] = ctrl_pred
                
                # Passo da física
                mujoco.mj_step(m, d)
                
                # Infinite Target Spawning Logic
                torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
                torso_x = d.xpos[torso_id, 0]
                torso_y = d.xpos[torso_id, 1]
                
                dist = np.sqrt((torso_x - tx)**2 + (torso_y - ty)**2)
                
                if dist < 0.5:
                    # Spawn new target!
                    d.mocap_pos[mocapid, 0] = torso_x + np.random.uniform(-2.5, 2.5)
                    d.mocap_pos[mocapid, 1] = torso_y + np.random.uniform(-2.5, 2.5)
                    
            # Sincroniza a tela consistentemente
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
