import mujoco
import mujoco.viewer
import time
import torch
import numpy as np

# Import the network architecture
from train import WalkerTeacherNet

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
    input_dim = 2 + m.nq + m.nv
    output_dim = m.nu
    
    print("Loading Trained PyTorch Model...")
    net = WalkerTeacherNet(input_dim=input_dim, output_dim=output_dim)
    try:
        net.load_state_dict(torch.load(model_path))
        net.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    # Mover a bolinha um pouco para frente inicialmente
    d.mocap_pos[3 * mocapid + 0] = 2.0
    d.mocap_pos[3 * mocapid + 1] = 0.0

    print("Starting Interactive Viewer...")
    print("Instruções:")
    print(" - Segure SHIFT e arraste a bolinha vermelha com o Botão Direito para movê-la.")
    print(" - A Rede Neural comandará o Walker para tentar segui-la!")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Extract current state
            tx = d.mocap_pos[3 * mocapid + 0]
            ty = d.mocap_pos[3 * mocapid + 1]
            qpos = d.qpos.copy()
            qvel = d.qvel.copy()
            
            # Create input vector
            x_np = np.concatenate(([tx, ty], qpos, qvel))
            x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
            
            # Predict control with Neural Network (No MPC overhead!)
            with torch.no_grad():
                ctrl_pred = net(x_tensor).squeeze(0).numpy()
            
            # Apply control
            d.ctrl[:] = ctrl_pred
            
            # Physics step
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # Time synchronization to run near real-time
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
