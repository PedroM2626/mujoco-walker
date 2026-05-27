import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from mujoco_mpc_walker.train import WalkerTeacherNet

df = pd.read_csv('d:/mujoco-walker/mujoco_mpc_walker/build/dataset.csv')
qpos_cols = [c for c in df.columns if c.startswith('qpos_') and c not in ('qpos_0', 'qpos_1')]
qvel_cols = [c for c in df.columns if c.startswith('qvel_')]
ctrl_cols = [c for c in df.columns if c.startswith('ctrl_')]

df['rel_tx'] = df['target_x'] - df['qpos_0']
df['rel_ty'] = df['target_y'] - df['qpos_1']
X_cols = ['rel_tx', 'rel_ty'] + qpos_cols + qvel_cols
X_np_full = df[X_cols].values

scaler1 = StandardScaler()
X_np_scaled = scaler1.fit_transform(X_np_full)

with open('d:/mujoco-walker/mujoco_mpc_walker/scaler.pkl', 'rb') as f:
    scaler2 = pickle.load(f)

print("Scaler mean match:", np.allclose(scaler1.mean_, scaler2.mean_))
print("Scaler scale match:", np.allclose(scaler1.scale_, scaler2.scale_))

# Also evaluate the model on the full training set!
net = WalkerTeacherNet(47, 17)
net.load_state_dict(torch.load('d:/mujoco-walker/mujoco_mpc_walker/teacher_model.pt', weights_only=True))
net.eval()

with torch.no_grad():
    pred_all = net(torch.tensor(X_np_scaled, dtype=torch.float32)).numpy()
    actual_all = df[ctrl_cols].values
    mse_all = np.mean((pred_all - actual_all)**2)
    print("MSE on full dataset:", mse_all)

    pred_0 = pred_all[0]
    actual_0 = actual_all[0]
    print("Pred 0:", pred_0[:5])
    print("Actual 0:", actual_0[:5])
