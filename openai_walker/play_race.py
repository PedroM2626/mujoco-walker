"""Grande corrida offline da Fase 4 (Walker2d-v5, com GUI).

NÃO confundir com `play_race_ragdoll.py` da raiz, que é a corrida multi-agente do
ragdoll customizado (Fases 1-3). Para servidores sem display, use a
contraparte headless `evaluate_all.py --episodes N`.
"""
import gymnasium as gym
import torch
import torch.nn as nn
import time
import os
import numpy as np
from stable_baselines3 import SAC
from train_irl_airl import SACActor as AIRLActor
from train_offline_bcq import VAE, PerturbationNetwork
from train_offline_dt import DecisionTransformer
from train_irl_maxent import SACActor as MaxEntActor
from train_irl_pqr import PolicyNet as PQRPolicy

# ==========================================================
# Architectures Definitions
# ==========================================================
class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SACActor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, output_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, output_dim))
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        return torch.tanh(mean) * self.max_action

class GAILActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        return torch.tanh(mean) * self.max_action

class PolicyNetIQL(nn.Module):
    def __init__(self, input_dim, output_dim, max_action=1.0):
        super(PolicyNetIQL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )
        self.mean_layer = nn.Linear(512, output_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, output_dim))
        self.max_action = max_action

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_layer(features)
        return torch.tanh(mean) * self.max_action

# ==========================================================
# Evaluation Wrapper
# ==========================================================
def evaluate_model(env, name, model, device, episodes=1, is_gail=False, is_airl=False, is_dt=False, is_maxent=False, dt_context=20):
    print(f"\n[{name}] Preparando para a corrida...")
    time.sleep(2)
    
    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        # Specific context for Decision Transformer
        if is_dt:
            state_seq = torch.zeros((1, dt_context, env.observation_space.shape[0]), dtype=torch.float32, device=device)
            action_seq = torch.zeros((1, dt_context, env.action_space.shape[0]), dtype=torch.float32, device=device)
            rtg_seq = torch.zeros((1, dt_context, 1), dtype=torch.float32, device=device)
            
            # Start with a high target return
            rtg = 4000.0
            
            step = 0
            
        while not done:
            with torch.no_grad():
                if is_dt:
                    # Shift sequences and append current obs
                    state_seq[0, :-1] = state_seq[0, 1:].clone()
                    state_seq[0, -1] = torch.FloatTensor(obs).to(device)
                    
                    rtg_seq[0, :-1] = rtg_seq[0, 1:].clone()
                    rtg_seq[0, -1] = torch.tensor([rtg], dtype=torch.float32).to(device)
                    
                    timesteps = torch.arange(0, dt_context, device=device).unsqueeze(0)
                    
                    action_preds = model(state_seq, action_seq, rtg_seq, timesteps)
                    action = action_preds[0, -1].cpu().data.numpy().flatten()
                    
                    action_seq[0, :-1] = action_seq[0, 1:].clone()
                    action_seq[0, -1] = torch.FloatTensor(action).to(device)
                    
                else:
                    state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    if is_airl or is_maxent:
                        mean, _ = model(state_t)
                        action = (torch.tanh(mean) * model.max_action).cpu().data.numpy().flatten()
                    elif is_gail:
                        action = model(state_t).squeeze(0).cpu().data.numpy()
                    else:
                        action = model(state_t).squeeze(0).cpu().data.numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if is_dt:
                rtg -= reward
            done = terminated or truncated
            env.render()
            
        total_rewards.append(ep_reward)
        print(f"[{name}] Episodio {ep+1} - Pontuacao: {ep_reward:.2f}")
    
    return sum(total_rewards)/len(total_rewards)

# ==========================================================
# Main Runner
# ==========================================================
def main():
    print("==================================================")
    print(" GRANDE CORRIDA OFFLINE RL: WALKER2D-V5 ")
    print("==================================================")
    
    env = gym.make("Walker2d-v5", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cpu")
    
    scores = {}

    if os.path.exists("sac_walker2d_final.zip"):
        teacher = SAC.load("sac_walker2d_final.zip", device=device)
        class TeacherWrapper(nn.Module):
            def forward(self, state):
                action, _ = teacher.predict(state.cpu().numpy(), deterministic=True)
                return torch.FloatTensor(action).to(device)
        scores["Teacher (Upper Bound)"] = evaluate_model(env, "Teacher (Upper Bound)", TeacherWrapper(), device)

    # 1. Behavioral Cloning Puro
    if os.path.exists("bc_model.pt"):
        bc = BCPolicy(state_dim, action_dim).to(device)
        bc.load_state_dict(torch.load("bc_model.pt", map_location=device))
        bc.eval()
        scores["Behavioral Cloning"] = evaluate_model(env, "Behavioral Cloning Puro", bc, device)

    # 2. IQL
    if os.path.exists("iql_full_ckpt.pt"):
        iql = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        checkpoint = torch.load("iql_full_ckpt.pt", map_location=device)
        iql.load_state_dict(checkpoint['policy'])
        iql.eval()
        scores["IQL Offline"] = evaluate_model(env, "Implicit Q-Learning", iql, device)

    # 3. CQL
    if os.path.exists("cql_full_ckpt.pt"):
        cql = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        checkpoint = torch.load("cql_full_ckpt.pt", map_location=device)
        cql.load_state_dict(checkpoint['policy'])
        cql.eval()
        scores["CQL Offline"] = evaluate_model(env, "Conservative Q-Learning", cql, device)

    # 4. BC + SAC (Naive)
    if os.path.exists("bc_sac_naive_model.pt"):
        bc_sac = SACActor(state_dim, action_dim, max_action).to(device)
        bc_sac.load_state_dict(torch.load("bc_sac_naive_model.pt", map_location=device))
        bc_sac.eval()
        scores["BC+SAC (Naive)"] = evaluate_model(env, "BC+SAC (Naive Initialization)", bc_sac, device)

    # 5. BC + SAC (Regularized)
    if os.path.exists("bc_sac_regularized_model.pt"):
        bc_reg = SACActor(state_dim, action_dim, max_action).to(device)
        bc_reg.load_state_dict(torch.load("bc_sac_regularized_model.pt", map_location=device))
        bc_reg.eval()
        scores["BC+SAC (Regularized)"] = evaluate_model(env, "BC+SAC (Regularization Penalty)", bc_reg, device)

    # 6. BC + SAC (Constrained)
    if os.path.exists("bc_sac_constrained_model.pt"):
        bc_con = SACActor(state_dim, action_dim, max_action).to(device)
        bc_con.load_state_dict(torch.load("bc_sac_constrained_model.pt", map_location=device))
        bc_con.eval()
        scores["BC+SAC (Constrained)"] = evaluate_model(env, "BC+SAC (Action Constraints)", bc_con, device)

    # 7. IQL + SAC
    if os.path.exists("iql_sac_model.pt"):
        iql_sac = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        iql_sac.load_state_dict(torch.load("iql_sac_model.pt", map_location=device))
        iql_sac.eval()
        scores["IQL+SAC"] = evaluate_model(env, "IQL+SAC (Offline-to-Online)", iql_sac, device)

    # 8. CQL + SAC
    if os.path.exists("cql_sac_model.pt"):
        cql_sac = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        cql_sac.load_state_dict(torch.load("cql_sac_model.pt", map_location=device))
        cql_sac.eval()
        scores["CQL+SAC"] = evaluate_model(env, "CQL+SAC (Offline-to-Online)", cql_sac, device)
    
    # 9. GAIL
    gail = GAILActor(state_dim, action_dim, max_action).to(device)
    if os.path.exists("gail_model.pt"):
        gail.load_state_dict(torch.load("gail_model.pt", map_location=device))
        gail.eval()
        scores["Inverse RL (GAIL)"] = evaluate_model(env, "Inverse RL (GAIL)", gail, device, is_gail=True)

    # 10. AIRL
    airl = AIRLActor(state_dim, action_dim, max_action).to(device)
    if os.path.exists("airl_model.pt"):
        airl.load_state_dict(torch.load("airl_model.pt", map_location=device))
        airl.eval()
        scores["Inverse RL (AIRL)"] = evaluate_model(env, "Inverse RL (AIRL)", airl, device, is_airl=True)

    # 11. BCQ
    if os.path.exists("bcq_vae.pt"):
        vae = VAE(state_dim, action_dim, action_dim*2, max_action).to(device)
        vae.load_state_dict(torch.load("bcq_vae.pt", map_location=device))
        perturbation = PerturbationNetwork(state_dim, action_dim, max_action).to(device)
        perturbation.load_state_dict(torch.load("bcq_perturbation.pt", map_location=device))
        vae.eval()
        perturbation.eval()
        class BCQWrapper(nn.Module):
            def forward(self, state):
                a = vae.decode(state)
                return perturbation(state, a)
        scores["Batch-Constrained Q-learning (BCQ)"] = evaluate_model(env, "Batch-Constrained Q-learning (BCQ)", BCQWrapper(), device)

    # 12. Decision Transformer
    if os.path.exists("dt_model.pt"):
        dt = DecisionTransformer(state_dim, action_dim, hidden_size=128, max_length=20).to(device)
        dt.load_state_dict(torch.load("dt_model.pt", map_location=device))
        dt.eval()
        scores["Decision Transformer (DT)"] = evaluate_model(env, "Decision Transformer (DT)", dt, device, is_dt=True)

    # 13. MaxEnt IRL
    maxent = MaxEntActor(state_dim, action_dim, max_action).to(device)
    if os.path.exists("maxent_model.pt"):
        maxent.load_state_dict(torch.load("maxent_model.pt", map_location=device))
        maxent.eval()
        scores["MaxEnt IRL"] = evaluate_model(env, "MaxEnt IRL", maxent, device, is_maxent=True)

    # 14. Deep PQR
    pqr = PQRPolicy(state_dim, action_dim, max_action).to(device)
    if os.path.exists("pqr_policy.pt"):
        pqr.load_state_dict(torch.load("pqr_policy.pt", map_location=device))
        pqr.eval()
        scores["Deep PQR"] = evaluate_model(env, "Deep PQR", pqr, device)

    # 15. Extra Trees Cloner (sklearn, CPU). Opt-in automático se o .pkl existir.
    if os.path.exists("extratrees_model.pkl"):
        try:
            import joblib

            et_model = joblib.load("extratrees_model.pkl")

            class ExtraTreesWrapper(nn.Module):
                def forward(self, state):
                    obs = state.cpu().numpy()
                    if obs.ndim == 1:
                        obs = obs.reshape(1, -1)
                    action = np.asarray(et_model.predict(obs)).reshape(-1)
                    return torch.FloatTensor(action).to(device)

            scores["Extra Trees Cloner"] = evaluate_model(env, "Extra Trees Cloner", ExtraTreesWrapper().to(device), device)
        except Exception as e:
            print(f"[Extra Trees Cloner] skipped: {e}")

    env.close()

    print("\n==================================================")
    print(" PLACAR FINAL ")
    print("==================================================")
    for model_name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{model_name:30s}: {score:.2f} pontos")
    print("==================================================")

if __name__ == "__main__":
    main()
