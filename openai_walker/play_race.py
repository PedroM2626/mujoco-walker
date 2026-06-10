import gymnasium as gym
import torch
import torch.nn as nn
import time
import os
from stable_baselines3 import SAC

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
def evaluate_model(env, name, get_action_fn, episodes=1):
    print(f"\n[{name}] Preparando para a corrida...")
    time.sleep(2) # Pausa para o usuario ler o terminal
    
    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = get_action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            env.render()
            time.sleep(0.01) # Desacelera para visualizacao
            
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
    
    device = torch.device("cpu") # Avaliacao visual sempre na CPU para evitar gargalos de sync
    
    scores = {}

    # 0. Professor SAC (Teto)
    if os.path.exists("sac_walker2d_final.zip"):
        teacher = SAC.load("sac_walker2d_final.zip", device=device)
        scores["Teacher (Upper Bound)"] = evaluate_model(env, "Teacher (Upper Bound)", lambda obs: teacher.predict(obs, deterministic=True)[0])

    # 1. Behavioral Cloning Puro
    if os.path.exists("bc_model.pt"):
        bc = BCPolicy(state_dim, action_dim).to(device)
        bc.load_state_dict(torch.load("bc_model.pt", map_location=device))
        bc.eval()
        with torch.no_grad():
            scores["Behavioral Cloning"] = evaluate_model(env, "Behavioral Cloning Puro", lambda obs: bc(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 2. IQL
    if os.path.exists("iql_full_ckpt.pt"):
        iql = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        checkpoint = torch.load("iql_full_ckpt.pt", map_location=device)
        iql.load_state_dict(checkpoint['policy'])
        iql.eval()
        with torch.no_grad():
            scores["IQL Offline"] = evaluate_model(env, "Implicit Q-Learning", lambda obs: iql(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 3. CQL
    if os.path.exists("cql_full_ckpt.pt"):
        cql = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        checkpoint = torch.load("cql_full_ckpt.pt", map_location=device)
        cql.load_state_dict(checkpoint['policy'])
        cql.eval()
        with torch.no_grad():
            scores["CQL Offline"] = evaluate_model(env, "Conservative Q-Learning", lambda obs: cql(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 4. BC + SAC (Naive)
    if os.path.exists("bc_sac_naive_model.pt"):
        bc_sac = SACActor(state_dim, action_dim, max_action).to(device)
        bc_sac.load_state_dict(torch.load("bc_sac_naive_model.pt", map_location=device))
        bc_sac.eval()
        with torch.no_grad():
            scores["BC+SAC (Naive)"] = evaluate_model(env, "BC+SAC (Naive Initialization)", lambda obs: bc_sac(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 5. BC + SAC (Regularized)
    if os.path.exists("bc_sac_regularized_model.pt"):
        bc_reg = SACActor(state_dim, action_dim, max_action).to(device)
        bc_reg.load_state_dict(torch.load("bc_sac_regularized_model.pt", map_location=device))
        bc_reg.eval()
        with torch.no_grad():
            scores["BC+SAC (Regularized)"] = evaluate_model(env, "BC+SAC (Regularization Penalty)", lambda obs: bc_reg(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 6. BC + SAC (Constrained)
    if os.path.exists("bc_sac_constrained_model.pt"):
        bc_con = SACActor(state_dim, action_dim, max_action).to(device)
        bc_con.load_state_dict(torch.load("bc_sac_constrained_model.pt", map_location=device))
        bc_con.eval()
        with torch.no_grad():
            scores["BC+SAC (Constrained)"] = evaluate_model(env, "BC+SAC (Action Constraints)", lambda obs: bc_con(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 7. IQL + SAC
    if os.path.exists("iql_sac_model.pt"):
        iql_sac = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        iql_sac.load_state_dict(torch.load("iql_sac_model.pt", map_location=device))
        iql_sac.eval()
        with torch.no_grad():
            scores["IQL+SAC"] = evaluate_model(env, "IQL+SAC (Offline-to-Online)", lambda obs: iql_sac(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())

    # 8. CQL + SAC
    if os.path.exists("cql_sac_model.pt"):
        cql_sac = PolicyNetIQL(state_dim, action_dim, max_action).to(device)
        cql_sac.load_state_dict(torch.load("cql_sac_model.pt", map_location=device))
        cql_sac.eval()
        with torch.no_grad():
            scores["CQL+SAC"] = evaluate_model(env, "CQL+SAC (Offline-to-Online)", lambda obs: cql_sac(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())
    
    # 9. GAIL
    gail = GAILActor(state_dim, action_dim, max_action).to(device)
    try:
        gail.load_state_dict(torch.load("gail_model.pt", map_location=device))
        gail.eval()
        with torch.no_grad():
            scores["Inverse RL (GAIL)"] = evaluate_model(env, "Inverse RL (GAIL)", lambda obs: gail(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).numpy())
    except Exception as e:
        print(f"GAIL skip: {e}")

    env.close()

    print("\n==================================================")
    print(" PLACAR FINAL ")
    print("==================================================")
    for model_name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{model_name:30s}: {score:.2f} pontos")
    print("==================================================")

if __name__ == "__main__":
    main()
