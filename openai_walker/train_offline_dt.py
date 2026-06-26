import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import mlflow

class SequenceDataset(Dataset):
    def __init__(self, csv_file, context_len=20):
        df = pd.read_csv(csv_file)
        self.context_len = context_len
        
        # Flattened dataset into sequential episodes
        states = df.iloc[:, :17].values
        actions = df.iloc[:, 34:40].values
        rewards = df.iloc[:, 40:41].values
        dones = df.iloc[:, 41:42].values
        
        self.trajectories = []
        curr_traj = {"states": [], "actions": [], "rewards": []}
        
        for i in range(len(states)):
            curr_traj["states"].append(states[i])
            curr_traj["actions"].append(actions[i])
            curr_traj["rewards"].append(rewards[i])
            
            if dones[i][0] == 1.0 or i == len(states) - 1:
                # Compute returns-to-go
                rtg = np.zeros_like(np.array(curr_traj["rewards"]))
                discounted_sum = 0
                for t in reversed(range(len(rtg))):
                    discounted_sum = curr_traj["rewards"][t][0] + discounted_sum
                    rtg[t] = [discounted_sum]
                    
                curr_traj["rtg"] = rtg
                self.trajectories.append({
                    "states": np.array(curr_traj["states"]),
                    "actions": np.array(curr_traj["actions"]),
                    "rtg": curr_traj["rtg"]
                })
                curr_traj = {"states": [], "actions": [], "rewards": []}
                
        # Build samples
        self.samples = []
        for traj in self.trajectories:
            traj_len = len(traj["states"])
            for t in range(traj_len):
                if t + self.context_len <= traj_len:
                    self.samples.append({
                        "s": traj["states"][t:t+self.context_len],
                        "a": traj["actions"][t:t+self.context_len],
                        "rtg": traj["rtg"][t:t+self.context_len]
                    })
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = torch.FloatTensor(self.samples[idx]["s"])
        a = torch.FloatTensor(self.samples[idx]["a"])
        rtg = torch.FloatTensor(self.samples[idx]["rtg"])
        return s, a, rtg

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, max_length=20):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        self.embed_timestep = nn.Embedding(1000, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=4*hidden_size, 
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        B, T, _ = states.shape
        
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        
        # Sequence: R_1, s_1, a_1, R_2, s_2, a_2 ...
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3*T, self.hidden_size)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Causal Mask
        mask = torch.triu(torch.ones(3*T, 3*T) * float('-inf'), diagonal=1).to(states.device)
        
        x = self.transformer(stacked_inputs, mask=mask)
        
        # Reshape to (B, T, 3, hidden_size)
        x = x.reshape(B, T, 3, self.hidden_size)
        
        # Predict next action using the state representation (index 1)
        # Note: at step t, state representation contains R_t, s_t
        action_preds = self.predict_action(x[:, :, 1])
        return action_preds

def train_dt():
    mlflow.set_tracking_uri("sqlite:///../mlruns.db")
    mlflow.set_experiment("Walker2d_Offline_to_Online")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("Walker2d-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    context_len = 20
    dataset = SequenceDataset("dataset_openai.csv", context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    model = DecisionTransformer(state_dim, action_dim, hidden_size=128, max_length=context_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 10
    
    with mlflow.start_run(run_name="Decision_Transformer"):
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for s, a, rtg in dataloader:
                s, a, rtg = s.to(device), a.to(device), rtg.to(device)
                
                # Mock timesteps for simplicity
                timesteps = torch.arange(0, context_len, device=device).unsqueeze(0).repeat(64, 1)
                
                action_preds = model(s, a, rtg, timesteps)
                
                loss = F.mse_loss(action_preds, a)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("dt_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            
        torch.save(model.state_dict(), "dt_model.pt")
        print("DT Model saved as dt_model.pt")

if __name__ == "__main__":
    train_dt()
