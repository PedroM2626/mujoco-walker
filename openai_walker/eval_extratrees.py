"""Evaluate the Extra Trees behavioural clone on Walker2d-v5.

Same protocol as `evaluate_all.py` (seeded resets, N episodes, mean reported), so the
scores are comparable with the rest of the Phase-4 table instead of being a one-off
number. Paths resolve next to this file: the original hard-coded "extratrees_model.pkl"
only worked when the process happened to be started from `openai_walker/`, and the reset
was unseeded, so repeated runs did not agree with each other.
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

import joblib  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Eval do Extra Trees (Walker2d-v5).")
    p.add_argument("--model", default=os.path.join(HERE, "extratrees_model.pkl"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def evaluate_extratrees(model_path, episodes, seed):
    if not os.path.exists(model_path):
        print(f"Modelo ausente: {model_path}")
        print("Treine com: python train_extratrees.py")
        raise SystemExit(2)

    print("Loading Extra Trees model...")
    model = joblib.load(model_path)

    env = gym.make("Walker2d-v5")
    total = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            while not done:
                action = model.predict(obs.reshape(1, -1))[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += float(reward)
                done = terminated or truncated
            total.append(ep_reward)
    finally:
        env.close()

    arr = np.asarray(total)
    print("==================================================")
    print(f"Extra Trees: {arr.mean():.2f} avg over {episodes} ep "
          f"(std {arr.std():.2f}, min {arr.min():.2f}, max {arr.max():.2f})")
    print("==================================================")
    return arr.mean(), arr


if __name__ == "__main__":
    args = parse_args()
    evaluate_extratrees(args.model, args.episodes, args.seed)
