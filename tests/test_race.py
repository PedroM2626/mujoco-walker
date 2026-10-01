import os
import sys
import unittest
import shutil
import subprocess
import torch
import numpy as np
import gymnasium as gym
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_walker import SACAgent, PPOAgent, TD3Agent
from play_race_ragdoll import build_race_xml, get_agent_observation


class TestRaceSystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_race_assets")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy checkpoints for SAC, PPO, and TD3
        self.sac_path = os.path.join(self.test_dir, "dummy_sac.pt")
        self.ppo_path = os.path.join(self.test_dir, "dummy_ppo.pt")
        self.td3_path = os.path.join(self.test_dir, "dummy_td3.pt")

        # Set up spaces
        self.obs_dim_recovery = 46
        self.obs_dim_target = 49
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,))
        
        # Save dummy SAC
        sac_agent = SACAgent(self.obs_dim_recovery, self.action_space)
        torch.save({
            "algo": "sac",
            "actor_state_dict": sac_agent.state_dict(),
            "task_phase": "recovery",
            "target_forward_velocity": 0.8,
        }, self.sac_path)
        
        # Save dummy PPO (with target phase obs_dim = 49)
        ppo_agent = PPOAgent(self.obs_dim_target, 17)
        torch.save({
            "algo": "ppo",
            "agent_state_dict": ppo_agent.state_dict(),
            "task_phase": "target",
            "target_forward_velocity": 0.8,
        }, self.ppo_path)
        
        # Save dummy TD3
        td3_agent = TD3Agent(self.obs_dim_recovery, self.action_space)
        torch.save({
            "algo": "td3",
            "agent_state_dict": td3_agent.state_dict(),
            "task_phase": "recovery",
            "target_forward_velocity": 0.8,
        }, self.td3_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_xml_builder(self):
        base_xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "walker_ragdoll.xml")
        temp_xml = build_race_xml(base_xml, num_agents=3, target_x=10.0, lane_distance=2.0)
        
        self.assertTrue(os.path.exists(temp_xml), "Race XML was not generated.")
        
        # Verify it loads in MuJoCo
        model = mujoco.MjModel.from_xml_path(temp_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        # We should find 3 duplicate torsos (agent0_torso, agent1_torso, agent2_torso)
        for i in range(3):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}_torso")
            self.assertGreaterEqual(body_id, 0, f"agent{i}_torso body not found in model.")
            
            # Verify target marker exists
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}_target_marker")
            self.assertGreaterEqual(target_id, 0, f"agent{i}_target_marker not found in model.")
            
        os.remove(temp_xml)

    def test_observation_extraction(self):
        base_xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "walker_ragdoll.xml")
        temp_xml = build_race_xml(base_xml, num_agents=2, target_x=3.0, lane_distance=1.5)
        
        model = mujoco.MjModel.from_xml_path(temp_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        # Test extraction for agent 0 (recovery phase, size 46)
        obs_rec = get_agent_observation(model, data, agent_idx=0, target_x=3.0, target_y_initial=-0.75, obs_dim=46)
        self.assertEqual(obs_rec.shape, (46,), "Extracted observation shape mismatch for recovery phase.")
        
        # Test extraction for agent 1 (target phase, size 49)
        obs_tgt = get_agent_observation(model, data, agent_idx=1, target_x=3.0, target_y_initial=0.75, obs_dim=49)
        self.assertEqual(obs_tgt.shape, (49,), "Extracted observation shape mismatch for target phase.")
        
        # Check target observation content values (dx relative should be 3.0)
        self.assertAlmostEqual(obs_tgt[-3], 3.0, places=2)  # dx
        self.assertAlmostEqual(obs_tgt[-2], 0.0, places=2)   # dy relative
        self.assertAlmostEqual(obs_tgt[-1], 3.0, places=2)  # distance
        
        os.remove(temp_xml)

    def test_play_race_execution(self):
        # Run the race script as a subprocess in headless mode with low steps
        result = subprocess.run(
            [
                sys.executable,
                "play_race_ragdoll.py",
                "--checkpoints", self.sac_path, self.ppo_path, self.td3_path,
                "--names", "FastSAC", "AccuratePPO", "RobustTD3",
                "--target-x", "5.0",
                "--max-steps", "10",
                "--device", "cpu",
                "--headless"
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"play_race_ragdoll.py failed with: {result.stderr}")
        self.assertIn("FINAL RACE RESULTS", result.stdout, "Results table not found in output.")
        self.assertIn("FastSAC", result.stdout, "Custom name FastSAC not found in output.")
        self.assertIn("AccuratePPO", result.stdout, "Custom name AccuratePPO not found in output.")
        self.assertIn("RobustTD3", result.stdout, "Custom name RobustTD3 not found in output.")


if __name__ == "__main__":
    unittest.main()
