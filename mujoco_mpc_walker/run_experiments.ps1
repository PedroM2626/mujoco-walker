cd d:\mujoco-walker\mujoco_mpc_walker

echo "Training IQL (Offline) to get full checkpoint..."
python train_iql.py

echo "Training CQL (Offline) to get full checkpoint..."
python train_cql.py

echo "Training IQL+SAC (Offline-to-Online)..."
python train_iql_sac.py

echo "Training CQL+SAC (Offline-to-Online)..."
python train_cql_sac.py

echo "Evaluating all 6 models..."
python evaluate_all.py
