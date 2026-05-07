import os
import glob
import re
import json
import torch
import shutil


def get_checkpoint_dir(run_id, base_dir="checkpoints"):
    """Returns the checkpoint directory for a given run_id."""
    return os.path.join(base_dir, run_id)


def get_run_dir(run_id, base_dir="runs"):
    """Returns the tensorboard/wandb run directory for a given run_id."""
    return os.path.join(base_dir, run_id)


def save_checkpoint(agent, optimizer, global_step, run_id, base_dir="checkpoints", keep_last_n=3):
    """Save a training checkpoint with metadata."""
    ckpt_dir = get_checkpoint_dir(run_id, base_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
    metadata_path = os.path.join(ckpt_dir, f"ckpt_{global_step}_metadata.json")

    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }

    metadata = {
        "global_step": global_step,
        "run_id": run_id,
    }

    torch.save(checkpoint, checkpoint_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    cleanup_old_checkpoints(ckpt_dir, keep_last_n)
    print(f"[CHECKPOINT] Saved at step {global_step} -> {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(agent, optimizer, run_id, global_step=None, base_dir="checkpoints"):
    """Load a training checkpoint. If global_step is None, loads the latest."""
    ckpt_dir = get_checkpoint_dir(run_id, base_dir)

    if not os.path.exists(ckpt_dir):
        return None, 0

    if global_step is not None:
        checkpoint_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"[CHECKPOINT] Requested step {global_step} not found. Falling back to latest.")
            checkpoint_path = find_latest_checkpoint(ckpt_dir)
    else:
        checkpoint_path = find_latest_checkpoint(ckpt_dir)

    if checkpoint_path is None:
        return None, 0

    print(f"[CHECKPOINT] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    agent.load_state_dict(checkpoint["agent_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loaded_step = checkpoint.get("global_step", 0)

    return checkpoint, loaded_step


def find_latest_checkpoint(ckpt_dir):
    """Find the most recent checkpoint in a directory."""
    checkpoints = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not checkpoints:
        return None

    def extract_step(path):
        match = re.search(r"ckpt_(\d+)\.pt", os.path.basename(path))
        return int(match.group(1)) if match else -1

    checkpoints.sort(key=extract_step, reverse=True)
    return checkpoints[0]


def cleanup_old_checkpoints(ckpt_dir, keep_last_n=3):
    """Remove old checkpoints, keeping only the last N."""
    checkpoints = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if len(checkpoints) <= keep_last_n:
        return

    def extract_step(path):
        match = re.search(r"ckpt_(\d+)\.pt", os.path.basename(path))
        return int(match.group(1)) if match else -1

    checkpoints.sort(key=extract_step, reverse=True)

    for ckpt in checkpoints[keep_last_n:]:
        try:
            os.remove(ckpt)
            meta = ckpt.replace(".pt", "_metadata.json")
            if os.path.exists(meta):
                os.remove(meta)
            print(f"[CHECKPOINT] Removed old checkpoint: {ckpt}")
        except OSError:
            pass


def force_delete_run(run_id, ckpt_base="checkpoints", run_base="runs"):
    """Delete all data for a run (checkpoints and logs)."""
    ckpt_dir = get_checkpoint_dir(run_id, ckpt_base)
    run_dir = get_run_dir(run_id, run_base)

    for d in [ckpt_dir, run_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"[FORCE] Deleted {d}")
        else:
            print(f"[FORCE] Directory not found: {d}")
