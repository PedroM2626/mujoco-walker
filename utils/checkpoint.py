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


def save_checkpoint(agent, optimizer, global_step, run_id, envs=None, base_dir="checkpoints", keep_last_n=3):
    """Save a training checkpoint with metadata and environment stats."""
    ckpt_dir = get_checkpoint_dir(run_id, base_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
    metadata_path = os.path.join(ckpt_dir, f"ckpt_{global_step}_metadata.json")

    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }

    # Save environment normalization statistics if available
    if envs is not None:
        # For Gymnasium VectorEnv with NormalizeObservation wrapper
        if hasattr(envs, "obs_rms"):
            checkpoint["obs_rms"] = envs.obs_rms
        # For single environment with NormalizeObservation wrapper
        elif hasattr(envs, "get_wrapper_attr") and hasattr(envs, "obs_rms"):
             checkpoint["obs_rms"] = envs.obs_rms
        elif hasattr(envs, "unwrapped") and hasattr(envs, "obs_rms"):
             checkpoint["obs_rms"] = envs.obs_rms

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


def load_checkpoint(agent, optimizer, run_id, envs=None, global_step=None, base_dir="checkpoints"):
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
    # Checkpoint local (pode conter RunningMeanStd em obs_rms).
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    agent.load_state_dict(checkpoint["agent_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load environment normalization statistics if available and envs provided
    if envs is not None and "obs_rms" in checkpoint:
        if hasattr(envs, "obs_rms"):
            envs.obs_rms = checkpoint["obs_rms"]
            print("[CHECKPOINT] Loaded observation normalization statistics.")
        elif hasattr(envs, "unwrapped") and hasattr(envs, "obs_rms"):
             envs.obs_rms = checkpoint["obs_rms"]
             print("[CHECKPOINT] Loaded observation normalization statistics.")

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


def _dir_is_in_use(directory):
    """True if some other process still holds a file open inside `directory`.

    TensorBoard's writer keeps its event file open, and on Windows opening it for append from
    a second process fails. Checking is much cheaper than the alternative: deleting the
    directory out from under a live run makes that run die with a FileNotFoundError from
    inside the writer thread, several hundred steps and one wasted training run later.
    """
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, "ab"):
                    pass
            except PermissionError:
                return True
            except OSError:
                continue
    return False


def force_delete_run(run_id, run_name=None, ckpt_base="checkpoints", run_base="runs"):
    """Delete a run's data before a fresh start.

    `run_name` is the caller's own `<run_id>__<seed>` log directory. It used to be ignored:
    the function globbed `runs/<run_id>__*` and deleted every match, so starting seed 8
    destroyed seed 7's tensorboard logs - and two processes launched with the same run id
    killed each other outright. Without `run_name` the old behaviour is kept for callers that
    really mean "the whole run id", but it now skips directories another process is writing.
    """
    ckpt_dir = get_checkpoint_dir(run_id, ckpt_base)

    # Delete checkpoints
    if os.path.exists(ckpt_dir):
        if _dir_is_in_use(ckpt_dir):
            print(f"[FORCE] {ckpt_dir} is open by another process - leaving it alone.")
        else:
            shutil.rmtree(ckpt_dir)
            print(f"[FORCE] Deleted checkpoints: {ckpt_dir}")

    # Delete this process's own log directory, or every directory of the run when asked for
    targets = [os.path.join(run_base, run_name)] if run_name \
        else glob.glob(os.path.join(run_base, f"{run_id}__*"))
    for d in targets:
        if not os.path.isdir(d):
            continue
        if _dir_is_in_use(d):
            print(f"[FORCE] {d} is open by another process - leaving it alone. "
                  "Use a distinct --run-id per concurrent run.")
            continue
        shutil.rmtree(d)
        print(f"[FORCE] Deleted log directory: {d}")
