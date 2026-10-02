"""The reward shaping the online-RL trainers apply to WalkerRagdoll-v0, in one place.

This lives outside `walker_ragdoll_env.py` on purpose: it is a property of the *trainer*, not of
the environment revision. Evaluators need to import it while an older environment revision is
aliased in as `envs.walker_ragdoll_env` (`eval_phase1.py --env-commit`), and a dict defined in
that module would then be the old revision's dict - or missing entirely.

Why it matters: the shaping overrides the environment defaults, so an evaluator that calls
`gym.make()` without it scores the agent against a reward function it never optimised. Measured
on the 40M SAC policy (`benchmarks/reward_term_breakdown_v9.json`): the defaults pay
`standing_reward=50/step` that training sets to 0, and weight velocity-toward-target at 80
instead of 200, so the same trajectory scores differently by design and the ranking of postures
against locomotion changes.
"""

TRAINING_REWARD_KWARGS = {
    # No survival bonus: standing still must not be a paying strategy.
    "standing_reward": 0.0,
    # Strong signal: velocity directed at the target.
    "target_direction_reward_weight": 200.0,
    # Moderate progress reward (clipped at 0.15 m/step).
    "target_progress_reward_weight": 300.0,
    # Posture and steadiness stay bonuses; posture is never punished (env v9).
    "stand_height_reward_weight": 100.0,
    "stability_reward_weight": 20.0,
    # Small stillness penalty - enough to prevent freezing, not enough to force falls.
    "stillness_penalty_weight": 5.0,
    # Moderate lateral drift penalty to keep the agent heading at the target.
    "lateral_drift_penalty_weight": 3.0,
}


def reward_kwargs_for(checkpoint):
    """The reward kwargs a checkpoint should be scored with, and where that decision came from.

    New checkpoints carry their own `reward_kwargs`; the historical SAC/TD3/PPO runs were all made
    by `train_walker.py`, so they are scored with TRAINING_REWARD_KWARGS; anything else (Dreamer,
    REDQ, ARS) trained against the environment defaults and must be scored against those.
    """
    recorded = checkpoint.get("reward_kwargs") if isinstance(checkpoint, dict) else None
    tfv = checkpoint.get("target_forward_velocity") if isinstance(checkpoint, dict) else None
    algo = str(checkpoint.get("algo") or "").lower() if isinstance(checkpoint, dict) else ""
    kwargs = {}
    if recorded:
        kwargs, source = dict(recorded), "checkpoint"
    elif algo.startswith(("sac", "td3", "ppo")):
        kwargs, source = dict(TRAINING_REWARD_KWARGS), "train_walker shaping (checkpoint predates recording)"
    else:
        source = "environment defaults"
    if tfv is not None and "target_forward_velocity" not in kwargs:
        kwargs["target_forward_velocity"] = float(tfv)
    return kwargs, source
