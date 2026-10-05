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

    New checkpoints carry their own `reward_kwargs`. For the ones that do not, the fallback used to
    be "SAC/TD3/PPO were trained by `train_walker.py` so they get the shaping, and Dreamer/REDQ/ARS
    trained against the environment defaults" - and that split was wrong. `train_walker.make_env`
    has applied these kwargs to *every* sub-environment since 8d37846 (2026-05-20), and
    `train_dreamer.py`, `train_redq.py` and `train_ars.py` all build their environments through it,
    so a Phase-1 checkpoint that does not record its reward was still trained under this one. The
    cost of the wrong fallback was that every published Dreamer/ARS/REDQ return was computed against
    a reward nobody optimised - the exact failure this module's docstring warns about, on the three
    algorithms it did not cover.
    """
    recorded = checkpoint.get("reward_kwargs") if isinstance(checkpoint, dict) else None
    tfv = checkpoint.get("target_forward_velocity") if isinstance(checkpoint, dict) else None
    if recorded:
        kwargs, source = dict(recorded), "checkpoint"
    else:
        kwargs, source = dict(TRAINING_REWARD_KWARGS), (
            "train_walker.make_env shaping (applied to every Phase-1 trainer since 8d37846; this "
            "checkpoint predates recording it)")
    if tfv is not None and "target_forward_velocity" not in kwargs:
        kwargs["target_forward_velocity"] = float(tfv)
    return kwargs, source
