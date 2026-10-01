"""Legacy alias module kept so old merged checkpoints still load.

History matters here. This file used to define *empty stub* classes
(`class RunningMeanStd: pass`) and inject them over
`gymnasium.wrappers.normalize`, so that `torch.load` on a checkpoint holding
`obs_rms` would not raise. Loading succeeded, but normalisation silently became a
no-op, which is how the merging/MoE scripts ended up evaluating a different policy
than `play.py` runs.

The stubs are gone and the real classes are aliased instead - see
`envs/normalize_compat.py`. But `merged_avg_model.pt` and `merged_ta_model.pt` were
pickled *while the stubs were installed*, so their embedded objects literally name
`mock_wrappers.RunningMeanStd`. Deleting this file makes those two committed
artifacts unloadable (confirmed: `ModuleNotFoundError: No module named
'mock_wrappers'`). Keeping it as an alias means they load again and the statistics
they carry are functional.

Do not import this module for new code; import the real wrapper from gymnasium.
"""

from envs.normalize_compat import NormalizeObservation, NormalizeReward, RunningMeanStd

__all__ = ["RunningMeanStd", "NormalizeObservation", "NormalizeReward"]
