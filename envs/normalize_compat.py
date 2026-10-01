"""Pickle compatibility for checkpoints that stored observation-normalisation stats.

Why this exists: `utils/checkpoint.py` writes the `RunningMeanStd` object owned by the
`NormalizeObservation` wrapper straight into the `.pt` file, so loading one pins the
module path it was saved under. Gymnasium has moved that class between releases
(`gymnasium.wrappers.normalize` in 0.29, re-exported through `gymnasium.wrappers`
in 1.x), and older runs may even reference legacy `gym.wrappers.normalize`.

The previous version of this file (`mock_wrappers.py`) answered by injecting empty stub
classes over `gymnasium.wrappers.normalize`. That made `torch.load` succeed while
silently turning observation normalisation into a no-op, so the merging/MoE scripts ran
on un-normalised states. Aliasing the *real* classes fixes loading and keeps the
statistics meaningful.

Import for its side effects before unpickling a checkpoint.
"""

import importlib
import sys
import types

_NAMES = ("RunningMeanStd", "NormalizeObservation", "NormalizeReward")
_CANDIDATES = (
    "gymnasium.wrappers.normalize",
    "gymnasium.wrappers.common",
    "gymnasium.wrappers",
)


def _real_source():
    for path in _CANDIDATES:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        if all(hasattr(module, name) for name in _NAMES):
            return module
    raise ImportError(
        "could not locate RunningMeanStd/NormalizeObservation/NormalizeReward in this "
        "gymnasium install; checkpoint stats cannot be unpickled faithfully"
    )


def install():
    """Point every historical module path at the real classes. Returns the source module."""
    source = _real_source()

    # Gymnasium >= 1.0 dropped the submodule these objects were pickled under.
    if "gymnasium.wrappers.normalize" not in sys.modules:
        shim = types.ModuleType("gymnasium.wrappers.normalize")
        for name in _NAMES:
            setattr(shim, name, getattr(source, name))
        sys.modules["gymnasium.wrappers.normalize"] = shim
        shim.__source__ = source.__name__

    # Pre-gymnasium runs pickled these objects under the OpenAI Gym module path.
    if "gym" not in sys.modules:
        gym_pkg = types.ModuleType("gym")
        gym_pkg.__path__ = []
        sys.modules["gym"] = gym_pkg
    for name in ("gym.wrappers", "gym.wrappers.normalize"):
        if name not in sys.modules:
            alias = types.ModuleType(name)
            alias.__path__ = []
            for attr in _NAMES:
                setattr(alias, attr, getattr(source, attr))
            sys.modules[name] = alias
    sys.modules["gym"].wrappers = sys.modules["gym.wrappers"]
    sys.modules["gym.wrappers"].normalize = sys.modules["gym.wrappers.normalize"]

    return source


RunningMeanStd = None  # populated below, exported for callers that want the real class

source = install()
RunningMeanStd = source.RunningMeanStd
NormalizeObservation = source.NormalizeObservation
NormalizeReward = source.NormalizeReward
