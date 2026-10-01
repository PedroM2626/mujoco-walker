"""Pickle compatibility for checkpoints that stored observation-normalisation stats.

Why this exists: `utils/checkpoint.py` writes the `RunningMeanStd` object owned by the
`NormalizeObservation` wrapper straight into the `.pt` file, so loading one pins the
module path it was saved under. Gymnasium has moved that class between releases
(`gymnasium.wrappers.normalize` in 0.29; in 1.0 it is `gymnasium.wrappers.utils`, with the
wrappers themselves in `gymnasium.wrappers.stateful_observation` / `stateful_reward`), and
older runs may even reference legacy `gym.wrappers.normalize`.

The previous version of this file (`mock_wrappers.py`) answered by injecting empty stub
classes over `gymnasium.wrappers.normalize`. That made `torch.load` succeed while silently
turning observation normalisation into a no-op, so the merging/MoE scripts ran on
un-normalised states. Aliasing the *real* classes fixes loading and keeps the statistics
meaningful.

Import for its side effects before unpickling a checkpoint.
"""

import importlib
import importlib.util
import sys
import types

# Looked up name by name: gymnasium 1.0 split these across modules, so no single import
# provides all three.
_SOURCES = {
    "RunningMeanStd": (
        "gymnasium.wrappers.normalize",
        "gymnasium.wrappers.utils",
        "gymnasium.wrappers",
    ),
    "NormalizeObservation": (
        "gymnasium.wrappers.normalize",
        "gymnasium.wrappers.stateful_observation",
        "gymnasium.wrappers",
    ),
    "NormalizeReward": (
        "gymnasium.wrappers.normalize",
        "gymnasium.wrappers.stateful_reward",
        "gymnasium.wrappers",
    ),
}


def _resolve():
    resolved = {}
    for name, paths in _SOURCES.items():
        for path in paths:
            try:
                module = importlib.import_module(path)
            except ImportError:
                continue
            if hasattr(module, name):
                resolved[name] = getattr(module, name)
                break
        else:
            raise ImportError(
                f"could not find {name} in this gymnasium install; checkpoint statistics "
                "cannot be unpickled faithfully"
            )
    return resolved


def install():
    """Point every historical module path at the real classes. Returns the class map."""
    classes = _resolve()

    # Gymnasium >= 1.0 dropped the submodule these objects were pickled under. On 0.29 the
    # real module is already importable, so leave it alone rather than shadow it.
    if importlib.util.find_spec("gymnasium.wrappers.normalize") is None:
        shim = types.ModuleType("gymnasium.wrappers.normalize")
        for name, obj in classes.items():
            setattr(shim, name, obj)
        sys.modules["gymnasium.wrappers.normalize"] = shim

    # Pre-gymnasium runs pickled these objects under the OpenAI Gym module path.
    if "gym" not in sys.modules:
        gym_pkg = types.ModuleType("gym")
        gym_pkg.__path__ = []
        sys.modules["gym"] = gym_pkg
    for name in ("gym.wrappers", "gym.wrappers.normalize"):
        if name not in sys.modules:
            alias = types.ModuleType(name)
            alias.__path__ = []
            for attr, obj in classes.items():
                setattr(alias, attr, obj)
            sys.modules[name] = alias
    sys.modules["gym"].wrappers = sys.modules["gym.wrappers"]
    sys.modules["gym.wrappers"].normalize = sys.modules["gym.wrappers.normalize"]

    return classes


_classes = install()
RunningMeanStd = _classes["RunningMeanStd"]
NormalizeObservation = _classes["NormalizeObservation"]
NormalizeReward = _classes["NormalizeReward"]
