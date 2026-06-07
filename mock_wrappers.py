import sys
import types
import gymnasium.wrappers

class RunningMeanStd:
    pass

class NormalizeObservation:
    pass

mock_n = types.ModuleType('gym.wrappers.normalize')
mock_n.RunningMeanStd = RunningMeanStd
mock_n.NormalizeObservation = NormalizeObservation

# Inject into legacy gym
sys.modules['gym.wrappers'] = types.ModuleType('gym.wrappers')
sys.modules['gym.wrappers.normalize'] = mock_n

# DO NOT overwrite gymnasium.wrappers, just append normalize to it
sys.modules['gymnasium.wrappers.normalize'] = mock_n
gymnasium.wrappers.normalize = mock_n
