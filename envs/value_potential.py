"""A frozen planner-value regressor, used as a potential over states during training.

The measured facts this exists to exploit: the MPC planner's *action* is not imitable - no regressor of
it from the 49-wide observation beats a constant, because the executed action is the argmin of a
stochastic search - while its *lookahead value* is: ridge predicts that scalar with R2 0.77 on held-out
episodes and with -0.0013 when restricted to the target components of the observation
(`bench_bc_teacher.py`). So the planner's usable product is a judgement about states, not a set of labels
over actions. Cloning it failed, and training a policy offline against it by Q-ascent diverged. What is
left is to put the judgement *inside* the online loop as a support signal.

The form is potential-based shaping: the learner sees `r + alpha * (gamma * Phi(s') - Phi(s))`. With a
state-only potential and the terminal state scored at zero, that term telescopes over an episode and
leaves the optimal policy of the MDP unchanged (Ng, Harada & Russell, 1999), so the arm answers "does the
planner's judgement make this task easier to learn?" instead of "did we quietly change the task?". Two
consequences are enforced here rather than assumed:

  * the shaping is applied to the reward the *trainer* reads. `RecordEpisodeStatistics` sits inside this
    wrapper, so `episode["r"]` - and every return scored by `eval_phase1.py`, which builds its own env -
    stays the published reward. A shaped arm is comparable to an unshaped one because both are graded on
    the same function;
  * the potential is frozen. Nothing here fits weights while the agent learns; the file is produced by
    `fit_planner_potential.py` and this module only evaluates it.

The value is fitted on states the *planner* visits, so on states a falling ragdoll reaches it is
extrapolation. Two guards follow from that: hidden activations are clipped by the same normalisation the
regressor was trained under, and the output is clamped to the range observed in the demonstrations,
because a potential that extrapolates to a huge number would dominate the task reward it is meant to
nudge.
"""

import json
import os

import gymnasium as gym
import numpy as np

from evaluate_merging import CLIP

# The arrays a potential file must carry besides its layers. Checked on load: a half-written or
# hand-edited npz would otherwise produce a potential that silently evaluates to something different
# from what was fitted. The layer keys (W0/B0, W1/B1, ...) are derived from the file, so the check does
# not have to know the architecture in advance.
BASE_KEYS = ("obs_mean", "obs_scale", "value_mean", "value_scale")


class ValuePotential:
    """Frozen MLP Phi(s) over the raw 49-wide observation, evaluated in numpy.

    Numpy rather than torch because this runs once per environment per step, inside every worker process
    of a vectorised environment: the weights travel as plain arrays, which also makes the object trivially
    picklable and keeps autograd out of the collection loop.
    """

    def __init__(self, layers, obs_mean, obs_scale, value_mean, value_scale, clamp, provenance=None):
        self.layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64))
                       for w, b in layers]
        self.obs_mean = np.asarray(obs_mean, dtype=np.float64)
        self.obs_scale = np.asarray(obs_scale, dtype=np.float64)
        self.value_mean = float(value_mean)
        self.value_scale = float(value_scale)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        self.obs_dim = int(self.obs_mean.shape[0])
        self.provenance = dict(provenance or {})

    def __call__(self, obs):
        """Phi of one raw observation, in the planner's own lookahead-value units."""
        return self.clamped(obs)[0]

    def clamped(self, obs):
        """(value, hit_low, hit_high): whether this state's value was truncated to the demo range.

        The third answer is what a coverage measurement counts. The regressor only ever saw states the
        planner visited, so a state whose value has to be clamped is a state the fit is extrapolating on,
        and the shaping there is an opinion the potential was never asked to have.
        """
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        if x.shape != self.obs_mean.shape:
            raise ValueError(f"Phi was fitted on {self.obs_mean.shape[0]}-wide observations and this "
                             f"state has {x.shape[0]}; the observation width follows task_phase, so a "
                             "potential fitted in one phase cannot score another")
        x = np.clip((x - self.obs_mean) / self.obs_scale, -CLIP, CLIP)
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if i + 1 < len(self.layers):
                x = np.tanh(x)
        raw = float(x.reshape(-1)[0]) * self.value_scale + self.value_mean
        return float(np.clip(raw, self.clamp[0], self.clamp[1])), raw < self.clamp[0], raw > self.clamp[1]

    @classmethod
    def load(cls, path):
        """Read a potential written by `fit_planner_potential.py`: arrays in the npz, provenance beside it."""
        with np.load(path) as handle:
            files = list(handle.files)
            width = [int(key[1:]) for key in files if key.startswith("W") and key[1:].isdigit()]
            n_layers = max(width) + 1 if width else 0
            missing = [f"{prefix}{i}" for i in range(n_layers)
                       for prefix in ("W", "B") if f"{prefix}{i}" not in files]
            missing += [key for key in BASE_KEYS if key not in files]
            if n_layers == 0 or missing:
                raise SystemExit(f"{path}: potential file carries {sorted(files)} and is missing "
                                 f"{missing or 'every layer'}; refit it with fit_planner_potential.py "
                                 "rather than editing it")
            layers = [(handle[f"W{i}"], handle[f"B{i}"]) for i in range(n_layers)]
            data = {key: handle[key] for key in BASE_KEYS}
        sidecar = path + ".json"
        meta = {}
        if os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as handle:
                meta = json.load(handle)
        clamp = meta.get("clamp_value") or [float("-inf"), float("inf")]
        potential = cls(layers, data["obs_mean"], data["obs_scale"], data["value_mean"],
                        data["value_scale"], clamp, provenance=meta)
        if meta.get("obs_dim") and meta["obs_dim"] != potential.obs_dim:
            raise SystemExit(f"{path}: sidecar says obs_dim={meta['obs_dim']} but the stored mean has "
                             f"{potential.obs_dim} entries")
        return potential


class PotentialShaping(gym.Wrapper):
    """Add `weight * (gamma * Phi(s') - Phi(s))` to the reward the trainer reads.

    `terminated` scores the successor potential as zero, which is the condition under which the shaping
    term telescopes and the optimal policy is preserved. `truncated` is not the end of the MDP - it is the
    TimeLimit - so the potential is carried across it instead of punished; the alternative would put a
    standing agent in debt every time the horizon runs out, which is a bias, not a signal.

    The wrapper sits outside `RecordEpisodeStatistics`, so the episode return it reports is the published
    reward and only the reward handed upward - the one that reaches the replay buffer - is shaped.
    """

    def __init__(self, env, potential, weight, gamma):
        super().__init__(env)
        if weight == 0.0:
            raise ValueError("weight 0 shapes nothing; build the environment without this wrapper")
        self.potential = potential
        self.weight = float(weight)
        self.gamma = float(gamma)
        self._prev = None
        self.last_shaping = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = self.potential(obs)
        successor = 0.0 if terminated else self.gamma * phi_next
        self.last_shaping = self.weight * (successor - self._prev)
        # After a done the vectorised environment resets this env and its `reset()` re-seeds `_prev`,
        # so carrying the terminal state's own potential forward is never read as a successor.
        self._prev = phi_next
        return obs, float(reward) + self.last_shaping, terminated, truncated, info
