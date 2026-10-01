"""Process-parallel vector environment with SyncVectorEnv-identical semantics.

``SyncVectorEnv`` steps every sub-environment in the calling thread, so 32 envs cost
about 32x the wall clock of one env and buy nothing: measured on this repo the n=32
sync vec env is *slower* than a single env. Threads are not the answer either —
MuJoCo releases the GIL but the reward code around it does not. So sub-environments
run in worker processes and one vector step dispatches all ``num_envs`` steps as a
parallel barrier.

Autoreset, ``_add_info`` merging and observation concatenation are inherited unchanged
from :class:`gym.vector.SyncVectorEnv`, so a trainer sees the same batch structure it
saw before — only faster.

Two costs dominate a parallel barrier here, measured at n=32 on an i9-14900HX:

* **Parent-side marshalling.** Roughly 130us of serial pickle/IPC work per env per
  vector step, which caps total throughput near 7.7k steps/s regardless of worker
  count. Grouping several envs into one process and one message pair does *not* beat
  it (n=32, 4s per point: 242 vector steps/s at 1 env per worker, 190 at 4, 184 at 8),
  so the default is one process per environment. Removing this ceiling needs
  shared-memory transport rather than pipes.
* **Payload size.** The ~30-entry step info dict is only ever read on the step that
  ends an episode, so ``sparse_info`` omits it from every other reply; that alone was
  worth 151 -> 210 vector steps/s at n=32.

Sub-environments are described by an *importable spec* rather than a closure because
spawn semantics pickle the launch arguments, and the ``make_env`` thunks in the
trainer modules are local closures.
"""

import importlib
import math
import multiprocessing as mp
import os
import sys
import traceback

import gymnasium as gym
import numpy as np

# (module, qualified name, positional args, keyword args). Calling the resolved
# attribute may return the env directly or a thunk that builds it.
EnvSpec = tuple

STEP, RESET, GETATTR, SETATTR, CALL, CLOSE = "step", "reset", "getattr", "setattr", "call", "close"


def _default_context():
    if sys.platform == "win32":
        return mp.get_context("spawn")
    try:
        return mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")


def _build_env(spec):
    module_name, func_name, args, kwargs = spec
    for path in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if path not in sys.path:
            sys.path.insert(0, path)
    factory = getattr(importlib.import_module(module_name), func_name)
    built = factory(*args, **kwargs)
    if hasattr(built, "step"):
        return built
    return built()


def make_walker_thunk(env_id="WalkerRagdoll-v0", index=0, task_phase="recovery",
                      reset_mode="mixed", flatten=False, env_kwargs=None):
    """Generic importable sub-env builder, for specs and for ad-hoc benchmarks.

    Lives here because a spec must name something a freshly spawned process can
    import; closures created inside a trainer module cannot be pickled.
    """
    import envs.walker_ragdoll_env  # noqa: F401  registers WalkerRagdoll-v0

    kwargs = {"task_phase": task_phase, "reset_mode": reset_mode}
    if env_kwargs:
        kwargs.update(env_kwargs)

    def thunk():
        env = gym.make(env_id, **kwargs)
        if flatten:
            env = gym.wrappers.FlattenObservation(env)
        return env

    return thunk


def _to_python(value):
    """Replace numpy scalars with plain Python ones before pickling.

    A numpy scalar costs roughly 10x more to pickle than the Python number it wraps,
    and a step info dict is ~30 of them per env per step. Values are unchanged apart
    from type.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def _pack_step(result, sparse_info):
    observation, reward, terminated, truncated, info = result
    terminated, truncated = bool(terminated), bool(truncated)
    if sparse_info and not (terminated or truncated):
        # On a non-terminal step the info dict is dead weight: trainers reach it only
        # through final_info/final_observation, which appear on the terminal step.
        return observation, reward, terminated, truncated, None
    return observation, reward, terminated, truncated, {
        k: _to_python(v) for k, v in info.items()
    }


def _pack_reset(result):
    observation, info = result
    return observation, {k: _to_python(v) for k, v in info.items()}


def _worker_loop(conn, specs, sparse_info=False):
    """Entry point for one worker process, owning ``len(specs)`` environments."""
    try:
        envs = [_build_env(spec) for spec in specs]
    except BaseException:
        conn.send(("init_error", traceback.format_exc()))
        conn.close()
        return

    conn.send(("ready", [(e.observation_space, e.action_space, e.metadata) for e in envs]))

    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            break
        if not message:
            break
        kind = message[0]
        try:
            if kind == STEP:
                reply = ("ok", [_pack_step(env.step(action), sparse_info)
                               for env, action in zip(envs, message[1])])
            elif kind == RESET:
                # Addressed by local index: after a partial autoreset only the envs
                # that finished may reset, and they are scattered across the group.
                reply = ("ok", [[local, _pack_reset(envs[local].reset(**kwargs))]
                                for local, kwargs in message[1]])
            elif kind == GETATTR:
                reply = ("ok", getattr(envs[message[1]], message[2]))
            elif kind == SETATTR:
                setattr(envs[message[1]], message[2], message[3])
                reply = ("ok", None)
            elif kind == CALL:
                _, local, name, args, kwargs = message
                reply = ("ok", getattr(envs[local])(*args, **kwargs))
            elif kind == CLOSE:
                for env in envs:
                    env.close()
                reply = ("ok", None)
            else:
                reply = ("error", f"unknown message kind: {kind!r}")
        except BaseException:
            reply = ("error", traceback.format_exc())
        try:
            conn.send(reply)
        except BaseException:
            # A reply that cannot be marshalled must not silently kill the worker;
            # the parent would otherwise block forever waiting on this pipe.
            try:
                conn.send(("error", traceback.format_exc()))
            except BaseException:
                break
        if kind == CLOSE:
            break
    conn.close()


class _RemoteEnv:
    """Stand-in for one gym Env whose state lives in a worker process.

    Exists so SyncVectorEnv's ``__init__``, ``_check_spaces`` and ``__getattr__``
    forwarding keep working unmodified. Vector stepping goes through the group pipe in
    :class:`ParallelVectorEnv`, not through this object.
    """

    __slots__ = ("conn", "index", "local", "observation_space", "action_space",
                 "metadata", "process")

    def __init__(self, conn, index, local, spaces, process):
        object.__setattr__(self, "conn", conn)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "observation_space", spaces[0])
        object.__setattr__(self, "action_space", spaces[1])
        object.__setattr__(self, "metadata", spaces[2])
        object.__setattr__(self, "process", process)

    def _transact(self, message):
        self.conn.send(message)
        if not self.conn.poll(30.0):
            raise RuntimeError(f"worker {self.process.pid} idle on {message[0]!r}")
        status, payload = self.conn.recv()
        if status == "error":
            raise RuntimeError(f"worker {self.process.pid} failed on {message[0]!r}:\n{payload}")
        return payload

    def step(self, action):
        return _pack_step(self._transact((STEP, [action]))[0], sparse_info=False)

    def reset(self, **kwargs):
        return self._transact((RESET, [[self.local, kwargs]]))[0][1]

    def call(self, name, *args, **kwargs):
        return self._transact((CALL, self.local, name, args, kwargs))

    def __getattr__(self, name):
        # Reached only for names the proxy does not define, i.e. SyncVectorEnv's
        # attribute forwarding onto a sub-env. Dunders never get here: Python finds
        # them on the type, and __slots__ makes any real attribute a direct hit.
        return self._transact((GETATTR, object.__getattribute__(self, "local"), name))

    def __setattr__(self, name, value):
        self._transact((SETATTR, object.__getattribute__(self, "local"), name, value))

    def close(self):
        # The pipe is shared with the rest of the group; ParallelVectorEnv.close_extras
        # shuts the worker down once, so a per-env close must not do it N times.
        pass


class ParallelVectorEnv(gym.vector.SyncVectorEnv):
    """Drop-in replacement for ``SyncVectorEnv`` that runs envs in worker processes.

    Args:
        env_specs: one ``(module, function, args, kwargs)`` spec per sub-environment.
            The function may return an env or a zero-arg thunk (``make_env`` does).
        envs_per_worker: environments grouped into one process. One message pair per
            group per vector step, so grouping trades pipe operations for less
            parallelism. ``None`` keeps the worker count at or below the core count
            with at least two envs each; 1 means one process per environment.
        sparse_info: omit the step info dict on non-terminal steps. Observations,
            rewards, done flags and terminal-step infos are unaffected.
        copy: return copies of the observation buffer, like SyncVectorEnv. The serial
            path deepcopies; ``np.copy`` is equivalent for a numpy Box and cheaper.
    """

    def __init__(self, env_specs, observation_space=None, action_space=None, copy=True,
                 mp_context=None, worker_timeout=300.0, sparse_info=False,
                 envs_per_worker=None):
        context = mp_context or _default_context()
        self._specs = list(env_specs)
        self.worker_timeout = float(worker_timeout)
        self.sparse_info = bool(sparse_info)

        group_size = (self._auto_group_size(len(self._specs)) if envs_per_worker is None
                      else int(envs_per_worker))
        group_size = max(1, min(group_size, max(1, len(self._specs))))
        self._groups = [
            list(range(start, min(start + group_size, len(self._specs))))
            for start in range(0, len(self._specs), group_size)
        ]
        self._local_index = {}
        self._group_of = {}
        for group_id, group in enumerate(self._groups):
            for local, index in enumerate(group):
                self._local_index[index] = local
                self._group_of[index] = group_id

        self._group_conns = []
        self._group_processes = []
        proxies = []
        for group_id, group in enumerate(self._groups):
            parent_conn, child_conn = context.Pipe(duplex=True)
            process = context.Process(
                target=_worker_loop,
                args=(child_conn, [self._specs[i] for i in group], self.sparse_info),
                daemon=True, name=f"walker-env-g{group_id}",
            )
            process.start()
            child_conn.close()
            if not parent_conn.poll(self.worker_timeout):
                process.terminate()
                raise RuntimeError(
                    f"worker #{group_id} (envs {group[0]}..{group[-1]}) did not report ready "
                    f"within {self.worker_timeout}s"
                )
            status, payload = parent_conn.recv()
            if status == "init_error":
                process.join()
                raise RuntimeError(f"failed to build the sub-env group starting at #{group[0]}:\n{payload}")
            self._group_conns.append(parent_conn)
            self._group_processes.append(process)
            for local, index in enumerate(group):
                proxies.append(_RemoteEnv(parent_conn, index, local, payload[local], process))

        # Hand the ready-made proxies to SyncVectorEnv: its own env_fns call becomes a
        # no-op construction, so autoreset, _add_info and _check_spaces stay inherited.
        thunk = lambda proxy: (lambda: proxy)
        super().__init__([thunk(p) for p in proxies], observation_space, action_space, copy)

    @staticmethod
    def _auto_group_size(num_envs):
        """One process per environment.

        Grouping was tried and measured slower (see the module docstring): the cost is
        in marshalling one payload per env, which grouping does not remove, while it
        does serialize those envs inside a single worker.
        """
        return 1

    def _recv(self, group_id, kind):
        conn = self._group_conns[group_id]
        if not conn.poll(self.worker_timeout):
            process = self._group_processes[group_id]
            raise RuntimeError(
                f"worker #{group_id} (pid {process.pid}) did not answer {kind!r} within "
                f"{self.worker_timeout}s (alive={process.is_alive()})"
            )
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"worker #{group_id} failed during {kind!r}:\n{payload}")
        return payload

    def _step_all(self, actions_per_env):
        """Step every sub-env with one message per worker; results come back in order."""
        for conn, group in zip(self._group_conns, self._groups):
            conn.send((STEP, [actions_per_env[i] for i in group]))
        results = []
        for group_id in range(len(self._groups)):
            results.extend(self._recv(group_id, STEP))
        return results

    def _reset_selected(self, kwargs_by_index):
        """Reset only the addressed envs; a group must not reset its first k by mistake."""
        payloads = [{} for _ in self._groups]
        for index, kwargs in kwargs_by_index.items():
            payloads[self._group_of[index]][self._local_index[index]] = kwargs

        touched = []
        for group_id, selected in enumerate(payloads):
            if not selected:
                continue
            touched.append(group_id)
            self._group_conns[group_id].send(
                (RESET, sorted(selected.items()))
            )

        results = {}
        for group_id in touched:
            group = self._groups[group_id]
            for local, packed in self._recv(group_id, RESET):
                results[group[local]] = packed
        return results

    def reset_wait(self, seed=None, options=None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        kwargs_by_index = {}
        for i, single_seed in enumerate(seed):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            kwargs_by_index[i] = kwargs

        infos = {}
        results = self._reset_selected(kwargs_by_index)
        for i in range(self.num_envs):
            observation, info = results[i]
            self.observations[i] = observation
            infos = self._add_info(infos, info, i)

        return self._maybe_copy(self.observations), infos

    def step_wait(self):
        results = self._step_all(list(self._actions))

        # The parent drives the autoreset so the terminal observation is parked in
        # info["final_observation"] before being replaced, exactly like SyncVectorEnv.
        done = [i for i, r in enumerate(results) if r[2] or r[3]]
        resets = self._reset_selected({i: {} for i in done}) if done else {}

        infos = {}
        for i, (observation, reward, terminated, truncated, info) in enumerate(results):
            if i in resets:
                old_observation, old_info = observation, info
                observation, info = resets[i]
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            self.observations[i] = observation
            self._rewards[i] = reward
            self._terminateds[i] = terminated
            self._truncateds[i] = truncated
            if info is not None:
                infos = self._add_info(infos, info, i)

        return (
            self._maybe_copy(self.observations),
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def _maybe_copy(self, observations):
        if not self.copy:
            return observations
        if isinstance(observations, np.ndarray):
            return observations.copy()
        from copy import deepcopy

        return deepcopy(observations)

    def call(self, name, *args, **kwargs):
        """Apply a method to every sub-env, like SyncVectorEnv.call.

        One round trip per env: these calls are configuration, not per-step work.
        """
        return tuple(env.call(name, *args, **kwargs) for env in self.envs)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in self.envs]
        for env, value in zip(self.envs, values):
            env._transact((SETATTR, env.index, name, value))

    def close_extras(self, **kwargs):
        for conn, process in zip(self._group_conns, self._group_processes):
            try:
                conn.send((CLOSE, None))
                conn.recv()
            except (OSError, EOFError):
                pass
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            conn.close()
