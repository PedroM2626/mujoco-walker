"""Would a JAX learner beat the CUDA-graph-captured torch update? Measure the loop, don't argue.

Item context. The physics question is already answered in this repository: MJX runs WalkerRagdoll
at 0.19x of `ParallelVectorEnv` even on GPU (dense collision pairs, RK4 ~4x), so porting the
*environment* to JAX is not the move. The open question is the *learner*: after capture, the
shipped Dreamer update costs 12.41 ms and about 3.93 ms of that is the 15-step imagination loop
(benchmarks/dreamer_update_scaling.json, measured on this same RTX 4070 Laptop under Windows). That
loop is a recurrence, so nothing can be stacked over time - but XLA can fuse the ~10 tiny kernels
each step launches, which is the one thing graph capture cannot do.

This script rebuilds that loop faithfully in JAX - actor -> sampled action -> GRU transition ->
sampled prior -> reward and continue heads, with the sampled z feeding the next step, exactly as
`RSSM.transition` and the imagination loop in `train_dreamer.py` do - runs it under `jit` over a
`lax.scan`, and times it. It also times a large matmul so the comparison has a floor: the measured
arithmetic throughput says how much of any loop time is work at all.

Run it where JAX has a CUDA device - native Windows jaxlib ships no CUDA wheel, so on this machine
that means WSL2 (the repo's `requirements-mjx.txt` environment):

    wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/mujoco-walker && \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 /root/venv-mjx/bin/python bench_jax_update.py'

The fraction is not cosmetic: on this 8 GB card XLA's default preallocation asks for 6 GiB and
fails outright (CUDA_ERROR_OUT_OF_MEMORY) because the desktop already holds a context, which is the
same constraint the MJX measurements recorded for `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`.

The torch side is not measured here (that venv has no torch); it is read from the scaling artifact,
and the two were taken on the same GPU on different hosts, so the comparison is a bound on what a
port could win, not a paired A/B.

Writes benchmarks/jax_rssm_imagination.json.
"""

import argparse
import functools
import json
import os
import statistics as st
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "benchmarks", "jax_rssm_imagination.json")

# The shipped Dreamer shapes: hidden 256, stochastic 32, action 17, trunk width 128.
HIDDEN, STOCH, ACTION, TRUNK = 256, 32, 17, 128
BATCH, HORIZON = 16, 15


def init_params(rng):
    """Random parameters with `nn.GRUCell`'s own split: input-to-gate, hidden-to-gate, two biases."""
    keys = list(jax.random.split(rng, 16))   # 9 MLP layers + the GRU's two matrices
    it = iter(keys)

    def layer(fan_in, fan_out):
        scale = 1.0 / np.sqrt(fan_in)
        return {"w": jax.random.normal(next(it), (fan_out, fan_in)) * scale,
                "b": jnp.zeros((fan_out,), dtype=jnp.float32)}

    gru_in = STOCH + ACTION
    return {
        "actor": [layer(HIDDEN + STOCH, TRUNK), layer(TRUNK, TRUNK), layer(TRUNK, ACTION * 2)],
        "gru": {"w_ih": jax.random.normal(next(it), (3 * HIDDEN, gru_in)),
                "w_hh": jax.random.normal(next(it), (3 * HIDDEN, HIDDEN)),
                "b_ih": jnp.zeros((3 * HIDDEN,)), "b_hh": jnp.zeros((3 * HIDDEN,))},
        "prior": [layer(HIDDEN, TRUNK), layer(TRUNK, STOCH * 2)],
        "reward": [layer(HIDDEN + STOCH, TRUNK), layer(TRUNK, 1)],
        "continue": [layer(HIDDEN + STOCH, TRUNK), layer(TRUNK, 1)],
    }


def mlp(params, x):
    """ELU on every layer but the last, exactly as the Sequential blocks in train_dreamer.py."""
    for layer in params[:-1]:
        x = jax.nn.elu(jnp.add(jnp.dot(x, layer["w"].T), layer["b"]))
    last = params[-1]
    return jnp.add(jnp.dot(x, last["w"].T), last["b"])


def gru_cell(params, x, h):
    """torch's GRUCell formulation, gate for gate: n sees r applied to the hidden term only."""
    xo = jnp.add(jnp.dot(x, params["w_ih"].T), params["b_ih"])
    ho = jnp.add(jnp.dot(h, params["w_hh"].T), params["b_hh"])
    gates = xo + ho
    r, z_gate, n_pre = jnp.split(gates, 3, axis=-1)
    r, z_gate = jax.nn.sigmoid(r), jax.nn.sigmoid(z_gate)
    n = jnp.tanh(n_pre + r * ho[:, (2 * HIDDEN):])
    return (1.0 - z_gate) * n + z_gate * h


def imagination_step(carry, _, params):
    """One rollout step, matching the loop in train_dreamer.dreamer_update."""
    h, z, key = carry
    mean, log_std = jnp.split(mlp(params["actor"], jnp.concatenate([h, z], axis=-1)), 2, axis=-1)
    key, k1 = jax.random.split(key)
    action = jnp.tanh(mean + jax.random.normal(k1, mean.shape) * jnp.exp(log_std))
    h = gru_cell(params["gru"], jnp.concatenate([z, action], axis=-1), h)
    key, k2 = jax.random.split(key)
    p_mean, p_log_std = jnp.split(mlp(params["prior"], h), 2, axis=-1)
    z = p_mean + jax.random.normal(k2, p_mean.shape) * (jax.nn.softplus(p_log_std) + 0.1)
    state = jnp.concatenate([h, z], axis=-1)
    r = mlp(params["reward"], state)
    c = jax.nn.sigmoid(mlp(params["continue"], state))
    return (h, z, key), (r, c)


def build_run(params, horizon):
    def run(h0, z0, key):
        step = functools.partial(imagination_step, params=params)
        (h, z, _), (r, c) = jax.lax.scan(step, (h0, z0, key), None, length=horizon)
        # Return every branch, tied into one array: an output the XLA dead-code eliminator can drop
        # would make the timing optimistic, and the reward/continue heads are the point.
        return jnp.concatenate([h.reshape(-1), z.reshape(-1), r.reshape(-1), c.reshape(-1)])
    return jax.jit(run)


def flops_per_step(params, batch):
    """2 * MACs over every weight, counting the GRU's input-to-gate and h-to-gate products."""
    total = 0
    for head in ("actor", "prior", "reward", "continue"):
        for layer in params[head]:
            total += layer["w"].size
    total += params["gru"]["w_ih"].size + params["gru"]["w_hh"].size
    return 2 * total * batch


def build_grad_run(params, horizon):
    """The same loop differentiated - the half a Dreamer update actually spends its time on.

    A scalar loss over the imagined rewards stands in for the actor's return objective; what
    matters for the bound is that the gradient flows back through every step of the recurrence.
    """
    def loss_fn(p, h0, z0, key):
        step = functools.partial(imagination_step, params=p)
        _, (r, c) = jax.lax.scan(step, (h0, z0, key), None, length=horizon)
        return jnp.sum(r) + jnp.sum(c)
    return jax.jit(jax.value_and_grad(loss_fn, argnums=0))


def time_gpu_flops(dtype=jnp.float32, n=4096, reps=10):
    """Achieved throughput on a square matmul - the floor the loop is measured against."""
    a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=dtype)
    b = jax.random.normal(jax.random.PRNGKey(1), (n, n), dtype=dtype)
    f = jax.jit(lambda x, y: x @ y)
    f(a, b).block_until_ready()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(a, b).block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e3)
    ms = st.median(samples)
    return 2 * n ** 3 / (ms / 1e3) / 1e12, ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--allow-cpu", action="store_true",
                   help="run even if JAX has no CUDA device (the artifact then is not comparable "
                        "to the committed one)")
    p.add_argument("--out", default=OUT, help="where to write the measurement; the default is the "
                                              "committed artifact, so a re-run under a different jax "
                                              "or host should name its own file")
    args = p.parse_args()

    devices = jax.devices()
    print("jax", jax.__version__, "devices", devices)
    # jax names its CUDA platform "gpu" from 0.11, and "cuda" in the 0.10 build that produced the
    # committed artifact; both mean the device is the RTX, and anything else means the numbers would
    # be CPU ones written over a GPU measurement.
    if devices[0].platform not in ("cuda", "gpu") and not args.allow_cpu:
        raise SystemExit(
            f"the first JAX device is {devices[0].platform}, not cuda: this would overwrite "
            "benchmarks/jax_rssm_imagination.json - a GPU measurement - with CPU numbers. Run it "
            "under WSL2 (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 /root/venv-mjx/bin/python "
            "bench_jax_update.py`) or pass --allow-cpu if that is really what you want.")
    tflops, matmul_ms = time_gpu_flops()
    print(f"fp32 matmul floor: {tflops:.1f} TFLOP/s ({matmul_ms:.2f} ms for {4096}^3)")

    params = init_params(jax.random.PRNGKey(0))
    run = build_run(params, args.horizon)
    h = jnp.zeros((args.batch, HIDDEN))
    z = jnp.zeros((args.batch, STOCH))
    key = jax.random.PRNGKey(7)
    run(h, z, key).block_until_ready()          # compile + warm
    samples = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        run(h, z, key).block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e3)
    median = st.median(samples)

    grad_run = build_grad_run(params, args.horizon)
    val, grads = grad_run(params, h, z, key)      # compile the backward pass
    jax.block_until_ready([val, *jax.tree_util.tree_leaves(grads)])
    grad_samples = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        val, grads = grad_run(params, h, z, key)
        jax.block_until_ready([val, *jax.tree_util.tree_leaves(grads)])
        grad_samples.append((time.perf_counter() - t0) * 1e3)
    grad_median = st.median(grad_samples)
    flops = flops_per_step(params, args.batch) * args.horizon
    arithmetic_ms = flops / (tflops * 1e12) * 1e3

    scaling_path = os.path.join(ROOT, "benchmarks", "dreamer_update_scaling.json")
    torch_captured = None
    if os.path.exists(scaling_path):
        with open(scaling_path, encoding="utf-8") as handle:
            marg = json.load(handle)["marginal_ms_per_unit"]
        torch_captured = {"per_imag_step_ms": marg["imag_horizon_captured"],
                          "loop_ms_at_15_steps": round(marg["imag_horizon_captured"] * args.horizon, 3)}

    payload = {
        "protocol": f"jax {jax.__version__} on {devices[0]}, imagination loop only, "
                    f"batch {args.batch}, horizon {args.horizon}, median of {args.reps} jit calls",
        "jax_loop_ms": round(median, 3),
        "jax_ms_per_step": round(median / args.horizon, 4),
        "jax_forward_backward_ms": round(grad_median, 3),
        "jax_forward_backward_per_step": round(grad_median / args.horizon, 4),
        "gpu_fp32_tflops_measured": round(tflops, 1),
        "matmul_4096_ms": round(matmul_ms, 3),
        "loop_flops": flops,
        "arithmetic_floor_ms": round(arithmetic_ms, 4),
        "fraction_of_time_that_is_arithmetic": round(arithmetic_ms / median, 4),
        "torch_captured_for_reference": torch_captured,
        "torch_source": "benchmarks/dreamer_update_scaling.json, captured path, measured on the "
                        "same RTX 4070 Laptop under Windows. Different host, so this is a bound on "
                        "what a port could win, not a paired A/B.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    print("wrote", os.path.relpath(args.out, ROOT).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
