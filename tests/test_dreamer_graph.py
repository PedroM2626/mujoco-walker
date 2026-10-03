"""The captured Dreamer update has to be the same function as the eager one it replaces.

`--update-graph` records one whole update as a CUDA graph and replays it, because the update
dispatches ~14,400 aten calls (benchmarks/dreamer_update_profile.json) through six 256-unit MLPs
and spends its time in Python rather than in arithmetic. Capture is not free of obligations - the
optimizer changes configuration, gradients are cleared in place, the starting latents come from the
caller - so this file pins what they are worth:

* the extracted `dreamer_update` reproduces a recorded loss vector on CPU (the refactor gate);
* on CUDA, one captured update lands on the eager update's numbers, and the *control* arm - the
  same update eagerly with the capture-only Adam configuration - shows that the residual difference
  belongs to that configuration and not to capture;
* replays refresh their noise, which is the failure mode that would leave the model imagining the
  same trajectory forever;
* and every parameter receives a gradient, which is what makes `zero_grad(set_to_none=False)` safe.
"""

import contextlib
import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_dreamer as td  # noqa: E402

L, B, OBS, ACT, IMAG, WARMUP = 6, 3, 49, 17, 4, 3


class _OpaqueRMS:
    """Stand-in for the running-mean object a real Dreamer checkpoint pickles: no tensor, so a
    weights-only load rejects the whole file - which is the point of the test that builds it."""

    def __init__(self):
        self.count = 1
GOLDEN = {  # dreamer_update on CPU, frozen noise, seeds 0/1/2 below
    "rec_loss": 0.4104127287864685, "reward_loss": 1.271732211112976,
    "continue_loss": 0.6954019069671631, "kl_loss": 0.42284923791885376,
    "actor_loss": 0.15299589931964874, "critic_loss": 0.019534481689333916,
}


class _Args:
    seq_len, batch_size, imag_horizon, gamma, kl_weight = L, B, IMAG, 0.99, 1.0


@contextlib.contextmanager
def frozen_noise():
    """Every z in this model is mean + randn_like(mean) * std, so this makes the update exact."""
    real = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    try:
        yield
    finally:
        torch.randn_like = real


def build(device, graph=False):
    torch.manual_seed(0)
    mods = [td.WorldModel(OBS, ACT).to(device), td.DreamerActor(256, 32, ACT).to(device),
            td.DreamerCritic(256, 32).to(device)]
    return mods + [td.make_adam(m.parameters(), 3e-4, graph) for m in mods]


def inputs(device):
    torch.manual_seed(1)
    batch = (torch.randn(L, B, OBS, device=device), torch.randn(L - 1, B, ACT, device=device),
             torch.randn(L - 1, B, 1, device=device),
             torch.zeros(L - 1, B, 1, device=device, dtype=torch.bool))
    torch.manual_seed(2)
    return batch, torch.randperm(L * B)[:B].to(device)


def params(arm):
    return [p for mod in arm[:3] for p in mod.parameters()]


def one_update(arm, batch, idx):
    with frozen_noise():
        return td.dreamer_update(*arm, batch, _Args(), idx)


def reset_adam(arm):
    for opt in arm[3:]:
        for p in [q for g in opt.param_groups for q in g["params"]]:
            for v in opt.state.get(p, {}).values():
                if torch.is_tensor(v):
                    v.zero_()


class TestDreamerUpdateFunction(unittest.TestCase):
    """The extraction out of the training loop must not have changed the arithmetic."""

    def test_update_reproduces_the_recorded_losses(self):
        arm = build(torch.device("cpu"))
        batch, idx = inputs(torch.device("cpu"))
        losses = one_update(arm, batch, idx)
        for key, want in GOLDEN.items():
            self.assertAlmostEqual(float(losses[key]), want, places=6, msg=f"{key} drifted")

    def test_every_parameter_gets_a_gradient(self):
        """The precondition for clearing gradients in place instead of setting them to None."""
        arm = build(torch.device("cpu"))
        batch, idx = inputs(torch.device("cpu"))
        one_update(arm, batch, idx)
        for mod, name in zip(arm[:3], ("model", "actor", "critic")):
            for pname, p in mod.named_parameters():
                self.assertIsNotNone(p.grad, f"{name}.{pname} has no gradient")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph capture needs a CUDA device")
class TestCapturedDreamerUpdate(unittest.TestCase):
    def setUp(self):
        self.dev = torch.device("cuda")
        self.batch, self.idx = inputs(self.dev)
        self.eager = build(self.dev)

    def capture(self, arm):
        call = lambda: td.dreamer_update(*arm, self.batch, _Args(), self.idx,
                                         zero_grad_set_to_none=False)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP):
                with frozen_noise():
                    call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with frozen_noise():
                outputs = call()
        return graph, outputs

    def test_capture_matches_the_eager_update(self):
        graph_arm = build(self.dev, graph=True)
        init = [p.detach().clone() for p in params(graph_arm)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(init, params(self.eager))),
                        "the two arms did not start from the same weights")
        graph, outputs = self.capture(graph_arm)
        for p, q in zip(params(graph_arm), init):
            p.data.copy_(q)
        reset_adam(graph_arm)
        graph.replay()
        torch.cuda.synchronize()

        with frozen_noise():
            eager_losses = td.dreamer_update(*self.eager, self.batch, _Args(), self.idx)
        got = [float(v) for v in outputs.values()]
        want = [float(v) for v in eager_losses.values()]
        # Control: the same update eagerly, with only the capture-only Adam configuration changed.
        control = build(self.dev, graph=True)
        with frozen_noise():
            control_losses = td.dreamer_update(*control, self.batch, _Args(), self.idx)
        cw = [float(v) for v in control_losses.values()]

        loss_gap = max(abs(a - b) for a, b in zip(got, want))
        control_gap = max(abs(a - b) for a, b in zip(cw, want))
        param_gap = max((a - b).abs().max().item() for a, b in
                        zip(params(graph_arm), params(self.eager)))
        self.assertLess(loss_gap, 1e-5, f"captured losses drifted {loss_gap:.2e}")
        self.assertLess(param_gap, 1e-4, f"captured weights drifted {param_gap:.2e}")
        # The point of the control: capture itself adds nothing beyond what the Adam configuration
        # already costs. If this ever fails, the graph - not the optimizer - changed the numbers.
        self.assertLess(loss_gap, 4 * max(control_gap, 1e-9),
                        f"graph gap {loss_gap:.2e} exceeds the Adam-config control {control_gap:.2e}")

    def test_replays_do_not_freeze_the_noise(self):
        graph_arm = build(self.dev, graph=True)
        graph, outputs = self.capture(graph_arm)
        seen = []
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            seen.append(float(outputs["actor_loss"]))
        self.assertEqual(3, len(set(seen)), f"three replays produced the same loss {seen}: "
                                            "the captured RNG is not advancing")


class TestUpdateGraphResolution(unittest.TestCase):
    """Which update a run gets is now a three-way decision, and the default changed.

    Unset `--update-graph` means capture on CUDA and eager elsewhere. That flip is the measured
    8.4x on an isolated update (benchmarks/dreamer_update_scaling.json), but it would have silently
    stranded every checkpoint written by the eager path, because Adam's state is laid out
    differently - hence the checkpoint's own configuration outranks the device.
    """

    def test_an_explicit_flag_outranks_the_device_and_the_checkpoint(self):
        self.assertIs(True, td.resolve_update_graph(True, "cuda", False))
        self.assertIs(False, td.resolve_update_graph(False, "cuda", True))
        self.assertIs(False, td.resolve_update_graph(False, "cpu", None))

    def test_forcing_capture_on_a_device_that_cannot_replay_is_refused(self):
        with self.assertRaises(SystemExit):
            td.resolve_update_graph(True, "cpu")

    def test_a_resumed_checkpoint_decides_when_nothing_was_asked(self):
        self.assertIs(True, td.resolve_update_graph(None, "cpu", True))
        self.assertIs(False, td.resolve_update_graph(None, "cuda", False))

    def test_unasked_and_unresumed_follows_the_device(self):
        self.assertIs(True, td.resolve_update_graph(None, "cuda"))
        self.assertIs(False, td.resolve_update_graph(None, "cpu"))
        self.assertIs(False, td.resolve_update_graph(None, "mps"))

    def test_the_mode_is_readable_only_by_the_load_real_checkpoints_need(self):
        """Why the update mode is settled by the trainer's own load, not by a safe peek.

        These checkpoints carry `obs_rms` and `rng_state` objects, so `weights_only=True` refuses
        the file outright - a safe read would report "unknown" for every checkpoint that exists and
        the inheritance would never fire. This pins both halves: the mode is readable, and the safe
        variant genuinely cannot be used.
        """
        for graph, want in ((False, False), (True, True)):
            arm = build(torch.device("cpu"), graph=graph)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "dreamer_ckpt_20.pt")
                torch.save({"model_opt_state_dict": arm[3].state_dict(),
                            "obs_rms": _OpaqueRMS(), "global_step": 20}, path)
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                mode = bool(loaded["model_opt_state_dict"]["param_groups"][0].get("capturable"))
                self.assertIs(want, mode, f"a {want} checkpoint did not read back as {want}")
                with self.assertRaises(Exception):
                    torch.load(path, map_location="cpu", weights_only=True)

    def test_latest_checkpoint_picks_the_highest_step_and_survives_an_absent_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(td.latest_dreamer_checkpoint(d))
            for step in (100, 9_000, 2_000):
                open(os.path.join(d, f"dreamer_ckpt_{step}.pt"), "wb").close()
            open(os.path.join(d, "sac_ckpt_99999.pt"), "wb").close()
            self.assertEqual(os.path.join(d, "dreamer_ckpt_9000.pt"),
                             td.latest_dreamer_checkpoint(d))


class TestBenchArmsNameTheirMode(unittest.TestCase):
    """`bench_dreamer_update.py` cannot be allowed to measure the default by accident.

    Capture became the CUDA default, so an arm that passes nothing now runs captured. The A/B, the
    profile and the reproducibility run all publish eager-vs-captured claims, and every one of them
    would have turned into two identical arms without saying so.
    """

    def setUp(self):
        import bench_dreamer_update as bench
        self.bench = bench

    def test_every_ab_arm_states_which_update_it_measures(self):
        for label, tag, extra in self.bench.AB_ARMS:
            flags = [f for f in extra if f in ("--update-graph", "--no-update-graph")]
            self.assertEqual(1, len(flags), f"arm {label!r} does not name its update mode: {extra}")
        eager = [e for l, t, e in self.bench.AB_ARMS if l == "eager"]
        graph = [e for l, t, e in self.bench.AB_ARMS if l == "graph"]
        self.assertNotEqual(sorted(map(str, eager)), sorted(map(str, graph)),
                            "the two arms would run the same code and report a speedup of nothing")

    def test_the_profile_and_reproducibility_arms_stay_on_the_path_they_publish(self):
        src = open(os.path.join(ROOT, "bench_dreamer_update.py"), encoding="utf-8").read()
        self.assertIn("--no-update-graph", src[src.index("def run_profile"):],
                      "the dispatch-count profile would silently start measuring the graph")
        self.assertIn('run_trainer("bdr_a", args, EAGER)', src[src.index("def reproducibility"):],
                      "the reproducibility artifact says CUDA-graph-free; keep it that way")


class TestFusedImaginationHeads(unittest.TestCase):
    """`fused_heads` must be the same two networks, not a faster different function.

    The reward and continue heads read the same `[h, z]` once per imagination step, so they are the
    only launch-level win left after graph capture. That is worth doing only if it is exactly
    equivalent, so this pins the difference, the gradient path and the checkpoint format.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.model = td.WorldModel(OBS, ACT)
        self.x = torch.randn(64, 256 + 32)
        self.w1, self.b1, self.w2, self.b2 = td.fused_heads(self.model.reward_net,
                                                            self.model.continue_net)

    def fused(self, x):
        import torch.nn.functional as F
        return F.linear(F.elu(F.linear(x, self.w1, self.b1)), self.w2, self.b2)

    def test_the_two_heads_come_back_within_float_reassociation(self):
        heads = self.fused(self.x)
        r_diff = (heads[:, 0:1] - self.model.reward_net(self.x)).abs().max().item()
        # The continue head ends in a Sigmoid; the fused pass returns the logit, so compare after it.
        c_diff = (torch.sigmoid(heads[:, 1:2]) - self.model.continue_net(self.x)).abs().max().item()
        self.assertLess(r_diff, 1e-6, f"reward differs by {r_diff}")
        self.assertLess(c_diff, 1e-6, f"continue probability differs by {c_diff}")

    def test_gradients_land_on_the_original_parameters(self):
        self.fused(self.x).sum().backward()
        for name, mod in (("reward", self.model.reward_net), ("continue", self.model.continue_net)):
            for i, p in enumerate(mod.parameters()):
                self.assertIsNotNone(p.grad, f"{name}[{i}] received no gradient through the fused pass")
                self.assertGreater(float(p.grad.abs().sum()), 0.0, f"{name}[{i}] got a zero gradient")

    def test_no_parameters_are_introduced(self):
        # The checkpoint format is the reason the fusion is a function and not a module.
        keys = set(self.model.state_dict())
        self.assertTrue(all("." in k for k in keys))
        self.assertEqual(0, len([k for k in keys if "fused" in k]))
        self.assertIn("reward_net.0.weight", keys)
        self.assertIn("continue_net.2.weight", keys)

    def test_the_parameter_gradients_agree_with_the_separate_heads(self):
        """Forward equality is not enough: the optimizer consumes these gradients, not the outputs."""
        import torch.nn.functional as F
        x = self.x.clone()
        params = (list(self.model.reward_net.parameters())
                  + list(self.model.continue_net.parameters()))

        (self.model.reward_net(x).sum() + self.model.continue_net(x).sum()).backward()
        separate = [p.grad.detach().clone() for p in params]

        self.model.zero_grad(set_to_none=True)
        w1, b1, w2, b2 = td.fused_heads(self.model.reward_net, self.model.continue_net)
        heads = F.linear(F.elu(F.linear(x, w1, b1)), w2, b2)
        # Same objective on both sides: reward is linear in the fused output, continue is not -
        # the module's last layer is a Sigmoid, so the fused logit has to go through it first.
        (heads[:, 0:1].sum() + torch.sigmoid(heads[:, 1:2]).sum()).backward()

        worst = max((p.grad - s).abs().max().item() / max(s.abs().max().item(), 1e-12)
                    for p, s in zip(params, separate))
        self.assertLess(worst, 1e-5, f"a fused gradient diverges from the separate one by {worst:.3e}")


    def test_the_whole_update_agrees_between_the_two_implementations(self):
        """Not just the heads: one full `dreamer_update` per implementation, same weights, same batch."""
        dev = torch.device("cpu")
        batch, idx = inputs(dev)
        got = {}
        for impl in ("separate", "fused"):
            arm = build(dev)
            a = _Args()
            a.heads_impl = impl
            with frozen_noise():
                got[impl] = {k: float(v) for k, v in td.dreamer_update(*arm, batch, a, idx).items()}
        for key in got["separate"]:
            base = max(abs(got["separate"][key]), 1e-12)
            rel = abs(got["separate"][key] - got["fused"][key]) / base
            self.assertLess(rel, 1e-5,
                            f"{key}: separate {got['separate'][key]!r} vs fused "
                            f"{got['fused'][key]!r} (relative {rel:.3e})")


if __name__ == "__main__":
    unittest.main()
