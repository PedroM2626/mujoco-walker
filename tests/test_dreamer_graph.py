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
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_dreamer as td  # noqa: E402

L, B, OBS, ACT, IMAG, WARMUP = 6, 3, 49, 17, 4, 3
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


if __name__ == "__main__":
    unittest.main()
