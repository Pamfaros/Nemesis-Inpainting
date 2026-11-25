import unittest

import torch

from model import ManhwaUNet, context, flat_fill, restore, sample, training_loss


class Contract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.rand(1, 3, 33, 49)
        self.m = torch.zeros_like(self.x[:, :1], dtype=torch.bool)
        self.m[..., 5:27, 10:31] = True

    def test_flat_gate(self):
        x = torch.full_like(self.x, .8)
        dirty = torch.where(self.m, self.x, x)
        self.assertTrue(torch.equal(flat_fill(dirty, self.m), x))
        self.assertIsNone(flat_fill(self.x, self.m))
        self.assertIsNone(flat_fill(self.x, torch.ones_like(self.m)))
        self.assertTrue(torch.equal(flat_fill(self.x, ~torch.ones_like(self.m)), self.x))

    def test_context_hidden_and_mask_edges(self):
        x = torch.full_like(self.x, .3)
        dirty = torch.where(self.m, self.x, x)
        self.assertTrue(torch.equal(context(x, self.m), context(dirty, self.m)))
        self.assertEqual(context(dirty, self.m)[:, -2:].abs().max().item(), 0.)

    def test_backprop_and_sampling(self):
        dirty = torch.where(self.m, 1 - self.x, self.x)
        for mode in ("rgb", "diffusion"):
            net = ManhwaUNet(mode)
            loss = training_loss(net, dirty, self.m, self.x)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(net.stem.weight.grad).all())
            self.assertGreater(net.stem.weight.grad.norm().item(), 0)
            for steps in ((4, 8, 16) if mode == "diffusion" else (1,)):
                y, branch = restore(net, self.x, self.m, steps, 27)
                self.assertEqual(branch, mode)
                self.assertTrue(torch.equal(y, sample(net, dirty, self.m, steps, 27)))
                self.assertTrue(torch.equal(y * ~self.m, self.x * ~self.m))
                self.assertTrue(torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
