import unittest

import torch

from surrokit import Constraint, InfeasibleError
from surrokit.pickers import _constrained_max

BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 10.0]], dtype=torch.float64)


class FakePost:
    def __init__(self, mean, var):
        self.mean = mean
        self.variance = var


class RampModel:
    """mean0 = x0 (objective), mean1 = x1 (constrained axis), tiny sigma."""
    def posterior(self, X):
        mean = torch.stack([X[:, 0], X[:, 1]], dim=-1)
        var = torch.full_like(mean, 1e-6)
        return FakePost(mean, var)


class InfeasibleModel:
    """Constrained axis pinned far below any threshold."""
    def posterior(self, X):
        mean = torch.stack([X[:, 0], torch.full_like(X[:, 1], -100.0)], dim=-1)
        var = torch.full_like(mean, 1e-6)
        return FakePost(mean, var)


class TestConstrainedMax(unittest.TestCase):
    def test_feasible_picks_rank_by_axis0(self):
        c = Constraint(axis=1, min=5.0, k_sigma=1.0)
        picks = _constrained_max(RampModel(), BOUNDS, q=3, seed=0,
                                 pending=None, constraint=c)
        self.assertEqual(picks.shape, (3, 2))
        # Every pick satisfies the constraint (mean1 = x1 >= 5).
        self.assertTrue((picks[:, 1] >= 5.0 - 1e-6).all())
        # Ranked by axis-0 mean (= x0), descending -- top pick has largest x0.
        self.assertGreaterEqual(float(picks[0, 0]), float(picks[-1, 0]))

    def test_min_spacing_enforced(self):
        c = Constraint(axis=1, min=0.0, k_sigma=0.0)
        picks = _constrained_max(RampModel(), BOUNDS, q=4, seed=0,
                                 pending=None, constraint=c, min_spacing=0.2)
        norm = (picks - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                d = float((norm[i] - norm[j]).pow(2).sum().sqrt())
                self.assertGreaterEqual(d, 0.2 - 1e-9)

    def test_infeasible_raises(self):
        c = Constraint(axis=1, min=5.0, k_sigma=1.0)
        with self.assertRaises(InfeasibleError):
            _constrained_max(InfeasibleModel(), BOUNDS, q=2, seed=0,
                             pending=None, constraint=c)

    def test_k_ladder_relaxes_before_failing(self):
        # Threshold sits so that k=1 excludes everything but k=0 passes:
        # mean1 in [0, 10], sigma1 = 3 -> mean-1*sigma max = 7 < 9.5,
        # but mean max = 10 >= 9.5.
        class WideSigma:
            def posterior(self, X):
                mean = torch.stack([X[:, 0], X[:, 1]], dim=-1)
                var = torch.stack([torch.full_like(X[:, 0], 1e-6),
                                   torch.full_like(X[:, 1], 9.0)], dim=-1)
                return FakePost(mean, var)
        c = Constraint(axis=1, min=9.5, k_sigma=1.0)
        picks = _constrained_max(WideSigma(), BOUNDS, q=1, seed=0,
                                 pending=None, constraint=c)
        self.assertEqual(picks.shape, (1, 2))

    def test_deterministic_per_seed(self):
        c = Constraint(axis=1, min=2.0, k_sigma=1.0)
        a = _constrained_max(RampModel(), BOUNDS, q=2, seed=3,
                             pending=None, constraint=c)
        b = _constrained_max(RampModel(), BOUNDS, q=2, seed=3,
                             pending=None, constraint=c)
        self.assertTrue(torch.equal(a, b))
