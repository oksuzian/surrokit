import unittest

import torch

from surrokit.sampling import emit_picks, sobol_cold_start

BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 10.0]], dtype=torch.float64)


class TestSampling(unittest.TestCase):
    def test_cold_start_deterministic_and_in_bounds(self):
        a = sobol_cold_start(BOUNDS, q=4, seed=7)
        b = sobol_cold_start(BOUNDS, q=4, seed=7)
        c = sobol_cold_start(BOUNDS, q=4, seed=8)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(a.shape, (4, 2))
        self.assertTrue((a >= BOUNDS[0]).all() and (a <= BOUNDS[1]).all())

    def test_emit_picks_native_types_and_int_rounding(self):
        cands = torch.tensor([[0.25, 3.6], [0.75, 7.4]], dtype=torch.float64)
        out = emit_picks(cands, int_dims=[1])
        self.assertEqual(out, [[0.25, 4], [0.75, 7]])
        self.assertIsInstance(out[0][0], float)
        self.assertIsInstance(out[0][1], int)

    def test_emit_picks_no_int_dims(self):
        cands = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        self.assertEqual(emit_picks(cands, int_dims=[]), [[0.5, 1.5]])
