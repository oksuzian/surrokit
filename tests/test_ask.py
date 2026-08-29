import unittest

import torch

from surrokit import Constraint, Problem, ask
from surrokit.pickers import emit_picks, sobol_cold_start
from tests.common import PROB, PROB_C, history, in_bounds


class TestAskColdStart(unittest.TestCase):
    def test_cold_start_below_two_rows(self):
        for X, Y in (([], []), ([[0.5, 5.0]], [[1.0, 2.0]])):
            picks = ask(PROB, X, Y, q=3, picker="hybrid", seed=1)
            self.assertEqual(len(picks), 3)
            for row in picks:
                self.assertTrue(in_bounds(row))
                self.assertIsInstance(row[1], int)  # int_dims rounded

    def test_cold_start_deterministic(self):
        a = ask(PROB, [], [], q=2, seed=5)
        b = ask(PROB, [], [], q=2, seed=5)
        self.assertEqual(a, b)


class TestAskValidation(unittest.TestCase):
    def test_unknown_picker(self):
        X, Y = history(10)
        with self.assertRaises(ValueError):
            ask(PROB, X, Y, picker="nope")

    def test_multiobj_pickers_require_m2(self):
        X, Y = history(10)
        Y1 = [[r[0]] for r in Y]
        for p in ("qnehvi", "qnparego", "hybrid"):
            with self.assertRaises(ValueError):
                ask(PROB, X, Y1, picker=p)

    def test_constrained_max_requires_constraint(self):
        X, Y = history(10)
        with self.assertRaises(ValueError):
            ask(PROB, X, Y, picker="constrained_max")
        bad = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0),
                      constraint=Constraint(axis=2, min=0.0))
        with self.assertRaises(ValueError):
            ask(bad, X, Y, picker="constrained_max")

    def test_pending_dim_mismatch(self):
        X, Y = history(10)
        with self.assertRaises(ValueError):
            ask(PROB, X, Y, pending=[[0.5]])


class TestAskPickers(unittest.TestCase):
    def _smoke(self, picker, **kw):
        X, Y = history(10)
        picks = ask(PROB if picker != "constrained_max" else PROB_C,
                    X, Y, q=2, picker=picker, seed=0, **kw)
        self.assertEqual(len(picks), 2)
        for row in picks:
            self.assertTrue(in_bounds(row))
            self.assertIsInstance(row[1], int)
        return picks

    def test_qnehvi(self):
        self._smoke("qnehvi")

    def test_qlnei_accepts_1d_y(self):
        X, Y = history(10)
        Y1 = [[r[0]] for r in Y]
        picks = ask(PROB, X, Y1, q=2, picker="qlnei", seed=0)
        self.assertEqual(len(picks), 2)

    def test_qnparego(self):
        self._smoke("qnparego")

    def test_hybrid_and_hv_frac_zero_is_pure_parego(self):
        self._smoke("hybrid")
        X, Y = history(10)
        pure = ask(PROB, X, Y, q=2, picker="hybrid", seed=0, hv_frac=0.0)
        parego = ask(PROB, X, Y, q=2, picker="qnparego", seed=0)
        self.assertEqual(pure, parego)

    def test_constrained_max_through_ask(self):
        self._smoke("constrained_max")

    def test_seed_determinism_qlnei(self):
        X, Y = history(10)
        Y1 = [[r[0]] for r in Y]
        a = ask(PROB, X, Y1, q=2, picker="qlnei", seed=9)
        b = ask(PROB, X, Y1, q=2, picker="qlnei", seed=9)
        self.assertEqual(a, b)

    def test_pending_accepted(self):
        X, Y = history(10)
        picks = ask(PROB, X, Y, q=2, picker="qnehvi", seed=0,
                    pending=[[0.5, 5.0]])
        self.assertEqual(len(picks), 2)


class TestSamplingHelpers(unittest.TestCase):
    BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 10.0]], dtype=torch.float64)

    def test_cold_start_deterministic_and_in_bounds(self):
        a = sobol_cold_start(self.BOUNDS, q=4, seed=7)
        b = sobol_cold_start(self.BOUNDS, q=4, seed=7)
        c = sobol_cold_start(self.BOUNDS, q=4, seed=8)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(a.shape, (4, 2))
        self.assertTrue((a >= self.BOUNDS[0]).all()
                        and (a <= self.BOUNDS[1]).all())

    def test_emit_picks_native_types_and_int_rounding(self):
        cands = torch.tensor([[0.25, 3.6], [0.75, 7.4]], dtype=torch.float64)
        out = emit_picks(cands, int_dims=[1])
        self.assertEqual(out, [[0.25, 4], [0.75, 7]])
        self.assertIsInstance(out[0][0], float)
        self.assertIsInstance(out[0][1], int)

    def test_emit_picks_no_int_dims(self):
        cands = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        self.assertEqual(emit_picks(cands, int_dims=[]), [[0.5, 1.5]])
