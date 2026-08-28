import unittest

from surrokit import Constraint, InfeasibleError, NotEnoughData, Problem


class TestProblem(unittest.TestCase):
    def test_valid_problem_and_dim(self):
        p = Problem(bounds_lo=(0.0, 1.0), bounds_hi=(1.0, 2.0))
        self.assertEqual(p.dim, 2)
        self.assertEqual(p.int_dims, ())
        self.assertIsNone(p.noise)
        self.assertIsNone(p.constraint)

    def test_bounds_length_mismatch(self):
        with self.assertRaises(ValueError):
            Problem(bounds_lo=(0.0,), bounds_hi=(1.0, 2.0))

    def test_empty_bounds(self):
        with self.assertRaises(ValueError):
            Problem(bounds_lo=(), bounds_hi=())

    def test_lo_not_below_hi(self):
        with self.assertRaises(ValueError):
            Problem(bounds_lo=(0.0, 5.0), bounds_hi=(1.0, 5.0))

    def test_int_dims_out_of_range(self):
        with self.assertRaises(ValueError):
            Problem(bounds_lo=(0.0,), bounds_hi=(1.0,), int_dims=(1,))

    def test_constraint_validation(self):
        with self.assertRaises(ValueError):
            Constraint(axis=-1, min=0.0)
        with self.assertRaises(ValueError):
            Constraint(axis=0, min=0.0, k_sigma=-1.0)

    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(NotEnoughData, RuntimeError))
        self.assertTrue(issubclass(InfeasibleError, RuntimeError))
