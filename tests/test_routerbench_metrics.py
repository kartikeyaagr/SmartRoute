"""AIQ / hull math used by harness/routerbench.py."""

import numpy as np
import pytest

from harness.routerbench import aiq, cost_to_reach, nd_hull


def test_hull_drops_dominated_interior_points():
    pts = [(0.0, 0.5), (1.0, 0.55), (2.0, 0.9)]
    assert nd_hull(pts) == [(0.0, 0.5), (2.0, 0.9)]  # (1, .55) is under the chord


def test_aiq_of_a_line_is_its_mean_height():
    v, _, _ = aiq([(0.0, 0.5), (2.0, 0.9)], 0.0, 2.0)
    assert v == pytest.approx(0.7, abs=1e-6)


def test_curve_is_non_decreasing():
    v, grid, vals = aiq([(0.0, 0.5), (1.0, 0.95), (2.0, 0.9)], 0.0, 2.0)
    assert np.all(np.diff(vals) >= -1e-12)
    assert vals[-1] == pytest.approx(0.95)


def test_adding_a_better_point_raises_aiq():
    base, _, _ = aiq([(0.0, 0.5), (2.0, 0.9)], 0.0, 2.0)
    better, _, _ = aiq([(0.0, 0.5), (0.5, 0.88), (2.0, 0.9)], 0.0, 2.0)
    assert better > base


def test_cost_to_reach():
    _, grid, vals = aiq([(0.0, 0.5), (2.0, 0.9)], 0.0, 2.0)
    assert cost_to_reach(grid, vals, 0.7) == pytest.approx(1.0, abs=1e-3)
    assert cost_to_reach(grid, vals, 0.99) is None
