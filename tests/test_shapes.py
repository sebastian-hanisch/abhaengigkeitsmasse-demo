"""Die neun Formen: Gestalt, Rauschfaktor, Reproduzierbarkeit und die Nullkorrelation des Schachbretts (die erste 4 × 4-Fassung des Vorgängerstücks war nicht unkorreliert)."""

import math

import numpy as np
import pytest

import abm_constants as C
import abm_shapes as SH
import prs_algorithm as PA
import prs_scenario as S


@pytest.mark.parametrize("shape", list(C.SHAPES))
def test_every_shape_is_reproducible_finite_and_has_the_right_size(shape):
    x, y = SH.generate(shape, 50, S.VRng(3))
    x2, y2 = SH.generate(shape, 50, S.VRng(3))
    assert x.shape == y.shape == (50,) and np.isfinite(x).all() and np.isfinite(y).all()
    assert (x == x2).all() and (y == y2).all()
    assert (SH.generate(shape, 50, S.VRng(4))[0] != x).any()


def test_the_checkerboard_is_uncorrelated_and_fills_exactly_the_five_cells():
    x, y = SH.generate("schachbrett", 200_000, S.VRng(11))
    assert abs(PA.pearson_r(x, y)) < 0.01 and abs(PA.spearman_r(x, y)) < 0.01
    cx, cy = np.minimum(((x + 1) * 1.5).astype(int), 2), np.minimum(((y + 1) * 1.5).astype(int), 2)
    counts = {(i, j): int(((cy == i) & (cx == j)).sum()) for i in range(3) for j in range(3)}
    assert {cell for cell, c in counts.items() if c > 0} == set(SH.CHECKER_CELLS)
    assert all(abs(counts[cell] / 200_000 - 0.2) < 0.005 for cell in SH.CHECKER_CELLS)
    assert x.min() >= -1.0 and x.max() <= 1.0 and y.min() >= -1.0 and y.max() <= 1.0


def test_the_circle_without_noise_lies_on_the_unit_circle():
    x, y = SH.generate("kreis", 500, S.VRng(5), noise=0.0)
    assert x**2 + y**2 == pytest.approx(np.ones(500), abs=1e-12)


@pytest.mark.parametrize("shape", ["linear", "monoton", "parabel", "sinus"])
def test_a_function_shape_without_noise_equals_its_curve(shape):
    x, y = SH.generate(shape, 100, S.VRng(6), noise=0.0)
    assert y == pytest.approx(SH.curve(shape, x), abs=1e-12) and x.min() >= -1.0 and x.max() < 1.0


@pytest.mark.parametrize("shape", ["unabhaengig", "kreis", "streuung", "schachbrett", "gemeinsam"])
def test_shapes_without_a_function_have_no_curve(shape):
    assert SH.curve(shape, np.linspace(-1, 1, 5)) is None


@pytest.mark.parametrize("shape,base", [("linear", 0.8), ("parabel", 0.25), ("sinus", 0.6), ("monoton", 1.0)])
@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_the_noise_factor_scales_the_residual(shape, base, factor):
    x, y = SH.generate(shape, 20_000, S.VRng(7), noise=factor)
    assert (y - SH.curve(shape, x)).std() == pytest.approx(base * factor, rel=0.03)


def test_the_noise_factor_scales_the_spread_of_the_widening_shape_and_the_circle():
    x, y = SH.generate("streuung", 20_000, S.VRng(8), noise=2.0)
    assert (y / x).std() == pytest.approx(2.0, rel=0.03)
    x, y = SH.generate("kreis", 20_000, S.VRng(9), noise=2.0)
    assert (np.hypot(x, y) - 1).std() > (np.hypot(*SH.generate("kreis", 20_000, S.VRng(9), noise=0.5)) - 1).std() * 2


@pytest.mark.parametrize("shape", C.NOISE_FREE_SHAPES)
def test_the_noise_factor_does_not_act_on_the_noise_free_shapes(shape):
    a = SH.generate(shape, 30, S.VRng(2), noise=0.5)
    b = SH.generate(shape, 30, S.VRng(2), noise=4.0)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_the_widening_shape_is_uncorrelated_and_symmetric_in_y():
    x, y = SH.generate("streuung", 100_000, S.VRng(12))
    assert abs(PA.pearson_r(x, y)) < 0.01 and abs(PA.pearson_r(x, y**2)) < 0.01
    assert abs(PA.pearson_r(np.abs(x), np.abs(y))) > 0.3                                 # |y| wächst mit |x|: Abhängigkeit ohne Korrelation


def test_the_shared_scale_shape_is_the_t3_pair_with_rho_zero():
    x, y = SH.generate("gemeinsam", 40, S.VRng(13))
    X, Y = S.draw_pairs("t3", 1, 40, 0.0, S.VRng(13))
    assert (x == X[0]).all() and (y == Y[0]).all()
    big_x, big_y = SH.generate("gemeinsam", 100_000, S.VRng(14))
    assert abs(PA.pearson_r(big_x, big_y)) < 0.05 and PA.pearson_r(np.abs(big_x), np.abs(big_y)) > 0.2     # unkorreliert, aber die Beträge hängen zusammen


def test_the_independent_shape_is_independent():
    x, y = SH.generate("unabhaengig", 100_000, S.VRng(15))
    assert abs(PA.pearson_r(x, y)) < 0.01 and abs(PA.pearson_r(x**2, y**2)) < 0.01 and abs(x.mean()) < 0.01 and x.std() == pytest.approx(1 / math.sqrt(3), rel=0.01)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        SH.generate("würfel", 10, S.VRng(1))
    with pytest.raises(ValueError):
        SH.generate("linear", 1, S.VRng(1))
    with pytest.raises(ValueError):
        SH.generate("linear", 10, S.VRng(1), noise=-0.1)
    assert SH.generate("linear", 2, S.VRng(1))[0].shape == (2,)           # n = 2 ist erlaubt (kleinster Fall)
