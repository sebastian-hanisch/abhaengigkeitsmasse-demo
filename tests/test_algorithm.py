"""Einheiten des Kerns an Handrechnungen (Mini-Instanzen in conftest.py) und ihre Grundeigenschaften."""

import math

import numpy as np
import pytest

import abm_algorithm as K
import prs_scenario as S


def test_distance_covariance_and_correlation_by_hand(dist_mini):
    x, y = dist_mini
    assert K.dcov2(x, y) == pytest.approx(8 / 81, abs=1e-14)
    assert K.dcor2(x, y) == pytest.approx(8 / math.sqrt(640), abs=1e-14)
    assert K.dcor(x, y) == pytest.approx(0.5623413251903491, abs=1e-12)


def test_distance_correlation_of_an_independent_table_is_zero(full_table):
    x, y = full_table
    assert K.dcov2(x, y) == pytest.approx(0.0, abs=1e-15) and K.dcor(x, y) == pytest.approx(0.0, abs=1e-7)


def test_distance_correlation_of_four_points(four):
    assert K.dcor(*four) == pytest.approx(0.8320502943378437, abs=1e-12)


def test_distance_correlation_is_one_for_a_line():
    x = np.arange(1.0, 9.0)
    assert K.dcor(x, 3 * x + 1) == pytest.approx(1.0, abs=1e-12) and K.dcor(x, -x) == pytest.approx(1.0, abs=1e-12)


def test_double_centering_gives_zero_row_and_column_sums():
    A = K.double_center(K.distance_matrix(S.VRng(3).normal(9)))
    assert A.sum(0) == pytest.approx(np.zeros(9), abs=1e-12) and A.sum(1) == pytest.approx(np.zeros(9), abs=1e-12)


def test_u_centering_zeroes_the_diagonal_and_the_row_sums():
    U = K.u_center(K.distance_matrix(S.VRng(4).normal(9)))
    assert np.diag(U) == pytest.approx(np.zeros(9)) and U.sum(1) == pytest.approx(np.zeros(9), abs=1e-12)


def test_distance_matrix_in_one_and_several_dimensions():
    assert K.distance_matrix([0.0, 3.0]).tolist() == [[0.0, 3.0], [3.0, 0.0]]
    D = K.distance_matrix(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert D[0, 1] == pytest.approx(5.0) and D[1, 0] == pytest.approx(5.0) and D[0, 0] == 0.0


def test_xi_by_hand(four):
    assert K.chatterjee_xi(*four) == pytest.approx(0.0, abs=1e-15)


def test_xi_of_a_monotone_function_is_close_to_one_and_the_formula_is_exact_without_ties():
    x = np.arange(1.0, 11.0)
    assert K.chatterjee_xi(x, x**3) == pytest.approx(1.0 - 3 * 9 / 99)           # Ränge 1..10: Σ|Δr| = 9, ξ = 1 − 3·9/(100−1) = 0,7273


def test_xi_is_not_symmetric():
    x = np.linspace(-1, 1, 41)
    y = x**2
    assert K.chatterjee_xi(x, y) > 0.8 and K.chatterjee_xi(y, x) < 0.3


def test_hoeffding_of_identical_columns_is_one_and_of_the_mini_table_negative():
    x = S.VRng(5).normal(30)
    assert K.hoeffding_d(x, x) == pytest.approx(1.0, abs=1e-12)
    assert K.hoeffding_d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) == pytest.approx(1.0, abs=1e-12)


def test_mutual_information_of_two_equal_binary_columns_is_ln2():
    x = np.array([0.0, 1.0] * 6)
    assert K.mutual_information(x, x, 2) == pytest.approx(math.log(2), abs=1e-12)


def test_mutual_information_of_a_full_table_is_zero():
    x = np.repeat(np.arange(4.0), 4)
    y = np.tile(np.arange(4.0), 4)
    assert K.mutual_information(x, y, 4) == pytest.approx(0.0, abs=1e-12)


def test_quantile_classes_by_hand():
    assert K.quantile_classes([5.0, 1.0, 3.0, 4.0, 2.0], 2).tolist() == [1, 0, 0, 1, 0]       # Plätze 4,0,2,3,1 · 2 // 5 = 1,0,0,1,0
    assert K.quantile_classes(np.arange(10.0), 5).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert K.quantile_classes([3.0, 1.0, 2.0], 3).tolist() == [2, 0, 1]


def test_p_value_counts_with_the_plus_one_rule():
    null = np.array([0.1, 0.5, 0.9, 0.3])
    assert K.p_value(0.5, null, two_sided=False) == pytest.approx((1 + 2) / 5)                  # 0,5 und 0,9
    assert K.p_value(1.5, null, two_sided=False) == pytest.approx(1 / 5)
    assert K.p_value(-0.5, np.array([-0.9, 0.2, 0.5]), two_sided=True) == pytest.approx((1 + 2) / 4)
    assert math.isnan(K.p_value(float("nan"), null, two_sided=False))


def test_statistics_returns_the_requested_measures_in_the_fixed_order():
    x, y = S.VRng(1).normal(12), S.VRng(2).normal(12)
    out = K.statistics(x, y, np.arange(12)[None], which=("mi", "pearson", "xi"))
    assert list(out) == ["pearson", "xi", "mi"] and all(v.shape == (1,) for v in out.values())


def test_the_same_permutations_give_the_same_null_for_all_measures():
    x, y = S.VRng(1).normal(15), S.VRng(2).normal(15)
    perms = K.random_permutations(S.VRng(3), 7, 15)
    full = K.statistics(x, y, perms)
    for key in K.MEASURE_KEYS:
        assert K.statistics(x, y, perms, which=(key,))[key] == pytest.approx(full[key], abs=1e-12)


def test_random_permutations_are_permutations_and_reproducible():
    p = K.random_permutations(S.VRng(7), 20, 9)
    assert all(sorted(row) == list(range(9)) for row in p) and (p == K.random_permutations(S.VRng(7), 20, 9)).all()


def test_permutation_test_is_reproducible_and_depends_on_the_stream():
    x, y = S.VRng(1).normal(30), S.VRng(2).normal(30)
    a = K.permutation_test(x, y, 99, S.VRng(5))
    assert a == K.permutation_test(x, y, 99, S.VRng(5)) and a[1] != K.permutation_test(x, y, 99, S.VRng(6))[1]
    assert all(1 / 100 <= p <= 1.0 for p in a[1].values())
