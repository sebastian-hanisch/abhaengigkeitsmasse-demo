"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Fälle, inklusive Grenzen, Blöcke, Gleichstände, Fehlerwege."""

import math

import numpy as np
import pytest

import abm_algorithm as K
import abm_constants as C
import abm_evaluation as ev
import abm_shapes as SH
import prs_scenario as S


def test_exactly_five_observations_is_the_smallest_case_for_the_full_set():
    x, y = S.VRng(1).normal(5), S.VRng(2).normal(5)
    stats, p = K.permutation_test(x, y, 49, S.VRng(3), bins=2)
    assert set(stats) == set(K.MEASURE_KEYS) and all(0 < v <= 1 for v in p.values())


@pytest.mark.parametrize("n", [0, 1, 4])
def test_fewer_than_five_observations_are_rejected_by_the_test_and_by_hoeffding(n):
    x, y = np.arange(float(n)), np.arange(float(n))[::-1]
    with pytest.raises(ValueError):
        K.permutation_test(x, y, 9, S.VRng(1))
    with pytest.raises(ValueError):
        K.hoeffding_d(x, y)


def test_xi_needs_only_two_observations_and_mutual_information_too():
    assert K.chatterjee_xi([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0, abs=1e-15)                  # n = 2: Σ|Δr| = 1, 2 Σ l(n−l) = 2, ξ = 1 − 2·1/2
    assert K.chatterjee_xi([1.0, 2.0, 3.0], [3.0, 1.0, 2.0]) == pytest.approx(1 - 3 * 3 / 8)
    assert K.mutual_information([1.0, 2.0], [1.0, 2.0], 2) == pytest.approx(math.log(2))


def test_constant_columns_are_rejected_everywhere():
    c, v = np.ones(8), np.arange(8.0)
    for f in (K.chatterjee_xi, K.hoeffding_d):
        for a, b in ((c, v), (v, c)):
            with pytest.raises(ValueError):
                f(a, b)
    for a, b in ((c, v), (v, c)):
        with pytest.raises(ValueError):
            K.dcor2(a, b)
        with pytest.raises(ValueError):
            K.permutation_test(a, b, 9, S.VRng(1))


def test_length_mismatch_and_wrong_dimension_are_rejected():
    with pytest.raises(ValueError):
        K.chatterjee_xi([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        K.permutation_test(np.zeros((3, 3)), np.zeros((3, 3)), 9, S.VRng(1))


def test_class_boundaries_are_inclusive_at_two_and_n():
    v = np.arange(6.0)
    assert K.quantile_classes(v, 2).tolist() == [0, 0, 0, 1, 1, 1]
    assert K.quantile_classes(v, 6).tolist() == [0, 1, 2, 3, 4, 5]               # k = n: jede Beobachtung eine Klasse
    for k in (1, 0, 7):
        with pytest.raises(ValueError):
            K.quantile_classes(v, k)


def test_a_block_boundary_does_not_change_the_result(monkeypatch):
    x, y = S.VRng(1).normal(9), S.VRng(2).normal(9)
    perms = K.random_permutations(S.VRng(3), 13, 9)
    whole = K.statistics(x, y, perms)
    monkeypatch.setattr(K, "CHUNK_CELLS", 9 * 9 * 2 + 5)                        # 2 Permutationen je Block, 13 = 6 Blöcke und ein Rest
    blocked = K.statistics(x, y, perms)
    for key in K.MEASURE_KEYS:
        assert blocked[key] == pytest.approx(whole[key], abs=1e-12)
    monkeypatch.setattr(K, "CHUNK_CELLS", 1)                                    # Mindestgröße 1: trotzdem lauffähig
    assert K.statistics(x, y, perms, which=("dcor",))["dcor"] == pytest.approx(whole["dcor"], abs=1e-12)


def test_u_centering_needs_four_observations_and_works_with_exactly_four():
    with pytest.raises(ValueError):
        K.u_center(np.zeros((3, 3)))
    x, y = np.array([0.0, 1.0, 3.0, 4.0]), np.array([1.0, 0.0, 2.0, 5.0])
    assert math.isfinite(K.dcor2_unbiased(x, y))


def test_distance_covariance_of_two_equal_points_is_defined():
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    assert K.dcor(x, x) == pytest.approx(1.0, abs=1e-12)                        # Gleichstände: Abstand null


def test_ties_in_y_do_not_break_xi_and_ties_in_x_are_resolved_by_input_order():
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    assert math.isfinite(K.chatterjee_xi(x, y))
    tied_x = np.array([1.0, 1.0, 2.0])
    assert K.chatterjee_xi(tied_x, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.25)                 # Reihenfolge der Eingabe: Ränge (1, 2, 3), Σ|Δr| = 2, 2 Σ l(n−l) = 8
    assert K.chatterjee_xi(tied_x, np.array([2.0, 1.0, 3.0])) == pytest.approx(-0.125)               # vertauscht: Ränge (2, 1, 3), Σ|Δr| = 3


def test_p_value_has_a_floor_of_one_over_b_plus_one_and_a_ceiling_of_one():
    null = np.zeros(99)
    assert K.p_value(5.0, null, False) == pytest.approx(1 / 100)
    assert K.p_value(-5.0, null, False) == pytest.approx(1.0)
    assert K.p_value(0.0, null, False) == pytest.approx(1.0)                     # ein Gleichstand zählt mit (Toleranz)


def test_one_permutation_still_gives_a_valid_p_value():
    x, y = S.VRng(1).normal(10), S.VRng(2).normal(10)
    _, p = K.permutation_test(x, y, 1, S.VRng(3))
    assert all(v in (0.5, 1.0) for v in p.values())


def test_affordable_perms_respects_the_budget_inclusively_and_the_floor():
    assert ev.affordable_perms(499, 300) == 499                                  # 499 · 300² = 4,49·10^7 liegt im Budget
    assert ev.affordable_perms(499, 300, budget=499 * 300 * 300) == 499          # genau am Budget
    assert ev.affordable_perms(499, 300, budget=499 * 300 * 300 - 1) == 498
    assert ev.affordable_perms(499, 300, budget=10) == 49                        # Untergrenze
    assert ev.affordable_perms(99, 20) == 99                                      # mehr Budget als verlangt: B bleibt


def test_settings_at_all_limits_run():
    for n, bins, perms in ((C.N_MIN, C.BINS_MAX, C.PERM_MIN), (C.N_MAX, C.BINS_MIN, C.PERM_MAX)):
        a = ev.analyse(ev.Settings("kreis", n, C.NOISE_MAX if n == C.N_MIN else C.NOISE_MIN, bins, perms, 0))
        assert len(a.x) == n and a.perms_used == perms and set(a.p) == set(K.MEASURE_KEYS)


def test_bins_larger_than_n_are_clamped_not_raised():
    a = ev.analyse(ev.Settings("linear", 20, 1.0, 20, 99, 3))                     # k = n ist erlaubt
    assert len(a.x) == 20


def test_every_shape_runs_through_the_analysis():
    for shape in C.SHAPES:
        a = ev.analyse(ev.Settings(shape, 30, 1.0, 5, 99, 4))
        assert a.dependent == (shape != "unabhaengig") and set(a.found) <= set(K.MEASURE_KEYS)


def test_the_dimension_cell_works_for_one_dimension_and_one_run():
    r = ev.dimension_cell(1, n=6, runs_per_seed=1, seeds=(1,))
    assert r["runs"] == 1 and 0 <= r["dcor"] <= 1


def test_the_independent_shape_with_two_points_generates():
    x, y = SH.generate("unabhaengig", 2, S.VRng(5))
    assert x.shape == (2,)


def test_unbiased_distance_correlation_is_nan_if_either_side_is_constant():
    c, v = np.ones(8), S.VRng(4).normal(8)
    assert math.isnan(K.dcor2_unbiased(c, v)) and math.isnan(K.dcor2_unbiased(v, c)) and math.isnan(K.dcor2_unbiased(c, c))


def test_statistics_gives_nan_for_the_distance_correlation_of_a_constant_side():
    c, v = np.ones(8), S.VRng(4).normal(8)
    perms = K.random_permutations(S.VRng(5), 4, 8)
    assert np.isnan(K.statistics(c, v, perms, which=("dcor",))["dcor"]).all() and np.isnan(K.statistics(v, c, perms, which=("dcor",))["dcor"]).all()
    assert np.isfinite(K.statistics(v, v[::-1].copy(), perms, which=("dcor",))["dcor"]).all()


def test_dcor_permutation_p_follows_the_plus_one_rule_including_exact_ties():
    x, y = S.VRng(1).normal(5), S.VRng(2).normal(5)
    B = 500
    perms = K.random_permutations(S.VRng(3), B, 5)
    obs = K.dcov2(x, y)
    null = np.array([K.dcov2(x, y[p]) for p in perms])
    expected = (1 + int((null >= obs - 1e-12).sum())) / (B + 1)
    assert (null >= obs - 1e-12).sum() > (null > obs + 1e-12).sum()                     # die Probe enthält Gleichstände (Umordnungen mit gleichem Wert)
    assert K.dcor_permutation_p(x, y, B, S.VRng(3)) == pytest.approx(expected, abs=1e-12)
    line = np.arange(20.0)
    assert K.dcor_permutation_p(line, line, 99, S.VRng(4)) == pytest.approx(1 / 100)      # perfekte Abhängigkeit: kleinster möglicher Wert


def test_hoeffding_with_ties_follows_the_documented_strict_comparison():
    scipy_stats = pytest.importorskip("scipy.stats")
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    n = len(x)
    R, Sr = scipy_stats.rankdata(x), scipy_stats.rankdata(y)
    Q = np.array([1 + sum(1 for j in range(n) if x[j] < x[i] and y[j] < y[i]) for i in range(n)], dtype=float)
    d1, d2, d3 = ((Q - 1) * (Q - 2)).sum(), ((R - 1) * (R - 2) * (Sr - 1) * (Sr - 2)).sum(), ((R - 2) * (Sr - 2) * (Q - 1)).sum()
    expected = 30 * ((n - 2) * (n - 3) * d1 + d2 - 2 * (n - 2) * d3) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    assert K.hoeffding_d(x, y) == pytest.approx(expected, abs=1e-12)
