"""Eigenschaften der Maße: Invarianz, Symmetrie, Asymmetrie von ξ, der Gegensatz Pearson gegen die allgemeinen Maße und die Dimensionseigenschaft der Distanzkorrelation."""

import numpy as np
import pytest

import abm_algorithm as K
import abm_shapes as SH
import prs_algorithm as PA
import prs_scenario as S


def _pair(seed, n=40):
    rng = S.VRng(seed)
    x = rng.normal(n)
    return x, x**2 + 0.7 * rng.normal(n)


@pytest.mark.parametrize("seed", range(5))
def test_distance_correlation_is_invariant_under_shifts_and_scaling_and_symmetric(seed):
    x, y = _pair(seed)
    base = K.dcor(x, y)
    assert K.dcor(3.0 * x + 5.0, 0.2 * y - 1.0) == pytest.approx(base, abs=1e-12)
    assert K.dcor(-2.0 * x, y) == pytest.approx(base, abs=1e-12)                       # auch ein Vorzeichenwechsel ändert nichts
    assert K.dcor(y, x) == pytest.approx(base, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_the_rank_based_measures_are_invariant_under_monotone_maps(seed):
    x, y = _pair(seed)
    fx, fy = np.exp(x), (y + 10.0) ** 3
    for key in ("spearman", "xi", "hoeffding", "mi"):
        a = K.statistics(x, y, np.arange(40)[None], bins=4, which=(key,))[key][0]
        b = K.statistics(fx, fy, np.arange(40)[None], bins=4, which=(key,))[key][0]
        assert b == pytest.approx(a, abs=1e-12), key
    assert abs(PA.pearson_r(x, y) - PA.pearson_r(fx, fy)) > 1e-3                          # Pearson ändert sich dagegen


@pytest.mark.parametrize("seed", range(5))
def test_hoeffding_and_mutual_information_are_symmetric(seed):
    x, y = _pair(seed)
    assert K.hoeffding_d(x, y) == pytest.approx(K.hoeffding_d(y, x), abs=1e-12)
    assert K.mutual_information(x, y, 5) == pytest.approx(K.mutual_information(y, x, 5), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_xi_is_asymmetric_for_a_function_that_is_not_invertible(seed):
    x = S.VRng(seed).uniform(300) * 2 - 1
    assert K.chatterjee_xi(x, x**2) > 0.9 and K.chatterjee_xi(x**2, x) < 0.5


def test_xi_is_symmetric_for_an_invertible_function():
    x = S.VRng(3).uniform(300) * 2 - 1
    assert K.chatterjee_xi(x, np.exp(3 * x)) == pytest.approx(K.chatterjee_xi(np.exp(3 * x), x), abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_the_linear_measures_are_blind_on_a_parabola_and_the_general_ones_are_not(seed):
    x = S.VRng(seed).uniform(400) * 2 - 1
    y = x**2
    assert abs(PA.pearson_r(x, y)) < 0.12 and abs(PA.spearman_r(x, y)) < 0.12
    assert K.dcor(x, y) > 0.4 and K.chatterjee_xi(x, y) > 0.9 and K.hoeffding_d(x, y) > 0.1 and K.mutual_information(x, y, 5) > 0.5


@pytest.mark.parametrize("shape", ["parabel", "sinus", "kreis", "streuung", "schachbrett", "gemeinsam"])
def test_for_every_nonlinear_shape_at_least_one_measure_rejects_independence_at_n_300(shape):
    x, y = SH.generate(shape, 300, S.VRng(35))
    _, p = K.permutation_test(x, y, 199, S.VRng(36))
    assert min(p.values()) <= 0.05, p


@pytest.mark.parametrize("shape", ["parabel", "kreis", "schachbrett"])
def test_the_linear_measures_stay_near_zero_where_the_shape_is_symmetric(shape):
    """Die Populationswerte sind null; die Teststärke einzelner Läufe liegt wegen der Abhängigkeit trotzdem über 5 % (Raster, Experiment 1), hier zählt die Größe der Maße im Mittel."""
    rng = S.VRng(35)
    vals = []
    for _ in range(12):
        x, y = SH.generate(shape, 300, rng)
        stats = K.statistics(x, y, np.arange(300)[None], which=("pearson", "spearman"))
        vals.append((abs(stats["pearson"][0]), abs(stats["spearman"][0])))
    assert np.mean(vals, axis=0) == pytest.approx([0.05, 0.05], abs=0.04)


def test_each_measure_differs_from_the_others_on_some_shape():
    """Zweig-Test des Plans: auf jeder dieser Formen schlägt das eine Maß in den meisten Läufen an und das andere in den wenigsten (30 Läufe, 99 Permutationen)."""
    expect = {"parabel": (300, "dcor", "spearman"), "sinus": (30, "xi", "pearson"), "kreis": (100, "mi", "pearson"), "gemeinsam": (300, "dcor", "spearman")}
    for shape, (n, finds, misses) in expect.items():
        rng = S.VRng(35)
        hits = dict.fromkeys((finds, misses), 0)
        for _ in range(30):
            x, y = SH.generate(shape, n, rng)
            _, p = K.permutation_test(x, y, 99, rng, which=(finds, misses))
            hits[finds] += p[finds] <= 0.05
            hits[misses] += p[misses] <= 0.05
        assert hits[finds] >= 18 and hits[misses] <= 9, (shape, n, hits)


def test_the_plain_distance_correlation_grows_with_the_dimension_and_the_corrected_one_does_not():
    plain, unbiased = {}, {}
    for p in (1, 20, 200):
        rng = S.VRng(p)
        pairs = [(rng.normal((50, p)), rng.normal((50, p))) for _ in range(20)]
        plain[p] = np.mean([K.dcor2(X, Y) for X, Y in pairs])
        unbiased[p] = np.mean([K.dcor2_unbiased(X, Y) for X, Y in pairs])
    assert plain[1] < plain[20] < plain[200] and plain[200] > 0.5
    assert all(abs(v) < 0.02 for v in unbiased.values())


def test_the_test_keeps_its_level_in_high_dimensions_although_the_plain_value_is_large():
    rejected = 0
    for seed in range(40):
        rng = S.VRng(500 + seed)
        X, Y = rng.normal((30, 100)), rng.normal((30, 100))
        rejected += K.dcor_permutation_p(X, Y, 99, rng) <= 0.05
    assert rejected <= 8                                                              # erwartet 2 von 40; großzügige obere Grenze


def test_the_false_alarm_rate_of_every_measure_is_near_five_percent_on_independent_data():
    rejected = dict.fromkeys(K.MEASURE_KEYS, 0)
    runs = 150
    rng = S.VRng(77)
    for _ in range(runs):
        x, y = SH.generate("unabhaengig", 40, rng)
        _, p = K.permutation_test(x, y, 99, rng)
        for k in K.MEASURE_KEYS:
            rejected[k] += p[k] <= 0.05
    assert all(0.0 <= v / runs <= 0.12 for v in rejected.values()), rejected
