"""Orakel (ORACLE-PLAYBOOK): jedes Maß gegen einen anderen Rechenweg.

* Distanzkovarianz: Schleifen über die Definition (Doppelsumme), bias-korrigierte Fassung als Schleife; Mehrdimensionales gegen scipy.spatial.
* Chatterjee-ξ: scipy.stats.chatterjeexi (mit und ohne Gleichstände).
* Hoeffding-D: die U-Statistik über alle geordneten 5-Tupel (Hoeffdings eigene Definition), mal 30.
* Transinformation: sklearn.metrics.mutual_info_score auf Klassen, die hier unabhängig aus scipy-Rängen gebildet werden.
* Pearson, Spearman: scipy.stats.
* Permutationstest: die vektorisierte Zeilenfassung gegen die Einzelrechnung je Anordnung; vollständige Aufzählung aller Anordnungen bei n = 5.
Die Bibliotheken stehen nur im Test-venv; das Orakel braucht sie nicht zur Laufzeit."""

import itertools
import math

import numpy as np
import pytest

import abm_algorithm as K
import prs_scenario as S

scipy_stats = pytest.importorskip("scipy.stats")
scipy_distance = pytest.importorskip("scipy.spatial.distance")
sklearn_metrics = pytest.importorskip("sklearn.metrics")


def _pair(seed, n, dependent=True):
    rng = S.VRng(seed)
    x = rng.normal(n)
    y = (x**2 + 0.5 * rng.normal(n)) if dependent else rng.normal(n)
    return x, y


# --- Distanzkorrelation: Doppelsumme --------------------------------------------------------------------------------------------------------

def loop_dcov2(x, y):
    n = len(x)
    a = [[abs(x[i] - x[j]) for j in range(n)] for i in range(n)]
    b = [[abs(y[i] - y[j]) for j in range(n)] for i in range(n)]
    ar = [sum(a[i]) / n for i in range(n)]
    ac = [sum(a[i][j] for i in range(n)) / n for j in range(n)]
    br = [sum(b[i]) / n for i in range(n)]
    bc = [sum(b[i][j] for i in range(n)) / n for j in range(n)]
    ag, bg = sum(map(sum, a)) / n**2, sum(map(sum, b)) / n**2
    tot = {"xy": 0.0, "xx": 0.0, "yy": 0.0}
    for i in range(n):
        for j in range(n):
            ca = a[i][j] - ar[i] - ac[j] + ag
            cb = b[i][j] - br[i] - bc[j] + bg
            tot["xy"] += ca * cb
            tot["xx"] += ca * ca
            tot["yy"] += cb * cb
    return tot["xy"] / n**2, tot["xx"] / n**2, tot["yy"] / n**2


@pytest.mark.parametrize("seed", range(10))
def test_distance_correlation_matches_the_double_sum(seed):
    x, y = _pair(seed, 9)
    cov, vx, vy = loop_dcov2(x, y)
    assert K.dcov2(x, y) == pytest.approx(cov, abs=1e-12)
    assert K.dcor2(x, y) == pytest.approx(cov / math.sqrt(vx * vy), abs=1e-12)
    assert K.dcor(x, y) == pytest.approx(math.sqrt(cov / math.sqrt(vx * vy)), abs=1e-12)


def loop_dcov2_unbiased(x, y):
    """Definition der bias-korrigierten Distanzkovarianz als Schleife (U-Zentrierung nur für i ≠ j)."""
    n = len(x)
    a = [[abs(x[i] - x[j]) for j in range(n)] for i in range(n)]
    b = [[abs(y[i] - y[j]) for j in range(n)] for i in range(n)]

    def centered(d):
        tot = sum(map(sum, d))
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    out[i][j] = d[i][j] - sum(d[i]) / (n - 2) - sum(d[k][j] for k in range(n)) / (n - 2) + tot / ((n - 1) * (n - 2))
        return out

    ca, cb = centered(a), centered(b)
    f = n * (n - 3)
    return (sum(ca[i][j] * cb[i][j] for i in range(n) for j in range(n)) / f, sum(ca[i][j] ** 2 for i in range(n) for j in range(n)) / f,
            sum(cb[i][j] ** 2 for i in range(n) for j in range(n)) / f)


@pytest.mark.parametrize("seed", range(8))
def test_unbiased_distance_correlation_matches_its_definition(seed):
    x, y = _pair(seed, 8)
    cov, vx, vy = loop_dcov2_unbiased(x, y)
    assert K.dcor2_unbiased(x, y) == pytest.approx(cov / math.sqrt(vx * vy), abs=1e-10)


def test_unbiased_distance_correlation_has_mean_zero_under_independence_and_the_plain_one_does_not():
    plain, unbiased = [], []
    for seed in range(300):
        x, y = S.VRng(1000 + seed).normal(30), S.VRng(5000 + seed).normal(30)
        plain.append(K.dcor2(x, y))
        unbiased.append(K.dcor2_unbiased(x, y))
    assert abs(np.nanmean(unbiased)) < 0.01 and np.mean(plain) > 0.04


@pytest.mark.parametrize("p", [2, 5])
def test_distance_matrix_matches_scipy_in_several_dimensions(p):
    X = S.VRng(p).normal((9, p))
    assert K.distance_matrix(X) == pytest.approx(scipy_distance.squareform(scipy_distance.pdist(X)), abs=1e-12)


# --- Chatterjee-ξ -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_xi_matches_scipy(seed):
    x, y = _pair(seed, 40)
    assert K.chatterjee_xi(x, y) == pytest.approx(scipy_stats.chatterjeexi(x, y).statistic, abs=1e-12)
    assert K.chatterjee_xi(y, x) == pytest.approx(scipy_stats.chatterjeexi(y, x).statistic, abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_xi_matches_scipy_with_ties_in_y(seed):
    x, y = _pair(seed, 40)
    yt = np.round(y)                                                  # viele Gleichstände in y
    assert len(np.unique(yt)) < 25
    assert K.chatterjee_xi(x, yt) == pytest.approx(scipy_stats.chatterjeexi(x, yt).statistic, abs=1e-12)


# --- Hoeffding-D: U-Statistik über geordnete 5-Tupel ------------------------------------------------------------------------------------------

def _psi(a, b, c):
    return float(b <= a) - float(c <= a)


def hoeffding_u(x, y):
    """30 · Mittel von ¼ ψ(x1; x2, x3) ψ(x1; x4, x5) ψ(y1; y2, y3) ψ(y1; y4, y5) über alle geordneten 5-Tupel verschiedener Indizes."""
    tot, cnt = 0.0, 0
    for i, j, k, l, m in itertools.permutations(range(len(x)), 5):
        tot += 0.25 * _psi(x[i], x[j], x[k]) * _psi(x[i], x[l], x[m]) * _psi(y[i], y[j], y[k]) * _psi(y[i], y[l], y[m])
        cnt += 1
    return 30.0 * tot / cnt


@pytest.mark.parametrize("seed", range(8))
def test_hoeffding_matches_the_u_statistic(seed):
    x, y = _pair(seed, 7)
    assert K.hoeffding_d(x, y) == pytest.approx(hoeffding_u(x, y), abs=1e-12)


def test_hoeffding_is_about_zero_under_independence_and_positive_for_a_circle():
    vals = [K.hoeffding_d(S.VRng(i).normal(60), S.VRng(i + 500).normal(60)) for i in range(150)]
    assert abs(np.mean(vals)) < 0.005
    t = S.VRng(9).uniform(200) * 2 * math.pi
    assert K.hoeffding_d(np.cos(t), np.sin(t)) > 0.05


# --- Transinformation -----------------------------------------------------------------------------------------------------------------------

def _classes(v, k):
    r = scipy_stats.rankdata(v, method="ordinal")
    return [min(int((ri - 1) * k // len(v)), k - 1) for ri in r]


@pytest.mark.parametrize("k", [2, 3, 5, 8])
@pytest.mark.parametrize("seed", range(4))
def test_mutual_information_matches_sklearn(seed, k):
    x, y = _pair(seed, 60)
    assert K.mutual_information(x, y, k) == pytest.approx(sklearn_metrics.mutual_info_score(_classes(x, k), _classes(y, k)), abs=1e-12)


# --- Pearson, Spearman ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_pearson_and_spearman_match_scipy(seed):
    x, y = _pair(seed, 25)
    out = K.statistics(x, y, np.arange(25)[None], which=("pearson", "spearman"))
    assert out["pearson"][0] == pytest.approx(scipy_stats.pearsonr(x, y)[0], abs=1e-12) and out["spearman"][0] == pytest.approx(scipy_stats.spearmanr(x, y)[0], abs=1e-12)


# --- Permutationstest: Zeilenfassung gegen Einzelrechnung, vollständige Aufzählung ---------------------------------------------------------------

def scalar_measures(x, y, k):
    return {
        "pearson": scipy_stats.pearsonr(x, y)[0],
        "spearman": scipy_stats.spearmanr(x, y)[0],
        "dcor": math.sqrt(max(loop_dcov2(x, y)[0], 0.0) / math.sqrt(loop_dcov2(x, y)[1] * loop_dcov2(x, y)[2])),
        "xi": scipy_stats.chatterjeexi(x, y).statistic,
        "hoeffding": hoeffding_u(x, y),
        "mi": sklearn_metrics.mutual_info_score(_classes(x, k), _classes(y, k)),
    }


@pytest.mark.parametrize("seed", range(3))
def test_every_row_equals_the_single_computation_on_all_orderings_of_five_points(seed):
    x, y = _pair(seed, 5)
    perms = np.array(list(itertools.permutations(range(5))))
    rows = K.statistics(x, y, perms, bins=2)
    for r, perm in enumerate(perms):
        expected = scalar_measures(x, y[perm], 2)
        for key in K.MEASURE_KEYS:
            assert rows[key][r] == pytest.approx(expected[key], abs=1e-10), (key, tuple(perm))


@pytest.mark.parametrize("measure", K.MEASURE_KEYS)
def test_the_sampled_p_value_agrees_with_the_exact_one_from_all_orderings(measure):
    n = 6
    x, y = _pair(3, n)
    all_perms = np.array(list(itertools.permutations(range(n))))
    exact_null = K.statistics(x, y, all_perms, bins=2, which=(measure,))[measure]
    obs = K.statistics(x, y, np.arange(n)[None], bins=2, which=(measure,))[measure][0]
    two = measure in ("pearson", "spearman")
    exact = float((np.abs(exact_null) >= abs(obs) - 1e-12).sum() if two else (exact_null >= obs - 1e-12).sum()) / len(all_perms)
    sampled = K.p_value(float(obs), K.statistics(x, y, K.random_permutations(S.VRng(11), 4000, n), bins=2, which=(measure,))[measure], two)
    assert sampled == pytest.approx(exact, abs=0.04), (measure, exact, sampled)
