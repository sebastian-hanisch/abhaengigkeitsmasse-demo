"""Auswertung: Einzeldatensatz, Zellen, Seeds, Zugriff auf das Raster (die Verdikte prüft test_verdicts.py)."""

import math

import numpy as np
import pytest

import abm_algorithm as K
import abm_constants as C
import abm_evaluation as ev
import abm_shapes as SH
import prs_scenario as S


def test_analysis_is_deterministic_and_internally_consistent():
    s = ev.Settings("sinus", 60, 1.0, 5, 99, 11)
    a = ev.analyse(s)
    assert a == ev.analyse(s) and a.settings == s and len(a.x) == len(a.y) == 60 and a.dependent
    x, y = np.array(a.x), np.array(a.y)
    assert a.stats["pearson"] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12) and a.stats["dcor"] == pytest.approx(K.dcor(x, y), abs=1e-12)
    assert a.stats["xi"] == pytest.approx(K.chatterjee_xi(x, y), abs=1e-12) and a.stats["mi"] == pytest.approx(K.mutual_information(x, y, 5), abs=1e-12)
    assert a.stats["hoeffding"] == pytest.approx(K.hoeffding_d(x, y), abs=1e-12)
    assert a.found == tuple(k for k in K.MEASURE_KEYS if a.p[k] <= 0.05) and all(1 / 100 <= v <= 1 for v in a.p.values())


def test_the_data_stream_and_the_permutation_stream_are_separate_and_follow_the_seed():
    data_seed, perm_seed = S.substream_seeds(11, 2)
    x, y = SH.generate("sinus", 60, S.VRng(data_seed), 1.0)
    a = ev.analyse(ev.Settings("sinus", 60, 1.0, 5, 99, 11))
    assert a.x == tuple(map(float, x)) and a.y == tuple(map(float, y))
    stats, p = K.permutation_test(x, y, 99, S.VRng(perm_seed), 5)
    assert a.p == p and a.stats == stats
    b = ev.analyse(ev.Settings("sinus", 60, 1.0, 5, 99, 12))
    assert b.x != a.x


@pytest.mark.parametrize("field,value", [("shape", "kreis"), ("n", 61), ("noise", 1.5), ("bins", 6), ("perms", 100), ("seed", 12)])
def test_every_setting_changes_the_analysis(field, value):
    base = ev.Settings("sinus", 60, 1.0, 5, 99, 11)
    changed = ev.Settings(**{**base.__dict__, field: value})
    a, b = ev.analyse(base), ev.analyse(changed)
    assert a != b
    if field == "perms":
        assert a.perms_used == 99 and b.perms_used == 100


def test_cell_seeds_are_distinct_across_kinds_shapes_n_extra_and_seed():
    seeds = {ev.cell_seed(kind, shape, n, extra, s)
             for kind in ("power", "noise", "raster", "dim", "dimpower", "xi", "probe") for shape in C.SHAPES for n in (30, 100, 300) for extra in (0.5, 1.0, 2.0, 5) for s in C.GRID_SEEDS}
    assert len(seeds) == 7 * 9 * 3 * 4 * 4


def test_power_cell_structure_reproducibility_and_bounds():
    kw = dict(seeds=(1, 2), B=19)
    a = ev.power_cell("parabel", 30, 5, **kw)
    assert a == ev.power_cell("parabel", 30, 5, **kw) and a["runs"] == 10 and a["which"] == list(K.MEASURE_KEYS) and a["B"] == 19
    assert all(v * 10 == pytest.approx(round(v * 10)) and 0 <= v <= 1 for v in a["rate"].values()) and set(a["sd"]) == set(K.MEASURE_KEYS)
    assert a != ev.power_cell("parabel", 30, 5, seeds=(3, 4), B=19)


def test_power_cell_noise_and_bins_change_the_cells():
    kw = dict(seeds=(1,), B=49, runs_per_seed=12)
    base = ev.power_cell("parabel", 100, which=("dcor",), noise=0.5, **kw)
    assert base["rate"].keys() == {"dcor"} and base["sd"] == {"dcor": 0.0} and base["noise"] == 0.5          # eine Seed: keine Streuung
    assert base["rate"]["dcor"] == 1.0 and ev.power_cell("parabel", 100, which=("dcor",), noise=4.0, **kw)["rate"]["dcor"] < 0.6
    coarse = ev.power_cell("sinus", 100, which=("mi",), bins=3, kind="raster", **kw)
    fine = ev.power_cell("sinus", 100, which=("mi",), bins=8, kind="raster", **kw)
    assert coarse["bins"] == 3 and fine["bins"] == 8 and coarse["rate"]["mi"] < 0.4 < 0.7 < fine["rate"]["mi"]


def test_the_independent_shape_rejects_about_five_percent_in_a_small_run():
    c = ev.power_cell("unabhaengig", 30, 60, seeds=(5, 6), B=49)
    assert all(v <= 0.15 for v in c["rate"].values())


def test_dimension_cell_is_reproducible_and_the_mean_root_is_below_the_root_of_the_mean():
    a = ev.dimension_cell(5, n=20, runs_per_seed=6, seeds=(1, 2))
    assert a == ev.dimension_cell(5, n=20, runs_per_seed=6, seeds=(1, 2)) and a["runs"] == 12
    assert a["dcor"] <= math.sqrt(a["dcor2_v"]) + 1e-12 and a["dcor2_v"] > a["dcor2_u"]
    assert ev.dimension_cell(50, n=20, runs_per_seed=6, seeds=(1, 2))["dcor"] > a["dcor"]            # mehr Dimensionen, größere Stichprobenwerte


def test_dimension_power_cell_is_reproducible_and_drops_with_the_dimension():
    a = ev.dimension_power_cell(1, n=60, B=29, runs_per_seed=6, seeds=(1, 2))
    assert a == ev.dimension_power_cell(1, n=60, B=29, runs_per_seed=6, seeds=(1, 2)) and a["runs"] == 12 and a["rate"] >= 0.9
    assert ev.dimension_power_cell(60, n=60, B=29, runs_per_seed=6, seeds=(1, 2))["rate"] < a["rate"]


def test_xi_cells():
    f = ev.xi_function_cell("x²", n=200, reps=5)
    assert f["xi_xy"] > 0.9 > 0.5 > f["xi_yx"] and f["spearman"] < 0.2
    e = ev.xi_function_cell("e^(3x)", n=200, reps=5)
    assert e["xi_xy"] == pytest.approx(e["xi_yx"], abs=1e-12) and e["spearman"] == pytest.approx(1.0, abs=1e-12)
    sizes = [ev.xi_size_cell(n, reps=5)["xi"] for n in (30, 300)]
    assert 0.3 < sizes[0] < sizes[1] < 1.0
    assert ev.xi_size_cell(30, reps=5) == ev.xi_size_cell(30, reps=5) and ev.xi_function_cell("|x|", 50, 3) != ev.xi_function_cell("x²", 50, 3)


def test_find_matches_exact_and_float_keys_and_returns_none_otherwise():
    rows = [{"shape": "a", "noise": 0.5, "n": 3}, {"shape": "a", "noise": 1.0, "n": 3}]
    assert ev.find(rows, shape="a", noise=1.0) is rows[1] and ev.find(rows, shape="a", noise=0.5 + 1e-12) is rows[0]
    assert ev.find(rows, shape="b") is None and ev.find(rows, shape="a", noise=0.51) is None and ev.find(rows, n=3) is rows[0]


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    broken = tmp_path / "kaputt.json"
    broken.write_text("{nicht json", encoding="utf-8")
    assert ev.load_grid(broken) is None
    good = tmp_path / "gut.json"
    good.write_text('{"a": [1, 2]}', encoding="utf-8")
    assert ev.load_grid(good) == {"a": [1, 2]}


def test_power_table_collects_one_cell_per_shape():
    grid = ev.load_grid()
    table = ev.power_table(grid, 100)
    assert list(table) == list(C.SHAPES) and all(c["n"] == 100 and c["shape"] == s for s, c in table.items())
