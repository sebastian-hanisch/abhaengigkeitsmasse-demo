"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren, und die Klassengrenzen im Streudiagramm passen zur Klasseneinteilung der Transinformation."""

import numpy as np
import pytest

import abm_algorithm as K
import abm_constants as C
import abm_evaluation as ev
import abm_shapes as SH
import abm_visualization as V
import prs_scenario as S


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


@pytest.mark.parametrize("n,k", [(20, 2), (20, 20), (37, 3), (100, 5), (100, 8), (300, 20)])
def test_raster_edges_reproduce_the_quantile_classes(n, k):
    v = S.VRng(n + k).normal(n)
    edges = V.raster_edges(v, k)
    assert len(edges) == k - 1 and edges == sorted(edges)
    assert np.digitize(v, edges).tolist() == K.quantile_classes(v, k).tolist()


def test_the_scatter_has_raster_lines_and_a_curve_where_one_exists():
    x, y = SH.generate("parabel", 60, S.VRng(1))
    fig = V.build_scatter(x, y, "parabel", 5)
    assert len(fig.layout.shapes) == 2 * 4 and [t.name for t in fig.data] == ["Beobachtungen", "Kurve ohne Rauschen"]
    x, y = SH.generate("kreis", 60, S.VRng(1))
    fig = V.build_scatter(x, y, "kreis", None)
    assert len(fig.layout.shapes) == 0 and [t.name for t in fig.data] == ["Beobachtungen"]


def test_the_pvalue_chart_colors_found_measures_green_and_the_rest_grey():
    p = {"pearson": 0.5, "spearman": 0.05, "dcor": 0.051, "xi": 0.01, "hoeffding": 0.9, "mi": 0.002}
    fig = V.build_pvalues(p)
    assert list(fig.data[0].marker.color) == [C.COLOR_MISSED, C.COLOR_FOUND, C.COLOR_MISSED, C.COLOR_FOUND, C.COLOR_MISSED, C.COLOR_FOUND]       # p = 0,05 zählt als gefunden
    assert list(fig.data[0].x) == [0.5, 0.05, 0.051, 0.01, 0.9, 0.002] and list(fig.data[0].y) == list(C.MEASURES.values())


def test_every_chart_builds_and_locks_its_axes(grid):
    figs = [V.build_power_heatmap(ev.power_table(grid, 100), 100), V.build_noise(grid["noise"], "kreis"), V.build_raster(grid["raster"]), V.build_shared_scale(grid),
            V.build_dimension(list(grid["dim"]) + [ev.dimension_cell(1, n=10, runs_per_seed=2, seeds=(1,))]), V.build_dimension_power([{"p": 1, "rate": 1.0}, {"p": 5, "rate": 0.5}]),
            V.build_xi_functions([ev.xi_function_cell("x²", 100, 2)]), V.build_xi_size([ev.xi_size_cell(30, 2), ev.xi_size_cell(100, 2)])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_heatmap_shows_percent_with_the_right_shape_and_orientation(grid):
    table = ev.power_table(grid, 100)
    fig = V.build_power_heatmap(table, 100)
    z = np.array(fig.data[0].z)
    assert z.shape == (9, 6) and list(fig.data[0].y) == [C.SHAPES[s] for s in C.SHAPES] and list(fig.data[0].x) == [C.MEASURE_SHORT[k] for k in C.MEASURES]
    assert z[3, 2] == pytest.approx(100 * table["parabel"]["rate"]["dcor"]) and z[5, 0] == pytest.approx(100 * table["kreis"]["rate"]["pearson"])
    assert fig.layout.yaxis.autorange == "reversed"


def test_the_noise_chart_has_one_line_per_measure_and_sorted_x(grid):
    fig = V.build_noise(grid["noise"], "linear")
    assert [t.name for t in fig.data][:6] == [C.MEASURE_SHORT[k] for k in C.MEASURES] and list(fig.data[0].x) == [1.0, 2.0, 4.0, 8.0]


def test_the_raster_chart_has_one_line_per_shape_sorted_by_k(grid):
    fig = V.build_raster(grid["raster"])
    assert [t.name for t in fig.data][: len(C.RASTER_SHAPES)] == [C.SHAPES[s] for s in C.RASTER_SHAPES] and list(fig.data[0].x) == list(C.RASTER_KS)


def test_the_shared_scale_chart_has_one_bar_group_per_n(grid):
    fig = V.build_shared_scale(grid)
    assert [t.name for t in fig.data] == [f"n = {n}" for n in C.POWER_NS] and len(fig.data[0].x) == 6


def test_the_dimension_chart_sorts_by_p_and_uses_a_log_axis():
    rows = [{"p": 20, "dcor": 0.7, "dcor2_v": 0.5, "dcor2_u": 0.0}, {"p": 1, "dcor": 0.2, "dcor2_v": 0.05, "dcor2_u": 0.0}]
    fig = V.build_dimension(rows)
    assert list(fig.data[0].x) == [1, 20] and fig.layout.xaxis.type == "log" and len(fig.data) == 3
