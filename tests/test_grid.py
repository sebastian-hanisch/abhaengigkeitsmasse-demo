"""Das vorgerechnete Raster (data/dependence_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probezelle)."""

import pytest

import abm_algorithm as K
import abm_constants as C
import abm_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/dependence_grid.json fehlt: python tools/make_grid.py"
    return g


def test_probe_cell_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    probe = ev.grid_probe()
    assert probe["cell"]["runs"] == grid["probe"]["cell"]["runs"] and probe["cell"]["rate"] == grid["probe"]["cell"]["rate"]     # Trefferanteile und p-Werte exakt
    assert probe["p"] == grid["probe"]["p"]
    assert probe["cell"]["sd"] == pytest.approx(grid["probe"]["cell"]["sd"], rel=1e-9)                                           # abgeleitete Zahlen: Summationsreihenfolge je Plattform (DEMO-PLAYBOOK §4)
    assert probe["stats"] == pytest.approx(grid["probe"]["stats"], rel=1e-9)
    assert 0 < sum(grid["probe"]["cell"]["rate"].values()) < 6                                                                   # die Probe sieht Treffer und Nichttreffer


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return sorted(tuple(r[n] for n in names) for r in rows)

    assert keys(grid["power"], "shape", "n") == sorted((s, n) for s in C.SHAPES for n in C.POWER_NS)
    assert keys(grid["noise"], "shape", "noise") == sorted((s, nz) for s, levels in C.NOISE_CELLS.items() for nz in levels)
    assert keys(grid["raster"], "shape", "bins") == sorted((s, k) for s in C.RASTER_SHAPES for k in C.RASTER_KS)
    assert keys(grid["dim"], "p") == sorted((p,) for p in C.DIM_GRID_PS)


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(r["runs"] == 4 * C.POWER_RUNS[r["n"]] and r["B"] == C.POWER_B and r["bins"] == C.DEFAULT_BINS for r in grid["power"])
    assert all(r["runs"] == 4 * C.NOISE_RUNS and r["n"] == C.NOISE_N for r in grid["noise"])
    assert all(r["runs"] == 4 * C.RASTER_RUNS and r["n"] == C.RASTER_N and r["which"] == ["mi"] for r in grid["raster"])
    assert all(r["n"] == C.DIM_N and r["runs"] == 4 * C.DIM_RUNS for r in grid["dim"])


def test_every_rate_is_a_probability_and_a_multiple_of_the_run_count(grid):
    for r in grid["power"] + grid["noise"] + grid["raster"]:
        per_seed_runs = r["runs"] // len(C.GRID_SEEDS)
        for k, v in r["rate"].items():
            assert 0.0 <= v <= 1.0 and 0.0 <= r["sd"][k] <= 0.5
            assert v * r["runs"] == pytest.approx(round(v * r["runs"]), abs=1e-6) and per_seed_runs > 0


def test_the_false_alarm_rate_of_independent_data_stays_near_five_percent(grid):
    for n in C.POWER_NS:
        cell = ev.find(grid["power"], shape="unabhaengig", n=n)
        se = (0.05 * 0.95 / cell["runs"]) ** 0.5
        assert all(abs(v - 0.05) < 4 * se for v in cell["rate"].values()), (n, cell["rate"])


def test_the_linear_measures_are_blind_on_the_symmetric_shapes_at_every_n(grid):
    for shape in ("parabel", "kreis", "schachbrett"):
        for n in C.POWER_NS:
            r = ev.find(grid["power"], shape=shape, n=n)["rate"]
            assert r["pearson"] <= 0.13 and r["spearman"] <= 0.14, (shape, n, r)


def test_no_measure_wins_everywhere(grid):
    wins = {}
    for shape in C.SHAPES:
        if shape != "unabhaengig":
            for k in ev.winners(ev.find(grid["power"], shape=shape, n=100)):
                wins[k] = wins.get(k, 0) + 1
    assert set(wins) == set(K.MEASURE_KEYS) and max(wins.values()) < 8             # jedes Maß gewinnt irgendwo, keines überall (n = 100, acht abhängige Formen)


def test_the_best_measure_changes_with_the_shape(grid):
    rate = lambda shape, k: ev.find(grid["power"], shape=shape, n=100)["rate"][k]          # noqa: E731
    assert rate("sinus", "xi") > rate("sinus", "dcor") + 0.3
    assert rate("kreis", "mi") > rate("kreis", "dcor") + 0.6 and rate("kreis", "hoeffding") > rate("kreis", "dcor") + 0.5
    assert rate("streuung", "mi") > rate("streuung", "dcor") + 0.25
    assert rate("gemeinsam", "dcor") > rate("gemeinsam", "spearman") + 0.25


def test_the_price_of_generality_at_small_n(grid):
    r = ev.find(grid["power"], shape="linear", n=30)["rate"]
    assert r["pearson"] > 0.95 and r["xi"] < 0.6 and r["mi"] < 0.4


def test_shared_scale_is_found_best_by_the_distance_correlation(grid):
    for n in (100, 300):
        r = ev.find(grid["power"], shape="gemeinsam", n=n)["rate"]
        assert r["dcor"] == max(r.values()) and r["dcor"] > r["pearson"] > r["spearman"]
        assert max(r["spearman"], r["xi"], r["hoeffding"]) < 0.11


def test_the_noise_ranking_changes_and_everything_fades_at_high_noise(grid):
    circle = {r["noise"]: r["rate"] for r in grid["noise"] if r["shape"] == "kreis"}
    assert circle[2.0]["mi"] > 0.6 and max(v for k, v in circle[2.0].items() if k != "mi") < 0.15
    assert max(max(r["rate"].values()) for r in grid["noise"] if r["noise"] == 4.0 and r["shape"] != "linear") < 0.26


def test_the_raster_is_a_dial_with_an_optimum(grid):
    sinus = {r["bins"]: r["rate"]["mi"] for r in grid["raster"] if r["shape"] == "sinus"}
    assert sinus[3] < 0.2 and sinus[8] > 0.85 and sinus[20] < sinus[8] - 0.3
    linear = {r["bins"]: r["rate"]["mi"] for r in grid["raster"] if r["shape"] == "linear"}
    assert linear[3] > 0.95 and linear[20] < 0.3
    flat = {r["bins"]: r["rate"]["mi"] for r in grid["raster"] if r["shape"] == "unabhaengig"}
    assert all(abs(v - 0.05) < 0.03 for v in flat.values())


def test_the_large_dimensions_show_the_inflated_distance_correlation(grid):
    big = {r["p"]: r for r in grid["dim"]}
    assert big[1000]["dcor"] > big[500]["dcor"] > 0.98 and abs(big[500]["dcor2_u"]) < 0.01 and abs(big[1000]["dcor2_u"]) < 0.01


def test_verdicts_run_on_every_grid_row(grid):
    for n in C.POWER_NS:
        table = ev.power_table(grid, n)
        assert ev.verdict_power(table, n)[0] in ("info", "success")
    assert ev.verdict_price(ev.power_table(grid, 30))[0] == "info" and ev.verdict_false_alarm(grid)[0] in ("success", "warning")
    for shape in C.NOISE_CELLS:
        assert ev.verdict_noise(grid["noise"], shape)[0] == "info"
    for shape in C.RASTER_SHAPES:
        assert ev.verdict_raster(grid["raster"], shape)[0] in ("info", "warning")
    assert ev.verdict_shared_scale(grid)[0] == "info"
