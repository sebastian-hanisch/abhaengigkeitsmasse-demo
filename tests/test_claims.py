"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/dependence_grid.json) oder über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Raster-Werte stehen in der README als Prozent mit einer Nachkommastelle; die Vergleiche prüfen die Rundung der App (`_p`) und nutzen
sonst eine Toleranz von einem halben Zehntelprozent. Live nachgerechnete Werte (Dimension, ξ) lassen drei Nachkommastellen zu."""

import re
from pathlib import Path

import pytest

import abm_algorithm as K
import abm_constants as C
import abm_evaluation as ev
import prs_scenario as S

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
KEYS = list(C.MEASURES)
TOL = 0.0006


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def _rate(grid, shape, n):
    return ev.find(grid["power"], shape=shape, n=n)["rate"]


def test_claim_the_table_at_n_100_is_the_grid_in_the_apps_rounding(grid):
    """README: die Tabelle „Teststärke bei n = 100“ (neun Zeilen, sechs Maße) Zeile für Zeile."""
    for shape in C.SHAPES:
        rates = _rate(grid, shape, 100)
        row = f"| {C.SHAPES[shape]} | " + " | ".join(_p(rates[k]) for k in KEYS) + " |"
        assert row in README, row


def test_claim_the_linear_measures_are_blind_in_four_shapes(grid):
    """README: Parabel, Kreis, Schachbrett, wachsende Streuung: Pearson höchstens 15,8 %, Spearman höchstens 13,3 % (n = 30, 100, 300); Parabel n = 100: 12,3 und 13,3 %."""
    cells = [(shape, n) for shape in ("parabel", "kreis", "schachbrett", "streuung") for n in C.POWER_NS]
    assert max(_rate(grid, s, n)["pearson"] for s, n in cells) == pytest.approx(0.158, abs=TOL)
    assert max(_rate(grid, s, n)["spearman"] for s, n in cells) == pytest.approx(0.133, abs=TOL)
    assert "**15,8 %**" in README and "**13,3 %**" in README and "15,8 %** (statt 100 %)" in README
    assert (_rate(grid, "parabel", 100)["pearson"], _rate(grid, "parabel", 100)["spearman"]) == pytest.approx((0.123, 0.133), abs=TOL)


def test_claim_the_distance_correlation_finds_what_the_linear_measures_miss(grid):
    """README: Parabel ab n = 100 und wachsende Streuung bei n = 300: Distanzkorrelation 100,0 %."""
    assert _rate(grid, "parabel", 100)["dcor"] == 1.0 and _rate(grid, "parabel", 300)["dcor"] == 1.0 and _rate(grid, "streuung", 300)["dcor"] == 1.0


def test_claim_no_measure_wins_everywhere(grid):
    """README: n = 100: Sinus ξ 100,0 (Distanzkorrelation 65,7); Kreis Transinformation 100,0 und Hoeffding 91,5 (Distanzkorrelation 29,1); wachsende Streuung Transinformation 98,1 gegen 64,6; gemeinsame Streuung Distanzkorrelation 42,4."""
    sinus, kreis, streuung, shared = (_rate(grid, s, 100) for s in ("sinus", "kreis", "streuung", "gemeinsam"))
    assert (sinus["xi"], sinus["dcor"]) == pytest.approx((1.0, 0.657), abs=TOL)
    assert (kreis["mi"], kreis["hoeffding"], kreis["dcor"], kreis["xi"]) == pytest.approx((1.0, 0.915, 0.291, 0.412), abs=TOL)
    assert (streuung["mi"], streuung["dcor"]) == pytest.approx((0.981, 0.646), abs=TOL) and shared["dcor"] == pytest.approx(0.424, abs=TOL)
    assert max(sinus, key=sinus.get) == "xi" and max(kreis, key=kreis.get) == "mi" and max(streuung, key=streuung.get) == "mi" and max(shared, key=shared.get) == "dcor"


def test_claim_the_price_of_generality(grid):
    """README: linear, n = 30: Pearson 96,1, Spearman 94,3, Distanzkorrelation 94,1, Hoeffding 90,6, ξ 53,0, Transinformation 37,7; n = 100: ξ 91,9 und Transinformation 99,0."""
    r30, r100 = _rate(grid, "linear", 30), _rate(grid, "linear", 100)
    assert [r30[k] for k in KEYS] == pytest.approx([0.961, 0.943, 0.941, 0.530, 0.906, 0.377], abs=TOL)
    assert (r100["xi"], r100["mi"]) == pytest.approx((0.919, 0.990), abs=TOL)


def test_claim_false_alarm_range(grid):
    """README: unabhängige Daten, alle Maße und drei n: 3,0 bis 6,2 %."""
    rates = [v for n in C.POWER_NS for v in _rate(grid, "unabhaengig", n).values()]
    assert (min(rates), max(rates)) == pytest.approx((0.030, 0.062), abs=TOL)


def test_claim_shared_scale(grid):
    """README: gemeinsame Streuung n = 300: Distanzkorrelation 77,2, Pearson 38,2, Transinformation 28,5, ξ 10,2, Hoeffding 8,5, Spearman 6,5; dCor bei n = 30 und 100: 24,1 und 42,4."""
    r = _rate(grid, "gemeinsam", 300)
    assert (r["dcor"], r["pearson"], r["mi"], r["xi"], r["hoeffding"], r["spearman"]) == pytest.approx((0.772, 0.382, 0.285, 0.102, 0.085, 0.065), abs=TOL)
    assert (_rate(grid, "gemeinsam", 30)["dcor"], _rate(grid, "gemeinsam", 100)["dcor"]) == pytest.approx((0.241, 0.424), abs=TOL)


def test_claim_noise(grid):
    """README: Parabel, Faktor 2: Pearson 7,2, Distanzkorrelation 86,8; Kreis, Faktor 2: Transinformation 73,2, übrige höchstens 12,2; Faktor 4: Parabel, Sinus, Kreis höchstens 24,5; linear, Faktor 2: 92,5 / 27,5 / 42,5."""
    cell = lambda shape, nz: ev.find(grid["noise"], shape=shape, noise=nz)["rate"]          # noqa: E731
    assert (cell("parabel", 2.0)["pearson"], cell("parabel", 2.0)["dcor"]) == pytest.approx((0.072, 0.868), abs=TOL)
    kreis = cell("kreis", 2.0)
    assert kreis["mi"] == pytest.approx(0.732, abs=TOL) and max(v for k, v in kreis.items() if k != "mi") == pytest.approx(0.122, abs=TOL)
    assert max(max(cell(s, 4.0).values()) for s in ("parabel", "sinus", "kreis")) == pytest.approx(0.245, abs=TOL)
    assert (cell("linear", 2.0)["pearson"], cell("linear", 2.0)["xi"], cell("linear", 2.0)["mi"]) == pytest.approx((0.925, 0.275, 0.425), abs=TOL)


def test_claim_the_raster_is_a_dial(grid):
    """README: Sinus 10,2 / 76,8 / 91,8 / 80,5 / 50,0 bei k = 3, 5, 8, 12, 20; linear 100,0 / 98,8 / 86,5 / 62,0 / 25,0; unabhängig 4,0 bis 6,2."""
    mi = lambda shape: [ev.find(grid["raster"], shape=shape, bins=k)["rate"]["mi"] for k in C.RASTER_KS]          # noqa: E731
    assert mi("sinus") == pytest.approx([0.102, 0.768, 0.918, 0.805, 0.500], abs=TOL)
    assert mi("linear") == pytest.approx([1.0, 0.988, 0.865, 0.620, 0.250], abs=TOL)
    assert (min(mi("unabhaengig")), max(mi("unabhaengig"))) == pytest.approx((0.040, 0.062), abs=TOL)


def test_claim_seed_spread_is_at_most_seven_points(grid):
    """README: die Streuung über die Seeds liegt in den Zellen bei höchstens 7,0 Punkten."""
    assert max(max(c["sd"].values()) for c in grid["power"]) == pytest.approx(0.070, abs=TOL)
    assert max(max(c["sd"].values()) for c in grid["noise"] + grid["raster"]) <= 0.07


def test_claim_dimension_live_and_grid(grid):
    """README: dCor bei p = 1 / 10 / 50 / 200 / 1 000: 0,241 / 0,660 / 0,894 / 0,970 / 0,994; dCor² (V) 0,061 bis 0,988; korrigiert zwischen −0,003 und +0,004."""
    live = {p: ev.dimension_cell(p) for p in C.DIM_PS}
    rows = list(live.values()) + list(grid["dim"])
    big = {r["p"]: r for r in grid["dim"]}
    assert [live[1]["dcor"], live[10]["dcor"], live[50]["dcor"], live[200]["dcor"], big[1000]["dcor"]] == pytest.approx([0.241, 0.660, 0.894, 0.970, 0.994], abs=0.0006)
    assert (min(r["dcor2_v"] for r in rows), max(r["dcor2_v"] for r in rows)) == pytest.approx((0.061, 0.988), abs=0.0006)
    assert (min(r["dcor2_u"] for r in rows), max(r["dcor2_u"] for r in rows)) == pytest.approx((-0.003, 0.004), abs=0.0006)


def test_claim_dimension_power():
    """README: nur die erste von p Spalten abhängig, n = 100: p = 1, 2, 5: 100,0 %, p = 20: 91,0 %, p = 100: 13,0 %."""
    rates = {p: ev.dimension_power_cell(p)["rate"] for p in C.DIM_POWER_PS}
    assert [rates[p] for p in (1, 2, 5, 20, 100)] == pytest.approx([1.0, 1.0, 1.0, 0.91, 0.13], abs=1e-9)


def test_claim_xi():
    """README: ξ(x, y) / ξ(y, x): x² 0,994 / 0,246, sin(3πx) 0,982 / 0,075, e^(3x) 0,997 / 0,997; ξ gegen n für sin(3πx): 0,529 / 0,833 / 0,941 / 0,982 / 0,994."""
    f = {name: ev.xi_function_cell(name) for name in ev.XI_FUNCTIONS}
    assert (f["x²"]["xi_xy"], f["x²"]["xi_yx"]) == pytest.approx((0.994, 0.246), abs=0.0006)
    assert (f["sin(3πx)"]["xi_xy"], f["sin(3πx)"]["xi_yx"]) == pytest.approx((0.982, 0.075), abs=0.0006)
    assert (f["e^(3x)"]["xi_xy"], f["e^(3x)"]["xi_yx"]) == pytest.approx((0.997, 0.997), abs=0.0006)
    assert [ev.xi_size_cell(n)["xi"] for n in C.XI_NS] == pytest.approx([0.529, 0.833, 0.941, 0.982, 0.994], abs=0.0006)


def test_claim_hand_values(dist_mini, four, full_table):
    """README: dCor((0, 1, 2), (0, 1, 0)) = 0,562 (dCov² = 8/81); dCor der vollständigen 2 × 2-Tafel = 0; ξ((1, 2, 3, 4), (1, 3, 2, 4)) = 0; Transinformation zweier gleicher Binärmerkmale = ln 2; D(x, x) = 1."""
    assert K.dcor(*dist_mini) == pytest.approx(0.562, abs=0.0006) and K.dcov2(*dist_mini) == pytest.approx(8 / 81)
    assert K.dcor(*full_table) == pytest.approx(0.0, abs=1e-7) and K.chatterjee_xi(*four) == pytest.approx(0.0, abs=1e-15)
    x = S.VRng(5).normal(20)
    assert K.hoeffding_d(x, x) == pytest.approx(1.0, abs=1e-12)


def test_claim_preset_help_numbers(grid):
    """Hilfetexte der Presets: jede Prozentzahl steht im Raster (n und Form des Presets) in der Rundung der App."""
    expected = {
        "Standardfall (Voreinstellung)": ("parabel", 100, ("pearson", "spearman", "dcor", "xi", "hoeffding", "mi")),
        "Kreis": ("kreis", 100, KEYS),
        "Sinus": ("sinus", 100, ("xi", "dcor", "pearson")),
        "Schachbrett": ("schachbrett", 100, KEYS),
        "Gemeinsame Streuung (ρ = 0)": ("gemeinsam", 300, ("dcor", "pearson", "spearman", "xi")),          # „6,5 bis 10,2 %“: Hoeffding-D liegt dazwischen
        "Linear (der Preis der Allgemeinheit)": ("linear", 30, ("pearson", "xi", "mi")),
    }
    assert set(expected) == set(C.PRESETS)
    shared = _rate(grid, "gemeinsam", 300)
    assert min(shared["spearman"], shared["xi"], shared["hoeffding"]) == shared["spearman"] and max(shared["spearman"], shared["xi"], shared["hoeffding"]) == shared["xi"]
    for name, (shape, n, keys) in expected.items():
        assert (C.PRESETS[name]["shape"], C.PRESETS[name]["n"]) == (shape, n)
        text = C.PRESET_HELP[name]
        for k in keys:
            value = _p(_rate(grid, shape, n)[k]).replace("100,0", "100")                  # „100 %“ statt „100,0 %“
            assert re.search(rf"(?<![\d,]){re.escape(value)}(?![\d,])", text), (name, k, value)
        mentioned = [float(m.replace(",", ".")) for m in re.findall(r"(\d+,\d+|\d+) %", text)]
        allowed = {float(_p(v).replace(",", ".")) for v in _rate(grid, shape, n).values()}
        assert all(m in allowed for m in mentioned), (name, mentioned)


def test_claim_readme_mentions_its_key_numbers():
    """Verbindet die README mit den Tests: die Zahlen, die dort als Kernbefunde stehen, kommen im Text vor."""
    for needle in ("15,8 %", "13,3 %", "77,2 %", "6,5 %", "91,8 %", "0,994", "3,0 bis 6,2 %", "73,2 %", "86,8 %", "0,529 / 0,833 / 0,941 / 0,982 / 0,994", "218 Sekunden", "höchstens 7,0 Punkten"):
        assert needle in README, needle


def test_claim_grid_build_time(grid):
    """README: 218 Sekunden auf 10 Prozessen."""
    assert grid["meta"]["seconds"] == 218
