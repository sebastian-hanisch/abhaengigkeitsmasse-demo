"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds. Damit prüft der Test die Grenzen (≤ 12 %, > 0,25, 3 Standardfehler) und jede genannte Zahl direkt."""

import abm_algorithm as K
import abm_constants as C
import abm_evaluation as ev

KEYS = K.MEASURE_KEYS


def cell(shape="parabel", n=100, runs=1000, noise=1.0, bins=5, **rates):
    rate = dict.fromkeys(KEYS, 0.05)
    rate.update(rates)
    return {"shape": shape, "n": n, "runs": runs, "noise": noise, "bins": bins, "rate": rate}


def analysis(shape="parabel", found=(), n=100):
    stats = {"pearson": 0.031, "spearman": -0.012, "dcor": 0.4567, "xi": 0.3, "hoeffding": 0.0612, "mi": 0.3333}
    p = {k: (0.01 if k in found else 0.5) for k in KEYS}
    return ev.Analysis(ev.Settings(shape, n), (0.0,), (0.0,), 199, stats, p, tuple(found), shape != "unabhaengig")


# --- Hauptansicht -------------------------------------------------------------------------------------------------------------------------

def test_main_reports_all_six_measures_with_their_rounding():
    kind, text = ev.verdict_main(analysis(found=KEYS))
    assert kind == "success" and "alle sechs Maße finden sie" in text
    for needle in ("r = **0.03**", "ρ_S = **-0.01**", "dCor = **0.46**", "ξ = **0.30**", "D = **0.061**", "Transinformation = **0.33** nat", "(Parabel, n = 100)"):
        assert needle in text, needle


def test_main_with_no_finder_is_a_warning():
    kind, text = ev.verdict_main(analysis(found=()))
    assert kind == "warning" and "kein Maß findet sie" in text


def test_main_names_finders_and_missers_and_the_linear_blindness():
    kind, text = ev.verdict_main(analysis(found=("dcor", "xi")))
    assert kind == "info" and "**dCor, ξ**" in text and "Pearson, Spearman, Hoeffding D, Transinformation" in text
    assert "r und ρ_S sehen nichts" in text


def test_main_has_no_blindness_remark_if_one_linear_measure_finds_it():
    for found in (("pearson",), ("spearman", "dcor")):
        kind, text = ev.verdict_main(analysis(found=found))
        assert kind == "info" and "r und ρ_S sehen nichts" not in text


def test_main_for_independent_data_names_false_alarms():
    kind, text = ev.verdict_main(analysis("unabhaengig", found=("mi",)))
    assert kind == "info" and "unabhängig erzeugt" in text and "**Transinformation** schlägt trotzdem an" in text
    kind, text = ev.verdict_main(analysis("unabhaengig", found=()))
    assert kind == "success" and "kein Maß schlägt an" in text


# --- E1 ---------------------------------------------------------------------------------------------------------------------------------

def _table(n=100, rates=None, default=0.9):
    """Zelle je Form: alle Maße bei `default`, abweichende Werte je Form in `rates`; „unabhängig“ bei 0,05."""
    out = {}
    for shape in C.SHAPES:
        base = dict.fromkeys(KEYS, 0.05 if shape == "unabhaengig" else default)
        base.update((rates or {}).get(shape, {}))
        out[shape] = cell(shape, n, **base)
    return out


LONE = {"parabel": {"pearson": 0.10, "spearman": 0.05, "dcor": 1.0, "xi": 0.8, "hoeffding": 0.8, "mi": 0.8},
        "kreis": {"pearson": 0.10, "spearman": 0.05, "dcor": 0.3, "xi": 0.4, "hoeffding": 0.9, "mi": 1.0},
        "schachbrett": {"pearson": 0.10, "spearman": 0.05, "dcor": 0.2, "xi": 0.3, "hoeffding": 0.2, "mi": 0.96},
        "sinus": {"xi": 1.0}}


def test_power_counts_the_blind_shapes_and_names_the_shapes_led_by_one_or_two_measures():
    kind, text = ev.verdict_power(_table(rates=LONE), 100)
    assert kind == "info"
    assert "Bei n = 100 bleiben Pearson und Spearman in **3 von 8** abhängigen Formen bei höchstens 20 % (Parabel, Kreis, Schachbrett)." in text
    assert text.endswith("**Kein Maß** liegt in allen Formen vorn: nur ein oder zwei Maße führen bei Parabel (dCor); Sinus (ξ); Kreis (Transinformation); Schachbrett (Transinformation).")


def test_power_without_a_lone_leader_ends_after_the_first_clause():
    rates = {shape: {KEYS[i % 6]: 0.1} for i, shape in enumerate(s for s in C.SHAPES if s != "unabhaengig")}       # in jeder Form fällt ein anderes Maß heraus
    kind, text = ev.verdict_power(_table(rates=rates), 100)
    assert kind == "info" and text.endswith("**Kein Maß** liegt in allen Formen vorn.") and "nur ein oder zwei" not in text


def test_power_threshold_is_inclusive_at_twenty_percent():
    def table(pearson):
        t = _table(rates=LONE)
        for s in ("parabel", "kreis", "schachbrett"):
            t[s]["rate"]["pearson"] = pearson
        return t
    assert "**3 von 8**" in ev.verdict_power(table(0.20), 100)[1]
    assert "**0 von 8**" in ev.verdict_power(table(0.201), 100)[1]
    spear = _table(rates=LONE)
    spear["parabel"]["rate"]["spearman"] = 0.201
    assert "**2 von 8**" in ev.verdict_power(spear, 100)[1]


def test_power_says_success_if_one_measure_is_always_among_the_best():
    table = {s: cell(s, 100, **dict.fromkeys(KEYS, 1.0)) for s in C.SHAPES}
    kind, text = ev.verdict_power(table, 300)
    assert kind == "success" and text.endswith("Pearson, Spearman, dCor, ξ, Hoeffding D, Transinformation.") and "Bei n = 300" in text
    table["sinus"]["rate"]["pearson"] = 0.5                                  # ein Maß fällt in einer Form heraus
    kind, text = ev.verdict_power(table, 100)
    assert kind == "success" and "Überall unter den Besten (höchstens 2 Punkte hinter dem stärksten): Spearman" in text and "Pearson" not in text.split("Besten")[1]


def test_winners_margin_is_inclusive():
    c = cell("sinus", **{"dcor": 0.5, "xi": 0.25, "mi": 0.249})
    assert ev.winners(c, margin=0.25) == ["dcor", "xi"]                         # 0,5 − 0,25 = 0,25: ξ liegt genau auf der Grenze und zählt mit, 0,249 nicht
    assert ev.winners(cell(**{"dcor": 0.9}), margin=0.0) == ["dcor"]


def test_price_names_the_rates():
    table = {"linear": cell("linear", 30, pearson=0.961, xi=0.53, mi=0.377, dcor=0.941)}
    kind, text = ev.verdict_price(table)
    assert kind == "info" and "n = 30" in text and "Pearson in **96.1 %**" in text and "ξ **53.0 %**" in text and "Transinformation **37.7 %**" in text and "Distanzkorrelation 94.1 %" in text


def _alarm_grid(rates, runs=(1000, 1000, 400)):
    return {"power": [cell("unabhaengig", n, runs=r, **dict.fromkeys(KEYS, rate)) for n, r, rate in zip(C.POWER_NS, runs, rates)]}


def test_false_alarm_is_a_success_within_three_standard_errors():
    kind, text = ev.verdict_false_alarm(_alarm_grid((0.05, 0.05, 0.05)))
    assert kind == "success" and "**5.0 bis 5.0 %**" in text and "±3.3 Punkte" in text


def test_false_alarm_thresholds_use_the_smallest_run_count():
    assert ev.verdict_false_alarm(_alarm_grid((0.05, 0.05, 0.018)))[0] == "success"
    assert ev.verdict_false_alarm(_alarm_grid((0.05, 0.05, 0.017)))[0] == "warning"
    assert ev.verdict_false_alarm(_alarm_grid((0.05, 0.05, 0.082)))[0] == "success"
    assert ev.verdict_false_alarm(_alarm_grid((0.05, 0.05, 0.084)))[0] == "warning"
    assert ev.verdict_false_alarm(_alarm_grid((0.05, 0.05, 0.084), runs=(1000, 1000, 10**6)))[0] == "warning"


def test_false_alarm_names_the_lowest_and_the_highest_rate():
    grid = _alarm_grid((0.036, 0.062, 0.05))
    kind, text = ev.verdict_false_alarm(grid)
    assert kind == "success" and "**3.6 bis 6.2 %**" in text


# --- E2 bis E6 --------------------------------------------------------------------------------------------------------------------------

def test_noise_names_best_measure_at_lowest_and_highest_noise():
    rows = [cell("kreis", noise=0.5, hoeffding=1.0, mi=0.9), cell("kreis", noise=2.0, mi=0.7), cell("kreis", noise=4.0, mi=0.058, xi=0.048), cell("sinus", noise=1.0, xi=1.0)]
    kind, text = ev.verdict_noise(rows, "kreis")
    assert kind == "info"
    assert "beim kleinsten Rauschen (0.5) findet Hoeffding D die Abhängigkeit in **100.0 %**" in text
    assert "beim größten (4) das stärkste Maß (Transinformation) nur in **5.8 %**" in text


def test_raster_warns_if_the_optimum_beats_the_worse_edge_by_more_than_a_quarter():
    def rows(first, best, last):
        return [cell("sinus", bins=3, mi=first), cell("sinus", bins=8, mi=best), cell("sinus", bins=20, mi=last), cell("kreis", bins=3, mi=1.0)]
    kind, text = ev.verdict_raster(rows(0.103, 0.917, 0.5), "sinus")
    assert kind == "warning" and "bei k = 3 in **10.3 %**" in text and "bei k = 8 (am besten) in **91.7 %**" in text and "bei k = 20 in **50.0 %**" in text
    assert ev.verdict_raster(rows(0.5, 0.75, 0.5), "sinus")[0] == "info"                      # genau 0,25 Vorsprung: keine Warnung
    assert ev.verdict_raster(rows(0.5, 0.751, 0.5), "sinus")[0] == "warning"
    assert ev.verdict_raster(rows(0.2, 0.9, 0.6), "sinus")[0] == "warning" and ev.verdict_raster(rows(0.6, 0.9, 0.2), "sinus")[0] == "warning"


def test_shared_scale_names_the_two_stronger_measures_and_the_rest():
    grid = {"power": [cell("gemeinsam", 300, runs=400, dcor=0.7725, pearson=0.3825, mi=0.285, spearman=0.065, xi=0.1025, hoeffding=0.085)]}
    kind, text = ev.verdict_shared_scale(grid)
    assert kind == "info" and "Bei n = 300 verwirft der Distanzkorrelationstest" in text and "**77.2 %** der 400 Läufe" in text and "Pearson-Permutationstest in **38.2 %**" in text
    for needle in ("Spearman 6.5 %", "ξ 10.2 %", "Hoeffding D 8.5 %", "Transinformation 28.5 %"):
        assert needle in text, needle


def test_dimension_reports_first_and_last_rows_and_the_corrected_range():
    rows = [{"p": 200, "n": 50, "dcor": 0.97, "dcor2_v": 0.94, "dcor2_u": -0.0011}, {"p": 1, "n": 50, "dcor": 0.241, "dcor2_v": 0.061, "dcor2_u": 0.0033}, {"p": 20, "n": 50, "dcor": 0.78, "dcor2_v": 0.61, "dcor2_u": -0.0024}]
    kind, text = ev.verdict_dimension(rows)
    assert kind == "warning" and "n = 50" in text and "**0.24** (p = 1)" in text and "**0.97** (p = 200)" in text
    assert "-0.002 bis +0.003" in text


def test_dimension_power_reports_first_and_last_rows():
    rows = [{"p": 100, "n": 100, "rate": 0.13}, {"p": 1, "n": 100, "rate": 1.0}, {"p": 20, "n": 100, "rate": 0.91}]
    kind, text = ev.verdict_dimension_power(rows)
    assert kind == "info" and "n = 100" in text and "p = 1 in **100.0 %**" in text and "p = 100 in **13.0 %**" in text


def test_xi_verdict_names_the_function_with_the_largest_asymmetry():
    rows = [{"function": "e^(3x)", "xi_xy": 0.997, "xi_yx": 0.997, "spearman": 1.0}, {"function": "sin(3πx)", "xi_xy": 0.982, "xi_yx": 0.075, "spearman": 0.249},
            {"function": "x²", "xi_xy": 0.994, "xi_yx": 0.246, "spearman": 0.030}]
    kind, text = ev.verdict_xi(rows)
    assert kind == "info" and "y = sin(3πx)" in text and "ξ(x, y) = **0.982**" in text and "ξ(y, x) = **0.075**" in text and "|Spearman| 0.249" in text


def test_xi_size_verdict_names_the_smallest_and_largest_n():
    rows = [{"n": 3000, "xi": 0.994}, {"n": 30, "xi": 0.529}, {"n": 300, "xi": 0.941}]
    kind, text = ev.verdict_xi_size(rows)
    assert kind == "info" and "bei n = 30 im Mittel **0.529**" in text and "bei n = 3000 erst **0.994**" in text
