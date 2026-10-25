"""App: jedes Preset, der Permalink, jeder Experiment-Knopf einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d: jeder Knopf einzeln im AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import abm_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["n_slider"] == preset["n"] and at.session_state["shape_select"] == preset["shape"]
    assert len(at.get("plotly_chart")) == 8                       # zwei Abbildungen der Hauptansicht, vier Raster-Experimente, zwei von ξ; die Dimension erst nach dem Knopf


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["shape"] = "kreis"
    at.query_params["n"] = "99999"
    at.query_params["noise"] = "0.1"
    at.query_params["k"] = "abc"          # ungültig: wird ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["shape_select"] == "kreis" and at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["noise_slider"] == C.NOISE_MIN and at.session_state["bins_slider"] == C.DEFAULT_BINS


def test_six_metrics_show_the_measures_of_the_current_data():
    at = _app()
    assert [m.label for m in at.metric] == list(C.MEASURES.values()) and len(at.metric) == 6
    pearson = float(at.metric[0].value)
    at.sidebar.selectbox(key="shape_select").set_value("linear").run()
    assert not at.exception and float(at.metric[0].value) > 0.5 > abs(pearson)


def test_the_headline_verdict_changes_with_the_shape():
    at = _app()
    assert "r und ρ_S sehen nichts" in _texts(at)                         # Parabel: die linearen Maße sind blind
    at.sidebar.selectbox(key="shape_select").set_value("unabhaengig").run()
    assert "unabhängig erzeugt" in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("power_n", 30), ("power_n", 300), ("noise_shape", "linear"), ("noise_shape", "parabel"), ("raster_shape", "linear"), ("raster_shape", "unabhaengig")):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, [str(e) for e in at.exception])


def test_the_dimension_button_shows_chart_table_and_verdict():
    at = _app()
    assert "Bei unabhängigen Daten und n = 50" not in _texts(at)
    at.button(key="btn_dim").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    assert "Bei unabhängigen Daten und n = 50" in text and "(p = 1000)" in text and "| 200 |" in text and "| 500 |" in text


def test_the_dimension_power_button_shows_chart_and_verdict():
    at = _app()
    assert "Hängt nur die erste Spalte ab" not in _texts(at)
    at.button(key="btn_dimpower").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Hängt nur die erste Spalte ab (n = 100)" in _texts(at)


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Bei n = 100 bleiben Pearson und Spearman", "Bei linearer Abhängigkeit und n = 30", "Bei unabhängigen Daten liegt der Fehlalarm", "beim kleinsten Rauschen", "die Transinformation findet die Abhängigkeit bei k = 3",
                   "Bei n = 300 verwirft der Distanzkorrelationstest", "ist ξ(x, y) =", "Für y = sin(3πx) ohne jedes Rauschen"):
        assert needle in text, needle


def test_corner_combination_of_all_minima_and_all_maxima():
    for edge in ("min", "max"):
        at = _app()
        for key in ("n_slider", "noise_slider", "bins_slider", "perm_slider"):
            w = at.sidebar.slider(key=key)
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.selectbox(key="shape_select").set_value("kreis" if edge == "min" else "gemeinsam")
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
