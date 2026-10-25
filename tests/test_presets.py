"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte, Adresszeile."""

import pytest

import abm_constants as C
import abm_evaluation as ev
import abm_presets as P


def test_presets_are_complete():
    assert "Standardfall (Voreinstellung)" in C.PRESETS
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name         # ein echter Text, nicht der Name
        assert set(preset) == set(P.PRESET_KEYS), name
        assert preset["shape"] in C.SHAPES
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.NOISE_MIN <= preset["noise"] <= C.NOISE_MAX and C.BINS_MIN <= preset["bins"] <= C.BINS_MAX
        assert C.PERM_MIN <= preset["perms"] <= C.PERM_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_the_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("shape_select", "n_slider", "noise_slider", "bins_slider", "perm_slider", "seed_input")] == [s.shape, s.n, s.noise, s.bins, s.perms, s.seed]


def test_every_preset_shows_a_different_shape_or_n():
    keys = {(p["shape"], p["n"]) for p in C.PRESETS.values()}
    assert len(keys) == len(C.PRESETS)


@pytest.mark.parametrize("raw,expected", [("99999", C.N_MAX), ("1", C.N_MIN), ("150", 150), ("abc", None), ("", None)])
def test_clean_value_clamps_integers(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["n_slider"], raw) == expected


@pytest.mark.parametrize("raw,expected", [("0.123", 0.5), ("9", C.NOISE_MAX), ("1.234", 1.23), ("x", None)])
def test_clean_value_clamps_and_rounds_floats(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["noise_slider"], raw) == expected


def test_clean_value_accepts_only_known_choices():
    spec = P.SETTING_SPECS["shape_select"]
    assert P.clean_value(spec, "kreis") == "kreis" and P.clean_value(spec, "würfel") is None and P.clean_value(spec, "") is None
    assert P.clean_value(P.SETTING_SPECS["seed_input"], "-5") == 0 and P.clean_value(P.SETTING_SPECS["seed_input"], str(C.SEED_MAX + 1)) == C.SEED_MAX
