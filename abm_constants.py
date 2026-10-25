"""Konstanten: Regler-Grenzen, Formen und Maße, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Pearson-Kern kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): ein Datensatz mit n = 300 und B = 499 Permutationen kostet für die Distanzkorrelation und Hoeffdings D je
B·n² = 4,5·10^7 Zellen (rund 0,5 s für alle sechs Maße); die Wiederholungsexperimente stehen vorgerechnet im Raster (data/dependence_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

N_MIN, N_MAX, DEFAULT_N = 20, 300, 100
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0.5, 4.0, 1.0
BINS_MIN, BINS_MAX, DEFAULT_BINS = 3, 20, 5
PERM_MIN, PERM_MAX, DEFAULT_PERM = 99, 499, 199
DEFAULT_SEED = 35
ALPHA = 0.05
PERM_BUDGET = 5 * 10**7          # B · n² höchstens (Distanzkorrelation und Hoeffding-D rechnen je Permutation eine n × n-Matrix)

# Formen: Schlüssel -> Anzeigename (die Reihenfolge ist die der Tabellen)
SHAPES = {
    "unabhaengig": "unabhängig",
    "linear": "linear",
    "monoton": "monoton (e^(3x))",
    "parabel": "Parabel",
    "sinus": "Sinus",
    "kreis": "Kreis",
    "streuung": "Streuung wächst",
    "schachbrett": "Schachbrett",
    "gemeinsam": "gemeinsame Streuung",
}
NOISE_FREE_SHAPES = ("unabhaengig", "schachbrett", "gemeinsam")      # hier wirkt der Rauschregler nicht

# Maße: Schlüssel -> Anzeigename
MEASURES = {
    "pearson": "Pearson r",
    "spearman": "Spearman ρ_S",
    "dcor": "Distanzkorrelation",
    "xi": "Chatterjee ξ",
    "hoeffding": "Hoeffding D",
    "mi": "Transinformation",
}
MEASURE_SHORT = {"pearson": "Pearson", "spearman": "Spearman", "dcor": "dCor", "xi": "ξ", "hoeffding": "Hoeffding D", "mi": "Transinformation"}

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
POWER_NS = (30, 100, 300)
POWER_RUNS = {30: 250, 100: 250, 300: 100}          # Läufe je Seed
POWER_B = 199
NOISE_N, NOISE_RUNS = 100, 100
NOISE_CELLS = {"parabel": (0.5, 1.0, 2.0, 4.0), "sinus": (0.5, 1.0, 2.0, 4.0), "kreis": (0.5, 1.0, 2.0, 4.0), "linear": (1.0, 2.0, 4.0, 8.0)}
RASTER_SHAPES = ("unabhaengig", "linear", "sinus", "kreis", "schachbrett", "streuung")
RASTER_KS = (3, 5, 8, 12, 20)
RASTER_N, RASTER_RUNS = 100, 100

# Live klein: Dimension (E5) und Chatterjee-ξ (E6)
DIM_N, DIM_PS, DIM_RUNS = 50, (1, 2, 5, 10, 20, 50, 100, 200), 25     # Läufe je Seed
DIM_GRID_PS = (500, 1000)                                               # die größten Dimensionen stehen nur im Raster
DIM_POWER_N, DIM_POWER_B, DIM_POWER_PS = 100, 99, (1, 2, 5, 20, 100)
XI_N, XI_REPS = 1000, 40
XI_FUNCTIONS = ("x²", "sin(3πx)", "|x|", "e^(3x)")
XI_NS = (30, 100, 300, 1000, 3000)

COLOR_POINTS = "#4c78a8"
COLOR_RASTER = "rgba(128,128,128,0.7)"
COLOR_TRUE = "#222222"
COLOR_FOUND = "#2F6B65"
COLOR_MISSED = "#b0b0b0"
MEASURE_COLORS = {"pearson": "#4c78a8", "spearman": "#f58518", "dcor": "#54a24b", "xi": "#b279a2", "hoeffding": "#e45756", "mi": "#72b7b2"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"shape": "parabel", "n": 100, "noise": 1.0, "bins": 5, "perms": 199, "seed": DEFAULT_SEED},
    "Kreis": {"shape": "kreis", "n": 100, "noise": 1.0, "bins": 5, "perms": 199, "seed": DEFAULT_SEED},
    "Sinus": {"shape": "sinus", "n": 100, "noise": 1.0, "bins": 5, "perms": 199, "seed": DEFAULT_SEED},
    "Schachbrett": {"shape": "schachbrett", "n": 100, "noise": 1.0, "bins": 5, "perms": 199, "seed": DEFAULT_SEED},
    "Gemeinsame Streuung (ρ = 0)": {"shape": "gemeinsam", "n": 300, "noise": 1.0, "bins": 5, "perms": 199, "seed": DEFAULT_SEED},
    "Linear (der Preis der Allgemeinheit)": {"shape": "linear", "n": 30, "noise": 1.0, "bins": 5, "perms": 199, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen des vorgerechneten Rasters (data/dependence_grid.json); tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Parabel, n = 100: Pearson und Spearman finden die Abhängigkeit in der Messreihe nur in 12,3 % und 13,3 % der Läufe, Distanzkorrelation, ξ, Hoeffding-D und Transinformation in 100 %.",
    "Kreis": "Kreis, n = 100: Pearson findet die Abhängigkeit in 0,7 % der Läufe, Spearman in 0,2 %, die Distanzkorrelation in 29,1 %, ξ in 41,2 %, Hoeffding-D in 91,5 % und die Transinformation in 100 %.",
    "Sinus": "Sinus, n = 100: ξ findet die Abhängigkeit in 100 % der Läufe, die Distanzkorrelation in 65,7 %, Pearson in 50,7 %: hier gewinnt ein anderes Maß als beim Kreis.",
    "Schachbrett": "Schachbrett (Pearson-Korrelation der Verteilung exakt null), n = 100: Pearson 7,1 %, Spearman 8,0 %, Distanzkorrelation 13,8 %, ξ 32,7 %, Hoeffding-D 20,9 %, Transinformation 95,9 % der Läufe.",
    "Gemeinsame Streuung (ρ = 0)": "t(3) mit gemeinsamer Skala, ρ = 0, n = 300: die Distanzkorrelation findet die Abhängigkeit in 77,2 % der Läufe, der Pearson-Permutationstest in 38,2 %, Spearman, ξ und Hoeffding-D in 6,5 bis 10,2 %.",
    "Linear (der Preis der Allgemeinheit)": "Linear, n = 30: Pearson findet die Abhängigkeit in 96,1 % der Läufe, ξ nur in 53,0 % und die Transinformation in 37,7 %: die Allgemeinheit kostet Teststärke.",
}
