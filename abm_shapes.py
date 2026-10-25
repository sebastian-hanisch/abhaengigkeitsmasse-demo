"""Szenario: neun Formen der Abhängigkeit zwischen zwei Kennzahlen (x: Abweichung des Auftragsvolumens vom Plan, y: Überstunden) einer Standort-Woche.

Zufall nur über den übergebenen VRng (prs_scenario, SplitMix64): gleiche Eingabe, gleicher Datensatz, auf jeder Plattform. Die Reihenfolge der Ziehungen ist je Form
festgelegt (Docstring von `generate`), damit Messreihen reproduzierbar bleiben.

Formen mit x ~ U(−1, 1) und standardnormalem ε: linear y = x + 0,8ε; monoton y = e^(3x) + ε; Parabel y = x² + 0,25ε; Sinus y = sin(3πx) + 0,6ε; Streuung y = x·ε; Kreis
(cos t, sin t) mit Rauschen 0,15 in beiden Koordinaten; Schachbrett: gleichverteilt auf den fünf Zellen (Ecken und Mitte) eines 3 × 3-Rasters über [−1, 1]²
(Pearson-Korrelation der Verteilung exakt null); gemeinsame Streuung: t(3) mit gemeinsamem Skalenfaktor und ρ = 0 (unkorreliert, aber abhängig).
Der Rauschfaktor skaliert den Rauschterm der Form; bei unabhängig, Schachbrett und gemeinsamer Streuung wirkt er nicht."""

import math

import numpy as np

import abm_constants as C
import prs_scenario as S

CHECKER_CELLS = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))      # besetzte Zellen des 3 × 3-Rasters (Zeile, Spalte): Ecken und Mitte
NOISE_OF = {"linear": 0.8, "monoton": 1.0, "parabel": 0.25, "sinus": 0.6, "streuung": 1.0, "kreis": 0.15}


def generate(shape: str, n: int, rng: S.VRng, noise: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Ein Datensatz (x, y) der Größe n. Reihenfolge der Ziehungen:
    gemeinsame Streuung: draw_pairs("t3", ρ = 0); Schachbrett: Zellenindex (below), dann Gleichverteilte (n, 2);
    Kreis: Winkel (gleichverteilt), Rauschen in x, Rauschen in y; alle übrigen: x gleichverteilt in (−1, 1), dann ε standardnormal."""
    if shape not in C.SHAPES:
        raise ValueError(f"unbekannte Form: {shape}")
    if n < 2:
        raise ValueError("mindestens zwei Beobachtungen nötig")
    if noise < 0:
        raise ValueError("der Rauschfaktor darf nicht negativ sein")
    if shape == "gemeinsam":
        X, Y = S.draw_pairs("t3", 1, n, 0.0, rng)
        return X[0], Y[0]
    if shape == "schachbrett":
        cell = rng.below(len(CHECKER_CELLS), n)
        u = rng.uniform((n, 2))
        rows, cols = np.array(CHECKER_CELLS).T
        return -1.0 + (cols[cell] + u[:, 0]) * 2.0 / 3.0, -1.0 + (rows[cell] + u[:, 1]) * 2.0 / 3.0
    if shape == "kreis":
        t = rng.uniform(n) * 2.0 * math.pi
        s = NOISE_OF["kreis"] * noise
        return np.cos(t) + s * rng.normal(n), np.sin(t) + s * rng.normal(n)
    x = rng.uniform(n) * 2.0 - 1.0
    e = rng.normal(n)
    if shape == "unabhaengig":
        return x, e
    if shape == "streuung":
        return x, x * e * noise
    s = NOISE_OF[shape] * noise
    base = {"linear": x, "monoton": np.exp(3.0 * x), "parabel": x**2, "sinus": np.sin(3.0 * math.pi * x)}[shape]
    return x, base + s * e


def curve(shape: str, x: np.ndarray) -> np.ndarray | None:
    """Die rauschfreie Kurve y = f(x) der Form (für die Anzeige); None, wo es keine Funktion von x gibt (Kreis, Schachbrett, Streuung, unabhängig, gemeinsame Streuung)."""
    if shape == "linear":
        return x
    if shape == "monoton":
        return np.exp(3.0 * x)
    if shape == "parabel":
        return x**2
    if shape == "sinus":
        return np.sin(3.0 * math.pi * x)
    return None
