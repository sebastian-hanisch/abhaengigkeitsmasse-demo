"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def dist_mini():
    """x = (0, 1, 2), y = (0, 1, 0). Abstandsmatrix x: [[0,1,2],[1,0,1],[2,1,0]], Zeilenmittel (1, 2/3, 1), Gesamtmittel 8/9; y: Zeilenmittel (1/3, 2/3, 1/3), Gesamtmittel 4/9.
    Doppelt zentriert: Σ A_ij B_ij = 72/81 (Diagonale 20+32+20, Nebendiagonalen 8−16+8 = 0), also dCov² = 8/81; dVar²(x) = 40/81, dVar²(y) = 16/81;
    dCor² = (8/81)/√(40/81 · 16/81) = 8/√640 = 0,3162, dCor = 0,5623."""
    return [0.0, 1.0, 2.0], [0.0, 1.0, 0.0]


@pytest.fixture
def four():
    """x = (1, 2, 3, 4), y = (1, 3, 2, 4). ξ: nach x sortiert sind die Ränge von y (1, 3, 2, 4), Σ|Δr| = 2 + 1 + 2 = 5, ξ = 1 − 3·5/(16−1) = 0.
    dCor ohne Gleichstandsstruktur: 0,8321 (Handrechnung: Abstandsmatrizen zentrieren, wie bei dist_mini)."""
    return [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]


@pytest.fixture
def full_table():
    """Vollständige 2 × 2-Tafel x = (0, 0, 1, 1), y = (0, 1, 0, 1): x und y sind unabhängig, dCov² = 0 (jede der vier Kombinationen kommt einmal vor)."""
    return [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]
