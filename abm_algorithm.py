"""Kern: sechs Abhängigkeitsmaße (Pearson, Spearman, Distanzkorrelation, Chatterjee-ξ, Hoeffding-D, Transinformation) und ihre Permutationstests.

Pearson und Durchschnittsränge kommen aus prs_algorithm.py (wörtliche Kopie). Alle Maße laufen über EINE Funktion `statistics(x, y, perms, bins)`: Zeile r von `perms` ist eine
Anordnung von y (y[perms[r]]), x bleibt fest. Der beobachtete Wert ist die Identität als einzige Zeile, der Nullwert unter Unabhängigkeit sind zufällige Permutationen; alle
sechs Maße sehen dieselben Permutationen. Zufall nur über einen übergebenen VRng (prs_scenario).

Gleichstände: Pearson und Spearman (Durchschnittsränge) und ξ (Formel mit l_i) behandeln sie; Hoeffding-D und die Klasseneinteilung der Transinformation sind für Daten ohne
Gleichstände gedacht (die Formen der Demo sind stetig). Bei ξ werden Gleichstände in x nach der Reihenfolge der Eingabe aufgelöst."""

import math

import numpy as np

import abm_constants as C
import prs_algorithm as PA

MEASURE_KEYS = tuple(C.MEASURES)
CHUNK_CELLS = 4 * 10**6          # so viele Matrixzellen je Block (Speicher der B × n × n-Felder)


def _check(x, y, minimum: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x und y müssen eindimensional und gleich lang sein")
    if len(x) < minimum:
        raise ValueError(f"mindestens {minimum} Beobachtungen nötig")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise ValueError("eine Kennzahl ist konstant: das Maß ist nicht definiert")
    return x, y


def _blocks(reps: int, n: int):
    step = max(1, CHUNK_CELLS // (n * n))
    for s in range(0, reps, step):
        yield slice(s, min(reps, s + step))


# --- Ränge und Klassen -----------------------------------------------------------------------------------------------------------------------

def quantile_classes(v, k: int) -> np.ndarray:
    """Klasse 0 bis k − 1 nach dem Platz in der Reihe (gleich viele Werte je Klasse, Rest nach oben; Gleichstände nach Eingabereihenfolge)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if k < 2 or k > n:
        raise ValueError("die Klassenzahl muss zwischen 2 und n liegen")
    place = np.empty(n, dtype=np.int64)
    place[np.argsort(v, kind="stable")] = np.arange(n)
    return place * k // n                                     # place ≤ n − 1, also höchstens k − 1


# --- Distanzkorrelation ----------------------------------------------------------------------------------------------------------------------

def distance_matrix(X) -> np.ndarray:
    """Euklidische Abstände aller Paare von Zeilen (X: (n,) oder (n, p))."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return np.abs(X[:, None] - X[None, :])
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))


def double_center(D: np.ndarray) -> np.ndarray:
    """A_ij = d_ij − Zeilenmittel_i − Spaltenmittel_j + Gesamtmittel."""
    return D - D.mean(0)[None, :] - D.mean(1)[:, None] + D.mean()


def u_center(D: np.ndarray) -> np.ndarray:
    """U-zentrierte Matrix (bias-korrigiert, Diagonale null): d_ij − Σ_k d_ik/(n−2) − Σ_k d_kj/(n−2) + Σ d/((n−1)(n−2)); braucht n > 3."""
    n = len(D)
    if n < 4:
        raise ValueError("die bias-korrigierte Fassung braucht mindestens vier Beobachtungen")
    U = D - D.sum(0)[None, :] / (n - 2) - D.sum(1)[:, None] / (n - 2) + D.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(U, 0.0)
    return U


def dcov2(X, Y) -> float:
    """Quadrierte Distanzkovarianz (V-Statistik): Mittel aller n² Produkte A_ij · B_ij der doppelt zentrierten Abstandsmatrizen."""
    return float((double_center(distance_matrix(X)) * double_center(distance_matrix(Y))).mean())


def dcor2(X, Y) -> float:
    """Quadrierte Distanzkorrelation (V-Statistik) dCov² / √(dVar²(X) · dVar²(Y)); Fehler, wenn eine Seite konstant ist."""
    A, B = double_center(distance_matrix(X)), double_center(distance_matrix(Y))
    vx, vy = float((A * A).mean()), float((B * B).mean())
    if vx <= 0.0 or vy <= 0.0:
        raise ValueError("eine Seite ist konstant: die Distanzkorrelation ist nicht definiert")
    return float((A * B).mean()) / math.sqrt(vx * vy)


def dcor(X, Y) -> float:
    """Distanzkorrelation (Wurzel aus dcor2, 0 bis 1)."""
    return math.sqrt(max(dcor2(X, Y), 0.0))


def dcor2_unbiased(X, Y) -> float:
    """Bias-korrigierte quadrierte Distanzkorrelation (U-Statistik); kann negativ werden, Mittel unter Unabhängigkeit etwa null. NaN, wenn eine korrigierte Varianz nicht positiv ist."""
    n = len(X)
    A, B = u_center(distance_matrix(X)), u_center(distance_matrix(Y))
    scale = n * (n - 3)
    vx, vy = float((A * A).sum()) / scale, float((B * B).sum()) / scale
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float((A * B).sum()) / scale / math.sqrt(vx * vy)


def _dcov2_rows(A: np.ndarray, B: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """Mittel von A_ij · B_{π(i)π(j)} je Permutation π (Zeile von perms), in Blöcken."""
    out = np.empty(len(perms))
    for sl in _blocks(len(perms), len(A)):
        p = perms[sl]
        out[sl] = (A[None] * B[p[:, :, None], p[:, None, :]]).mean((1, 2))
    return out


def dcor_permutation_p(X, Y, B: int, rng) -> float:
    """Permutationstest der Distanzkorrelation für Vektoren beliebiger Dimension: (1 + #{dCov²* ≥ dCov²}) / (B + 1)."""
    A, Bm = double_center(distance_matrix(X)), double_center(distance_matrix(Y))
    n = len(A)
    perms = np.argsort(rng.uniform((B, n)), axis=1)
    obs = float((A * Bm).mean())
    return float((1 + int((_dcov2_rows(A, Bm, perms) >= obs - 1e-12).sum())) / (B + 1))


# --- Chatterjee-ξ ----------------------------------------------------------------------------------------------------------------------------

def _xi_counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """r_j = #{k: y_k ≤ y_j} und l_j = #{k: y_k ≥ y_j} je Beobachtung j."""
    s = np.sort(y)
    return np.searchsorted(s, y, side="right").astype(float), (len(y) - np.searchsorted(s, y, side="left")).astype(float)


def chatterjee_xi(x, y) -> float:
    """ξ_n(x, y) = 1 − n Σ|r_{i+1} − r_i| / (2 Σ l_i (n − l_i)) mit Beobachtungen nach x sortiert. Nicht symmetrisch: ξ(x, y) misst, wie gut y eine Funktion von x ist."""
    x, y = _check(x, y, 2)
    order = np.argsort(x, kind="stable")
    r, l = _xi_counts(y)
    n = len(x)
    return float(1.0 - n * np.abs(np.diff(r[order])).sum() / (2.0 * (l * (n - l)).sum()))


# --- Hoeffding-D -----------------------------------------------------------------------------------------------------------------------------

def hoeffding_d(x, y) -> float:
    """Hoeffdings D (auf 30 skaliert): 30 [(n−2)(n−3) D1 + D2 − 2(n−2) D3] / (n(n−1)(n−2)(n−3)(n−4)) mit D1 = Σ(Q_i−1)(Q_i−2), D2 = Σ(R_i−1)(R_i−2)(S_i−1)(S_i−2),
    D3 = Σ(R_i−2)(S_i−2)(Q_i−1); R, S Ränge, Q_i = 1 + #{j: x_j < x_i und y_j < y_i}. Für Daten ohne Gleichstände."""
    x, y = _check(x, y, 5)
    return float(statistics(x, y, np.arange(len(x))[None], which=("hoeffding",))["hoeffding"][0])


# --- Transinformation ------------------------------------------------------------------------------------------------------------------------

def mutual_information(x, y, k: int) -> float:
    """Transinformation (natürlicher Logarithmus) der k × k-Klassentafel aus Quantilklassen beider Achsen."""
    x, y = _check(x, y, 2)
    return float(statistics(x, y, np.arange(len(x))[None], bins=k, which=("mi",))["mi"][0])


def _mi_rows(xb: np.ndarray, Yb: np.ndarray, k: int) -> np.ndarray:
    reps, n = Yb.shape
    idx = xb[None, :] * k + Yb + (np.arange(reps) * k * k)[:, None]
    counts = np.bincount(idx.ravel(), minlength=reps * k * k).reshape(reps, k, k) / n
    px, py = counts.sum(2, keepdims=True), counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = counts * np.log(counts / (px * py))
    return np.nansum(terms, axis=(1, 2))


# --- alle Maße auf Permutationen ------------------------------------------------------------------------------------------------------------

def statistics(x, y, perms, bins: int = C.DEFAULT_BINS, which=MEASURE_KEYS) -> dict[str, np.ndarray]:
    """Die gewählten Maße für (x, y[perms[r]]) je Zeile r von perms; Pearson und Spearman mit Vorzeichen, die übrigen wie definiert (dCor 0 bis 1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    perms = np.asarray(perms)
    n = len(x)
    reps = len(perms)
    out: dict[str, np.ndarray] = {}
    rx = ry = None
    if "pearson" in which:
        out["pearson"] = PA.pearson_rows(np.broadcast_to(x, (reps, n)), y[perms])
    if "spearman" in which or "hoeffding" in which:
        rx, ry = PA.rank_average(x), PA.rank_average(y)
    if "spearman" in which:
        out["spearman"] = PA.pearson_rows(np.broadcast_to(rx, (reps, n)), ry[perms])
    if "dcor" in which:
        A, Bm = double_center(distance_matrix(x)), double_center(distance_matrix(y))
        vx, vy = float((A * A).mean()), float((Bm * Bm).mean())
        with np.errstate(invalid="ignore", divide="ignore"):           # eine konstante Seite gibt 0/0 = NaN
            out["dcor"] = np.sqrt(np.maximum(_dcov2_rows(A, Bm, perms), 0.0) / math.sqrt(vx * vy))
    if "xi" in which:
        order = np.argsort(x, kind="stable")
        r, l = _xi_counts(y)
        den = 2.0 * (l * (n - l)).sum()
        out["xi"] = 1.0 - n * np.abs(np.diff(r[perms][:, order], axis=1)).sum(1) / den
    if "hoeffding" in which:
        less_x = x[None, :] < x[:, None]                                  # [i, j]: x_j < x_i
        D = np.empty(reps)
        for sl in _blocks(reps, n):
            Yp = y[perms[sl]]
            Q = 1.0 + (less_x[None] & (Yp[:, None, :] < Yp[:, :, None])).sum(2)
            S = ry[perms[sl]]
            d1 = ((Q - 1) * (Q - 2)).sum(1)
            d2 = ((rx - 1) * (rx - 2) * (S - 1) * (S - 2)).sum(1)
            d3 = ((rx - 2) * (S - 2) * (Q - 1)).sum(1)
            D[sl] = 30.0 * ((n - 2) * (n - 3) * d1 + d2 - 2.0 * (n - 2) * d3) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
        out["hoeffding"] = D
    if "mi" in which:
        out["mi"] = _mi_rows(quantile_classes(x, bins), quantile_classes(y, bins)[perms], bins)
    return {k: out[k] for k in MEASURE_KEYS if k in out}


def random_permutations(rng, B: int, n: int) -> np.ndarray:
    """B zufällige Anordnungen von 0..n−1 (Sortierung gleichverteilter Zahlen)."""
    return np.argsort(rng.uniform((B, n)), axis=1)


def p_value(observed: float, null: np.ndarray, two_sided: bool) -> float:
    """(1 + #{T* ≥ T}) / (B + 1); zweiseitig vergleicht Beträge. Die kleine Toleranz fängt Rundungsunterschiede gleicher Werte ab."""
    if not np.isfinite(observed):
        return float("nan")
    if two_sided:
        observed, null = abs(observed), np.abs(null)
    return float((1 + int((null >= observed - 1e-12).sum())) / (len(null) + 1))


def permutation_test(x, y, B: int, rng, bins: int = C.DEFAULT_BINS, which=MEASURE_KEYS) -> tuple[dict[str, float], dict[str, float]]:
    """(Maße, p-Werte) der gewählten Maße auf einem Datensatz. Pearson und Spearman zweiseitig, die übrigen Maße rechtsseitig (große Werte sprechen gegen Unabhängigkeit)."""
    x, y = _check(x, y, 5)
    n = len(x)
    obs = statistics(x, y, np.arange(n)[None], bins, which)
    null = statistics(x, y, random_permutations(rng, B, n), bins, which)
    p = {k: p_value(float(obs[k][0]), null[k], k in ("pearson", "spearman")) for k in obs}
    return {k: float(v[0]) for k, v in obs.items()}, p
