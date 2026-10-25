"""Auswertung: Einzeldatensatz, Zellen des vorgerechneten Rasters, kleine Live-Experimente, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import abm_algorithm as K
import abm_constants as C
import abm_shapes as SH
import prs_algorithm as PA
import prs_scenario as S

GRID_PATH = Path(__file__).resolve().parent / "data" / "dependence_grid.json"
BLIND_LIMIT = 0.20             # „blind“: höchstens das Vierfache des Fehlalarms von 5 %


# --- Einzeldatensatz -------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    shape: str = "parabel"
    n: int = C.DEFAULT_N
    noise: float = C.DEFAULT_NOISE
    bins: int = C.DEFAULT_BINS
    perms: int = C.DEFAULT_PERM
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: tuple[float, ...]
    y: tuple[float, ...]
    perms_used: int
    stats: dict               # Maß -> Wert
    p: dict                   # Maß -> p-Wert
    found: tuple[str, ...]    # Maße mit p ≤ 5 %
    dependent: bool           # ist die Form abhängig erzeugt?


def affordable_perms(perms: int, n: int, budget: int = C.PERM_BUDGET) -> int:
    """So viele Permutationen, wie das Budget B · n² erlaubt (mindestens 49)."""
    return max(49, min(perms, budget // max(1, n * n)))


def analyse(s: Settings) -> Analysis:
    data_seed, perm_seed = S.substream_seeds(s.seed, 2)
    x, y = SH.generate(s.shape, s.n, S.VRng(data_seed), s.noise)
    B = affordable_perms(s.perms, s.n)
    stats, p = K.permutation_test(x, y, B, S.VRng(perm_seed), min(s.bins, s.n))
    found = tuple(k for k in K.MEASURE_KEYS if p[k] <= C.ALPHA)
    return Analysis(s, tuple(map(float, x)), tuple(map(float, y)), B, stats, p, found, s.shape != "unabhaengig")


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, shape: str, n: int, extra: float, s: int) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art, Form, n und den dritten Parameter: Rauschfaktor, Klassenzahl oder Dimension)."""
    kinds = ("power", "noise", "raster", "dim", "dimpower", "xi", "probe")
    return s * 1_000_003 + n * 101 + int(round(extra * 10)) * 17 + list(C.SHAPES).index(shape) * 1013 + kinds.index(kind) * 100_000_007


def power_cell(shape: str, n: int, runs_per_seed: int, seeds=C.GRID_SEEDS, B: int = C.POWER_B, bins: int = C.DEFAULT_BINS, noise: float = 1.0,
               which=K.MEASURE_KEYS, kind: str = "power") -> dict:
    """Anteil der Läufe, in denen der Permutationstest die Unabhängigkeit verwirft (p ≤ 5 %), je Maß; Mittel und Standardabweichung über die Seeds."""
    per_seed = []
    for s in seeds:
        rng = S.VRng(cell_seed(kind, shape, n, bins if kind == "raster" else noise, s))
        hits = dict.fromkeys(which, 0)
        for _ in range(runs_per_seed):
            x, y = SH.generate(shape, n, rng, noise)
            _, p = K.permutation_test(x, y, B, rng, bins, which)
            for k in which:
                hits[k] += p[k] <= C.ALPHA
        per_seed.append([hits[k] / runs_per_seed for k in which])
    arr = np.array(per_seed)
    return {"shape": shape, "n": n, "noise": noise, "bins": bins, "B": B, "runs": runs_per_seed * len(seeds), "which": list(which),
            "rate": {k: float(arr[:, i].mean()) for i, k in enumerate(which)}, "sd": {k: float(arr[:, i].std(ddof=1)) if len(seeds) > 1 else 0.0 for i, k in enumerate(which)}}


def power_grid_cell(shape: str, n: int) -> dict:
    """E1 und E4: Teststärke aller sechs Maße für eine Form und ein n."""
    return power_cell(shape, n, C.POWER_RUNS[n])


def noise_grid_cell(shape: str, noise: float) -> dict:
    """E2: Teststärke bei einem Rauschfaktor (n = 100)."""
    return power_cell(shape, C.NOISE_N, C.NOISE_RUNS, noise=noise, kind="noise")


def raster_grid_cell(shape: str, k: int) -> dict:
    """E3: Teststärke der Transinformation bei k × k Klassen (n = 100)."""
    return power_cell(shape, C.RASTER_N, C.RASTER_RUNS, bins=k, which=("mi",), kind="raster")


def dimension_cell(p: int, n: int = C.DIM_N, runs_per_seed: int = C.DIM_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E5: unabhängige Normalvektoren der Dimension p (beide Seiten): Mittel von dCor (V-Statistik), dCor² (V) und dCor² (bias-korrigiert)."""
    v2, u2 = [], []
    for s in seeds:
        rng = S.VRng(cell_seed("dim", "unabhaengig", p, 0, s))
        for _ in range(runs_per_seed):
            X, Y = rng.normal((n, p)), rng.normal((n, p))
            v2.append(K.dcor2(X, Y))
            u2.append(K.dcor2_unbiased(X, Y))
    v2, u2 = np.array(v2), np.array(u2)
    return {"p": p, "n": n, "runs": len(v2), "dcor": float(np.sqrt(v2).mean()), "dcor2_v": float(v2.mean()), "dcor2_u": float(np.nanmean(u2))}


def dimension_power_cell(p: int, n: int = C.DIM_POWER_N, B: int = C.DIM_POWER_B, runs_per_seed: int = C.DIM_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E5: Anteil der Läufe, in denen der Distanzkorrelationstest anschlägt, wenn nur die erste von p Spalten abhängt (y₁ = x₁ + 0,8ε), die übrigen unabhängig sind."""
    hits = total = 0
    for s in seeds:
        rng = S.VRng(cell_seed("dimpower", "linear", p, 0, s))
        for _ in range(runs_per_seed):
            X, Y = rng.normal((n, p)), rng.normal((n, p))
            Y[:, 0] = X[:, 0] + 0.8 * rng.normal(n)
            hits += K.dcor_permutation_p(X, Y, B, rng) <= C.ALPHA
            total += 1
    return {"p": p, "n": n, "B": B, "runs": total, "rate": hits / total}


XI_FUNCTIONS = {"x²": lambda x: x**2, "sin(3πx)": lambda x: np.sin(3.0 * math.pi * x), "|x|": np.abs, "e^(3x)": lambda x: np.exp(3.0 * x)}


def xi_function_cell(name: str, n: int = C.XI_N, reps: int = C.XI_REPS) -> dict:
    """E6: y = f(x) ohne Rauschen, x gleichverteilt in (−1, 1): Mittel von ξ(x, y), ξ(y, x) und |Spearman|."""
    rng = S.VRng(cell_seed("xi", "linear", n, list(XI_FUNCTIONS).index(name), 1))
    fwd, back, rho = [], [], []
    for _ in range(reps):
        x = rng.uniform(n) * 2.0 - 1.0
        y = XI_FUNCTIONS[name](x)
        fwd.append(K.chatterjee_xi(x, y))
        back.append(K.chatterjee_xi(y, x))
        rho.append(abs(PA.spearman_r(x, y)))
    return {"function": name, "n": n, "reps": reps, "xi_xy": float(np.mean(fwd)), "xi_yx": float(np.mean(back)), "spearman": float(np.mean(rho))}


def xi_size_cell(n: int, reps: int = C.XI_REPS) -> dict:
    """E6: ξ gegen n für y = sin(3πx) ohne Rauschen (Populationswert 1)."""
    rng = S.VRng(cell_seed("xi", "sinus", n, 0, 2))
    vals = []
    for _ in range(reps):
        x = rng.uniform(n) * 2.0 - 1.0
        vals.append(K.chatterjee_xi(x, np.sin(3.0 * math.pi * x)))
    return {"n": n, "reps": reps, "xi": float(np.mean(vals))}


def grid_probe() -> dict:
    """Kleine feste Zelle und ein fester Einzeldatensatz, die der Test nachrechnet: ändert sich Erzeuger oder Statistik, passen sie nicht mehr zum gespeicherten Raster."""
    a = analyse(Settings("sinus", 60, 1.0, 5, 99, 11))
    return {"cell": power_cell("kreis", 40, 8, seeds=(7, 8), B=49, kind="probe"), "stats": a.stats, "p": a.p}


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for row in rows:
        if all(row.get(k) == v or (isinstance(v, float) and isinstance(row.get(k), (int, float)) and abs(row[k] - v) < 1e-9) for k, v in keys.items()):
            return row
    return None


def power_table(grid: dict, n: int) -> dict:
    """Form -> Zelle für ein n."""
    return {shape: find(grid["power"], shape=shape, n=n) for shape in C.SHAPES}


def winners(cell: dict, margin: float = 0.02) -> list[str]:
    """Maße, deren Teststärke höchstens `margin` unter der besten liegt."""
    best = max(cell["rate"].values())
    return [k for k in K.MEASURE_KEYS if cell["rate"][k] >= best - margin]


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------------

def _names(keys) -> str:
    return ", ".join(C.MEASURE_SHORT[k] for k in keys)


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Welche Maße verwerfen die Unabhängigkeit?"""
    s = a.settings
    base = (f"Gemessen: r = **{a.stats['pearson']:.2f}**, ρ_S = **{a.stats['spearman']:.2f}**, dCor = **{a.stats['dcor']:.2f}**, ξ = **{a.stats['xi']:.2f}**, "
            f"D = **{a.stats['hoeffding']:.3f}**, Transinformation = **{a.stats['mi']:.2f}** nat ({C.SHAPES[s.shape]}, n = {s.n}).")
    missed = tuple(k for k in K.MEASURE_KEYS if k not in a.found)
    if not a.dependent:
        if a.found:
            return "info", f"{base} Die Daten sind unabhängig erzeugt: **{_names(a.found)}** schlägt trotzdem an, ein Fehlalarm (erwartet: etwa 5 % je Maß)."
        return "success", f"{base} Die Daten sind unabhängig erzeugt, und kein Maß schlägt an."
    if not missed:
        return "success", f"{base} Die Abhängigkeit ist da, und alle sechs Maße finden sie (p ≤ 5 %)."
    if not a.found:
        return "warning", f"{base} Die Abhängigkeit ist da, aber kein Maß findet sie bei diesem n und diesem Rauschen."
    linear_blind = "pearson" in missed and "spearman" in missed
    tail = " r und ρ_S sehen nichts, obwohl y von x abhängt." if linear_blind else ""
    return "info", f"{base} Es finden die Abhängigkeit: **{_names(a.found)}**; es übersehen sie: {_names(missed)}.{tail}"


def verdict_power(table: dict, n: int) -> tuple[str, str]:
    """E1: in wie vielen abhängigen Formen die linearen Maße höchstens 20 % erreichen (das Vierfache der 5 % Fehlalarm) und ob ein Maß überall vorn liegt."""
    deps = [s for s in C.SHAPES if s != "unabhaengig"]
    blind = [s for s in deps if max(table[s]["rate"]["pearson"], table[s]["rate"]["spearman"]) <= BLIND_LIMIT]
    win = {s: winners(table[s]) for s in deps}
    always = [k for k in K.MEASURE_KEYS if all(k in w for w in win.values())]
    lone = [s for s in deps if len(win[s]) <= 2]
    text = (f"Bei n = {n} bleiben Pearson und Spearman in **{len(blind)} von {len(deps)}** abhängigen Formen bei höchstens {100 * BLIND_LIMIT:.0f} % ({', '.join(C.SHAPES[s] for s in blind)}). "
            + (f"Überall unter den Besten (höchstens 2 Punkte hinter dem stärksten): {_names(always)}." if always else "**Kein Maß** liegt in allen Formen vorn"
               + (": nur ein oder zwei Maße führen bei " + "; ".join(f"{C.SHAPES[s]} ({_names(win[s])})" for s in lone) + "." if lone else ".")))
    return ("info" if not always else "success"), text


def verdict_price(table: dict) -> tuple[str, str]:
    """E1: der Preis der Allgemeinheit bei linearer Abhängigkeit und kleinem n."""
    c = table["linear"]["rate"]
    return "info", (f"Bei linearer Abhängigkeit und n = {table['linear']['n']} finden Pearson in **{100 * c['pearson']:.1f} %** der Läufe, die allgemeinen Maße aber weniger: "
                    f"ξ **{100 * c['xi']:.1f} %**, Transinformation **{100 * c['mi']:.1f} %**, Distanzkorrelation {100 * c['dcor']:.1f} %.")


def verdict_false_alarm(grid: dict) -> tuple[str, str]:
    """E1: Fehlalarmrate bei unabhängigen Daten über alle Maße und n."""
    cells = [find(grid["power"], shape="unabhaengig", n=n) for n in C.POWER_NS]
    rates = [(c["rate"][k], c) for c in cells for k in K.MEASURE_KEYS]
    lo, hi = min(r for r, _ in rates), max(r for r, _ in rates)
    se = math.sqrt(0.05 * 0.95 / min(c["runs"] for c in cells))
    kind = "success" if lo > 0.05 - 3 * se and hi < 0.05 + 3 * se else "warning"
    return kind, f"Bei unabhängigen Daten liegt der Fehlalarm aller Maße bei **{100 * lo:.1f} bis {100 * hi:.1f} %** (erwartet 5 %, Monte-Carlo-Fehler bis ±{100 * 3 * se:.1f} Punkte)."


def verdict_noise(rows: list[dict], shape: str) -> tuple[str, str]:
    """E2: Teststärke gegen das Rauschen; genannt werden das gröbste und das feinste Rauschen."""
    rows = sorted((r for r in rows if r["shape"] == shape), key=lambda r: r["noise"])
    lo, hi = rows[0], rows[-1]
    best_lo, best_hi = max(lo["rate"], key=lo["rate"].get), max(hi["rate"], key=hi["rate"].get)
    text = (f"{C.SHAPES[shape]}: beim kleinsten Rauschen ({lo['noise']:g}) findet {C.MEASURE_SHORT[best_lo]} die Abhängigkeit in **{100 * lo['rate'][best_lo]:.1f} %** der Läufe, "
            f"beim größten ({hi['noise']:g}) das stärkste Maß ({C.MEASURE_SHORT[best_hi]}) nur in **{100 * hi['rate'][best_hi]:.1f} %**.")
    return "info", text


def verdict_raster(rows: list[dict], shape: str) -> tuple[str, str]:
    """E3: Teststärke der Transinformation gegen die Klassenzahl; das Optimum und die Ränder."""
    rows = sorted((r for r in rows if r["shape"] == shape), key=lambda r: r["bins"])
    best = max(rows, key=lambda r: r["rate"]["mi"])
    first, last = rows[0], rows[-1]
    text = (f"{C.SHAPES[shape]}: die Transinformation findet die Abhängigkeit bei k = {first['bins']} in **{100 * first['rate']['mi']:.1f} %**, bei k = {best['bins']} (am besten) in "
            f"**{100 * best['rate']['mi']:.1f} %** und bei k = {last['bins']} in **{100 * last['rate']['mi']:.1f} %** der Läufe.")
    return ("warning" if best["rate"]["mi"] - min(first["rate"]["mi"], last["rate"]["mi"]) > 0.25 else "info"), text


def verdict_shared_scale(grid: dict) -> tuple[str, str]:
    """E4: gemeinsame Streuung bei n = 300: wer findet die Abhängigkeit trotz ρ = 0?"""
    c = find(grid["power"], shape="gemeinsam", n=300)
    r = c["rate"]
    others = [k for k in K.MEASURE_KEYS if k not in ("pearson", "dcor")]
    return "info", (f"Bei n = {c['n']} verwirft der Distanzkorrelationstest die Unabhängigkeit in **{100 * r['dcor']:.1f} %** der {c['runs']} Läufe, der Pearson-Permutationstest in "
                    f"**{100 * r['pearson']:.1f} %**; {', '.join(C.MEASURE_SHORT[k] + ' ' + format(100 * r[k], '.1f') + ' %' for k in others)}.")


def verdict_dimension(rows: list[dict]) -> tuple[str, str]:
    """E5: dCor der Stichprobe bei unabhängigen Daten wächst mit der Dimension; die korrigierte Fassung bleibt bei null."""
    rows = sorted(rows, key=lambda r: r["p"])
    first, last = rows[0], rows[-1]
    return "warning", (f"Bei unabhängigen Daten und n = {first['n']} ist die Distanzkorrelation der Stichprobe im Mittel **{first['dcor']:.2f}** (p = {first['p']}) und **{last['dcor']:.2f}** "
                       f"(p = {last['p']}); die bias-korrigierte dCor² bleibt bei {min(r['dcor2_u'] for r in rows):+.3f} bis {max(r['dcor2_u'] for r in rows):+.3f}.")


def verdict_dimension_power(rows: list[dict]) -> tuple[str, str]:
    """E5: Leistung des Tests, wenn nur eine von p Spalten abhängt."""
    rows = sorted(rows, key=lambda r: r["p"])
    first, last = rows[0], rows[-1]
    return "info", (f"Hängt nur die erste Spalte ab (n = {first['n']}), findet der Test sie bei p = {first['p']} in **{100 * first['rate']:.1f} %** der Läufe und bei p = {last['p']} "
                    f"in **{100 * last['rate']:.1f} %**.")


def verdict_xi(functions: list[dict]) -> tuple[str, str]:
    """E6: ξ ist nicht symmetrisch; genannt wird die Funktion mit dem größten Unterschied."""
    worst = max(functions, key=lambda r: r["xi_xy"] - r["xi_yx"])
    return "info", (f"Für y = {worst['function']} ist ξ(x, y) = **{worst['xi_xy']:.3f}**, aber ξ(y, x) = **{worst['xi_yx']:.3f}** (|Spearman| {worst['spearman']:.3f}): "
                    "ξ fragt, ob y eine Funktion von x ist, und das ist keine symmetrische Frage.")


def verdict_xi_size(rows: list[dict]) -> tuple[str, str]:
    """E6: ξ erreicht bei exakter Abhängigkeit erst bei großem n den Wert 1."""
    rows = sorted(rows, key=lambda r: r["n"])
    first, last = rows[0], rows[-1]
    return "info", (f"Für y = sin(3πx) ohne jedes Rauschen ist ξ bei n = {first['n']} im Mittel **{first['xi']:.3f}** und erreicht bei n = {last['n']} erst **{last['xi']:.3f}** "
                    "(Populationswert 1): der Stichprobenwert unterschätzt die Funktionalität bei kleinem n.")
