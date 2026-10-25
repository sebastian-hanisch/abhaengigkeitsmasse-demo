"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import abm_constants as C
import abm_shapes as SH

SHAPE_COLORS = {"unabhaengig": "#9d9d9d", "linear": "#4c78a8", "monoton": "#72b7b2", "parabel": "#f58518", "sinus": "#b279a2", "kreis": "#e45756", "streuung": "#54a24b",
                "schachbrett": "#9c755f", "gemeinsam": "#222222"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def raster_edges(v, k: int) -> list[float]:
    """Die k − 1 Grenzen der Quantilklassen (Mitte zwischen den beiden Werten, an denen die Klasse wechselt); passt zu abm_algorithm.quantile_classes."""
    s = np.sort(np.asarray(v, dtype=float))
    n = len(s)
    firsts = [-(-j * n // k) for j in range(1, k)]                 # erster Platz der Klasse j: ⌈j·n/k⌉
    return [float((s[f - 1] + s[f]) / 2.0) for f in firsts]


def build_scatter(x, y, shape: str, bins: int | None = None, height: int = 360):
    """Streudiagramm der Messwerte; die rauschfreie Kurve gestrichelt, wo es sie gibt; das Klassenraster der Transinformation hellgrau."""
    x, y = np.asarray(x), np.asarray(y)
    fig = go.Figure()
    if bins:
        for e in raster_edges(x, bins):
            fig.add_vline(x=e, line=dict(color=C.COLOR_RASTER, width=1))
        for e in raster_edges(y, bins):
            fig.add_hline(y=e, line=dict(color=C.COLOR_RASTER, width=1))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Beobachtungen", marker=dict(color=C.COLOR_POINTS, size=7, opacity=0.8),
                             hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    xs = np.linspace(x.min(), x.max(), 200)
    ys = SH.curve(shape, xs) if shape in ("linear", "monoton", "parabel", "sinus") else None
    if ys is not None:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Kurve ohne Rauschen", line=dict(color=C.COLOR_TRUE, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="x: Abweichung des Auftragsvolumens vom Plan")
    fig.update_yaxes(title_text="y: Überstunden")
    return _base(fig, height)


def build_pvalues(p: dict, alpha: float = C.ALPHA, height: int = 300):
    """p-Werte der sechs Permutationstests als Punkte auf logarithmischer Achse; grün = verwirft die Unabhängigkeit, grau = nicht."""
    keys = list(C.MEASURES)
    names = [C.MEASURES[k] for k in keys]
    vals = [p[k] for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vals, y=names, mode="markers+text", text=[f"{v:.3f}" for v in vals], textposition="middle right", showlegend=False,
                             marker=dict(size=13, color=[C.COLOR_FOUND if v <= alpha else C.COLOR_MISSED for v in vals]), hovertemplate="p = %{x:.3f}<extra>%{y}</extra>"))
    fig.add_vline(x=alpha, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5), annotation_text="5 %", annotation_position="top")
    ticks = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    fig.update_xaxes(title_text="p-Wert des Permutationstests (logarithmisch)", type="log", range=[np.log10(0.0015), np.log10(2.5)], tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names, automargin=True)
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_power_heatmap(table: dict, n: int, height: int = 420):
    """Teststärke je Form (Zeilen) und Maß (Spalten) für ein n, als Heatmap mit Prozentangaben."""
    shapes = list(table)
    keys = list(C.MEASURES)
    z = [[100 * table[s]["rate"][k] for k in keys] for s in shapes]
    fig = go.Figure(go.Heatmap(z=z, x=[C.MEASURE_SHORT[k] for k in keys], y=[C.SHAPES[s] for s in shapes], zmin=0, zmax=100, colorscale="Greens", showscale=False,
                               texttemplate="%{z:.0f}", textfont=dict(size=13), hovertemplate="%{y}, %{x}: %{z:.1f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return _base(fig, height)


def build_noise(rows: list[dict], shape: str, height: int = 340):
    """Teststärke gegen den Rauschfaktor, eine Linie je Maß."""
    rows = sorted((r for r in rows if r["shape"] == shape), key=lambda r: r["noise"])
    fig = go.Figure()
    for k in C.MEASURES:
        fig.add_trace(go.Scatter(x=[r["noise"] for r in rows], y=[100 * r["rate"][k] for r in rows], mode="lines+markers", name=C.MEASURE_SHORT[k],
                                 line=dict(color=C.MEASURE_COLORS[k], width=3), hovertemplate="Rauschen %{x}: %{y:.1f} %<extra>" + C.MEASURE_SHORT[k] + "</extra>"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text="Rauschfaktor (1 = Standard der Form)", type="category")
    fig.update_yaxes(title_text="Anteil verworfener Unabhängigkeit (%)", range=[0, 102])
    return _base(fig, height)


def build_raster(rows: list[dict], height: int = 360):
    """Teststärke der Transinformation gegen die Klassenzahl k, eine Linie je Form."""
    fig = go.Figure()
    for shape in C.RASTER_SHAPES:
        sub = sorted((r for r in rows if r["shape"] == shape), key=lambda r: r["bins"])
        fig.add_trace(go.Scatter(x=[r["bins"] for r in sub], y=[100 * r["rate"]["mi"] for r in sub], mode="lines+markers", name=C.SHAPES[shape],
                                 line=dict(color=SHAPE_COLORS[shape], width=3), hovertemplate="k = %{x}: %{y:.1f} %<extra>" + C.SHAPES[shape] + "</extra>"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text="Klassen je Achse k", type="category")
    fig.update_yaxes(title_text="Anteil verworfener Unabhängigkeit (%)", range=[0, 102])
    return _base(fig, height)


def build_shared_scale(grid: dict, height: int = 340):
    """Gemeinsame Streuung (ρ = 0): Teststärke je Maß, gruppiert nach n."""
    fig = go.Figure()
    palette = {30: "#c6dbef", 100: "#6baed6", 300: "#2171b5"}
    for n in C.POWER_NS:
        cell = next(r for r in grid["power"] if r["shape"] == "gemeinsam" and r["n"] == n)
        fig.add_trace(go.Bar(x=[C.MEASURE_SHORT[k] for k in C.MEASURES], y=[100 * cell["rate"][k] for k in C.MEASURES], name=f"n = {n}", marker_color=palette[n]))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil verworfener Unabhängigkeit (%)", range=[0, 102])
    return _base(fig, height)


def build_dimension(rows: list[dict], height: int = 340):
    """Mittlere Distanzkorrelation der Stichprobe bei unabhängigen Vektoren gegen die Dimension p: dCor, dCor² (V-Statistik) und dCor² (bias-korrigiert)."""
    rows = sorted(rows, key=lambda r: r["p"])
    ps = [r["p"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("dcor", "dCor (V-Statistik)", "#54a24b"), ("dcor2_v", "dCor² (V-Statistik)", "#9ecae1"), ("dcor2_u", "dCor² (bias-korrigiert)", "#222222")):
        fig.add_trace(go.Scatter(x=ps, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3), hovertemplate="p = %{x}: %{y:.3f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Dimension p (beide Seiten, unabhängig)", type="log", tickvals=ps, ticktext=[str(p) for p in ps])
    fig.update_yaxes(title_text="Mittel über die Läufe", range=[-0.05, 1.02])
    return _base(fig, height)


def build_dimension_power(rows: list[dict], height: int = 300):
    """Teststärke des Distanzkorrelationstests, wenn nur die erste von p Spalten abhängt."""
    rows = sorted(rows, key=lambda r: r["p"])
    fig = go.Figure(go.Bar(x=[str(r["p"]) for r in rows], y=[100 * r["rate"] for r in rows], marker_color=C.MEASURE_COLORS["dcor"], hovertemplate="p = %{x}: %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Spalten p je Seite (nur die erste ist abhängig)", type="category")
    fig.update_yaxes(title_text="Anteil verworfener Unabhängigkeit (%)", range=[0, 102])
    return _base(fig, height)


def build_xi_functions(rows: list[dict], height: int = 320):
    """ξ(x, y), ξ(y, x) und |Spearman| für exakte Funktionen y = f(x)."""
    names = [r["function"] for r in rows]
    fig = go.Figure()
    for key, label, color in (("xi_xy", "ξ(x, y)", C.MEASURE_COLORS["xi"]), ("xi_yx", "ξ(y, x)", "#d4b9da"), ("spearman", "|Spearman|", C.MEASURE_COLORS["spearman"])):
        fig.add_trace(go.Bar(x=names, y=[r[key] for r in rows], name=label, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="y =", type="category")
    fig.update_yaxes(title_text="Mittel über die Läufe", range=[0, 1.02])
    return _base(fig, height)


def build_xi_size(rows: list[dict], height: int = 300):
    """ξ gegen n für eine exakte Funktion (Populationswert 1)."""
    rows = sorted(rows, key=lambda r: r["n"])
    ns = [r["n"] for r in rows]
    fig = go.Figure(go.Scatter(x=ns, y=[r["xi"] for r in rows], mode="lines+markers", line=dict(color=C.MEASURE_COLORS["xi"], width=3), showlegend=False,
                               hovertemplate="n = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text="Stichprobengröße n", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Mittel von ξ", range=[0, 1.05])
    return _base(fig, height)
