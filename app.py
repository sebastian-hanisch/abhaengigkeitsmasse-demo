"""Nichtlineare Abhängigkeit: wenn r und ρ_S null sagen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Korrelations-Reihe. Dieselben simulierten Standort-Wochen (x: Auftragsvolumen gegen den Plan, y: Überstunden), jetzt mit neun Formen der Abhängigkeit
und sechs Maßen (Pearson, Spearman, Distanzkorrelation, Chatterjee-ξ, Hoeffding-D, Transinformation), jedes mit einem Permutationstest auf Unabhängigkeit.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import abm_constants as C
import abm_evaluation as ev
from abm_layout import dunkles_design, kopf_css
from abm_pdf_export import build_pdf
from abm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from abm_visualization import (
    build_dimension,
    build_dimension_power,
    build_noise,
    build_power_heatmap,
    build_pvalues,
    build_raster,
    build_scatter,
    build_shared_scale,
    build_xi_functions,
    build_xi_size,
)

st.set_page_config(page_title="Nichtlineare Abhängigkeit – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


@st.cache_data(show_spinner=False)
def _xi():
    return [ev.xi_function_cell(name) for name in ev.XI_FUNCTIONS], [ev.xi_size_cell(n) for n in C.XI_NS]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in abm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Nichtlineare Abhängigkeit: wenn r und ρ_S null sagen")
st.markdown(
    """
**Drittes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios mit nichtlinearen Abhängigkeitsmaßen**. Dieselben simulierten Standort-Wochen wie in den beiden ersten Stücken
(*Pearson: eine Zahl, viele Daten* und *Rangkorrelation*): das **Auftragsvolumen** gegen den Plan (x) und die **Überstunden** (y). Dort ging es um die Frage, wann r und ρ_S kippen. Hier die
andere Seite: **Wenn r und ρ_S null sagen, ist dann wirklich nichts da?** Neun **Formen** der Abhängigkeit, sechs **Maße**, jedes mit einem **Permutationstest** auf Unabhängigkeit.

Weiter unten: sechs Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Der Abstand zweier **Rangfolgen** steht in `rangdistanz-demo`,
Ursachen in der Linie *Kausale Diagramme und Adjustierung*, gleichzeitige Extreme im Stück zur Randabhängigkeit.
"""
)

with st.expander("So funktionieren die sechs Maße", expanded=True):
    st.markdown(
        """
Jedes Maß gibt **eine Zahl**; ob sie „groß genug“ ist, entscheidet ein **Permutationstest**: y wird zufällig umgeordnet (dann ist die Abhängigkeit zerstört), die Zahl neu gerechnet, und der
**p-Wert** ist der Anteil der Umordnungen, die mindestens so groß ausfallen wie die gemessene. Alle sechs Maße sehen dieselben Umordnungen.

| Maß | Idee | Sieht | Blind für |
|---|---|---|---|
| **Pearson r** | Geradlinigkeit der Punkte | Gerade, Trend | Parabel, Kreis, Schachbrett, Streuung |
| **Spearman ρ_S** | Pearson der Ränge | monotone Kurven | alles Nichtmonotone |
| **Distanzkorrelation** | Abstände der Punkte untereinander, doppelt zentriert | fast jede Abhängigkeit, auch gemeinsame Streuung | wird bei vielen Dimensionen aufgebläht |
| **Chatterjee ξ** | wie glatt ist y als Funktion von x (Ränge der Nachbarn) | Funktionen, auch wellige | Kreis, Streuung (keine Funktion), kleines n |
| **Hoeffding D** | Abstand der gemeinsamen von der Produkt-Verteilungsfunktion | allgemeine Abhängigkeit | Gleichstände, flache Muster |
| **Transinformation** | Klassentafel aus k × k Quantilklassen | grobe Muster, Schachbrett | zu grobes oder zu feines Raster |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    shape = st.selectbox("Form der Abhängigkeit", list(C.SHAPES), format_func=C.SHAPES.get, key="shape_select",
                         help="Wie y von x abhängt. „Gemeinsame Streuung“: beide Kennzahlen teilen sich einen Skalenfaktor (t(3), ρ = 0): unkorreliert, aber nicht unabhängig.")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), key="n_slider", help="Standort-Wochen im Datensatz (höchstens 300: Distanzkorrelation und Hoeffding-D rechnen n × n-Matrizen).")
    noise = st.slider("Rauschfaktor", *bounds("noise_slider"), step=0.25, key="noise_slider",
                      help="Skaliert das Rauschen der Form (1 = Standard). Wirkt nicht bei „unabhängig“, „Schachbrett“ und „gemeinsame Streuung“.")
    st.markdown("**Verfahren**")
    bins = st.slider("Klassen k der Transinformation", *bounds("bins_slider"), key="bins_slider",
                     help="Zahl der Quantilklassen je Achse. Zu grob übersieht Muster, zu fein verteilt die Punkte auf zu viele Zellen (Experiment 3).")
    perms = st.slider("Permutationen B", *bounds("perm_slider"), step=100, key="perm_slider",
                      help="Wie oft y umgeordnet wird. Der kleinste mögliche p-Wert ist 1/(B + 1). Bei großem n begrenzt das Budget B · n² die Zahl.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Datensatz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"shape_select": shape, "n_slider": int(n), "noise_slider": noise, "bins_slider": int(bins), "perm_slider": int(perms), "seed_input": int(seed)})

settings = ev.Settings(shape, int(n), round(float(noise), 2), int(bins), int(perms), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
cols = st.columns(6)
fmt = {"pearson": "{:.2f}", "spearman": "{:.2f}", "dcor": "{:.2f}", "xi": "{:.2f}", "hoeffding": "{:.3f}", "mi": "{:.2f}"}
for col, key in zip(cols, C.MEASURES):
    col.metric(C.MEASURES[key], fmt[key].format(a.stats[key]), help=f"p-Wert des Permutationstests: {a.p[key]:.3f}" + (" – verwirft die Unabhängigkeit" if key in a.found else " – verwirft sie nicht"))
st.caption(f"{a.perms_used} Permutationen. Hoeffding D ist auf 30 skaliert, die Transinformation in nat (natürlicher Logarithmus). Die Hilfe an jedem Kästchen nennt den p-Wert.")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_scatter(a.x, a.y, a.settings.shape, min(a.settings.bins, a.settings.n)), width="stretch", key="main_scatter")
    st.caption("Messwerte (blau), die Kurve ohne Rauschen gestrichelt (wo es sie gibt) und das Klassenraster der Transinformation hellgrau.")
with right:
    st.plotly_chart(build_pvalues(a.p), width="stretch", key="main_pvalues")
    st.caption("p-Werte der sechs Permutationstests: grün = verwirft die Unabhängigkeit (p ≤ 5 %), grau = nicht. Gestrichelt: 5 %.")
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="abhaengigkeitsmasse-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wer findet die Abhängigkeit?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/dependence_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente 1 bis 4 zeigen es.")

# --- E1 Teststärke je Form -----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer findet welche Form? Teststärke je Maß und Form")
st.caption("Anteil der Läufe, in denen der Permutationstest die Unabhängigkeit verwirft (5 %-Niveau). Vorgerechnet: 4 Seeds, n = 30 und 100 mit 4 · 250 Läufen je Zelle, n = 300 mit 4 · 100, jeweils 199 Permutationen.")
if grid is not None:
    pn = st.selectbox("n", list(C.POWER_NS), index=1, key="power_n")
    table = ev.power_table(grid, pn)
    st.plotly_chart(build_power_heatmap(table, pn), width="stretch", key="power_heatmap")
    kind, text = ev.verdict_power(table, pn)
    getattr(st, kind)(text)
    kind, text = ev.verdict_price(ev.power_table(grid, 30))
    getattr(st, kind)(text)
    kind, text = ev.verdict_false_alarm(grid)
    getattr(st, kind)(text)
    st.caption("Die Zellen sind Anteile in Prozent. Bei „unabhängig“ ist 5 % richtig; alles darüber bei einer abhängigen Form ist Teststärke.")

st.markdown("---")

# --- E2 Rauschen ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Mit Rauschen wechselt die Rangfolge der Maße")
st.caption("n = 100, 4 · 100 Läufe je Zelle, 199 Permutationen. Der Rauschfaktor 1 ist der Standard der Form; bei „linear“ stehen die Faktoren 1, 2, 4 und 8.")
if grid is not None:
    nshape = st.selectbox("Form", list(C.NOISE_CELLS), format_func=C.SHAPES.get, index=2, key="noise_shape")
    st.plotly_chart(build_noise(grid["noise"], nshape), width="stretch", key="noise_chart")
    kind, text = ev.verdict_noise(grid["noise"], nshape)
    getattr(st, kind)(text)

st.markdown("---")

# --- E3 Raster -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Das Raster der Transinformation ist ein Regler")
st.caption("n = 100, 4 · 100 Läufe je Zelle, 199 Permutationen. Dieselben Daten, nur die Zahl k der Klassen je Achse ändert sich.")
if grid is not None:
    st.plotly_chart(build_raster(grid["raster"]), width="stretch", key="raster_chart")
    rshape = st.selectbox("Form für das Verdikt", list(C.RASTER_SHAPES), format_func=C.SHAPES.get, index=2, key="raster_shape")
    kind, text = ev.verdict_raster(grid["raster"], rshape)
    getattr(st, kind)(text)
    st.caption("k ist keine Naturkonstante: zu grob (wenige große Zellen) glättet ein Muster weg, zu fein (viele fast leere Zellen) erreicht der Test wegen der kleinen Besetzung nichts.")

st.markdown("---")

# --- E4 Gemeinsame Streuung ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Unkorreliert heißt nicht unabhängig – gemeinsame Streuung")
st.markdown(
    "Im ersten Stück schlug der Pearson-Test auf ρ = 0 bei gemeinsamer Streuung (Spitzenwochen) zu oft an, weil die Kennzahlen nicht unabhängig sind. Hier ist es dieselbe Erzeugung, "
    "nur die Frage ist eine andere: Ist da **Abhängigkeit**? Ja, und die Maße unterscheiden sich darin, ob sie sie sehen."
)
if grid is not None:
    st.plotly_chart(build_shared_scale(grid), width="stretch", key="shared_chart")
    kind, text = ev.verdict_shared_scale(grid)
    getattr(st, kind)(text)
    st.caption("Ränge sehen nur die Reihenfolge, nicht die gemeinsame Skala: Spitzenwochen heben beide Kennzahlen zugleich, aber nicht in einer bestimmten Richtung. Die Distanzkorrelation sieht die Größe der Abstände.")

st.markdown("---")

# --- E5 Viele Spalten ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Viele Spalten – die Distanzkorrelation der Stichprobe wird aufgebläht")
st.markdown(
    "Unabhängige Normalvektoren der Dimension p auf beiden Seiten, n = 50 (4 · 25 Läufe je Zeile). Die **Stichprobe** der Distanzkorrelation (V-Statistik, wie sie die meisten Programme "
    "ausgeben) wächst mit p, obwohl nichts zusammenhängt; die **bias-korrigierte** Fassung bleibt bei null."
)
if st.button("🔁 Dimension rechnen (p bis 200, wenige Sekunden)", key="btn_dim"):
    with st.spinner("Rechne die Dimensionen …"):
        st.session_state["dim_result"] = [ev.dimension_cell(p) for p in C.DIM_PS]
dim = st.session_state.get("dim_result")
if dim:
    extra = list(grid["dim"]) if grid is not None else []
    st.plotly_chart(build_dimension(dim + extra), width="stretch", key="dim_chart")
    tab = ["| p | dCor (V) | dCor² (V) | dCor² (korrigiert) |", "|---|---|---|---|"]
    tab += [f"| {r['p']} | {r['dcor']:.3f} | {r['dcor2_v']:.3f} | {r['dcor2_u']:+.3f} |" for r in sorted(dim + extra, key=lambda r: r["p"])]
    st.markdown("\n".join(tab))
    kind, text = ev.verdict_dimension(dim + extra)
    getattr(st, kind)(text)
    st.caption("Die Zeilen p = 500 und 1 000 stammen aus dem vorgerechneten Raster (n = 50, 4 · 25 Läufe), die übrigen aus der Rechnung eben.")
st.markdown(
    "**Und die Leistung?** Hängt nur die erste von p Spalten ab (y₁ = x₁ + 0,8ε), die übrigen sind unabhängig, so verdünnt jede weitere Spalte das Signal im Abstand (n = 100, 99 Permutationen)."
)
if st.button("🔁 Leistung rechnen (rund 10 Sekunden, auf langsameren Rechnern länger)", key="btn_dimpower"):
    with st.spinner("Rechne die Leistung …"):
        st.session_state["dimpower_result"] = [ev.dimension_power_cell(p) for p in C.DIM_POWER_PS]
dimpower = st.session_state.get("dimpower_result")
if dimpower:
    st.plotly_chart(build_dimension_power(dimpower), width="stretch", key="dimpower_chart")
    kind, text = ev.verdict_dimension_power(dimpower)
    getattr(st, kind)(text)

st.markdown("---")

# --- E6 ξ ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Chatterjee-ξ ist asymmetrisch und wächst langsam")
st.caption("Exakte Funktionen y = f(x) ohne Rauschen, x gleichverteilt in (−1, 1), n = 1 000, Mittel über 40 Stichproben; rechts ξ gegen n für y = sin(3πx).")
xi_funcs, xi_sizes = _xi()
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(build_xi_functions(xi_funcs), width="stretch", key="xi_functions_chart")
with g2:
    st.plotly_chart(build_xi_size(xi_sizes), width="stretch", key="xi_size_chart")
kind, text = ev.verdict_xi(xi_funcs)
getattr(st, kind)(text)
kind, text = ev.verdict_xi_size(xi_sizes)
getattr(st, kind)(text)
st.caption("Eine umkehrbare Funktion (e^(3x)) ist in beiden Richtungen eine Funktion; bei x² und |x| erklärt x das y, aber y nicht das x. ξ ist deshalb keine Stärke („wie stark“), sondern ein Maß für „wie funktional“.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Test auf Unabhängigkeit sagt, ob** etwas da ist | nicht wie stark, nicht in welcher Richtung, nicht warum; ein kleiner p-Wert bei großem n kann eine winzige Abhängigkeit meinen | Effektgrößen, Intervalle (Stücke 1 und 2) |
| **Die neun Formen stehen für die Wirklichkeit** | es sind Beispiele; wie häufig sie in echten Daten vorkommen, zeigt die Demo nicht | – |
| **Allgemeiner ist besser** | bei linearer Abhängigkeit und kleinem n kostet die Allgemeinheit Teststärke (Experiment 1) | Maß nach der Frage wählen |
| **Das Raster ist egal** | die Transinformation hängt an k (Experiment 3); k hier ist ein Vorschlag, kein Gesetz | – |
| **Die Distanzkorrelation der Stichprobe misst Stärke** | bei vielen Spalten ist sie auch ohne Abhängigkeit groß (Experiment 5); der Test selbst bleibt gültig | bias-korrigierte Fassung |
| **ξ ist ein Stärkemaß** | es ist asymmetrisch und erreicht bei exakter Abhängigkeit erst bei großem n die 1 (Experiment 6) | – |
| **Unkorreliert heißt unabhängig** | gemeinsame Streuung ist abhängig, und nur manche Maße sehen sie (Experiment 4) | Stück 1 (Fehlalarm des Pearson-Tests) |
| **Gleichzeitige Extreme** | ob zwei Kennzahlen *zusammen* extrem werden, misst kein Maß hier | Randabhängigkeit und Kopulas |
| **Beobachtungen sind unabhängig** | Reihen in der Zeit verletzen das | Scheinkorrelation in Zeitreihen |
| **Ursachen** | Abhängigkeit sagt nichts darüber, welche Kennzahl die andere bewegt | Linie *Kausale Diagramme und Adjustierung* |
"""
)
st.caption("Alle Daten sind simuliert; die Formen sind Modellwahlen, kein Befund über echte Standorte. Hoeffding-D ist für Daten ohne Gleichstände gedacht, die Formen sind stetig.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Paare $(x_i, y_i)$; gefragt ist, ob $x$ und $y$ unabhängig sind.

| Symbol | Bedeutung |
|---|---|
| $a_{ij} = \lvert x_i - x_j\rvert$, $b_{ij} = \lvert y_i - y_j\rvert$ | Abstände der Beobachtungen untereinander |
| $A_{ij}, B_{ij}$ | doppelt zentrierte Abstände |
| $R_i, S_i$ | Ränge von $x_i$ und $y_i$ |
| $r_i, l_i$ | Zahl der $y_j \le y_i$ bzw. $y_j \ge y_i$ (bei Beobachtungen nach $x$ sortiert) |
| $Q_i$ | $1 + \#\{j : x_j < x_i,\ y_j < y_i\}$ |
| $p_{ab}$ | Anteil der Beobachtungen in Klasse $a$ von $x$ und Klasse $b$ von $y$ ($k \times k$ Quantilklassen) |

**Distanzkorrelation.** Doppelt zentrierte Abstände und die quadrierte Distanzkovarianz (V-Statistik):

$$ A_{ij} = a_{ij} - \bar a_{i\cdot} - \bar a_{\cdot j} + \bar a_{\cdot\cdot}, \qquad \mathrm{dCov}^2 = \frac{1}{n^2}\sum_{i,j} A_{ij} B_{ij} $$

$$ \mathrm{dCor} = \sqrt{\frac{\mathrm{dCov}^2(x,y)}{\sqrt{\mathrm{dCov}^2(x,x)\,\mathrm{dCov}^2(y,y)}}} $$

Die **bias-korrigierte** Fassung zentriert mit $n-2$ und $(n-1)(n-2)$ statt $n$ und lässt die Diagonale aus:

$$ \tilde A_{ij} = a_{ij} - \frac{1}{n-2}\sum_k a_{ik} - \frac{1}{n-2}\sum_k a_{kj} + \frac{1}{(n-1)(n-2)}\sum_{k,l} a_{kl}\ (i\neq j), \qquad \widetilde{\mathrm{dCov}}^2 = \frac{1}{n(n-3)}\sum_{i\neq j}\tilde A_{ij}\tilde B_{ij} $$

**Chatterjee-ξ** (Beobachtungen nach $x$ sortiert):

$$ \xi_n = 1 - \frac{n\sum_{i=1}^{n-1} \lvert r_{i+1} - r_i\rvert}{2\sum_{i=1}^{n} l_i\,(n - l_i)} $$

**Hoeffding-D** (auf 30 skaliert, für Daten ohne Gleichstände):

$$ D = 30\,\frac{(n-2)(n-3)D_1 + D_2 - 2(n-2)D_3}{n(n-1)(n-2)(n-3)(n-4)} $$

$$ D_1 = \sum_i (Q_i-1)(Q_i-2), \qquad D_2 = \sum_i (R_i-1)(R_i-2)(S_i-1)(S_i-2), \qquad D_3 = \sum_i (R_i-2)(S_i-2)(Q_i-1) $$

**Transinformation** der Klassentafel (natürlicher Logarithmus):

$$ \widehat{I} = \sum_{a,b} p_{ab}\,\ln\frac{p_{ab}}{p_{a\cdot}\,p_{\cdot b}} $$

**Permutationstest.** Mit einer Statistik $T$ und $B$ zufälligen Umordnungen $T^*_1, \dots, T^*_B$ von $y$:

$$ p = \frac{1 + \#\{b : T^*_b \ge T\}}{B + 1} $$

(bei Pearson und Spearman mit Beträgen). Der kleinste mögliche Wert ist $1/(B+1)$.

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Definitionen sind durch unabhängige Rechenwege (Tests) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
