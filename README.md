# Nichtlineare Abhängigkeit: wenn r und ρ_S null sagen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-abhaengigkeitsmasse-demo.streamlit.app/)** -->

Drittes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo) (Kante `fix`: Ränge sehen nur Monotonie) und [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo). Es ist die **erste Demo des Portfolios mit nichtlinearen Abhängigkeitsmaßen**: dieselben simulierten Standort-Wochen (x: Abweichung des Auftragsvolumens vom Plan, y: Überstunden), jetzt mit **neun Formen der Abhängigkeit** und **sechs Maßen** (Pearson, Spearman, Distanzkorrelation, Chatterjee-ξ, Hoeffding-D, Transinformation), jedes mit einem **Permutationstest** auf Unabhängigkeit. Die Frage „Wann kippt r?“ stand in den ersten beiden Stücken; hier steht die andere: **Wenn r und ρ_S null sagen, ist dann wirklich nichts da, und welches Maß findet was, zu welchem Preis?**

## Warum dieses Problem

Eine Korrelation von null gilt oft als „kein Zusammenhang“. Die Demo zeigt, dass das für Pearson und Spearman nicht stimmt, und dass die Maße, die mehr sehen, Preise haben. Befunde: bei **Parabel, Kreis, Schachbrett und wachsender Streuung** erreichen Pearson und Spearman in diesen vier Formen in keiner Zelle mehr als **15,8 %** (statt 100 %), das Maximum über zwölf verrauschte Zellen; **kein Maß gewinnt überall** (bei n = 100: Sinus: ξ 100,0 %, Distanzkorrelation 65,7 %; Kreis: Transinformation 100,0 %, Distanzkorrelation 29,1 %; bei n = 300 holen die anderen Maße auf); der **Preis der Allgemeinheit** bei linearer Abhängigkeit und n = 30 ist, dass ξ nur 53,0 % und die Transinformation 37,7 % erreichen (Pearson 96,1 %); bei **gemeinsamer Streuung** (ρ = 0, aber abhängig) findet die Distanzkorrelation die Abhängigkeit bei n = 300 in 77,2 % der Läufe (400 Läufe, Nachläufe rund 77 bis 81 %), Spearman nur in 6,5 % (Nachläufe 6 bis 9 %); die **Klassenzahl** der Transinformation ist ein Regler, dessen Optimum vom Muster abhängt (Sinus: 10,2 % bei k = 3, 91,8 % bei k = 8, 50,0 % bei k = 20; linear fällt die Teststärke mit k); die **Distanzkorrelation der Stichprobe** liegt bei unabhängigen Daten mit p = 1 000 Spalten bei 0,994.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-3-abhaengigkeitsmasse.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern, anderer Ziehungsreihenfolge und teils anderen Läufen. Gebaut ist alles mit SplitMix64 und eigenem Kern; die Zahlen weichen leicht ab, die Befunde bleiben, mit diesen Korrekturen:

- **Pearson und Spearman bei den Formen ohne Mittelwertabhängigkeit** (Plan: höchstens 12 %): in Parabel, Kreis, Schachbrett und wachsender Streuung **höchstens 15,8 % (Pearson) und 13,3 % (Spearman)** über n = 30, 100, 300. Bei der Parabel liegen sie über 5 % (n = 100: 12,3 und 13,3 %), obwohl die Populationswerte null sind; Nachläufe mit anderen Seeds geben für Pearson rund 9 bis 10 % und für Spearman rund 10 bis 11 %, einzelne Zellen des Rasters sind also Rauschen. Ursache: der Permutationstest prüft Unabhängigkeit, die Streuung von r ist bei der Parabel aber größer als unter Unabhängigkeit (Grenzwert für Pearson bei großem n etwa 9 %).
- **Kreis, n = 100:** Distanzkorrelation 29,1 % (Plan: 27,3 %), ξ 41,2 % (40,1 %), Hoeffding-D 91,5 % (92,2 %); Sinus: Hoeffding-D 69,8 % (69,2 %).
- **Fehlalarm** bei unabhängigen Daten: 3,0 bis 6,2 % über alle Maße und drei Werte von n (Plan: 3,5 bis 5,5 %).
- **Dimension:** die Plan-Zahlen 0,065 … 0,988 waren **dCor²** (V-Statistik); die Distanzkorrelation selbst, wie Programme sie ausgeben, ist 0,241 bei p = 1 und 0,994 bei p = 1 000. Die App zeigt beide und die bias-korrigierte dCor².
- **Schachbrett:** die Verteilung wird jetzt über die fünf besetzten Zellen gezogen statt per Ablehnung (gleiche Verteilung, Pearson-Korrelation der Verteilung exakt null).
- **„Blind“ in der App:** höchstens 20 % (das Vierfache des Fehlalarms), nicht 12 %. Die Schwelle ist eine Vereinbarung und wurde nach dem Befund so gewählt, dass auch die Parabel mit 13,3 % darunter fällt.
- **Presets:** sechs (Standardfall = Parabel, Kreis, Sinus, Schachbrett, gemeinsame Streuung, Linear); Hilfetexte nennen nur Zahlen des Rasters.

## Modell

Paare (x, y) aus neun Formen mit x gleichverteilt in (−1, 1) und standardnormalem Rauschen ε (Erzeuger `abm_shapes.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo`: `prs_rng.py`, `prs_constants.py`, `prs_scenario.py`, `prs_algorithm.py`): **unabhängig** (y = ε); **linear** y = x + 0,8ε; **monoton** y = e^(3x) + ε; **Parabel** y = x² + 0,25ε; **Sinus** y = sin(3πx) + 0,6ε; **Kreis** (cos t, sin t) mit Rauschen 0,15 in beiden Koordinaten; **Streuung wächst** y = x·ε; **Schachbrett** gleichverteilt auf den fünf besetzten Zellen (Ecken und Mitte) eines 3 × 3-Rasters über [−1, 1]²; **gemeinsame Streuung** t(3) mit gemeinsamem Skalenfaktor und ρ = 0 (unkorreliert, aber abhängig). Der Rauschfaktor skaliert den Rauschterm; bei unabhängig, Schachbrett und gemeinsamer Streuung wirkt er nicht.

## Methodik

- **Sechs Maße** in `abm_algorithm.py`, ohne scipy zur Laufzeit: Pearson und Spearman (aus der Kopie), **Distanzkorrelation** (doppelt zentrierte Abstandsmatrizen, V-Statistik; dazu die bias-korrigierte U-Fassung für die Dimensionsexperimente), **Chatterjee-ξ** (mit Gleichständen in y nach der Formel mit l_i), **Hoeffding-D** (für Daten ohne Gleichstände) und die **Transinformation** einer k × k-Tafel aus Quantilklassen (natürlicher Logarithmus).
- **Permutationstest** für alle Maße über **dieselben** Umordnungen von y: p = (1 + #{T\* ≥ T})/(B + 1), bei Pearson und Spearman mit Beträgen, sonst rechtsseitig. Eine einzige Funktion (`statistics`) rechnet den beobachteten Wert (Identität) und die Nullwerte (Umordnungen).
- **Experimente:** (1) Teststärke je Form und Maß, (2) Rauschen, (3) Raster der Transinformation, (4) gemeinsame Streuung, (5) viele Spalten (live klein), (6) ξ (live klein).
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/dependence_grid.json`, 218 Sekunden auf 10 Prozessen): Teststärke je Form · n · Maß (27 Zellen, n = 30 und 100 mit 4 · 250 Läufen, n = 300 mit 4 · 100, jeweils 199 Permutationen), Rauschen (16 Zellen, n = 100, 4 · 100 Läufe), Raster der Transinformation (30 Zellen, 4 · 100 Läufe), Distanzkorrelation bei p = 500 und 1 000 (4 · 25 Läufe, n = 50). Vier Seeds je Zelle (101 bis 104), die Streuung über die Seeds steht im Raster. **Live:** Dimension (p bis 200, n = 50, 4 · 25 Läufe, dazu die Leistung bei einer abhängigen von p Spalten) und ξ.

## Befunde (gemessen, keine Behauptungen)

Alle Messzahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, Live-Werte über dieselben Funktionen wie in der App); Laufzeit- und Mutationsangaben sind dort nicht erfasst. Anteile verworfener Unabhängigkeit bei 5 % Niveau; die Streuung über die Seeds liegt in den Zellen bei höchstens 7,0 Punkten.

**Teststärke bei n = 100** (Experiment 1; Anteil der 1 000 Läufe in %):

| Form | Pearson | Spearman | Distanzkorrelation | ξ | Hoeffding D | Transinformation |
|---|---|---|---|---|---|---|
| unabhängig | 3,6 | 4,0 | 3,9 | 3,9 | 4,3 | 5,0 |
| linear | 100,0 | 100,0 | 100,0 | 91,9 | 100,0 | 99,0 |
| monoton (e^(3x)) | 100,0 | 100,0 | 100,0 | 100,0 | 100,0 | 100,0 |
| Parabel | 12,3 | 13,3 | 100,0 | 100,0 | 100,0 | 100,0 |
| Sinus | 50,7 | 52,6 | 65,7 | 100,0 | 69,8 | 77,6 |
| Kreis | 0,7 | 0,2 | 29,1 | 41,2 | 91,5 | 100,0 |
| Streuung wächst | 14,4 | 9,1 | 64,6 | 29,9 | 29,3 | 98,1 |
| Schachbrett | 7,1 | 8,0 | 13,8 | 32,7 | 20,9 | 95,9 |
| gemeinsame Streuung | 29,5 | 7,3 | 42,4 | 7,2 | 7,9 | 11,7 |

| Frage | Befund |
|---|---|
| Sind r und ρ_S blind? (Experiment 1) | **Ja, bei vier Formen:** Parabel, Kreis, Schachbrett und wachsende Streuung erreichen Pearson höchstens **15,8 %** und Spearman höchstens **13,3 %** (alle drei n; Maximum über zwölf verrauschte Zellen mit 400 bis 1 000 Läufen); Pearson liegt im Grenzfall großer n bei etwa 9 % (Parabel), 14 % (Streuung), 7 % (Schachbrett) und 1 % (Kreis); die Distanzkorrelation findet die Parabel ab n = 100 in 100,0 % und die wachsende Streuung bei n = 300 in 100,0 %. |
| Gewinnt ein Maß überall? | **Nein.** n = 100: Sinus gewinnt ξ (100,0 %), Kreis die Transinformation (100,0 %) und Hoeffding-D (91,5 %), wachsende Streuung die Transinformation (98,1 % gegen 64,6 % der Distanzkorrelation), gemeinsame Streuung die Distanzkorrelation (42,4 %). Die Aussage gilt für n = 100; bei n = 300 erreichen beim Kreis auch die Distanzkorrelation und beim Sinus auch die Distanzkorrelation und Hoeffding-D 100 %. |
| Was kostet die Allgemeinheit? | Bei **linearer** Abhängigkeit und n = 30: Pearson 96,1 %, Spearman 94,3 %, Distanzkorrelation 94,1 %, Hoeffding-D 90,6 %, **ξ 53,0 %, Transinformation 37,7 %**; bei n = 100 haben ξ 91,9 % und die Transinformation 99,0 % aufgeholt. |
| Bleibt der Fehlalarm bei 5 %? | Bei unabhängigen Daten liegen alle sechs Maße bei **3,0 bis 6,2 %** (n = 30, 100, 300). |
| Ist ρ = 0 Unabhängigkeit? (Experiment 4) | **Nein.** Bei gemeinsamer Streuung und n = 300: Distanzkorrelation **77,2 %**, Pearson-Permutationstest 38,2 %, Transinformation 28,5 %, ξ 10,2 %, Hoeffding-D 8,5 %, Spearman 6,5 %; bei n = 30 und 100 findet die Distanzkorrelation sie in 24,1 % und 42,4 %. Der n = 300-Wert hat bei 400 Läufen einen Fehler von etwa ±4 Punkten und streut über die Seeds bis 7 Punkte; die Rangmaße liegen mit 6 bis 9 % in der Größenordnung des Fehlalarms. |
| Was macht Rauschen? (Experiment 2) | Parabel bei Faktor 2: Pearson 7,2 %, Distanzkorrelation **86,8 %**; Kreis bei Faktor 2: nur die Transinformation (**73,2 %**) sieht noch etwas, die übrigen höchstens 12,2 %; bei Faktor 4 erreicht für Parabel, Sinus und Kreis kein Maß mehr als 24,5 %; linear bei Faktor 2: Pearson 92,5 %, ξ 27,5 %, Transinformation 42,5 %. Bei 400 Läufen je Zelle beträgt der Standardfehler bis 2,5 Punkte; eine Nachrechnung weicht um 2 bis 3 Punkte ab (beim Kreis erreicht ξ dabei 15,2 % statt höchstens 12,2 %), die Aussagen bleiben qualitativ gleich. |
| Hängt die Transinformation am Raster? (Experiment 3) | **Ja, das Optimum hängt vom Muster ab:** beim Sinus liegt es bei k = 8 (10,2 / 76,8 / 91,8 / 80,5 / 50,0 % bei k = 3 / 5 / 8 / 12 / 20), bei linearer Abhängigkeit fällt die Teststärke mit k (100,0 / 98,8 / 86,5 / 62,0 / 25,0 %); bei unabhängigen Daten bleibt sie bei 4,0 bis 6,2 %. |
| Wächst die Distanzkorrelation mit den Spalten? (Experiment 5) | **Ja, die Stichprobe:** bei unabhängigen Vektoren (n = 50) ist dCor im Mittel 0,241 / 0,660 / 0,894 / 0,970 / 0,994 bei p = 1 / 10 / 50 / 200 / 1 000; dCor² (V) 0,061 bis 0,988; die **bias-korrigierte** dCor² liegt im Mittel bei null (die Mittelwerte der Messreihen bleiben zwischen −0,003 und +0,004, bei wenigen Läufen mit einer Auflösung von etwa ±0,01). Hängt nur die erste von p Spalten ab (n = 100), findet der Test sie bei p = 1, 2, 5 in 100,0 %, bei p = 20 in 91,0 % und bei p = 100 in 13,0 % (100 Läufe, Standardfehler rund 3 Punkte; ein Nachlauf mit 300 Läufen gab 86 % und 18 %). |
| Ist ξ symmetrisch? (Experiment 6) | **Nein:** für y = x² ist ξ(x, y) 0,994 und ξ(y, x) 0,246; für y = sin(3πx) 0,982 gegen 0,075; für die umkehrbare e^(3x) beide 0,997. Ohne Rauschen erreicht ξ bei y = sin(3πx) 0,529 / 0,833 / 0,941 / 0,982 / 0,994 bei n = 30 / 100 / 300 / 1 000 / 3 000. |
| Handwerte | dCor((0, 1, 2), (0, 1, 0)) = 0,562 (dCov² = 8/81); dCor der vollständigen 2 × 2-Tafel = 0; ξ((1, 2, 3, 4), (1, 3, 2, 4)) = 0; Transinformation zweier gleicher Binärmerkmale = ln 2; Hoeffding D(x, x) = 1. |

## Ehrliche Grenzen

- **Ein Test auf Unabhängigkeit** sagt, ob etwas da ist, nicht wie stark, in welcher Richtung oder warum. Die Teststärken gelten für diese neun Formen, diese Rauschpegel und diese n; wie häufig die Formen in echten Daten vorkommen, zeigt die Demo nicht.
- **Auflösung:** mit B = 199 Permutationen ist der kleinste p-Wert 0,005; die Läufe des Rasters nutzen sie, die App erlaubt B von 99 bis 499 und bei großem n ein Budget B · n² von höchstens 5·10^7.
- **Hoeffding-D** ist für Daten ohne Gleichstände gedacht (die Formen sind stetig, eine Korrektur ist nicht eingebaut); bei **ξ** werden Gleichstände in x nach der Eingabereihenfolge aufgelöst.
- **Transinformation** ist der einfache Plug-in-Schätzer auf einer Quantiltafel (nach oben verzerrt); der Permutationstest fängt das ab, die Zahl selbst ist nicht vergleichbar zwischen verschiedenen k. Die Wahl k = 5 ist ein Vorschlag.
- **Die Distanzkorrelation der Stichprobe** ist bei vielen Spalten auch ohne Abhängigkeit groß (Experiment 5); der Permutationstest bleibt gültig, die bias-korrigierte Fassung zeigt null. Die Leistung des Tests nimmt trotzdem ab, wenn nur wenige von vielen Spalten abhängen.
- **Pearson im Permutationstest:** der Test ist hier nicht der t-Test; bei den Formen mit gleichem Mittel und abhängiger Streuung liegen die Anteile über 5 %, weil der Permutationstest Unabhängigkeit prüft und die Streuung von r dort größer ist als unter Unabhängigkeit.
- **Quellen:** keine Literaturangaben; die Definitionen sind durch unabhängige Rechenwege gestützt (unten) und durch die Messungen. Aussagen zur Kritik an ξ oder zum Verhalten bei sehr großem n stehen nicht in dieser Demo.

## Verifikation

- **Orakel** (`tests/test_oracle_abhaengigkeitsmasse.py`): Distanzkovarianz gegen die Doppelsumme als Schleife, die bias-korrigierte Fassung gegen ihre Definition als Schleife, Abstandsmatrix gegen `scipy.spatial.distance`; **ξ gegen `scipy.stats.chatterjeexi`** (mit und ohne Gleichstände in y); **Hoeffding-D gegen die U-Statistik** über alle geordneten 5-Tupel (Hoeffdings Definition, mal 30); **Transinformation gegen `sklearn.metrics.mutual_info_score`**; Pearson und Spearman gegen scipy; jede Zeile des vektorisierten Permutationskerns gegen die Einzelrechnung auf **allen 120 Umordnungen** von fünf Punkten; der gezogene p-Wert gegen die vollständige Aufzählung bei n = 6.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`), Eigenschaften (Zentrierung, Blöcke ändern nichts), kleinste Fälle (n = 5), konstante Spalten, Fehlerwege.
- **Formen** (`tests/test_shapes.py`): Schachbrett mit Pearson-Korrelation null auf 200 000 Punkten und genau fünf besetzten Zellen, Kreis auf dem Einheitskreis, Rauschfaktor, Reproduzierbarkeit.
- **Eigenschaften** (`tests/test_eigenschaften_abhaengigkeitsmasse.py`): Invarianz von dCor unter Verschiebung und Skalierung, von ξ, D, MI und Spearman unter monotonen Abbildungen, Symmetrie, **Asymmetrie von ξ**, Gegensatz linear gegen allgemein, Dimensionsverhalten, Fehlalarm.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probezelle des Erzeugers gegen die gespeicherte, jede Messzahl dieser README gegen das Raster oder die Live-Funktion (nicht erfasst: Laufzeit, Mutationszahlen), Prozentreihen mit der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jeder Experiment-Knopf einzeln, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `abm_algorithm`: 150 Stellen, 9 davon als gleichwertig festgehalten (Blockgröße `CHUNK_CELLS`, `min`/`max` am Blockende, die Skalierung `n(n − 3)` kürzt sich in der bias-korrigierten Distanzkorrelation). Die Zustandsdatei enthält 151 Einträge (138 gefangen, 9 gleichwertig, 4 als „überlebt“ vermerkt); diese vier gehören zu Quelltextzeilen, die es in dieser Form nicht mehr gibt, also zu einem älteren Stand des Kerns. Ein Neulauf nur mit den Orakeltests ließ fünf Mutanten überleben (zwei NaN-Rückgaben, drei Schwellen `n < 4` in `u_center`); die ganze Suite fängt alle fünf (`tests/test_edge_cases.py`, `tests/test_algorithm.py`), sie sind also nicht gleichwertig. Der erste Lauf fand 20 Überlebende und führte zu Tests für die Permutations-p-Werte mit exakten Gleichständen, die konstante Seite und das Verhalten von Hoeffding-D bei Gleichständen (der Zustand steht in `tests/mutation_state.json`). Formen, Auswertung und Verdikte prüfen die übrigen Tests, sie werden nicht mutiert.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `abm_algorithm.py` | Kern: sechs Maße, Distanzmatrizen, Permutationstests |
| `abm_constants.py` | Regler-Grenzen, Formen und Maße, Raster der Messreihen, Farben, Presets |
| `abm_evaluation.py` | Einzeldatensatz, Zellen des Rasters, Live-Experimente, Verdikte |
| `abm_layout.py` | Fester Kopf (Stylesheet) |
| `abm_pdf_export.py` | PDF-Export der Hauptansicht |
| `abm_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `abm_shapes.py` | die neun Formen der Abhängigkeit |
| `abm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App |
| `data/dependence_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und scikit-learn für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Formen, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **MIC** und andere „Maximal-Information“-Maße: nicht geprüft, die Debatte gehört nicht in dieses Stück.
- **Blum–Kiefer–Rosenblatt-R** und **Bergsma–Dassios-τ\*** (verwandte Verteilungsfunktions-Maße), **kernbasierte Maße** (HSIC), **Maße für Kategorien** (Cramér-V): nur als Hinweis, nicht gebaut.
- **Asymptotische Tests und ξ-Bootstrap:** alle Tests sind Permutationstests; das hält die Vergleichbarkeit und braucht keine Verteilungsannahme.
- **Gleichzeitige Extreme** und Kopulas (Stück zur Randabhängigkeit), **Zeitreihen** (Scheinkorrelation), **gerichtete Abhängigkeit** (Kointegration, Granger, Transferentropie), **Ursachen** (Linie *Kausale Diagramme und Adjustierung*).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (rund vier Minuten auf zehn Prozessen); `tests/test_grid.py` rechnet eine feste Probezelle nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
