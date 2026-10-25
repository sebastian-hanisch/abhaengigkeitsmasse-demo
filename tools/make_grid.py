"""Rechnet das Raster der Messreihen vor (E1 bis E5, große Dimensionen) und schreibt data/dependence_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse (einige Minuten)
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/dependence_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probezelle nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import abm_constants as C  # noqa: E402
import abm_evaluation as ev  # noqa: E402

JOBS = {"power": ev.power_grid_cell, "noise": ev.noise_grid_cell, "raster": ev.raster_grid_cell, "dim": ev.dimension_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = []
    tasks += [("power", (shape, n)) for n in sorted(C.POWER_NS, reverse=True) for shape in C.SHAPES]               # die teuren Zellen zuerst
    tasks += [("noise", (shape, nz)) for shape, levels in C.NOISE_CELLS.items() for nz in levels]
    tasks += [("raster", (shape, k)) for shape in C.RASTER_SHAPES for k in C.RASTER_KS]
    tasks += [("dim", (p,)) for p in C.DIM_GRID_PS]
    if quick:
        tasks = [("power", ("sinus", 30)), ("noise", ("parabel", 1.0)), ("raster", ("sinus", 5)), ("dim", (500,))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("dependence_grid_quick.json" if quick else "dependence_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
