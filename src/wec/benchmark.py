"""Train and evaluate every model on every site, log runs to MLflow and write a results table.

    python -m wec.benchmark --data-dir data/raw
    python -m wec.benchmark --data-dir data/sample --epochs 3 --no-mlflow   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from wec import SITES
from wec.data import TOTAL, buoy_powers, clean, coordinates, load_site, split_indices
from wec.metrics import report
from wec.models_dl import TorchRegressor, TrainConfig
from wec.models_ml import ML_MODELS

DL_MODELS = ("mlp", "set_transformer")
ALL_MODELS = (*ML_MODELS, *DL_MODELS)


def run_site(path: Path, models, split: str, seed: int, epochs: int, max_rows: int | None, patience: int = 10):
    df, cleaning = clean(load_site(path))
    if max_rows:
        df = df.iloc[:max_rows]
    coords, total, buoys = coordinates(df), df[TOTAL].to_numpy(float), buoy_powers(df)
    tr, va, te = split_indices(len(df), split, seed)
    results = []
    for name in models:
        start = time.perf_counter()
        if name in ML_MODELS:
            feat, est = ML_MODELS[name](seed)
            # ML models use train+validation (HGB carves its own early-stopping split).
            fit_idx = np.concatenate([tr, va])
            est.fit(feat(coords[fit_idx]), total[fit_idx])
            pred = est.predict(feat(coords[te]))
            extra = {}
        else:
            model = TorchRegressor(name, TrainConfig(epochs=epochs, seed=seed, patience=patience))
            model.fit(coords[tr], total[tr], buoys[tr], coords[va], total[va])
            pred = model.predict(coords[te])
            extra = {"epochs_run": len(model.history)}
        results.append({
            "site": path.stem.replace("_Data", ""), "model": name, "split": split,
            "train_rows": int(len(tr) + len(va)), "test_rows": int(len(te)),
            "fit_seconds": round(time.perf_counter() - start, 1), **report(total[te], pred), **extra,
        })
        print(json.dumps(results[-1]))
    return cleaning, results


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Site | Split | Model | R² | RMSE (kW) | MAPE (%) | Fit time (s) |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['site']} | {r['split']} | {r['model']} | {r['r2']:.4f} | {r['rmse_kw']:.1f} "
            f"| {r['mape_pct']:.3f} | {r['fit_seconds']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    ap.add_argument("--splits", nargs="+", default=["random"], choices=["random", "blocked"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stopping patience in epochs for the neural networks")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    all_rows, cleaning = [], {}
    for site in args.sites:
        for split in args.splits:
            rep, rows = run_site(args.data_dir / f"{site}_Data.csv", args.models, split, args.seed,
                                 args.epochs, args.max_rows, args.patience)
            cleaning[site] = rep.__dict__
            all_rows += rows

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({"cleaning": cleaning, "results": all_rows}, indent=2))
    (args.out / "results.md").write_text(to_markdown(all_rows))

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("wave-energy-farm")
        for r in all_rows:
            with mlflow.start_run(run_name=f"{r['site']}-{r['model']}-{r['split']}"):
                mlflow.log_params({k: r[k] for k in ("site", "model", "split", "train_rows", "test_rows")}
                                  | {"seed": args.seed, "epochs": args.epochs, "patience": args.patience})
                mlflow.log_metrics({k: r[k] for k in ("r2", "rmse_kw", "mape_pct", "fit_seconds")})


if __name__ == "__main__":
    main()
