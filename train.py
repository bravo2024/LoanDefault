from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.evaluate import save_metrics, print_report
from src.persist import save_model


def main():
    parser = argparse.ArgumentParser(description="LoanDefault — Credit Risk Training Pipeline")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--n", type=int, default=10000, help="Sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    print("Loading data...")
    data = make_synthetic(n=args.n, seed=args.seed)
    print(f"  {data['n_samples']:,} samples, {data['n_features']} features, "
          f"default rate={data['positive_rate']:.2%}")

    print("Training models...")
    bundle = train_all_models(data, seed=args.seed)
    print_report({name: res["metrics"] for name, res in bundle["results"].items()})

    print("\nCross-validation...")
    cv = cross_validate(data, seed=args.seed, n_folds=args.cv)
    for name, scores in cv.items():
        print(f"  {name:25s} AUC={scores['roc_auc']['mean']:.4f} "
              f"±{scores['roc_auc']['std']:.4f}")

    best_name = max(bundle["results"], key=lambda n: bundle["results"][n]["metrics"].get("roc_auc", 0))
    print(f"\nBest model: {best_name}")
    save_model({
        "models": bundle["models"],
        "scaler": bundle["scaler"],
        "features": bundle["features"],
        "best_model": best_name,
        "seed": args.seed,
    })
    save_metrics({
        "holdout": {name: bundle["results"][name]["metrics"] for name in bundle["results"]},
        "cv": cv,
        "best_model": best_name,
    })
    print("Saved models/model.pkl and models/metrics.json")


if __name__ == "__main__":
    main()
