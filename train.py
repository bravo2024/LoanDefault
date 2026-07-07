"""Fit the competing-risks Cox model and save metrics."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import make_synthetic
from src.evaluate import save_metrics, print_report
from src.model import fit_and_evaluate
from src.persist import save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = make_synthetic(n=args.n, seed=args.seed)
    df = data["df"]
    print(f"{data['n_samples']:,} loans | defaults {(df.event == 1).mean():.1%} "
          f"| prepayments {(df.event == 2).mean():.1%}")

    model, metrics = fit_and_evaluate(data, seed=args.seed)
    print_report({"cox_competing_risks": metrics})
    save_model(model)
    save_metrics(metrics)
    print("Saved model + metrics under models/")


if __name__ == "__main__":
    main()
