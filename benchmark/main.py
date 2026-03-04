"""CLI entry point for the scaling law benchmark.

Usage:
    python -m benchmark.main --dataset sft_scaling_law --method random --fitter lbfgsb --repeat 5
    python -m benchmark.main --dataset parallel_scaling_law --method random --fitter lbfgsb --repeat 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import warnings

import numpy as np

from benchmark.task import load_tasks_for_dataset
from benchmark.method import METHOD_REGISTRY
from benchmark.fitter import FITTER_REGISTRY
from benchmark.runner import run_repeat, BUDGET_CHECKPOINTS, ALPHA_LEVELS


def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _aggregate_results(run_results: list) -> dict:
    """Aggregate RunResults for the same task across seeds into mean ± std."""
    task_id = run_results[0].task_id
    fracs = BUDGET_CHECKPOINTS

    r2_matrix = np.array([
        [r.r2_at_checkpoints[f] for f in fracs] for r in run_results
    ])  # (n_seeds, n_checkpoints)
    # Treat any non-finite R² as -1
    r2_matrix = np.where(np.isfinite(r2_matrix), r2_matrix, -1.0)
    mse_matrix = np.array([
        [r.mse_at_checkpoints[f] for f in fracs] for r in run_results
    ])  # (n_seeds, n_checkpoints)
    # Replace inf with nan for aggregation, but cap for JSON output
    mse_finite = np.where(np.isfinite(mse_matrix), mse_matrix, np.nan)
    auc_arr = np.array([r.auc for r in run_results])
    oracle_r2_arr = np.array([r.oracle_r2 for r in run_results])
    oracle_r2_arr = np.where(np.isfinite(oracle_r2_arr), oracle_r2_arr, -1.0)
    oracle_mse_arr = np.array([r.oracle_mse for r in run_results])
    oracle_mse_finite = np.where(np.isfinite(oracle_mse_arr), oracle_mse_arr, np.nan)

    btr_matrix = {
        alpha: np.array([r.budget_to_reach_values[alpha] for r in run_results])
        for alpha in ALPHA_LEVELS
    }

    def _safe_float(x):
        """Convert to float, replacing non-finite with None for JSON."""
        v = float(x)
        return v if np.isfinite(v) else None

    return {
        "task_id": task_id,
        "n_seeds": len(run_results),
        "budget_checkpoints": fracs,
        "r2_mean": r2_matrix.mean(axis=0).tolist(),
        "r2_std": r2_matrix.std(axis=0).tolist(),
        "r2_raw": r2_matrix.tolist(),
        "mse_mean": [_safe_float(v) for v in _nanmean(mse_finite, axis=0)],
        "mse_std": [_safe_float(v) for v in _nanstd(mse_finite, axis=0)],
        "auc_r2_mean": float(auc_arr.mean()),
        "auc_r2_std": float(auc_arr.std()),
        "oracle_r2_mean": float(oracle_r2_arr.mean()),
        "oracle_r2_std": float(oracle_r2_arr.std()),
        "oracle_mse_mean": _safe_float(oracle_mse_finite.mean()),
        "oracle_mse_std": _safe_float(oracle_mse_finite.std()),
        "budget_to_reach": {
            str(alpha): {
                "mean": float(v.mean()),
                "std": float(v.std()),
            }
            for alpha, v in btr_matrix.items()
        },
    }


def _plot_results(aggregated: list, output_dir: Path):
    """Plot test R² vs budget: scatter all seeds, median line, IQR shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for rec in aggregated:
        fracs = np.array(rec["budget_checkpoints"])
        r2_raw = np.array(rec["r2_raw"])  # (n_seeds, n_checkpoints)
        n_seeds = r2_raw.shape[0]

        median_r2 = np.median(r2_raw, axis=0)
        q25 = np.percentile(r2_raw, 25, axis=0)
        q75 = np.percentile(r2_raw, 75, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Scatter all individual seed points
        for s in range(n_seeds):
            ax.scatter(fracs, r2_raw[s], s=8, alpha=0.35, color="cornflowerblue",
                       edgecolors="none", zorder=2)

        # IQR shading
        ax.fill_between(fracs, q25, q75, alpha=0.25, color="salmon",
                        label="IQR", zorder=3)
        # Median line
        ax.plot(fracs, median_r2, "-", color="orangered", linewidth=2,
                label="median", zorder=4)
        # R²=0 reference
        ax.axhline(0, ls="--", color="grey", linewidth=0.8, label="R\u00b2=0", zorder=1)

        ax.set_xscale("log")
        ax.set_xlabel("Budget fraction (log scale)")
        ax.set_ylabel("Test R\u00b2")
        ax.set_title(f"Test R\u00b2 vs. budget  ({n_seeds} experiments)\n{rec['task_id']}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = rec["task_id"].replace("/", "__").replace(" ", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(aggregated)} plots to {plot_dir}/", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Scaling Law Benchmark")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, default="random",
                        choices=list(METHOD_REGISTRY.keys()))
    parser.add_argument("--fitter", type=str, default="lbfgsb",
                        choices=list(FITTER_REGISTRY.keys()))
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent repetitions (seeds 0..repeat-1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seeds (overrides --repeat)")
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of threads for parallel seed repetitions")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.repeat))

    # Create output directory: output/<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / args.dataset / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    args_record = {
        "dataset": args.dataset,
        "method": args.method,
        "fitter": args.fitter,
        "n_seeds": len(seeds),
        "n_restarts": args.n_restarts,
        "timestamp": timestamp,
    }
    (output_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n")

    tasks = load_tasks_for_dataset(args.dataset)
    method = METHOD_REGISTRY[args.method]()
    fitter = FITTER_REGISTRY[args.fitter](n_restarts=args.n_restarts)

    aggregated = []
    for task in tasks:
        print(f"Running {task.task_id} x{len(seeds)} seeds ...", file=sys.stderr)
        results = run_repeat(task, method, fitter, seeds, max_workers=args.workers)
        agg = _aggregate_results(results)
        aggregated.append(agg)

    # Save aggregated results
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(aggregated, indent=2) + "\n")
    print(f"Results saved to {result_path}", file=sys.stderr)

    # Plot
    if not args.no_plot:
        _plot_results(aggregated, output_dir)

    print(f"Output directory: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
