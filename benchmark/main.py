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

import numpy as np

from benchmark.task import load_tasks_for_dataset
from benchmark.method import METHOD_REGISTRY
from benchmark.fitter import FITTER_REGISTRY
from benchmark.runner import run_repeat, get_checkpoints


def _aggregate_results(run_results: list, checkpoints: list) -> dict:
    """Aggregate RunResults for the same task across seeds into mean ± std."""
    task_id = run_results[0].task_id
    fracs = checkpoints

    r2_matrix = np.array([
        [r.r2_at_checkpoints[f] for f in fracs] for r in run_results
    ])  # (n_seeds, n_checkpoints)
    # Treat any non-finite R² as -1
    r2_matrix = np.where(np.isfinite(r2_matrix), r2_matrix, -1.0)
    log_auc_arr = np.array([r.log_auc for r in run_results])

    return {
        "task_id": task_id,
        "n_seeds": len(run_results),
        "r2_mean": r2_matrix.mean(axis=0).tolist(),
        "r2_std": r2_matrix.std(axis=0).tolist(),
        "log_auc_mean": float(log_auc_arr.mean()),
        "log_auc_std": float(log_auc_arr.std()),
    }


def _plot_results(task_id: str, run_results: list, output_dir: Path, checkpoints: list):
    """Plot test R² vs budget: scatter all seeds, mean line, std shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fracs = np.array(checkpoints)
    r2_raw = np.array([
        [r.r2_at_checkpoints[f] for f in checkpoints] for r in run_results
    ])
    r2_raw = np.where(np.isfinite(r2_raw), r2_raw, -1.0)
    n_seeds = r2_raw.shape[0]

    mean_r2 = r2_raw.mean(axis=0)
    std_r2 = r2_raw.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for s in range(n_seeds):
        ax.scatter(fracs, r2_raw[s], s=8, alpha=0.35, color="cornflowerblue",
                   edgecolors="none", zorder=2)

    ax.fill_between(fracs, mean_r2 - std_r2, mean_r2 + std_r2,
                    alpha=0.25, color="salmon", label="mean \u00b1 std", zorder=3)
    ax.plot(fracs, mean_r2, "-", color="orangered", linewidth=2,
            label="mean", zorder=4)
    ax.axhline(0, ls="--", color="grey", linewidth=0.8, label="R\u00b2=0", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Budget fraction (log scale)")
    ax.set_ylabel("Test R\u00b2")
    ax.set_title(f"Test R\u00b2 vs. budget  ({n_seeds} experiments)\n{task_id}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_name = task_id.replace("/", "__").replace(" ", "_")
    fig.savefig(plot_dir / f"{safe_name}.png", dpi=150)
    plt.close(fig)

    print(f"Saved plot to {plot_dir}/{safe_name}.png", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Scaling Law Benchmark")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--fitter", type=str, default="lbfgsb",
                        choices=list(FITTER_REGISTRY.keys()))
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of independent repetitions (seeds 0..repeat-1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seeds (overrides --repeat)")
    parser.add_argument("--n-restarts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of processes for parallel multistart refit")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--task-filter", type=str, default=None,
                        help="Only run tasks whose task_id contains this substring")
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
    if args.task_filter:
        tasks = [t for t in tasks if args.task_filter in t.task_id.split("/")]
        if not tasks:
            print(f"No tasks match filter '{args.task_filter}'", file=sys.stderr)
            sys.exit(1)

    # Dynamic method loading from methods_evolve/
    if args.method not in METHOD_REGISTRY:
        evolve_dir = Path(__file__).resolve().parent.parent / "methods_evolve"
        method_file = evolve_dir / f"{args.method}.py"
        if method_file.exists():
            import importlib.util
            # Add methods_evolve to sys.path so workers can import the module
            str_evolve = str(evolve_dir)
            if str_evolve not in sys.path:
                sys.path.insert(0, str_evolve)
            spec = importlib.util.spec_from_file_location(args.method, str(method_file))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[args.method] = mod
            spec.loader.exec_module(mod)
            method_cls = mod.Method
        else:
            print(f"Unknown method '{args.method}'. Not in registry and no file "
                  f"{method_file} found.", file=sys.stderr)
            sys.exit(1)
    else:
        method_cls = METHOD_REGISTRY[args.method]

    try:
        method = method_cls(max_workers=args.workers)
    except TypeError:
        method = method_cls()
    fitter = FITTER_REGISTRY[args.fitter](n_restarts=args.n_restarts,
                                          max_workers=args.workers)

    result_path = output_dir / "result.json"
    aggregated = []
    for task in tasks:
        print(f"Running {task.task_id} x{len(seeds)} seeds ...", file=sys.stderr)
        results = run_repeat(task, method, fitter, seeds)
        cp = get_checkpoints(task)
        agg = _aggregate_results(results, cp)
        aggregated.append(agg)

        # Persist results and plot after every task
        result_path.write_text(json.dumps(aggregated, indent=2) + "\n")
        if not args.no_plot:
            _plot_results(task.task_id, results, output_dir, cp)
        print(f"  -> saved {task.task_id}", file=sys.stderr)

    print(f"Results saved to {result_path}", file=sys.stderr)
    print(f"Output directory: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
