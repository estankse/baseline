from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "FL_robust" / "logs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot-FL-robust"
PRIORITY_METRICS = [
    "loss",
    "accuracy",
    "robust_loss",
    "robust_accuracy",
    "num_clients",
    "total_samples",
    "num_eval_clients",
    "num_eval_samples",
    "num_pgd_batches",
    "num_pgd_samples",
]
COMPARE_METRICS = ["accuracy", "loss", "robust_accuracy", "robust_loss"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric_metrics(metrics: dict[str, object]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            extracted[key] = float(value)
    return extracted


def metric_names(records: list[dict[str, float]]) -> list[str]:
    discovered = set()
    for record in records:
        discovered.update(key for key in record.keys() if key != "round")
    ordered = [name for name in PRIORITY_METRICS if name in discovered]
    ordered.extend(sorted(discovered - set(ordered)))
    return ordered


def plot_single_line(
    rounds: list[float],
    values: list[float],
    title: str,
    ylabel: str,
    save_path: Path,
    color: str | None = None,
) -> None:
    if not rounds:
        return
    plt.figure(figsize=(8, 5))
    if color is None:
        plt.plot(rounds, values, linewidth=2)
    else:
        plt.plot(rounds, values, linewidth=2, color=color)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Round", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close()


def plot_multi_lines(
    series_dict: dict[str, list[tuple[float, float]]],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    if not series_dict:
        return
    plt.figure(figsize=(10, 6))
    labels = sorted(series_dict.keys())
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    for idx, label in enumerate(labels):
        points = sorted(series_dict[label], key=lambda item: item[0])
        if not points:
            continue
        rounds = [point[0] for point in points]
        values = [point[1] for point in points]
        plt.plot(rounds, values, linewidth=2, color=cmap(idx), label=label)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Round", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close()


def parse_log(log_path: Path) -> dict[str, object]:
    train_records: list[dict[str, float]] = []
    eval_records: list[dict[str, float]] = []
    pgd_config: dict[str, object] | None = None
    method_config: dict[str, object] | None = None
    method_name = log_path.stem.split("_", 1)[0]

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            record_type = entry.get("type")
            round_idx = float(entry.get("round", -1))
            metrics = numeric_metrics(entry.get("metrics", {}))
            if record_type == "train":
                train_records.append({"round": round_idx, **metrics})
            elif record_type == "eval":
                eval_records.append({"round": round_idx, **metrics})
                if pgd_config is None and isinstance(entry.get("pgd"), dict):
                    pgd_config = dict(entry["pgd"])
                if method_config is None and isinstance(entry.get(method_name), dict):
                    method_config = dict(entry[method_name])

    final_eval = eval_records[-1] if eval_records else {}
    best_eval: dict[str, float] = {}
    for metric in COMPARE_METRICS:
        values = [record[metric] for record in eval_records if metric in record]
        if not values:
            continue
        if "loss" in metric:
            best_eval[metric] = min(values)
        else:
            best_eval[metric] = max(values)

    return {
        "method": method_name,
        "train_records": train_records,
        "eval_records": eval_records,
        "pgd_config": pgd_config,
        "method_config": method_config,
        "final_eval": {key: value for key, value in final_eval.items() if key != "round"},
        "best_eval": best_eval,
    }


def analyse_single_log(log_path: Path, output_dir: Path) -> dict[str, object]:
    ensure_dir(output_dir)
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    ensure_dir(train_dir)
    ensure_dir(eval_dir)

    parsed = parse_log(log_path)
    train_records = parsed["train_records"]
    eval_records = parsed["eval_records"]

    plt.style.use("seaborn-v0_8-muted")

    for metric in metric_names(train_records):
        rounds = [record["round"] for record in train_records if metric in record]
        values = [record[metric] for record in train_records if metric in record]
        plot_single_line(
            rounds=rounds,
            values=values,
            title=f"Train {metric}",
            ylabel=metric,
            save_path=train_dir / f"{metric}.png",
        )

    for metric in metric_names(eval_records):
        rounds = [record["round"] for record in eval_records if metric in record]
        values = [record[metric] for record in eval_records if metric in record]
        plot_single_line(
            rounds=rounds,
            values=values,
            title=f"Eval {metric}",
            ylabel=metric,
            save_path=eval_dir / f"{metric}.png",
        )

    summary = {
        "log_file": str(log_path),
        "method": parsed["method"],
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "final_eval": parsed["final_eval"],
        "best_eval": parsed["best_eval"],
        "pgd_config": parsed["pgd_config"],
        "method_config": parsed["method_config"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return parsed


def latest_logs_by_method(log_paths: list[Path]) -> list[Path]:
    latest: dict[str, Path] = {}
    for log_path in sorted(log_paths):
        method = log_path.stem.split("_", 1)[0]
        current = latest.get(method)
        if current is None or log_path.stem > current.stem:
            latest[method] = log_path
    return sorted(latest.values(), key=lambda path: path.stem)


def build_series(eval_records: list[dict[str, float]], metric: str) -> list[tuple[float, float]]:
    return [
        (record["round"], record[metric])
        for record in eval_records
        if "round" in record and metric in record
    ]


def analyse_log_directory(log_dir: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    run_dir = output_dir / "runs"
    compare_dir = output_dir / "compare_latest"
    ensure_dir(run_dir)
    ensure_dir(compare_dir)

    log_paths = sorted(log_dir.glob("*.jsonl"))
    if not log_paths:
        raise FileNotFoundError(f"No .jsonl logs found under {log_dir}")

    parsed_by_log: dict[str, dict[str, object]] = {}
    for log_path in log_paths:
        parsed_by_log[log_path.stem] = analyse_single_log(log_path, run_dir / log_path.stem)

    latest_logs = latest_logs_by_method(log_paths)
    latest_summary: dict[str, dict[str, object]] = {}
    for metric in COMPARE_METRICS:
        series_dict: dict[str, list[tuple[float, float]]] = {}
        for log_path in latest_logs:
            parsed = parsed_by_log[log_path.stem]
            eval_records = parsed["eval_records"]
            series = build_series(eval_records, metric)
            if not series:
                continue
            label = parsed["method"]
            series_dict[label] = series
            latest_summary[label] = {
                "log_file": str(log_path),
                "final_eval": parsed["final_eval"],
                "best_eval": parsed["best_eval"],
            }
        if series_dict:
            plot_multi_lines(
                series_dict=series_dict,
                title=f"Latest Robust Runs - {metric}",
                ylabel=metric,
                save_path=compare_dir / f"{metric}.png",
            )

    with (compare_dir / "summary_latest.json").open("w", encoding="utf-8") as handle:
        json.dump(latest_summary, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse standalone FL_robust experiment logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyse_log_directory(log_dir=args.log_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
