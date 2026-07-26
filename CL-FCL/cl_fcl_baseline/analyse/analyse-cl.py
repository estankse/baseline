from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "CL" / "logs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot-CL"
AVERAGE_METRICS = [
    "average_accuracy",
    "average_forgetting",
    "backward_transfer",
]
TRAIN_METRIC_PRIORITY = [
    "loss",
    "task_loss",
    "new_loss",
    "classification_loss",
    "distillation_loss",
    "ewc_loss",
    "constraint_violations",
    "gradient_projected",
    "lr",
    "warmup",
]
TASK_METRIC_PRIORITY = ["accuracy"]


def numeric_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def ordered_metric_names(
    records: list[dict[str, object]], priority: list[str], ignored: set[str]
) -> list[str]:
    discovered: set[str] = set()
    for record in records:
        discovered.update(
            key
            for key, value in record.items()
            if key not in ignored and isinstance(value, (int, float))
        )
    ordered = [name for name in priority if name in discovered]
    ordered.extend(sorted(discovered - set(ordered)))
    return ordered


def task_index(task_id: str, fallback: object = None) -> int:
    if isinstance(fallback, (int, float)):
        return int(fallback)
    match = re.search(r"(\d+)$", task_id)
    return int(match.group(1)) if match else 0


def infer_method(log_path: Path, config: dict[str, object]) -> str:
    args = config.get("args", {})
    if isinstance(args, dict) and args.get("algorithm"):
        return str(args["algorithm"]).lower()
    stem = re.sub(r"_\d{8}_\d{6}$", "", log_path.stem)
    return re.sub(r"^cl_", "", stem).lower()


def add_task_progress(records: list[dict[str, object]]) -> None:
    totals = Counter(int(record["task_index"]) for record in records)
    positions: Counter[int] = Counter()
    for record in records:
        index = int(record["task_index"])
        positions[index] += 1
        record["progress"] = index + positions[index] / totals[index]


def finite_points(
    records: list[dict[str, object]], x_name: str, metric: str
) -> tuple[list[float], list[float]]:
    points = [
        (float(record[x_name]), float(record[metric]))
        for record in records
        if x_name in record
        and metric in record
        and math.isfinite(float(record[x_name]))
        and math.isfinite(float(record[metric]))
    ]
    return [point[0] for point in points], [point[1] for point in points]


def style_task_axis(max_task_index: int) -> None:
    if max_task_index < 0:
        return
    ticks = [index + 0.5 for index in range(max_task_index + 1)]
    plt.xticks(ticks, [f"task_{index}" for index in range(max_task_index + 1)], rotation=30)
    for boundary in range(1, max_task_index + 1):
        plt.axvline(boundary, color="grey", linewidth=0.8, alpha=0.25)


def plot_single_line(
    xs: list[float],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    max_task_index: int = -1,
) -> None:
    points = [
        (float(x), float(value))
        for x, value in zip(xs, values)
        if math.isfinite(float(x)) and math.isfinite(float(value))
    ]
    if not points:
        return
    xs = [point[0] for point in points]
    values = [point[1] for point in points]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, values, linewidth=2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    if xlabel == "Task progress":
        style_task_axis(max_task_index)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close()


def plot_multi_lines(
    series: dict[str, list[tuple[float, float]]],
    title: str,
    ylabel: str,
    save_path: Path,
    max_task_index: int,
) -> None:
    valid_series = {
        label: [
            (x, value)
            for x, value in points
            if math.isfinite(x) and math.isfinite(value)
        ]
        for label, points in series.items()
    }
    valid_series = {label: points for label, points in valid_series.items() if points}
    if not valid_series:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    labels = sorted(valid_series)
    colors = plt.get_cmap("tab20", max(len(labels), 1))
    for index, label in enumerate(labels):
        points = sorted(valid_series[label], key=lambda point: point[0])
        plt.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            linewidth=2,
            label=label,
            color=colors(index),
        )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Task progress", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    style_task_axis(max_task_index)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.close()


def parse_log(log_path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    train_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    eval_records: list[dict[str, object]] = []

    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{log_path}:{line_number}: invalid JSON: {error}") from error

            record_type = entry.get("type")
            if record_type == "config":
                config = dict(entry)
                continue
            if record_type not in {"train", "eval"}:
                continue

            task_id = str(entry.get("task_id", "unknown_task"))
            index = task_index(task_id, entry.get("task_index"))
            epoch = float(entry.get("epoch", entry.get("metrics", {}).get("epoch", -1)))
            if record_type == "train":
                train_by_task[task_id].append(
                    {
                        "task_index": index,
                        "epoch": epoch,
                        **numeric_metrics(entry.get("metrics")),
                    }
                )
                continue

            eval_records.append(
                {
                    "task_id": task_id,
                    "task_index": index,
                    "epoch": epoch,
                    **numeric_metrics(entry.get("metrics")),
                    "task_metrics": {
                        str(name): numeric_metrics(metrics)
                        for name, metrics in entry.get("task_metrics", {}).items()
                        if isinstance(metrics, dict)
                    },
                }
            )

    add_task_progress(eval_records)
    method = infer_method(log_path, config)
    max_index = max(
        [int(record["task_index"]) for record in eval_records]
        + [
            int(record["task_index"])
            for records in train_by_task.values()
            for record in records
        ],
        default=-1,
    )
    return {
        "method": method,
        "config": config,
        "train_by_task": dict(train_by_task),
        "eval_records": eval_records,
        "max_task_index": max_index,
    }


def task_metric_history(
    eval_records: list[dict[str, object]],
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    history: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in eval_records:
        progress = float(record["progress"])
        for name, metrics in record["task_metrics"].items():
            for metric, value in metrics.items():
                if metric == "num_samples":
                    continue
                history[name][metric].append((progress, float(value)))
    return {
        task: {metric: points for metric, points in metrics.items()}
        for task, metrics in history.items()
    }


def safe_number(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def evaluation_summary(
    eval_records: list[dict[str, object]], metric_names: list[str]
) -> tuple[dict[str, float | None], dict[str, float]]:
    final = {
        metric: safe_number(eval_records[-1].get(metric))
        for metric in metric_names
        if metric in eval_records[-1]
    } if eval_records else {}
    best: dict[str, float] = {}
    for metric in metric_names:
        values = [
            float(record[metric])
            for record in eval_records
            if metric in record and math.isfinite(float(record[metric]))
        ]
        if not values:
            continue
        best[metric] = min(values) if "loss" in metric or "forgetting" in metric else max(values)
    return final, best


def analyse_single_log(log_path: Path, output_dir: Path) -> dict[str, object]:
    parsed = parse_log(log_path)
    train_by_task = parsed["train_by_task"]
    eval_records = parsed["eval_records"]
    max_index = int(parsed["max_task_index"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-muted")

    train_dir = output_dir / "train"
    for task_id, records in train_by_task.items():
        metrics = ordered_metric_names(
            records,
            TRAIN_METRIC_PRIORITY,
            {"task_index", "epoch"},
        )
        for metric in metrics:
            xs, values = finite_points(records, "epoch", metric)
            plot_single_line(
                xs,
                values,
                f"{task_id} - train {metric}",
                "Epoch",
                metric,
                train_dir / f"{task_id}_{metric}.png",
            )

    eval_metrics = ordered_metric_names(
        eval_records,
        AVERAGE_METRICS,
        {"task_id", "task_index", "epoch", "progress", "task_metrics"},
    )
    for metric in eval_metrics:
        xs, values = finite_points(eval_records, "progress", metric)
        plot_single_line(
            xs,
            values,
            f"{parsed['method']} - {metric}",
            "Task progress",
            metric,
            output_dir / "eval_average" / f"{metric}.png",
            max_index,
        )

    history = task_metric_history(eval_records)
    all_task_metrics = sorted(
        {
            metric
            for metrics in history.values()
            for metric in metrics
        },
        key=lambda name: (
            TASK_METRIC_PRIORITY.index(name)
            if name in TASK_METRIC_PRIORITY
            else len(TASK_METRIC_PRIORITY),
            name,
        ),
    )
    for task_id, metrics in history.items():
        for metric, points in metrics.items():
            plot_single_line(
                [point[0] for point in points],
                [point[1] for point in points],
                f"{task_id} - eval {metric}",
                "Task progress",
                metric,
                output_dir / "eval_task" / f"{task_id}_{metric}.png",
                max_index,
            )
    for metric in all_task_metrics:
        plot_multi_lines(
            {
                task_id: metrics[metric]
                for task_id, metrics in history.items()
                if metric in metrics
            },
            f"All tasks - {metric}",
            metric,
            output_dir / "eval_compare" / f"all_tasks_{metric}.png",
            max_index,
        )

    final_eval, best_eval = evaluation_summary(eval_records, eval_metrics)
    config_args = parsed["config"].get("args", {})
    summary = {
        "log_file": str(log_path),
        "method": parsed["method"],
        "num_train_records": sum(len(records) for records in train_by_task.values()),
        "num_eval_records": len(eval_records),
        "num_completed_tasks": max_index + 1,
        "final_eval": final_eval,
        "best_eval": best_eval,
        "config": config_args if isinstance(config_args, dict) else {},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return parsed


def latest_by_method(
    log_paths: list[Path], parsed_by_log: dict[str, dict[str, object]]
) -> list[Path]:
    latest: dict[str, Path] = {}
    for log_path in sorted(log_paths):
        method = str(parsed_by_log[log_path.stem]["method"])
        if method not in latest or log_path.stem > latest[method].stem:
            latest[method] = log_path
    return sorted(latest.values(), key=lambda path: str(parsed_by_log[path.stem]["method"]))


def analyse_log_directory(log_dir: Path, output_dir: Path) -> None:
    log_paths = sorted(log_dir.glob("*.jsonl"))
    if not log_paths:
        raise FileNotFoundError(f"No .jsonl logs found under {log_dir}")

    runs_dir = output_dir / "runs"
    compare_dir = output_dir / "compare_latest"
    parsed_by_log = {
        log_path.stem: analyse_single_log(log_path, runs_dir / log_path.stem)
        for log_path in log_paths
    }
    latest_logs = latest_by_method(log_paths, parsed_by_log)
    max_index = max(
        (int(parsed_by_log[path.stem]["max_task_index"]) for path in latest_logs),
        default=-1,
    )

    for metric in AVERAGE_METRICS:
        series: dict[str, list[tuple[float, float]]] = {}
        for log_path in latest_logs:
            parsed = parsed_by_log[log_path.stem]
            points = [
                (float(record["progress"]), float(record[metric]))
                for record in parsed["eval_records"]
                if metric in record
            ]
            if points:
                series[str(parsed["method"])] = points
        plot_multi_lines(
            series,
            f"Latest CL runs - {metric}",
            metric,
            compare_dir / f"{metric}.png",
            max_index,
        )

    latest_summary = {}
    for log_path in latest_logs:
        parsed = parsed_by_log[log_path.stem]
        eval_records = parsed["eval_records"]
        final_eval, best_eval = evaluation_summary(eval_records, AVERAGE_METRICS)
        latest_summary[str(parsed["method"])] = {
            "log_file": str(log_path),
            "final_eval": final_eval,
            "best_eval": best_eval,
        }
    compare_dir.mkdir(parents=True, exist_ok=True)
    with (compare_dir / "summary_latest.json").open("w", encoding="utf-8") as handle:
        json.dump(latest_summary, handle, ensure_ascii=False, indent=2, allow_nan=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse standalone CL JSONL logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyse_log_directory(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
