import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import re

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_global_avg_series(eval_avg_by_task, metric_name):
    task_names = sorted(eval_avg_by_task.keys(), key=extract_task_order)

    series = []
    xticks = []
    xtick_labels = []
    current_offset = 0

    for task_name in task_names:
        records = sorted(eval_avg_by_task[task_name], key=lambda x: x["round"])
        points = [(r["round"], r[metric_name]) for r in records if metric_name in r]

        if not points:
            continue

        shifted_points = [(r + current_offset, v) for r, v in points]
        series.extend(shifted_points)

        shifted_rounds = [p[0] for p in shifted_points]
        center = (shifted_rounds[0] + shifted_rounds[-1]) / 2
        xticks.append(center)
        xtick_labels.append(task_name)

        current_offset = shifted_rounds[-1] + 2

    return series, xticks, xtick_labels


def plot_global_avg_metric(series, xticks, xtick_labels, title, ylabel, save_path):
    if not series:
        return

    xs = [p[0] for p in series]
    ys = [p[1] for p in series]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linewidth=2, label=title)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Task Progress", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    if xticks:
        plt.xticks(xticks, xtick_labels, rotation=30)

    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()

def plot_single_line(rounds, values, title, xlabel, ylabel, save_path, color=None):
    plt.figure(figsize=(8, 5))
    if color is not None:
        plt.plot(rounds, values, linewidth=2, label=title, color=color)
    else:
        plt.plot(rounds, values, linewidth=2, label=title)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()

def extract_task_order(task_name):
    """
    task_0 -> 0
    task_12 -> 12
    其他无法解析的名字放到最后
    """
    m = re.search(r'(\d+)$', task_name)
    return int(m.group(1)) if m else 10**9


def plot_multi_lines(series_dict, title, xlabel, ylabel, save_path, separate_tasks_on_x=True):
    """
    series_dict:
        {
            "task_0": [(0, 0.1), (1, 0.2)],
            "task_1": [(0, 0.15), (1, 0.3)],
        }

    separate_tasks_on_x=True:
        自动把不同 task 沿 x 轴错开，避免 round 重叠
    """
    plt.figure(figsize=(10, 6))

    # 用足够多的颜色，避免默认颜色循环复用太快
    task_names = sorted(series_dict.keys(), key=extract_task_order)
    cmap = plt.get_cmap("tab20", max(len(task_names), 1))

    current_offset = 0
    xticks = []
    xtick_labels = []

    for idx, task_name in enumerate(task_names):
        points = sorted(series_dict[task_name], key=lambda x: x[0])
        rounds = [p[0] for p in points]
        values = [p[1] for p in points]

        if separate_tasks_on_x:
            shifted_rounds = [r + current_offset for r in rounds]
            # 给下一个任务留一点空隙
            current_offset = max(shifted_rounds) + 2 if shifted_rounds else current_offset + 2

            # 记录任务中心位置，方便标 task 名
            if shifted_rounds:
                center = (shifted_rounds[0] + shifted_rounds[-1]) / 2
                xticks.append(center)
                xtick_labels.append(task_name)
        else:
            shifted_rounds = rounds

        plt.plot(
            shifted_rounds,
            values,
            linewidth=2,
            label=task_name,
            color=cmap(idx)
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    if separate_tasks_on_x and xticks:
        plt.xticks(xticks, xtick_labels, rotation=30)

    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()



def plots_fcl(log_text, output_dir="plot-FCL"):
    """
    支持 federated continual learning 日志格式：
    - train: {"type":"train","task_id":"task_0","round":0,"metrics":{...}}
    - eval : {"type":"eval","task_id":"task_0","round":0,"avg_metrics":{...},"task_metrics":{...}}
    """
    ensure_dir(output_dir)

    train_dir = os.path.join(output_dir, "train")
    eval_avg_dir = os.path.join(output_dir, "eval_avg")
    eval_task_dir = os.path.join(output_dir, "eval_task")
    eval_compare_dir = os.path.join(output_dir, "eval_compare")

    for d in [train_dir, eval_avg_dir, eval_task_dir, eval_compare_dir]:
        ensure_dir(d)

    # --------------------------
    # 1. 解析日志
    # --------------------------
    train_data_by_task = defaultdict(list)
    eval_avg_by_task = defaultdict(list)

    # task-wise evaluation history
    # 例如 eval_task_history["task_0"]["accuracy"] = [(round, value), ...]
    eval_task_history = defaultdict(lambda: defaultdict(list))

    for line in log_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        entry = json.loads(line)
        entry_type = entry.get("type")
        task_id = entry.get("task_id", "unknown_task")
        round_id = entry.get("round")

        if entry_type == "train":
            metrics = entry.get("metrics", {})
            train_data_by_task[task_id].append({
                "round": round_id,
                **metrics
            })

        elif entry_type == "eval":
            avg_metrics = entry.get("avg_metrics", {})
            task_metrics = entry.get("task_metrics", {})

            if avg_metrics:
                eval_avg_by_task[task_id].append({
                    "round": round_id,
                    **avg_metrics
                })

            for eval_task_name, metrics in task_metrics.items():
                for metric_name, metric_value in metrics.items():
                    # 只画标量指标
                    if isinstance(metric_value, (int, float)):
                        eval_task_history[eval_task_name][metric_name].append(
                            (round_id, metric_value)
                        )

    plt.style.use('seaborn-v0_8-muted')

    # --------------------------
    # 2. 绘制 train 曲线（每个训练 task 单独画）
    # --------------------------
    train_metrics_to_plot = [
        "client_loss",
        "client_ce_loss",
        "client_sparse_loss",
        "client_retro_loss",
        "client_accuracy",
        "client_kb_size",
        "client_shared_norm",
        "client_adaptive_norm",
        "client_shared_density",
        "client_adaptive_density",
    ]

    for task_id, records in train_data_by_task.items():
        records = sorted(records, key=lambda x: x["round"])

        for metric_name in train_metrics_to_plot:
            rounds = [r["round"] for r in records if metric_name in r]
            values = [r[metric_name] for r in records if metric_name in r]

            if not rounds:
                continue

            save_path = os.path.join(train_dir, f"{task_id}_{metric_name}.png")
            plot_single_line(
                rounds=rounds,
                values=values,
                title=f"{task_id} - Train {metric_name}",
                xlabel="Round",
                ylabel=metric_name,
                save_path=save_path
            )

    # --------------------------
    # 3. 绘制 eval avg_metrics（每个当前任务单独画）
    # --------------------------
    eval_avg_metrics_to_plot = ["accuracy", "loss"]

    for task_id, records in eval_avg_by_task.items():
        records = sorted(records, key=lambda x: x["round"])

        for metric_name in eval_avg_metrics_to_plot:
            rounds = [r["round"] for r in records if metric_name in r]
            values = [r[metric_name] for r in records if metric_name in r]

            if not rounds:
                continue

            save_path = os.path.join(eval_avg_dir, f"{task_id}_avg_{metric_name}.png")
            plot_single_line(
                rounds=rounds,
                values=values,
                title=f"{task_id} - Eval Avg {metric_name}",
                xlabel="Round",
                ylabel=metric_name,
                save_path=save_path
            )

    # --------------------------
    # 4. 绘制每个历史任务的 task_metrics 单独曲线
    # --------------------------
    # 例如：task_0_accuracy.png, task_0_loss.png
    for eval_task_name, metric_dict in eval_task_history.items():
        for metric_name, points in metric_dict.items():
            points = sorted(points, key=lambda x: x[0])
            rounds = [p[0] for p in points]
            values = [p[1] for p in points]

            save_path = os.path.join(eval_task_dir, f"{eval_task_name}_{metric_name}.png")
            plot_single_line(
                rounds=rounds,
                values=values,
                title=f"{eval_task_name} - Eval {metric_name}",
                xlabel="Round",
                ylabel=metric_name,
                save_path=save_path
            )

    # --------------------------
    # 5. 绘制“所有任务同图对比”
    #    这是看遗忘最关键的图
    # --------------------------
    compare_metrics = ["accuracy", "loss"]

    for metric_name in compare_metrics:
        series_dict = {}

        for eval_task_name, metric_dict in eval_task_history.items():
            if metric_name in metric_dict:
                series_dict[eval_task_name] = metric_dict[metric_name]

        if series_dict:
            save_path = os.path.join(eval_compare_dir, f"all_tasks_{metric_name}.png")
            plot_multi_lines(
                series_dict=series_dict,
                title=f"All Tasks Eval {metric_name}",
                xlabel="Task Progress",
                ylabel=metric_name,
                save_path=save_path,
                separate_tasks_on_x=True
            )
    # --------------------------
    # 绘制 avg_metrics 的全局跨任务曲线
    # --------------------------
    eval_avg_global_dir = os.path.join(output_dir, "eval_avg_global")
    ensure_dir(eval_avg_global_dir)
    for metric_name in ["accuracy", "loss"]:
        series, xticks, xtick_labels = build_global_avg_series(eval_avg_by_task, metric_name)

        if series:
            save_path = os.path.join(eval_avg_global_dir, f"global_avg_{metric_name}.png")
            plot_global_avg_metric(
                series=series,
                xticks=xticks,
                xtick_labels=xtick_labels,
                title=f"Global Avg {metric_name}",
                ylabel=metric_name,
                save_path=save_path
            )

if __name__ == "__main__":
    log_path = "../experiments/logs/fedweit_20260419_000346.jsonl"
    with open(log_path, "r", encoding="utf-8") as f:
        plots_fcl(f.read(), output_dir="plot-FCL")