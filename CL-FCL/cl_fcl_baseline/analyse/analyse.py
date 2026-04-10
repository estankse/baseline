import json
import matplotlib.pyplot as plt
import os

def plots(log_text, output_dir="plots"):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 解析数据
    train_data = []
    eval_data = []

    for line in log_text.strip().split('\n'):
        if not line.strip(): continue
        entry = json.loads(line)
        if entry['type'] == 'train':
            train_data.append({'round': entry['round'], **entry['metrics']})
        elif entry['type'] == 'eval':
            eval_data.append({'round': entry['round'], **entry['metrics']})

    # 提取坐标数据
    t_rounds = [d['round'] for d in train_data]
    e_rounds = [d['round'] for d in eval_data]

    # 3. 定义绘图配置: (数据来源, Key, 标题, 文件名, 颜色)
    plot_configs = [
        (train_data, 'server_distill_loss', 'Server Distill Loss', 'train_distill_loss.png', '#1f77b4'),
        (eval_data, 'loss', 'Evaluation Loss', 'eval_loss.png', '#ff7f0e'),
        (eval_data, 'accuracy', 'Evaluation Accuracy', 'eval_accuracy.png', '#2ca02c')
    ]

    plt.style.use('seaborn-v0_8-muted')

    for dataset, key, title, filename, color in plot_configs:
        # 提取当前指标的 X 和 Y
        rounds = [d['round'] for d in dataset]
        values = [d[key] for d in dataset]

        # 创建独立画布
        plt.figure(figsize=(8, 5))

        # 绘制曲线 (去掉了 marker，仅保留线条)
        plt.plot(rounds, values, color=color, linewidth=2, label=title)

        # 细节修饰
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel(key.replace('_', ' ').capitalize(), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存到指定文件夹
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")

        # 必须关闭当前 figure，防止内存占用和图像重叠
        plt.close()

if __name__ == "__main__":
    with open('../experiments/logs/fedkemf_20260410_000953.jsonl', 'r') as f:
        plots(f.read())