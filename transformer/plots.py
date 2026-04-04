import json
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_comparison(all_data):
    # 定义需要绘制的指标
    metrics = ['train_ppl', 'val_ppl', 'bleu', 'train_tps', 'peak_mem_mb']

    # 设置绘图风格
    plt.style.use('ggplot')  # 如果报错可以改回 'ggplot' 或 'seaborn-darkgrid'

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for k_val, data in all_data.items():
            history = data['history']
            epochs = [h['epoch'] for h in history]
            values = [h[metric] for h in history]

            plt.plot(epochs, values, marker='o', linestyle='-', label=f'K={k_val}', linewidth=2)

        plt.title(f'Comparison of {metric} across different K values', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 自动保存图片
        file_name = f'comparison_{metric}.png'
        plt.savefig(file_name)
        print(f"已生成图表: {file_name}")
        plt.close()


if __name__ == "__main__":
    # --- 请在此处修改你的文件名 ---
    file_map = {
        "2": "./results/K2_seed42/summary.json",
        "4": "./results/K4_seed42/summary.json",
        "6": "./results/K6_seed42/summary.json"
    }

    all_results = {}

    try:
        for k, path in file_map.items():
            if os.path.exists(path):
                all_results[k] = load_data(path)
                print(f"成功加载 K={k} 的数据")
            else:
                print(f"警告: 找不到文件 {path}，跳过该项。")

        if all_results:
            plot_comparison(all_results)
            print("\n所有图表已绘制完成！")
        else:
            print("错误: 没有加载到任何数据，请检查文件名。")

    except Exception as e:
        print(f"发生错误: {e}")