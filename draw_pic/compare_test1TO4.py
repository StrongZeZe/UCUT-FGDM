import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

# 设置后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# ===================== SCI期刊紧凑设置 =====================
fig_width_single = 3.5  # 双列期刊的单列宽度（3.5英寸）
base_fontsize = 11  # 基础字体大小，在紧凑图形中足够清晰

# 目标权重
target_weight = np.array([0.139, 0.267, 0.216, 0.133, 0.107, 0.138])

# 实验位置
positions = {
    'T1': r"E:\newManucript\python_code_rare\script2\data\simulation_data\test1",
    'T2': r"E:\newManucript\python_code_rare\script2\data\simulation_data\test2",
    'T3': r"E:\newManucript\python_code_rare\script2\data\simulation_data\test3",
    'T4': r"E:\newManucript\python_code_rare\script2\data\simulation_data\test4"
}

# 指标名称
indicator_names = ['Resource', 'Environment', 'External\nSupport', 'Risk', 'Economy', 'Geology']

# 存储所有数据
all_data = {}
differ_data = {}
comparison_results = {}

# 1. 加载数据并计算差值
print("Loading data and calculating differences...")
for exp_idx, (exp_name, pos) in enumerate(positions.items()):
    # 加载数据
    weight_file = os.path.join(pos, f"group_weight_{exp_idx + 1}.pkl")
    gcd_file = os.path.join(pos, f"GCD_final_name_{exp_idx + 1}.pkl")
    feedback_file = os.path.join(pos, f"feedback_Num_{exp_idx + 1}.pkl")

    with open(feedback_file, 'rb') as f:
        feed_data = pickle.load(f)  # (5, 500, 6)
    # 计算反馈次数

    for jj in range(len(feed_data)):
        feed_nums = 0
        for feed_num in feed_data[jj]:
            feed_nums += feed_num

        print("第", exp_idx + 1, "次实验的反馈均值为", (feed_nums / len(feed_data[jj])) - 1)
        print("__________________________________")

    with open(weight_file, 'rb') as f:
        weights = np.array(pickle.load(f))  # (5, 500, 6)
    with open(gcd_file, 'rb') as f:
        gcds = np.array(pickle.load(f))  # (5, 500)

    all_data[exp_name] = {'weights': weights, 'GCD': gcds}

    # 计算差值
    differ_list = []
    avg_differences = []

    for group_idx in range(5):  # 5个决策群体
        group_differences = []
        for sim_idx in range(500):  # 500次仿真
            weight = weights[group_idx, sim_idx]
            differ = 0.0
            for j in range(6):
                differ += abs(weight[j] - target_weight[j])
            group_differences.append(differ)

        # 排序并取中间460个值
        sorted_differ = sorted(group_differences)
        middle_460 = sorted_differ[20:480]  # 去掉20个最小值，20个最大值       #20:480
        avg_middle = sum(middle_460) / len(middle_460)

        differ_list.append(group_differences)
        avg_differences.append(avg_middle)

    differ_data[exp_name] = {
        'differ_list': np.array(differ_list),  # (5, 500)
        'avg_middle': np.array(avg_differences)  # (5,)
    }

# 2. 计算优势次数（比较四个实验）
print("\nCalculating advantage counts...")
advantage_matrix = np.zeros((4, 4, 5))  # [exp_i, exp_j, group] 实验i优于实验j的次数

for group_idx in range(5):
    for sim_idx in range(500):
        # 获取当前仿真中四个实验的差值
        diffs = []
        for exp_name in ['T1', 'T2', 'T3', 'T4']:
            diffs.append(differ_data[exp_name]['differ_list'][group_idx, sim_idx])

        diffs = np.array(diffs)

        # 两两比较
        for i in range(4):
            for j in range(4):
                if i != j and diffs[i] < diffs[j]:
                    advantage_matrix[i, j, group_idx] += 1

# 3. 二维GCD折线图 - 使用紧凑尺寸
print("\nGenerating GCD 2D line plots...")
for group_idx in range(5):
    # 使用SCI单列宽度，高度适当
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single * 0.7))

    # 实验颜色和样式
    exp_info = [
        ('T1', '#1f77b4', '-'),
        ('T2', '#ff7f0e', '--'),
        ('T3', '#2ca02c', '-.'),
        ('T4', '#d62728', ':')
    ]

    # 为每个实验绘制折线 - 使用更细的线条
    for exp_name, color, linestyle in exp_info:
        gcd_data = all_data[exp_name]['GCD'][group_idx]
        sorted_gcd = np.sort(gcd_data)
        indices = np.arange(1, 501)

        ax.plot(indices, sorted_gcd,
                color=color, linewidth=1.0, linestyle=linestyle,
                label=f'{exp_name} (Avg: {np.mean(gcd_data):.4f})')

    # 设置坐标轴和标签
    ax.set_xlabel('Simulation Index (Sorted by GCD)',
                  fontsize=base_fontsize - 1, labelpad=8)
    ax.set_ylabel('GCD',
                  fontsize=base_fontsize - 1, labelpad=8)

    # 简化刻度
    ax.set_xlim([1, 500])
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_xticklabels(['1', '100', '200', '300', '400', '500'],
                       fontsize=base_fontsize - 2)

    ax.set_ylim([0.9, 1.0])
    ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_yticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'],
                       fontsize=base_fontsize - 2)

    # 添加网格 - 使用更细的网格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # 添加图例 - 使用更小的字体和紧凑布局
    ax.legend(loc='lower right', fontsize=base_fontsize - 2,
              frameon=True, framealpha=0.9)

    # 添加标题
    ax.set_title(f'Group {group_idx + 1} - GCD Distribution',
                 fontsize=base_fontsize, pad=12)

    # 调整布局
    plt.tight_layout()

    # 保存图像 - 使用高DPI
    save_path = fr"E:\newManucript\manuscript2\image\12yue\Fig12_{group_idx + 1}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

# 4. 权重雷达图（简化版，只画均值） - 使用紧凑尺寸
print("\nGenerating simplified weight radar plots...")

# 雷达图角度设置
angles = np.linspace(0, 2 * np.pi, len(indicator_names), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

for group_idx in range(5):
    # 使用紧凑的正方形尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single),
                           subplot_kw=dict(polar=True))

    # 实验信息
    exp_info = [
        ('T1', '#1f77b4'),
        ('T2', '#ff7f0e'),
        ('T3', '#2ca02c'),
        ('T4', '#d62728')
    ]

    # 为每个实验绘制均值线 - 使用更细的线条
    for exp_name, color in exp_info:
        weights = all_data[exp_name]['weights'][group_idx]  # (500, 6)
        mean_weights = np.mean(weights, axis=0)

        # 闭合数据
        mean_plot = np.concatenate((mean_weights, [mean_weights[0]]))

        # 绘制均值线
        ax.plot(angles, mean_plot, color=color, linewidth=0.8,
                label=exp_name, marker='o', markersize=3)

    # 绘制目标权重线
    target_plot = np.concatenate((target_weight, [target_weight[0]]))
    ax.plot(angles, target_plot, color='black', linewidth=0.8,
            linestyle='--', label='Target', alpha=0.7)

    # 设置雷达图参数
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicator_names, fontsize=base_fontsize - 1)

    # 增加标签间距
    ax.tick_params(axis='x', pad=15)

    # 设置y轴范围
    ax.set_ylim([0, 0.35])
    ax.set_yticks([0.1, 0.2, 0.3])
    ax.set_yticklabels(['0.1', '0.2', '0.3'], fontsize=base_fontsize - 2)

    # 添加网格 - 使用更细的网格线
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

    # 添加图例 - 使用更小的字体
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
              fontsize=base_fontsize - 2, frameon=True)

    # 添加标题
    ax.set_title(f'Group {group_idx + 1} - Average Weight Distribution',
                 fontsize=base_fontsize, pad=20)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    save_path = fr"E:\newManucript\manuscript2\image\12yue\Fig13_{group_idx + 1}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

# 5. 权重差值折线图 - 使用紧凑尺寸
print("\nGenerating weight difference line plots...")

# 存储排序后的差值数据
sorted_diffs_all = {}

for group_idx in range(5):
    # 使用SCI单列宽度
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single * 0.7))

    # 实验信息
    exp_info = [
        ('T1', '#1f77b4', '-'),
        ('T2', '#ff7f0e', '--'),
        ('T3', '#2ca02c', '-.'),
        ('T4', '#d62728', ':')
    ]

    # 为每个实验绘制排序后的差值曲线 - 使用更细的线条
    for exp_name, color, linestyle in exp_info:
        diffs = differ_data[exp_name]['differ_list'][group_idx]
        sorted_diffs = np.sort(diffs)
        sorted_diffs_all[f'G{group_idx + 1}_{exp_name}'] = sorted_diffs

        # 计算中间460个值的平均值
        avg_middle = differ_data[exp_name]['avg_middle'][group_idx]

        # 绘制曲线
        indices = np.arange(1, 501)
        ax.plot(indices, sorted_diffs,
                color=color, linewidth=0.8, linestyle=linestyle,
                label=f'{exp_name} (Avg: {avg_middle:.4f})')

    # 设置坐标轴和标签
    ax.set_xlabel('Simulation Index (Sorted by Difference)',
                  fontsize=base_fontsize - 1, labelpad=8)
    ax.set_ylabel('Weight Difference from Target',
                  fontsize=base_fontsize - 1, labelpad=8)

    # 简化刻度
    ax.set_xlim([1, 500])
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_xticklabels(['1', '100', '200', '300', '400', '500'],
                       fontsize=base_fontsize - 2)

    # 自动设置y轴范围
    y_min = min([differ_data[exp]['differ_list'][group_idx].min() for exp in ['T1', 'T2', 'T3', 'T4']])
    y_max = max([differ_data[exp]['differ_list'][group_idx].max() for exp in ['T1', 'T2', 'T3', 'T4']])
    ax.set_ylim([y_min - 0.01, y_max + 0.01])
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    ax.set_yticklabels([f'{val:.3f}' for val in np.linspace(y_min, y_max, 5)],
                       fontsize=base_fontsize - 2)

    # 添加网格 - 使用更细的网格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # 添加图例 - 使用更小的字体
    ax.legend(loc='upper left', fontsize=base_fontsize - 2.5,
              frameon=True, framealpha=0.9)

    # 添加标题
    ax.set_title(f'Group {group_idx + 1} - Weight Difference from Target',
                 fontsize=base_fontsize, pad=12)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    save_path = fr"E:\newManucript\manuscript2\image\12yue\Fig14_{group_idx + 1}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

# 6. 输出结果
print("\n" + "=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)

# 6.1 平均差值（中间460个值）
print("\n1. AVERAGE DIFFERENCES (MIDDLE 460 VALUES)")
print("=" * 60)
print("Group |    T1     |    T2     |    T3     |    T4     |   Best   |")
print("-" * 60)

for group_idx in range(5):
    diffs = []
    best_exp = ''
    best_value = float('inf')

    for exp_idx, exp_name in enumerate(['T1', 'T2', 'T3', 'T4']):
        avg_val = differ_data[exp_name]['avg_middle'][group_idx]
        diffs.append(avg_val)
        if avg_val < best_value:
            best_value = avg_val
            best_exp = exp_name

    print(f"  {group_idx + 1}   | ", end="")
    for val in diffs:
        print(f"{val:.6f} | ", end="")
    print(f" {best_exp} ({best_value:.6f})")

# 6.2 优势次数矩阵
print("\n\n2. ADVANTAGE MATRIX (Experiment i vs Experiment j)")
print("=" * 80)
print("Note: Values represent times experiment i is better than experiment j")
print("-" * 80)

exp_names = ['T1', 'T2', 'T3', 'T4']
for i in range(4):
    print(f"\n{exp_names[i]} vs Others:")
    for j in range(4):
        if i != j:
            total_advantage = np.sum(advantage_matrix[i, j, :])
            avg_advantage = np.mean(advantage_matrix[i, j, :])
            print(f"  {exp_names[i]} > {exp_names[j]}: {int(total_advantage)} times "
                  f"(avg {avg_advantage:.1f} per group)")

    # 计算综合优势
    total_superiority = np.sum(advantage_matrix[i, :, :]) - np.sum(advantage_matrix[:, i, :])
    print(f"  Net advantage: {int(total_superiority)} times")

# 6.3 整体性能排名
print("\n\n3. OVERALL PERFORMANCE RANKING")
print("=" * 60)

# 计算每个实验的综合评分
scores = []
for exp_name in ['T1', 'T2', 'T3', 'T4']:
    # 1. 平均差值越小越好
    avg_diff = np.mean(differ_data[exp_name]['avg_middle'])

    # 2. 优势次数越多越好
    exp_idx = ['T1', 'T2', 'T3', 'T4'].index(exp_name)
    total_advantage = np.sum(advantage_matrix[exp_idx, :, :]) - np.sum(advantage_matrix[:, exp_idx, :])

    # 3. 综合评分（差值权重0.7，优势次数权重0.3）
    normalized_diff = 1 - (avg_diff - 0.02) / 0.02  # 假设差值范围在0.02-0.04之间
    normalized_advantage = (total_advantage + 2500) / 5000  # 归一化到0-1

    final_score = 0.7 * normalized_diff + 0.3 * normalized_advantage

    scores.append((exp_name, avg_diff, total_advantage, final_score))

# 按综合评分排序
scores.sort(key=lambda x: x[3], reverse=True)

print("Rank | Experiment | Avg Difference | Net Advantage | Final Score")
print("-" * 60)
for rank, (exp_name, avg_diff, advantage, score) in enumerate(scores, 1):
    print(f" {rank:2}  |     {exp_name}     |   {avg_diff:.6f}   |   {int(advantage):6}   |   {score:.4f}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETED!")
print("=" * 80)