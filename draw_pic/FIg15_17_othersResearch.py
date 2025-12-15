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

# 定义路径
position1 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\test1"
position3 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\test2"
position4 = r"E:\newManucript\python_code_rare\script2\data\compare_bayes"
position5 = r"E:\newManucript\python_code_rare\script2\data\compare_consensus"
position6 = r"E:\newManucript\python_code_rare\script2\data\compare_personalTrust"

# 1. 加载T1和T2的权重数据（只取第一组）
print("Loading T1 and T2 weight data...")
original_file_1_1 = os.path.join(position1, f"group_weight_1.pkl")
with open(original_file_1_1, 'rb') as f:
    original_data_1_1 = pickle.load(f)  # (5, 500, 6)
t1_weights = original_data_1_1[0]  # 取第一组决策群，形状(500, 6)

original_file_1_3 = os.path.join(position3, f"group_weight_2.pkl")
with open(original_file_1_3, 'rb') as f:
    original_data_1_3 = pickle.load(f)  # (5, 500, 6)
t3_weights = original_data_1_3[0]  # 取第一组决策群，形状(500, 6)

# 2. 加载三篇文献的数据
print("Loading literature data...")

# Dai's文章数据 (T1和T2)
priginal_file = os.path.join(position4, f"batch_results.pkl")
with open(priginal_file, 'rb') as f:
    priginal_data = pickle.load(f)  # Dai's T1

priginal_file_random = os.path.join(position4, f"batch_results_random.pkl")
with open(priginal_file_random, 'rb') as f:
    priginal_data_random = pickle.load(f)  # Dai's T2

# Xin's文章数据 (T1和T2)
priginal_file2 = os.path.join(position6, f"all_simulation_results.pkl")
with open(priginal_file2, 'rb') as f:
    priginal_data2 = pickle.load(f)  # Xin's T1

priginal_file2_random = os.path.join(position6, f"all_simulation_results_random.pkl")
with open(priginal_file2_random, 'rb') as f:
    priginal_data2_random = pickle.load(f)  # Xin's T2

# Guo's文章数据 (T1)
priginal_file3 = os.path.join(position5, f"consensus_results.pkl")
with open(priginal_file3, 'rb') as f:
    priginal_data3 = pickle.load(f)  # Guo's T1

# 3. 提取权重数据
print("Extracting weight data...")

# T1条件下各方法的权重
dai_t1_weights = np.array([priginal_data[i]['final_collective_evaluation']
                           for i in range(500)])  # (500, 6)

xin_t1_weights = np.array([priginal_data2['all_results'][i]['final_weights_group']
                           for i in range(500)])  # (500, 6)

guo_t1_weights = np.array([priginal_data3[i]['alternative_weights']
                           for i in range(500)])  # (500, 6)

# T2条件下各方法的权重
dai_t3_weights = np.array([priginal_data_random[i]['final_collective_evaluation']
                           for i in range(500)])  # (500, 6)

xin_t3_weights = np.array([priginal_data2_random['all_results'][i]['final_weights_group']
                           for i in range(500)])  # (500, 6)

# 4. 提取反馈次数数据
print("Extracting feedback iteration data...")

# T1条件下的反馈次数
dai_t1_feedback = np.array([priginal_data[i]['crp_iterations']
                            for i in range(500)])  # (500,)

xin_t1_feedback = np.array([priginal_data2['summary_data'][i]['feedback_iterations']
                            for i in range(500)])  # (500,)

guo_t1_feedback = np.array([priginal_data3[i]['feedback_iterations']
                            for i in range(500)])  # (500,)

t1_feedback = np.zeros(500)  # 全0数组

# T2条件下的反馈次数
dai_t3_feedback = np.array([priginal_data_random[i]['crp_iterations']
                            for i in range(500)])  # (500,)

xin_t3_feedback = np.array([priginal_data2_random['summary_data'][i]['feedback_iterations']
                            for i in range(500)])  # (500,)

t3_feedback = np.zeros(500)  # 全0数组


# 5. 绘制雷达图
def create_radar_chart(weights_list, labels, colors, title, save_path, target_weight=None):
    """
    创建雷达图
    """
    # 指标名称
    indicator_names = ['Resource', 'Environment', 'External\nSupport', 'Risk', 'Economy', 'Geology']

    # 计算平均权重
    mean_weights = []
    for weights in weights_list:
        if len(weights.shape) == 2:
            mean_weights.append(np.mean(weights, axis=0))
        else:
            mean_weights.append(weights)

    # 雷达图角度
    angles = np.linspace(0, 2 * np.pi, len(indicator_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 创建图形 - 使用紧凑尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single),
                           subplot_kw=dict(polar=True))

    # 绘制每个方法的均值线 - 使用更细的线条
    for idx, (mean_w, label, color) in enumerate(zip(mean_weights, labels, colors)):
        plot_data = np.concatenate((mean_w, [mean_w[0]]))
        ax.plot(angles, plot_data, color=color, linewidth=1.2,
                label=label, marker='o', markersize=4)

    # 绘制目标权重线（如果有）
    if target_weight is not None:
        target_plot = np.concatenate((target_weight, [target_weight[0]]))
        ax.plot(angles, target_plot, color='black', linewidth=1.0,
                linestyle='--', label='Target', alpha=0.8)

    # 设置雷达图参数
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicator_names, fontsize=base_fontsize - 2)
    ax.set_ylim([0, 0.35])
    ax.set_yticks([0.1, 0.2, 0.3])
    ax.set_yticklabels(['0.1', '0.2', '0.3'], fontsize=base_fontsize - 2)

    # 增加标签间距
    ax.tick_params(axis='x', pad=10)

    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

    # 添加图例 - 使用更小的字体
    ax.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.17),
              fontsize=base_fontsize - 3, frameon=True)

    # 添加标题
    #ax.set_title(title, fontsize=base_fontsize, pad=20)

    # 保存图形 - 使用高DPI
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# 绘制T1条件下的雷达图
print("\nCreating T1 radar chart...")
t1_radar_weights = [np.mean(t1_weights, axis=0),
                    np.mean(dai_t1_weights, axis=0),
                    np.mean(xin_t1_weights, axis=0),
                    np.mean(guo_t1_weights, axis=0)]
t1_labels = ['T1 (Our Model)', "Dai's Method", "Xing's Method", "Guo's Method"]
t1_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
save_path_t1_radar = r"E:\newManucript\manuscript2\image\12yue\Fig15_T1.png"
create_radar_chart(t1_radar_weights, t1_labels, t1_colors,
                   'Average Weight Distribution (T1 Condition)', save_path_t1_radar, target_weight)

# 绘制T2条件下的雷达图
print("Creating T2 radar chart...")
t3_radar_weights = [np.mean(t3_weights, axis=0),
                    np.mean(dai_t3_weights, axis=0),
                    np.mean(xin_t3_weights, axis=0)]
t3_labels = ['T2 (Our Model)', "Dai's Method", "Xing's Method"]
t3_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
save_path_t3_radar = r"E:\newManucript\manuscript2\image\12yue\Fig15_T2.png"
create_radar_chart(t3_radar_weights, t3_labels, t3_colors,
                   'Average Weight Distribution (T2 Condition)', save_path_t3_radar, target_weight)


# 6. 计算差值并绘制折线图
def calculate_differences(weights, target_weight):
    """
    计算权重与目标权重的差值
    """
    differences = []
    for weight in weights:
        diff = 0.0
        for j in range(len(weight)):
            diff += abs(weight[j] - target_weight[j])
        differences.append(diff)
    return np.array(differences)


def calculate_trimmed_mean(differences, trim_percent=0.04):
    """
    计算去除极端值后的均值（去除trim_percent比例的最大最小值）
    """
    sorted_diffs = np.sort(differences)
    n = len(sorted_diffs)
    trim_count = int(n * trim_percent)
    trimmed_diffs = sorted_diffs[trim_count:-trim_count]
    return np.mean(trimmed_diffs), trimmed_diffs


def plot_difference_comparison(diff_dict, labels, colors, title, save_path):
    """
    绘制差值对比折线图
    """
    # 使用紧凑尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single * 0.7))

    # 绘制每个方法的差值曲线
    for idx, (method, diffs) in enumerate(diff_dict.items()):
        # 排序差值
        sorted_diffs = np.sort(diffs)
        x_values = np.arange(1, len(sorted_diffs) + 1)

        # 绘制曲线 - 使用更细的线条
        ax.plot(x_values, sorted_diffs, color=colors[idx], linewidth=1.0,
                label=labels[idx], alpha=0.8)

        # 计算修剪均值并绘制水平线
        trimmed_mean, trimmed_diffs = calculate_trimmed_mean(diffs[20:480])
        ax.axhline(y=trimmed_mean, color=colors[idx], linestyle='--',
                   linewidth=0.7, alpha=0.7,
                   label=f'{labels[idx]} mean: {trimmed_mean:.4f}')

    # 设置坐标轴和标签
    ax.set_xlabel('Simulation Index (Sorted by Difference)',
                  fontsize=base_fontsize - 1, labelpad=8)
    ax.set_ylabel('D_target',
                  fontsize=base_fontsize - 1, labelpad=8)

    # 简化刻度
    ax.set_xlim([1, 500])
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_xticklabels(['1', '100', '200', '300', '400', '500'],
                       fontsize=base_fontsize - 2)

    # 自动设置y轴范围
    all_diffs = np.concatenate(list(diff_dict.values()))
    y_min, y_max = np.min(all_diffs), np.max(all_diffs)
    ax.set_ylim([y_min - 0.01, y_max + 0.01])

    # 设置网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 2)

    # 添加图例
    ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.4, 1.0), fontsize=base_fontsize - 4,
              frameon=True, framealpha=0.9)

    # 添加标题
    ax.set_title(title, fontsize=base_fontsize, pad=12)

    # 调整布局
    plt.tight_layout()

    # 保存图形 - 使用高DPI
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# 计算T1条件下的差值
print("\nCalculating T1 differences...")
t1_diffs = calculate_differences(t1_weights, target_weight)
dai_t1_diffs = calculate_differences(dai_t1_weights, target_weight)
xin_t1_diffs = calculate_differences(xin_t1_weights, target_weight)
guo_t1_diffs = calculate_differences(guo_t1_weights, target_weight)

# 绘制T1差值对比图
t1_diff_dict = {
    'T1': t1_diffs,
    'Dai_T1': dai_t1_diffs,
    'Xin_T1': xin_t1_diffs,
    'Guo_T1': guo_t1_diffs
}
t1_diff_labels = ['T1 (Our Model)', "Dai's Method", "Xing's Method", "Guo's Method"]
t1_diff_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
save_path_t1_diff = r"E:\newManucript\manuscript2\image\12yue\Fig16_T1.png"
plot_difference_comparison(t1_diff_dict, t1_diff_labels, t1_diff_colors,
                           'T1 Condition', save_path_t1_diff)

# 计算T2条件下的差值
print("Calculating T2 differences...")
t3_diffs = calculate_differences(t3_weights, target_weight)
dai_t3_diffs = calculate_differences(dai_t3_weights, target_weight)
xin_t3_diffs = calculate_differences(xin_t3_weights, target_weight)

# 绘制T2差值对比图
t3_diff_dict = {
    'T2': t3_diffs,
    'Dai_T2': dai_t3_diffs,
    'Xin_T2': xin_t3_diffs
}
t3_diff_labels = ['T2 (Our Model)', "Dai's Method", "Xing's Method"]
t3_diff_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
save_path_t3_diff = r"E:\newManucript\manuscript2\image\12yue\Fig16_T2.png"
plot_difference_comparison(t3_diff_dict, t3_diff_labels, t3_diff_colors,
                           'T2 Condition', save_path_t3_diff)


# 7. 绘制反馈次数折线图
def plot_feedback_comparison(feedback_data_dict, title, save_path):
    """
    绘制反馈次数对比折线图
    显示各组数据的最大值和均值
    """
    # 使用紧凑尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single * 0.7))

    # 将数据转换为numpy数组并确保长度一致
    methods = list(feedback_data_dict.keys())
    n_methods = len(methods)
    n_simulations = 500

    # 准备数据矩阵
    feedback_matrix = np.zeros((n_methods, n_simulations))

    for i, (method, feedbacks) in enumerate(feedback_data_dict.items()):
        # 确保反馈数据是numpy数组
        if isinstance(feedbacks, (int, float)):
            feedbacks = np.full(n_simulations, feedbacks)
        elif len(feedbacks) != n_simulations:
            feedbacks = np.full(n_simulations, 0)

        # 将趋近于0但不是0的值设置为0.1
        # 这里假设任何大于0小于0.5的值都视为趋近于0
        feedbacks = np.where((feedbacks > 0) & (feedbacks < 0.5), 0.1, feedbacks)

        feedback_matrix[i] = feedbacks

    # 计算每次仿真的最大值和均值
    max_values = np.max(feedback_matrix, axis=0)
    mean_values = np.mean(feedback_matrix, axis=0)

    # 对最大值和均值进行排序
    sorted_max_values = np.sort(max_values)
    sorted_mean_values = np.sort(mean_values)

    # 创建x轴
    x_values = np.arange(1, n_simulations + 1)

    # # 绘制最大值曲线 - 使用更细的线条
    # ax.plot(x_values, sorted_max_values, color='red', linewidth=1.2,
    #         label='Maximum of All Methods', alpha=0.8)
    #
    # # 绘制均值曲线 - 使用更细的线条
    # ax.plot(x_values, sorted_mean_values, color='blue', linewidth=1.2,
    #         label='Mean of All Methods', alpha=0.8)

    # 设置坐标轴和标签
    ax.set_xlabel('Simulation Index (Sorted by Feedback Iterations)',
                  fontsize=base_fontsize - 1, labelpad=8)
    ax.set_ylabel('Feedback Iterations', fontsize=base_fontsize - 1, labelpad=8)

    # 简化刻度
    ax.set_xlim([1, n_simulations])
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_xticklabels(['1', '100', '200', '300', '400', '500'],
                       fontsize=base_fontsize - 2)

    # 自动设置y轴范围
    y_min = min(np.min(sorted_max_values), np.min(sorted_mean_values))
    y_max = max(np.max(sorted_max_values), np.max(sorted_mean_values))

    # 确保y轴从0开始，如果不是0的话留一些边距
    if y_min > 0:
        y_min = 0
    if y_max == 0:
        y_max = 1  # 如果所有值都是0，设置最大值为1以便显示

    ax.set_ylim([y_min, y_max * 1.1])  # 留10%边距

    # 设置网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 2)

    # 添加图例
    ax.legend(loc='upper right', fontsize=base_fontsize - 2,
              frameon=True, framealpha=0.9)

    # 添加标题
    ax.set_title(title, fontsize=base_fontsize, pad=12)

    # 添加统计信息标注
    stats_text = f'Max: {np.max(max_values):.1f}, Mean: {np.mean(mean_values):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=base_fontsize - 2, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 调整布局
    plt.tight_layout()

    # 保存图形 - 使用高DPI
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

    # 返回统计信息
    return {
        'max_of_max': np.max(max_values),
        'mean_of_means': np.mean(mean_values),
        'max_values_distribution': sorted_max_values,
        'mean_values_distribution': sorted_mean_values
    }


def plot_feedback_comparison_with_methods(feedback_data_dict, labels, colors, title, save_path):
    """
    绘制反馈次数对比折线图（包含各方法单独曲线）
    """
    # 使用紧凑尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single * 0.7))

    # 将数据转换为numpy数组并确保长度一致
    methods = list(feedback_data_dict.keys())
    n_methods = len(methods)
    n_simulations = 500

    # 准备数据矩阵并绘制各方法曲线
    sorted_feedbacks = {}
    for i, (method, feedbacks) in enumerate(feedback_data_dict.items()):
        # 确保反馈数据是numpy数组
        if isinstance(feedbacks, (int, float)):
            feedbacks = np.full(n_simulations, feedbacks)
        elif len(feedbacks) != n_simulations:
            feedbacks = np.full(n_simulations, 0)

        # 将趋近于0但不是0的值设置为0.1
        feedbacks = np.where((feedbacks > 0) & (feedbacks < 0.5), 0.1, feedbacks)

        # 排序并存储
        sorted_feedbacks[method] = np.sort(feedbacks)

        # 绘制该方法曲线 - 使用更细的线条
        x_values = np.arange(1, n_simulations + 1)
        ax.plot(x_values, sorted_feedbacks[method],
                color=colors[i], linewidth=1.0,
                label=f'{labels[i]} (max={np.max(feedbacks):.1f}, mean={np.mean(feedbacks):.2f})',
                alpha=0.7)

    # 计算并绘制最大值和均值曲线
    feedback_matrix = np.array([sorted_feedbacks[method] for method in methods])
    max_values = np.max(feedback_matrix, axis=0)
    mean_values = np.mean(feedback_matrix, axis=0)

    # # 绘制最大值曲线 - 使用更细的线条
    # ax.plot(x_values, max_values, color='black', linewidth=1.0,
    #         label=f'Maximum (max={np.max(max_values):.1f})',
    #         linestyle='-', alpha=0.9)
    #
    # # 绘制均值曲线 - 使用更细的线条
    # ax.plot(x_values, mean_values, color='purple', linewidth=1.0,
    #         label=f'Mean (mean={np.mean(mean_values):.2f})',
    #         linestyle='--', alpha=0.9)

    # 设置坐标轴和标签
    ax.set_xlabel('Simulation Index (Sorted by Feedback Iterations)',
                  fontsize=base_fontsize - 1, labelpad=8)
    ax.set_ylabel('Feedback Iterations', fontsize=base_fontsize - 1, labelpad=8)

    # 简化刻度
    ax.set_xlim([1, n_simulations])
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_xticklabels(['1', '100', '200', '300', '400', '500'],
                       fontsize=base_fontsize - 2)

    # 自动设置y轴范围
    all_values = np.concatenate(list(sorted_feedbacks.values()) + [max_values, mean_values])
    y_min, y_max = np.min(all_values), np.max(all_values)

    if y_min > 0:
        y_min = 0
    if y_max == 0:
        y_max = 1

    ax.set_ylim([y_min, y_max * 1.1])

    # 设置网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 2)

    # 添加图例
    ax.legend(loc='upper left', fontsize=base_fontsize - 3.5,
              frameon=True, framealpha=0.9)

    # 添加标题
    ax.set_title(title, fontsize=base_fontsize, pad=12)

    # 调整布局
    plt.tight_layout()

    # 保存图形 - 使用高DPI
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# T1条件下的反馈次数对比
print("\nCreating T1 feedback comparison plot...")
t1_feedback_dict = {
    'T1': t1_feedback,
    'Dai_T1': dai_t1_feedback,
    'Xin_T1': xin_t1_feedback,
    'Guo_T1': guo_t1_feedback
}

# 版本1：只显示最大值和均值
# save_path_t1_feedback = r"E:\newManucript\manuscript2\image\12yue\Fig17_T1.png"
# t1_stats = plot_feedback_comparison(t1_feedback_dict,
#                                     'Feedback Iterations Comparison (T1 Condition)',
#                                     save_path_t1_feedback)

# 版本2：显示所有方法曲线 + 最大值和均值曲线
t1_labels = ['T1 (Our Model)', "Dai's Method", "Xing's Method", "Guo's Method"]
t1_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
save_path_t1_feedback_detailed = r"E:\newManucript\manuscript2\image\12yue\Fig17_T1_detailed.png"
plot_feedback_comparison_with_methods(t1_feedback_dict, t1_labels, t1_colors,
                                      'T1 Condition',
                                      save_path_t1_feedback_detailed)

# T2条件下的反馈次数对比
print("Creating T2 feedback comparison plot...")
t3_feedback_dict = {
    'T2': t3_feedback,
    'Dai_T2': dai_t3_feedback,
    'Xin_T2': xin_t3_feedback
}

# 版本1：只显示最大值和均值
# save_path_t3_feedback = r"E:\newManucript\manuscript2\image\12yue\Fig17_T2.png"
# t3_stats = plot_feedback_comparison(t3_feedback_dict,
#                                     'Feedback Iterations Comparison (T2 Condition)',
#                                     save_path_t3_feedback)

# 版本2：显示所有方法曲线 + 最大值和均值曲线
t3_labels = ['T2 (Our Model)', "Dai's Method", "Xing's Method"]
t3_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
save_path_t3_feedback_detailed = r"E:\newManucript\manuscript2\image\12yue\Fig17_T2_detailed.png"
plot_feedback_comparison_with_methods(t3_feedback_dict, t3_labels, t3_colors,
                                      'T2 Condition',
                                      save_path_t3_feedback_detailed)

# 8. 输出统计结果
print("\n" + "=" * 80)
print("STATISTICAL RESULTS")
print("=" * 80)

# T1条件下的修剪均值
print("\nT1 Condition - Trimmed Means (removing 4% extremes):")
print("-" * 60)
t1_trimmed_mean, _ = calculate_trimmed_mean(t1_diffs)
dai_t1_trimmed_mean, _ = calculate_trimmed_mean(dai_t1_diffs)
xin_t1_trimmed_mean, _ = calculate_trimmed_mean(xin_t1_diffs)
guo_t1_trimmed_mean, _ = calculate_trimmed_mean(guo_t1_diffs)

print(f"T1 (Our Model): {t1_trimmed_mean:.6f}")
print(f"Dai's Method: {dai_t1_trimmed_mean:.6f}")
print(f"Xing's Method: {xin_t1_trimmed_mean:.6f}")
print(f"Guo's Method: {guo_t1_trimmed_mean:.6f}")

# T2条件下的修剪均值
print("\nT2 Condition - Trimmed Means (removing 4% extremes):")
print("-" * 60)
t3_trimmed_mean, _ = calculate_trimmed_mean(t3_diffs)
dai_t3_trimmed_mean, _ = calculate_trimmed_mean(dai_t3_diffs)
xin_t3_trimmed_mean, _ = calculate_trimmed_mean(xin_t3_diffs)

print(f"T2 (Our Model): {t3_trimmed_mean:.6f}")
print(f"Dai's Method: {dai_t3_trimmed_mean:.6f}")
print(f"Xing's Method: {xin_t3_trimmed_mean:.6f}")

# 反馈次数统计
print("\nFeedback Iterations Statistics:")
print("-" * 60)

# T1反馈次数
print("\nT1 Condition:")
print(f"  T1 (Our Model): mean={np.mean(t1_feedback):.2f}, min={np.min(t1_feedback)}, max={np.max(t1_feedback)}")
print(
    f"  Dai's Method: mean={np.mean(dai_t1_feedback):.2f}, min={np.min(dai_t1_feedback)}, max={np.max(dai_t1_feedback)}")
print(
    f"  Xing's Method: mean={np.mean(xin_t1_feedback):.2f}, min={np.min(xin_t1_feedback)}, max={np.max(xin_t1_feedback)}")
print(
    f"  Guo's Method: mean={np.mean(guo_t1_feedback):.2f}, min={np.min(guo_t1_feedback)}, max={np.max(guo_t1_feedback)}")

# T2反馈次数
print("\nT2 Condition:")
print(f"  T2 (Our Model): mean={np.mean(t3_feedback):.2f}, min={np.min(t3_feedback)}, max={np.max(t3_feedback)}")
print(
    f"  Dai's Method: mean={np.mean(dai_t3_feedback):.2f}, min={np.min(dai_t3_feedback)}, max={np.max(dai_t3_feedback)}")
print(
    f"  Xing's Method: mean={np.mean(xin_t3_feedback):.2f}, min={np.min(xin_t3_feedback)}, max={np.max(xin_t3_feedback)}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETED!")
print("=" * 80)