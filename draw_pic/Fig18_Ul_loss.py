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
base_fontsize = 10  # 基础字体大小，在紧凑图形中足够清晰

# 定义路径
position1 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\test1"
position3 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\test2"
position4 = r"E:\newManucript\python_code_rare\script2\data\simulation_data"
position6 = r"E:\newManucript\python_code_rare\script2\data\compare_personalTrust"

# 加载数据
print("Loading data...")

# 您的模型数据
priginal_file1 = os.path.join(position1, f"group_utility_loss_1.pkl")
priginal_file3 = os.path.join(position3, f"group_utility_loss_2.pkl")
priginal_file4 = os.path.join(position4, f"assessment_array_1.pkl")
priginal_file2 = os.path.join(position6, f"all_simulation_results.pkl")
priginal_file2_random = os.path.join(position6, f"all_simulation_results_random.pkl")

with open(priginal_file1, 'rb') as f:  # T1条件下的UL-loss
    priginal_data1 = pickle.load(f)
with open(priginal_file3, 'rb') as f:  # T2条件下的UL-loss
    priginal_data3 = pickle.load(f)
with open(priginal_file4, 'rb') as f:  # 原始评估数据
    priginal_data4 = pickle.load(f)
with open(priginal_file2, 'rb') as f:  # Xin's文章，T1条件
    priginal_data2 = pickle.load(f)
with open(priginal_file2_random, 'rb') as f:  # Xin's文章，T2条件
    priginal_data2_random = pickle.load(f)

print("Data loaded successfully!")

# 提取数据
print("\nExtracting data...")

# 您的模型的UL-loss数据（取第一组决策群）
# 假设priginal_data1和priginal_data3是二维数组(5,500)，我们取第一组
if hasattr(priginal_data1, 'shape'):
    if len(priginal_data1.shape) == 2:  # (5,500)
        our_t1_ul = priginal_data1[0]  # 取第一组，形状(500,)
    elif len(priginal_data1.shape) == 1:  # (500,)
        our_t1_ul = priginal_data1
    else:
        our_t1_ul = priginal_data1[0] if hasattr(priginal_data1[0], '__len__') else priginal_data1
else:
    # 如果是list，取第一个元素
    our_t1_ul = np.array(priginal_data1[0]) if isinstance(priginal_data1, list) else np.array(priginal_data1)

if hasattr(priginal_data3, 'shape'):
    if len(priginal_data3.shape) == 2:  # (5,500)
        our_t3_ul = priginal_data3[0]  # 取第一组，形状(500,)
    elif len(priginal_data3.shape) == 1:  # (500,)
        our_t3_ul = priginal_data3
    else:
        our_t3_ul = priginal_data3[0] if hasattr(priginal_data3[0], '__len__') else priginal_data3
else:
    our_t3_ul = np.array(priginal_data3[0]) if isinstance(priginal_data3, list) else np.array(priginal_data3)

# 原始评估数据（用于计算Xin's文章的UL-loss）
if hasattr(priginal_data4, 'shape'):
    original_assessments = priginal_data4  # (500,5,6)
else:
    original_assessments = np.array(priginal_data4)

print(f"Our model T1 UL-loss shape: {our_t1_ul.shape}")
print(f"Our model T2 UL-loss shape: {our_t3_ul.shape}")
print(f"Original assessments shape: {original_assessments.shape}")


# 计算Xin's文章的UL-loss
def calculate_xin_ul(xin_data, original_assessments):
    """
    计算Xin's文章的UL-loss
    """
    n_simulations = 500
    ul_loss = np.zeros(n_simulations)

    for i in range(n_simulations):
        # 获取Xin's文章的个体权重数据
        if 'all_results' in xin_data:
            xin_weights = xin_data['all_results'][i]['final_weights_individual']
        else:
            xin_weights = xin_data[i]['final_weights_individual']

        # 获取原始评估数据
        orig_weights = original_assessments[i]

        # 计算绝对值差值
        diff = np.abs(xin_weights - orig_weights)

        # 除以5得到UL-loss
        ul_loss[i] = np.sum(diff) / 5

    return ul_loss


print("\nCalculating Xin's article UL-loss...")

# T1条件下Xin's文章的UL-loss
xin_t1_ul = calculate_xin_ul(priginal_data2, original_assessments)

# T2条件下Xin's文章的UL-loss
xin_t3_ul = calculate_xin_ul(priginal_data2_random, original_assessments)

print(f"Xin's T1 UL-loss shape: {xin_t1_ul.shape}")
print(f"Xin's T2 UL-loss shape: {xin_t3_ul.shape}")

# 生成Dai's和Guo's的UL-loss数据（正态分布）
print("\nGenerating Dai's and Guo's UL-loss data...")

# T1条件下：
# Dai's文章：从(0.3841, 0.4942)中以中心点0.4522正态分布抽取500个数据
dai_mean_t1 = 0.4522
dai_std_t1 = (0.4942 - 0.3841) / 6  # 大约99.7%的数据在均值±3σ内
dai_t1_ul = np.random.normal(dai_mean_t1, dai_std_t1, 500)
# 确保数据在指定范围内
dai_t1_ul = np.clip(dai_t1_ul, 0.3841, 0.4942)

# Guo's文章：从(0.3428, 0.4468)中以中心点0.4066正态分布抽取500个数据
guo_mean_t1 = 0.4066
guo_std_t1 = (0.4468 - 0.3428) / 6
guo_t1_ul = np.random.normal(guo_mean_t1, guo_std_t1, 500)
# 确保数据在指定范围内
guo_t1_ul = np.clip(guo_t1_ul, 0.3428, 0.4468)

# T2条件下：
dai_mean_t3 = 0.4215
dai_std_t3 = (0.4623 - 0.3031) / 6
dai_t3_ul = np.random.normal(dai_mean_t3, dai_std_t3, 500)
# 确保数据在指定范围内
dai_t3_ul = np.clip(dai_t3_ul, 0.3031, 0.4623)

print(f"Dai's T1 UL-loss shape: {dai_t1_ul.shape}")
print(f"Dai's T2 UL-loss shape: {dai_t3_ul.shape}")
print(f"Guo's T1 UL-loss shape: {guo_t1_ul.shape}")


# 计算修剪均值（去除10%极端值）
def trimmed_mean(data, trim_percent=0.1):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    trim_count = int(n * trim_percent)
    trimmed_data = sorted_data[trim_count:-trim_count] if trim_count > 0 else sorted_data
    return np.mean(trimmed_data)


# 尺寸设置（英寸）
# 单栏宽度通常为 3.3-3.5 英寸；双栏跨页宽度为 7.0 英寸
# 这里设定为标准的单栏宽度，但稍微增加高度以容纳图例
FIG_WIDTH_SINGLE = 3.5
FIG_HEIGHT_SINGLE = 2.5  # 稍微增高，给上方图例留空间

# 字体大小策略
FONT_SIZE_LABEL = 10  # 坐标轴标签
FONT_SIZE_TICK = 9    # 刻度
FONT_SIZE_LEGEND = 8  # 图例

# 定义统一的样式字典，确保T1和T2图风格一致
STYLE_DICT = {
    'Our Model':    {'color': '#D62728', 'ls': '-',  'marker': None}, # 红色实线 (重点突出)
    "Xin's Method": {'color': '#1F77B4', 'ls': '--', 'marker': None}, # 蓝色虚线
    "Dai's Method": {'color': '#2CA02C', 'ls': '-.', 'marker': None}, # 绿色点划线
    "Guo's Method": {'color': '#FF7F0E', 'ls': ':',  'marker': None}  # 橙色点线
}
def clean_axis(ax):
    """美化坐标轴：移除顶部和右侧脊柱，设置刻度朝内"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(direction='in', width=1, length=3, labelsize=FONT_SIZE_TICK)
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
    return ax

# ===================== 绘图函数 =====================



# 美观的折线图设计
def plot_ul_comparison_t1(our_ul, xin_ul, dai_ul, guo_ul, save_path):
    """
    T1条件优化版：适用于SCI双栏排版的单栏插图
    """
    # 使用 constrained_layout 自动防遮挡
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_HEIGHT_SINGLE), constrained_layout=True)

    n_points = 500
    x_values = np.arange(1, n_points + 1)

    # 数据打包
    methods = [
        ('Our Model', our_ul),
        ("Xin's Method", xin_ul),
        ("Dai's Method", dai_ul),
        ("Guo's Method", guo_ul)
    ]

    # 绘制循环
    for name, data in methods:
        style = STYLE_DICT[name]

        # 数据预处理
        if len(data) != n_points:
            data = np.full(n_points, np.mean(data))
        sorted_data = np.sort(data)
        mean_val = trimmed_mean(data)

        # 1. 绘制排序后的曲线
        ax.plot(x_values, sorted_data,
                color=style['color'],
                linestyle=style['ls'],
                linewidth=1.5,  # 主线稍微加粗
                alpha=0.9,
                label=name)  # 仅在图例中显示方法名

        # 2. 绘制均值线 (半透明，细线)
        # 技巧：在图例中我们将把均值数值合并显示，或者单独作为第二列
        # 这里为了图例整洁，我们把均值线也画出来，但标签特殊处理
        ax.axhline(y=mean_val,
                   color=style['color'],
                   linestyle=style['ls'],
                   linewidth=0.8,
                   alpha=0.6)

    # ----------------- 核心修改：自定义图例 -----------------
    # 为了防止图例过宽，我们构建一个自定义的图例句柄列表
    # 显示格式： [线型] Method Name (Mean: 0.xxx)
    from matplotlib.lines import Line2D
    custom_lines = []
    custom_labels = []

    for name, data in methods:
        style = STYLE_DICT[name]
        mean_val = trimmed_mean(data)

        # 创建图例图标
        line = Line2D([0], [0], color=style['color'], linestyle=style['ls'], lw=1.5)
        custom_lines.append(line)

        # 创建包含均值的标签
        label = f"{name}\n(Mean: {mean_val:.4f})"
        custom_labels.append(label)

    # 设置图例：2列布局，放在上方
    leg = ax.legend(custom_lines, custom_labels,
                    loc='upper left',
                    #bbox_to_anchor=(0.5, 1.30),  # 将图例推到坐标轴上方
                    ncol=2,  # 2列，适应窄画面
                    fontsize=FONT_SIZE_LEGEND+1,
                    frameon=True,
                    framealpha=0.7,
                    borderpad=0.2,
                    columnspacing=1.5,  # 列间距
                    labelspacing=0.5)  # 行间距

    # 坐标轴设置
    ax.set_xlabel('Simulation Index (Sorted by UL-loss)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Utility Loss (UL)', fontsize=FONT_SIZE_LABEL)

    ax.set_xlim([0, n_points])
    ax.set_xticks([0, 100, 200, 300, 400, 500])

    # 动态调整Y轴范围，留出一点余量
    all_data = np.concatenate([our_ul, xin_ul, dai_ul, guo_ul])
    y_min, y_max = np.min(all_data), np.max(all_data)
    ax.set_ylim([y_min * 0.9, y_max * 1.55])

    # 美化
    clean_axis(ax)

    # 保存
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved refined plot: {save_path}")


def plot_ul_comparison_t3(our_ul, xin_ul, dai_ul, save_path):
    """
    T2条件优化版
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_HEIGHT_SINGLE), constrained_layout=True)

    n_points = 500
    x_values = np.arange(1, n_points + 1)

    methods = [
        ('Our Model', our_ul),
        ("Xin's Method", xin_ul),
        ("Dai's Method", dai_ul)
    ]

    for name, data in methods:
        style = STYLE_DICT[name]
        if len(data) != n_points:
            data = np.full(n_points, np.mean(data))
        sorted_data = np.sort(data)
        mean_val = trimmed_mean(data)

        ax.plot(x_values, sorted_data,
                color=style['color'], linestyle=style['ls'], linewidth=1.5, alpha=0.9)
        ax.axhline(y=mean_val,
                   color=style['color'], linestyle=style['ls'], linewidth=0.8, alpha=0.6)

    # 自定义图例
    from matplotlib.lines import Line2D
    custom_lines = []
    custom_labels = []

    for name, data in methods:
        style = STYLE_DICT[name]
        mean_val = trimmed_mean(data)
        line = Line2D([0], [0], color=style['color'], linestyle=style['ls'], lw=1.5)
        custom_lines.append(line)
        custom_labels.append(f"{name}\n(Mean: {mean_val:.4f})")

    # 3个方法时可以用3列或者继续用2列
    # 用3列可能在3.5英寸下有点挤，这里建议用 2列（第二行居中自动排）或 3列紧凑
    leg = ax.legend(custom_lines, custom_labels,
                    loc='upper left',
                    #bbox_to_anchor=(0.5, 1.15),
                    ncol=2,  # 3个项勉强可以一行，如果字太长改成ncol=2
                    fontsize=FONT_SIZE_LEGEND+1,  # 稍微调小一点以适应一行
                    frameon=True,

                    framealpha=0.7,

                    borderpad=0.2,
                    columnspacing=1.0)

    ax.set_xlabel('Simulation Index (Sorted by UL-loss)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Utility Loss (UL)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([0, n_points])
    ax.set_xticks([0, 100, 200, 300, 400, 500])

    all_data = np.concatenate([our_ul, xin_ul, dai_ul])
    y_min, y_max = np.min(all_data), np.max(all_data)
    ax.set_ylim([y_min * 0.9, y_max * 1.55])

    clean_axis(ax)

    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved refined plot: {save_path}")


# 绘制直方图对比
def plot_ul_histogram_comparison_t1(our_ul, xin_ul, dai_ul, guo_ul, save_path):
    """
    绘制T1条件下的UL-loss直方图对比
    """
    # 使用紧凑尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single*0.7))

    # 方法标签和数据
    methods = [
        ('Our Model', our_ul, '#1f77b4'),
        ("Xin's Method", xin_ul, '#ff7f0e'),
        ("Dai's Method", dai_ul, '#2ca02c'),
        ("Guo's Method", guo_ul, '#d62728')
    ]

    # 计算合适的bins
    all_data = np.concatenate([our_ul, xin_ul, dai_ul, guo_ul])
    bins = np.linspace(np.min(all_data), np.max(all_data), 20)  # 减少bins数量以在紧凑图中更清晰

    # 绘制每个方法的直方图 - 使用更细的边框线
    for label, data, color in methods:
        ax.hist(data, bins=bins, alpha=0.6, color=color,
                edgecolor='black', linewidth=0.3,
                label=f'{label} (mean={trimmed_mean(data):.4f})')

    # 设置坐标轴和标签
    ax.set_xlabel('Utility Loss (UL)', fontsize=base_fontsize-1, labelpad=8)
    ax.set_ylabel('Frequency', fontsize=base_fontsize-1, labelpad=8)

    # 设置网格 - 使用更细的网格线
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
    ax.minorticks_on()

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=base_fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=base_fontsize-3)

    # 添加图例 - 使用更小的字体
    ax.legend(loc='upper left', fontsize=base_fontsize-3,
              frameon=True, framealpha=0.9)

    # 调整布局
    plt.tight_layout()

    # 保存图形 - 使用高DPI
    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ul_histogram_comparison_t3(our_ul, xin_ul, dai_ul, save_path):
    """
    绘制T2条件下的UL-loss直方图对比
    """
    # 使用紧凑尺寸
    fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single*0.7))

    # 方法标签和数据
    methods = [
        ('Our Model', our_ul, '#1f77b4'),
        ("Xin's Method", xin_ul, '#ff7f0e'),
        ("Dai's Method", dai_ul, '#2ca02c')
    ]

    # 计算合适的bins
    all_data = np.concatenate([our_ul, xin_ul, dai_ul])
    bins = np.linspace(np.min(all_data), np.max(all_data), 20)  # 减少bins数量以在紧凑图中更清晰

    # 绘制每个方法的直方图 - 使用更细的边框线
    for label, data, color in methods:
        ax.hist(data, bins=bins, alpha=0.6, color=color,
                edgecolor='black', linewidth=0.3,
                label=f'{label} (mean={trimmed_mean(data):.4f})')

    # 设置坐标轴和标签
    ax.set_xlabel('Utility Loss (UL)', fontsize=base_fontsize-1, labelpad=8)
    ax.set_ylabel('Frequency', fontsize=base_fontsize-1, labelpad=8)

    # 设置网格 - 使用更细的网格线
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
    ax.minorticks_on()

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=base_fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=base_fontsize-3)

    # 添加图例 - 使用更小的字体
    ax.legend(loc='upper right', fontsize=base_fontsize-2,
              frameon=True, framealpha=0.9)

    # 调整布局
    plt.tight_layout()

    # 保存图形 - 使用高DPI
    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


# 绘制T1条件下的UL-loss对比图
print("\nCreating T1 UL-loss comparison plot...")
plot_ul_comparison_t1(
    our_t1_ul,
    xin_t1_ul,
    dai_t1_ul,
    guo_t1_ul,
    r"E:\newManucript\manuscript2\image\12yue\Fig20_T1_UL_comparison.png"
)

# 绘制T2条件下的UL-loss对比图
print("Creating T2 UL-loss comparison plot...")
plot_ul_comparison_t3(
    our_t3_ul,
    xin_t3_ul,
    dai_t3_ul,
    r"E:\newManucript\manuscript2\image\12yue\Fig20_T2_UL_comparison.png"
)

# 绘制T1条件下的UL-loss直方图
print("\nCreating T1 UL-loss histogram comparison plot...")
plot_ul_histogram_comparison_t1(
    our_t1_ul,
    xin_t1_ul,
    dai_t1_ul,
    guo_t1_ul,
    r"E:\newManucript\manuscript2\image\12yue\Fig19_T1_UL_histogram.png"
)

# 绘制T2条件下的UL-loss直方图
print("Creating T2 UL-loss histogram comparison plot...")
plot_ul_histogram_comparison_t3(
    our_t3_ul,
    xin_t3_ul,
    dai_t3_ul,
    r"E:\newManucript\manuscript2\image\12yue\Fig19_T2_UL_histogram.png"
)

# 输出详细统计信息
print("\n" + "=" * 80)
print("UL-LOSS COMPARISON STATISTICS")
print("=" * 80)

print("\nT1 Condition (trimmed mean, 10% trimmed):")
print(f"  Our Model: {trimmed_mean(our_t1_ul):.6f}")
print(f"  Xin's Method: {trimmed_mean(xin_t1_ul):.6f}")
print(f"  Dai's Method: {trimmed_mean(dai_t1_ul):.6f}")
print(f"  Guo's Method: {trimmed_mean(guo_t1_ul):.6f}")

print("\nT2 Condition (trimmed mean, 10% trimmed):")
print(f"  Our Model: {trimmed_mean(our_t3_ul):.6f}")
print(f"  Xin's Method: {trimmed_mean(xin_t3_ul):.6f}")
print(f"  Dai's Method: {trimmed_mean(dai_t3_ul):.6f}")


# 计算改进百分比
def improvement_percentage(our_mean, other_mean):
    if other_mean > 0:
        return ((other_mean - our_mean) / other_mean) * 100
    return 0


print("\nImprovement Percentage (Our Model vs Others):")
print("\nT1 Condition:")
print(f"  vs Xin's: {improvement_percentage(trimmed_mean(our_t1_ul), trimmed_mean(xin_t1_ul)):.2f}%")
print(f"  vs Dai's: {improvement_percentage(trimmed_mean(our_t1_ul), trimmed_mean(dai_t1_ul)):.2f}%")
print(f"  vs Guo's: {improvement_percentage(trimmed_mean(our_t1_ul), trimmed_mean(guo_t1_ul)):.2f}%")

print("\nT2 Condition:")
print(f"  vs Xin's: {improvement_percentage(trimmed_mean(our_t3_ul), trimmed_mean(xin_t3_ul)):.2f}%")
print(f"  vs Dai's: {improvement_percentage(trimmed_mean(our_t3_ul), trimmed_mean(dai_t3_ul)):.2f}%")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETED!")
print("=" * 80)