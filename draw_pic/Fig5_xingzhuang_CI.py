import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

# 设置后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设置为Times风格

# 设置紧凑的图形尺寸和字体（SCI期刊常用尺寸）
# 单栏宽度：3.5英寸，双栏宽度：7英寸，高度根据需要调整
fig_width = 3.5  # 单栏宽度（英寸）
fig_height = 3.0  # 紧凑高度
base_fontsize = 9  # 基础字体大小（在缩小图片时会显得足够大）




# 加载数据
position3 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility1"
ariginal_file = os.path.join(position3, f"group_CI_1.pkl")

with open(ariginal_file, 'rb') as f:
    list_data1 = pickle.load(f)  # group_weight_3

# 转换数据为numpy数组
list_data = np.array(list_data1)

# 确保数据是一维的
if list_data.ndim > 1:
    list_data = list_data.flatten()

# 创建形状参数数组
shape_params = np.linspace(0.01, 10, len(list_data))

# 创建图形 - 使用紧凑尺寸
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

# 绘制折线图 - 使用更细的线条和更小的标记
line = ax.plot(shape_params, list_data,
               color='#1f77b4', linewidth=1.2,
               marker='o', markersize=2.5, markeredgecolor='black',
               markeredgewidth=0.3, markerfacecolor='white',
               alpha=0.9, zorder=3)

# 添加网格 - 使用更细更浅的网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5, zorder=1)

# 设置坐标轴标签
ax.set_xlabel('Shape Parameter', fontsize=base_fontsize, labelpad=4)
ax.set_ylabel('CI', fontsize=base_fontsize, labelpad=4)

# 设置坐标轴范围
ax.set_xlim([0, 10])
# 设置y轴范围，留出一些边距
y_min = np.min(list_data) - 0.003
y_max = np.max(list_data) + 0.003
ax.set_ylim([y_min, y_max])

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=base_fontsize - 1, pad=2)
ax.tick_params(axis='both', which='minor', labelsize=base_fontsize - 2, pad=2)

# 设置刻度线参数
ax.tick_params(axis='both', which='major', width=0.5, length=3)
ax.tick_params(axis='both', which='minor', width=0.3, length=2)

# 设置科学记数法格式
from matplotlib.ticker import ScalarFormatter


class FixedScalarFormatter(ScalarFormatter):
    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        super().__init__(useOffset=useOffset, useMathText=useMathText, useLocale=useLocale)

    def __call__(self, x, pos=None):
        if abs(x) < 0.01:
            return "0"
        return super().__call__(x, pos)


# 应用自定义格式化器
y_formatter = FixedScalarFormatter()
y_formatter.set_powerlimits((-3, 3))
y_formatter.set_scientific(True)
y_formatter.set_useMathText(True)  # 使用数学文本格式
ax.yaxis.set_major_formatter(y_formatter)

# 调整科学记数法偏移量的字体大小
ax.yaxis.get_offset_text().set_fontsize(base_fontsize - 2)

# 标记最大值和最小值
max_idx = np.argmax(list_data)
min_idx = np.argmin(list_data)

# 添加数据点标记
if len(list_data) < 30:  # 如果数据点较少，可以标记所有点
    for i, (x, y) in enumerate(zip(shape_params, list_data)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=base_fontsize )
else:  # 数据点较多时，标记最大值和最小值
    max_idx = np.argmax(list_data)
    min_idx = np.argmin(list_data)

    ax.annotate(f'Max: {list_data[max_idx]:.3f}',
                (shape_params[max_idx], list_data[max_idx]),
                textcoords="offset points",
                xytext=(45, 40), ha='center', fontsize=base_fontsize ,
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5))

    ax.annotate(f'Min: {list_data[min_idx]:.3f}',
                (shape_params[min_idx], list_data[min_idx]),
                textcoords="offset points",
                xytext=(0, -20), ha='center', fontsize=base_fontsize ,
                arrowprops=dict(arrowstyle="->", color='green', lw=1.5))

    # 标记起点和终点
    ax.annotate(f'Start: {list_data[0]:.3f}',
                (shape_params[0], list_data[0]),
                textcoords="offset points",
                xytext=(20, 10), ha='left', fontsize=base_fontsize )

    ax.annotate(f'End: {list_data[-1]:.3f}',
                (shape_params[-1], list_data[-1]),
                textcoords="offset points",
                xytext=(-15, 10), ha='right', fontsize=base_fontsize )


# 设置图形边框
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)

# 添加紧凑图例
# 选择是否添加图例取决于空间和清晰度
# 如果有足够空间，可以添加紧凑图例
if fig_width >= 5:  # 如果图形宽度足够
    ax.legend(fontsize=base_fontsize - 2, loc='best', framealpha=0.8,
              edgecolor='black', fancybox=False)

# 使用更紧凑的布局
plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)  # 减少内边距

# 保存图像 - 使用高DPI以确保印刷质量
save_path = r"E:\newManucript\manuscript2\image\12yue\Fig5.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white',
            pad_inches=0.02)  # 进一步减少保存时的边距
print(f"Figure saved to: {save_path}")

plt.show()

# 可选：创建另一个版本，使用更大的字体用于双栏图形
# 如果你需要双栏宽度图形，可以取消注释下面的代码
"""
fig_width_double = 7.0  # 双栏宽度（英寸）
fig_height_double = 3.0  # 相应的高度

fig2, ax2 = plt.subplots(figsize=(fig_width_double, fig_height_double), dpi=300)

# 复制上面的绘图代码，使用ax2而不是ax
# ...（与上面相同的绘图代码，但使用ax2）...

plt.tight_layout(pad=1.2, h_pad=0.6, w_pad=0.6)

save_path_double = r"E:\newManucript\manuscript2\image\12yue\Fig5_double.png"
plt.savefig(save_path_double, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.03)
print(f"Double column figure saved to: {save_path_double}")
"""