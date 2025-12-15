import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
from matplotlib.patches import Polygon

# 设置后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设置为Times风格

# ===================== 关键修改：紧凑的SCI期刊尺寸 =====================
# SCI期刊单栏宽度通常为3.5英寸（约8.9厘米）
# 双栏宽度通常为7英寸（约17.8厘米）
# 我们将使用单栏宽度，但保持足够大的字体
fig_width_single = 3.5  # 单栏宽度（英寸）
base_fontsize = 9  # 基础字体大小 - 在3.5英寸图形中足够清晰

# 加载数据
position1 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility2_whiteBox"
position2 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility2_blackBox"

# 不考虑u-KDD的数据
original_file = os.path.join(position1, f"group_weight_2.pkl")
original_file4 = os.path.join(position1, f"GCD_final_name_2.pkl")
original_file6 = os.path.join(position1, f"group_CI_2.pkl")
original_file7 = os.path.join(position1, f"group_utility_loss_2.pkl")

with open(original_file4, 'rb') as f:
    original_data4 = pickle.load(f)  # GCD
with open(original_file6, 'rb') as f:
    original_data6 = pickle.load(f)  # 一致性水平
with open(original_file7, 'rb') as f:
    original_data7 = pickle.load(f)  # 效用损失
with open(original_file, 'rb') as f:
    original_data = pickle.load(f)  # group_weight_2

# 考虑u-KDD的数据
driginal_file = os.path.join(position2, f"group_weight_3.pkl")
driginal_file4 = os.path.join(position2, f"GCD_final_name_3.pkl")
driginal_file6 = os.path.join(position2, f"group_CI_3.pkl")
driginal_file7 = os.path.join(position2, f"group_utility_loss_3.pkl")

with open(driginal_file4, 'rb') as f:
    driginal_data4 = pickle.load(f)  # GCD
with open(driginal_file6, 'rb') as f:
    driginal_data6 = pickle.load(f)  # 一致性水平
with open(driginal_file7, 'rb') as f:
    driginal_data7 = pickle.load(f)  # 效用损失
with open(driginal_file, 'rb') as f:
    driginal_data = pickle.load(f)  # group_weight_3




# 转换为numpy数组
original_data = np.array(original_data)
original_data4 = np.array(original_data4)
original_data6 = np.array(original_data6)
original_data7 = np.array(original_data7)

driginal_data = np.array(driginal_data)
driginal_data4 = np.array(driginal_data4)
driginal_data6 = np.array(driginal_data6)
driginal_data7 = np.array(driginal_data7)

# 指标名称
indicator_names = ['Resource', 'Environment', 'External\nSupport', 'Risk', 'Economy', 'Geology']

# 1. 权重对比图 - 使用雷达图和箱线图组合
# 修改图形尺寸为SCI单栏宽度
fig, axes = plt.subplots(1, 2, figsize=(fig_width_single, fig_width_single*0.5), subplot_kw=dict(polar=True))

# 计算每个指标的最大值、最小值和平均值
max_original = np.max(original_data, axis=0)
min_original = np.min(original_data, axis=0)
mean_original = np.mean(original_data, axis=0)

max_driginal = np.max(driginal_data, axis=0)
min_driginal = np.min(driginal_data, axis=0)
mean_driginal = np.mean(driginal_data, axis=0)

# 设置雷达图参数
angles = np.linspace(0, 2*np.pi, len(indicator_names), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 准备数据（闭合）
max_original_plot = np.concatenate((max_original, [max_original[0]]))
min_original_plot = np.concatenate((min_original, [min_original[0]]))
mean_original_plot = np.concatenate((mean_original, [mean_original[0]]))

max_driginal_plot = np.concatenate((max_driginal, [max_driginal[0]]))
min_driginal_plot = np.concatenate((min_driginal, [min_driginal[0]]))
mean_driginal_plot = np.concatenate((mean_driginal, [mean_driginal[0]]))

# 设置雷达图参数
angles = np.linspace(0, 2*np.pi, len(indicator_names), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 准备数据（闭合）
max_original_plot = np.concatenate((max_original, [max_original[0]]))
min_original_plot = np.concatenate((min_original, [min_original[0]]))
mean_original_plot = np.concatenate((mean_original, [mean_original[0]]))

max_driginal_plot = np.concatenate((max_driginal, [max_driginal[0]]))
min_driginal_plot = np.concatenate((min_driginal, [min_driginal[0]]))
mean_driginal_plot = np.concatenate((mean_driginal, [mean_driginal[0]]))

# ===================== 图1: 不考虑u-KDD的雷达图 =====================
fig1, ax1 = plt.subplots(figsize=(fig_width_single, fig_width_single),
                        subplot_kw=dict(polar=True), dpi=300)

# 填充最大值和最小值之间的区域
ax1.fill_between(angles, min_original_plot, max_original_plot,
                color='#1f77b4', alpha=0.2, label='Range')

# 绘制最大值和最小值线
ax1.plot(angles, max_original_plot, color='#1f77b4', linewidth=0.5,
        linestyle='--', alpha=0.7)
ax1.plot(angles, min_original_plot, color='#1f77b4', linewidth=0.5,
        linestyle='--', alpha=0.7)

# 绘制平均值线（放在最上层）
ax1.plot(angles, mean_original_plot, color='#1f77b4', linewidth=1.0,
        label='Mean', zorder=5)

# 设置刻度
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(indicator_names, fontsize=base_fontsize+2, fontweight='medium')

# 设置半径范围和刻度
ax1.set_ylim([0.05, 0.3])
ax1.set_yticks([0.1, 0.2, 0.3])
ax1.set_yticklabels(['0.1', '0.2', '0.3'], fontsize=base_fontsize+1)

# 设置网格
ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)

# 设置标题 - 使用更大的字体和适当的边距
ax1.set_title('Without u-KDD', fontsize=base_fontsize+4, pad=25, fontweight='bold')

# 添加数据长度标注
ax1.text(0.5, -0.15, f'Feedback Nums: {len(original_data)}',
         fontsize=base_fontsize+1, color='#1f77b4', ha='center',
         transform=ax1.transAxes)

# 添加图例 - 调整位置和大小
ax1.legend(loc='upper right', fontsize=base_fontsize+1,
          bbox_to_anchor=(1.3, 1.15), frameon=True, framealpha=0.9,
          edgecolor='black', facecolor='white')

# 调整布局
plt.tight_layout()

# 保存第一幅雷达图
save_path1 = r"E:\newManucript\manuscript2\image\12yue\Fig8a_Weight_Without_uKDD.png"
plt.savefig(save_path1, dpi=600, bbox_inches='tight', facecolor='white',
           pad_inches=0.05)
print(f"Radar figure (Without u-KDD) saved to: {save_path1}")
plt.close()

# ===================== 图2: 考虑u-KDD的雷达图 =====================
fig2, ax2 = plt.subplots(figsize=(fig_width_single, fig_width_single),
                        subplot_kw=dict(polar=True), dpi=300)

# 填充最大值和最小值之间的区域
ax2.fill_between(angles, min_driginal_plot, max_driginal_plot,
                color='#ff7f0e', alpha=0.2, label='Range')

# 绘制最大值和最小值线
ax2.plot(angles, max_driginal_plot, color='#ff7f0e', linewidth=0.5,
        linestyle='--', alpha=0.7)
ax2.plot(angles, min_driginal_plot, color='#ff7f0e', linewidth=0.5,
        linestyle='--', alpha=0.7)

# 绘制平均值线（放在最上层）
ax2.plot(angles, mean_driginal_plot, color='#ff7f0e', linewidth=1.0,
        label='Mean', zorder=5)

# 设置刻度
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(indicator_names, fontsize=base_fontsize+2, fontweight='medium')

# 设置半径范围和刻度
ax2.set_ylim([0.05, 0.3])
ax2.set_yticks([0.1, 0.2, 0.3])
ax2.set_yticklabels(['0.1', '0.2', '0.3'], fontsize=base_fontsize+1)

# 设置网格
ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)

# 设置标题 - 使用更大的字体和适当的边距
ax2.set_title('With u-KDD', fontsize=base_fontsize+4, pad=25, fontweight='bold')

# 添加数据长度标注
ax2.text(0.5, -0.15, f'Feedback Nums: {len(driginal_data)}',
         fontsize=base_fontsize+1, color='#ff7f0e', ha='center',
         transform=ax2.transAxes)

# 添加图例 - 调整位置和大小
ax2.legend(loc='upper right', fontsize=base_fontsize+1,
          bbox_to_anchor=(1.3, 1.15), frameon=True, framealpha=0.9,
          edgecolor='black', facecolor='white')

# 调整布局
plt.tight_layout()

# 保存第二幅雷达图
save_path2 = r"E:\newManucript\manuscript2\image\12yue\Fig8b_Weight_With_uKDD.png"
plt.savefig(save_path2, dpi=600, bbox_inches='tight', facecolor='white',
           pad_inches=0.05)
print(f"Radar figure (With u-KDD) saved to: {save_path2}")
plt.close()

print("Both radar figures have been generated and saved successfully!")

# 2. 3D平面图 - 权重数据对比
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# ===================== 3D平面图（SCI一区级） =====================
# 修改图形尺寸为SCI单栏宽度
fig = plt.figure(figsize=(fig_width_single, fig_width_single*0.8), dpi=100, facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# 获取数据维度
n_feedback_original = original_data.shape[0]  # 不考虑u-KDD的反馈数: 41
n_feedback_driginal = driginal_data.shape[0]   # 考虑u-KDD的反馈数: 38
n_indicator = original_data.shape[1]           # 指标数: 6

# 创建网格 - 处理维度不一致问题
# 对于不考虑u-KDD的数据
X1 = np.arange(n_feedback_original)  # 0到40
Y1 = np.arange(n_indicator)          # 0到5
X1, Y1 = np.meshgrid(X1, Y1)         # 维度: (6, 41)
Z1 = original_data.T                 # 转置后维度: (6, 41)

# 对于考虑u-KDD的数据
X2 = np.arange(n_feedback_driginal)  # 0到37
Y2 = np.arange(n_indicator)          # 0到5
X2, Y2 = np.meshgrid(X2, Y2)         # 维度: (6, 38)
Z2 = driginal_data.T                 # 转置后维度: (6, 38)

# ===================== 核心优化：3D表面样式 =====================
# 配色选择：SCI顶刊偏好低饱和度、高对比度的专业配色
# Without u-KDD：深蓝（#2C3E50），With u-KDD：深橙（#E67E22）

# 绘制不考虑u-KDD的平面
surf1 = ax.plot_surface(
    X1, Y1, Z1,
    alpha=0.8,           # 透明度优化，避免遮挡
    color='#2C3E50',     # 低饱和深蓝，专业且打印清晰
    shade=True,          # 光影效果，提升立体感
    edgecolor='#E0E0E0', # 浅灰边缘，精致不突兀
    linewidth=0.1,       # 更细的边缘线，避免杂乱
    antialiased=True,    # 抗锯齿，提升清晰度
    zorder=1             # 底层显示
)

# 绘制考虑u-KDD的平面
surf2 = ax.plot_surface(
    X2, Y2, Z2,
    alpha=0.75,          # 略低透明度，区分两个平面
    color='#E67E22',     # 低饱和橙，与深蓝形成高对比
    shade=True,
    edgecolor='#E0E0E0',
    linewidth=0.1,
    antialiased=True,
    zorder=2             # 上层显示
)

# ===================== 坐标轴优化（核心要求+SCI规范） =====================
# 1. 坐标轴标签：更大的labelpad，字体加粗，字号适配
ax.set_xlabel('Feedback Number', fontsize=base_fontsize, labelpad=10, fontweight='medium')
ax.set_ylabel('Indicator', fontsize=base_fontsize, labelpad=12, fontweight='medium')
ax.set_zlabel('Index Weight', fontsize=base_fontsize, labelpad=10, fontweight='medium')

# 2. 刻度设置
# Y轴刻度（指标名）
ax.set_yticks(np.arange(n_indicator))
ax.set_yticklabels(indicator_names, fontsize=base_fontsize-2, fontweight='light')

# X轴刻度（反馈数） - 使用两个数据集的合并范围
max_feedback = max(n_feedback_original, n_feedback_driginal)
x_ticks = np.arange(0, max_feedback, 5)  # 每5个显示一个刻度
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, fontsize=base_fontsize-2, fontweight='light')

# Z轴刻度（权重） - 使用两个数据集的合并范围
z_min = min(Z1.min(), Z2.min())
z_max = max(Z1.max(), Z2.max())
z_ticks = np.linspace(z_min, z_max, 5)  # 均匀分布5个刻度
ax.set_zticks(z_ticks)
ax.set_zticklabels([f'{z:.2f}' for z in z_ticks], fontsize=base_fontsize-2, fontweight='light')

# 3. 坐标轴样式：精细网格线
ax.xaxis._axinfo['grid'].update(linewidth=0.3, color='#F0F0F0')
ax.yaxis._axinfo['grid'].update(linewidth=0.3, color='#F0F0F0')
ax.zaxis._axinfo['grid'].update(linewidth=0.3, color='#F0F0F0')

# 4. 设置坐标轴范围
ax.set_xlim([0, max_feedback-1])
ax.set_ylim([0, n_indicator-1])
ax.set_zlim([z_min - 0.05, z_max + 0.05])  # 添加小边距

# ===================== 视角与图例优化 =====================
# 视角调整：优化视角以清晰展示两个平面
ax.view_init(elev=25, azim=55)

# 图例优化：SCI风格
legend_elements = [
    Patch(facecolor='#2C3E50', alpha=0.8, label=f'Without u-KDD (N={n_feedback_original})', edgecolor='none'),
    Patch(facecolor='#E67E22', alpha=0.75, label=f'With u-KDD (N={n_feedback_driginal})', edgecolor='none')
]
ax.legend(
    handles=legend_elements,
    loc='upper left',          # 左上角，避免遮挡
    fontsize=base_fontsize-2,
    frameon=True,              # 带边框
    framealpha=0.9,            # 半透明背景
    facecolor='white',
    edgecolor='#E0E0E0',
    handlelength=1.5,          # 图例块长度优化
    borderaxespad=0.5
)

# 添加重要标注
# 标注反馈次数差异
fig.text(0.02, 0.02, f'Feedback reduction: {n_feedback_original - n_feedback_driginal}',
         fontsize=base_fontsize-3, color='red', transform=fig.transFigure)

# ===================== 保存设置（SCI期刊要求） =====================
save_path_3d = r"E:\newManucript\manuscript2\image\12yue\Fig8_3d_weight_SCI.png"
plt.savefig(
    save_path_3d,
    dpi=600,                    # 高分辨率
    bbox_inches='tight',        # 紧凑布局
    facecolor='white',          # 白色背景
    edgecolor='none',           # 无边框
    pad_inches=0.1              # 微小内边距
)
print(f"Optimized 3D weight comparison figure saved to: {save_path_3d}")

plt.close()

# 3. 一致性水平对比图
# 修改图形尺寸为SCI单栏宽度
fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single*0.7))

# 创建迭代序列（注意数据长度可能不同）
iter_original = np.arange(1, len(original_data6) + 1)
iter_driginal = np.arange(1, len(driginal_data6) + 1)

# 绘制折线图 - 调整线条粗细
line1, = ax.plot(iter_original, original_data6,
                color='#1f77b4', linewidth=0.8,
                marker='o', markersize=2, markeredgecolor='black',
                markeredgewidth=0.3, markerfacecolor='white',
                label=f'Without u-KDD (Feedback Nums={len(original_data6)})')

line2, = ax.plot(iter_driginal, driginal_data6,
                color='#ff7f0e', linewidth=0.8,
                marker='s', markersize=2, markeredgecolor='black',
                markeredgewidth=0.3, markerfacecolor='white',
                label=f'With u-KDD (Feedback Nums={len(driginal_data6)})')

# 添加共识水平线（修改标签名称）
ax.axhline(y=driginal_data6[0], color='red', linestyle='--', linewidth=0.8,
           alpha=0.7, label='with u-KDD\'s start level')

# 标记达到目标共识的点
target_idx_original = np.where(original_data6 >= 0.99)[0]
if len(target_idx_original) > 0:
    ax.plot(target_idx_original[0] + 1, original_data6[target_idx_original[0]],
            'o', markersize=4, markeredgecolor='red',
            markerfacecolor='none', markeredgewidth=1.5)

target_idx_driginal = np.where(driginal_data6 >= 0.99)[0]
if len(target_idx_driginal) > 0:
    ax.plot(target_idx_driginal[0] + 1, driginal_data6[target_idx_driginal[0]],
            's', markersize=4, markeredgecolor='red',
            markerfacecolor='none', markeredgewidth=1.5)

# 添加标注
if len(target_idx_original) > 0:
    ax.annotate(f'Reached at feedback {target_idx_original[0]+1}',
                (target_idx_original[0]+1, original_data6[target_idx_original[0]]),
                textcoords="offset points", xytext=(0, 10),
                ha='center', fontsize=base_fontsize-1, color='red')

if len(target_idx_driginal) > 0:
    ax.annotate(f'Reached at feedback {target_idx_driginal[0]+1}',
                (target_idx_driginal[0]+1, driginal_data6[target_idx_driginal[0]]),
                textcoords="offset points", xytext=(0, -15),
                ha='center', fontsize=base_fontsize-2, color='red')

# 设置坐标轴和标签（修改横坐标标签）
ax.set_xlabel('Feedback Nums', fontsize=base_fontsize, labelpad=8)
ax.set_ylabel('Consistency Index (CI)', fontsize=base_fontsize, labelpad=8)
ax.set_xlim([1, max(len(original_data6), len(driginal_data6))])
ax.set_ylim([min(np.min(original_data6), np.min(driginal_data6)) - 0.005, 0.99])

# 设置网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=base_fontsize-1)

# 添加图例
ax.legend(loc='upper right', fontsize=base_fontsize-2, frameon=True)

# 调整布局
plt.tight_layout()
save_path = r"E:\newManucript\manuscript2\image\12yue\Fig9_compareCI.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"CI comparison figure saved to: {save_path}")
plt.close()

# 4. 效用损失对比图
fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single*0.7))

# 绘制折线图 - 调整线条粗细
line1, = ax.plot(iter_original, original_data7,
                color='#1f77b4', linewidth=0.8,
                marker='o', markersize=2, markeredgecolor='black',
                markeredgewidth=0.3, markerfacecolor='white',
                label=f'Without u-KDD (Feedback Nums={len(original_data7)})')

line2, = ax.plot(iter_driginal, driginal_data7,
                color='#ff7f0e', linewidth=0.8,
                marker='s', markersize=2, markeredgecolor='black',
                markeredgewidth=0.3, markerfacecolor='white',
                label=f'With u-KDD (Feedback Nums={len(driginal_data7)})')

# 设置坐标轴和标签
ax.set_xlabel('Feedback Nums', fontsize=base_fontsize, labelpad=8)
ax.set_ylabel('Utility Loss (UL)', fontsize=base_fontsize, labelpad=8)
ax.set_xlim([1, max(len(original_data7), len(driginal_data7))])

# 计算y轴范围
y_min = min(np.min(original_data7), np.min(driginal_data7))
y_max = max(np.max(original_data7), np.max(driginal_data7))
y_range = y_max - y_min
ax.set_ylim([y_min - 0.05*y_range, y_max + 0.05*y_range])

# 设置网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=base_fontsize-1)

# 修改图例位置到右下角
ax.legend(loc='lower right', fontsize=base_fontsize-2, frameon=True)

# 添加最终效用损失值标注
ax.annotate(f'Final UL: {original_data7[-1]:.4f}',
            (len(original_data7), original_data7[-1]),
            textcoords="offset points", xytext=(-50, -20),
            ha='left', fontsize=base_fontsize-2, color='#1f77b4')

ax.annotate(f'Final UL: {driginal_data7[-1]:.4f}',
            (len(driginal_data7), driginal_data7[-1]),
            textcoords="offset points", xytext=(-40, -10),
            ha='left', fontsize=base_fontsize-2, color='#ff7f0e')

# 调整布局
plt.tight_layout()
save_path = r"E:\newManucript\manuscript2\image\12yue\Fig10_compareUL-loss.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"UL-loss comparison figure saved to: {save_path}")
plt.close()

# 5. 群共识水平对比图（修复数据维度问题）
# 检查数据长度
print(f"original_data4 length: {len(original_data4)}")
print(f"driginal_data4 length: {len(driginal_data4)}")

# 根据实际数据长度创建迭代序列
iter_original_gcd = np.arange(1, len(original_data4) + 1)
iter_driginal_gcd = np.arange(1, len(driginal_data4) + 1)

fig, ax = plt.subplots(figsize=(fig_width_single, fig_width_single*0.7))

# 绘制折线图 - 调整线条粗细
line1, = ax.plot(iter_original_gcd, original_data4,
                color='#1f77b4', linewidth=0.8,
                marker='o', markersize=2, markeredgecolor='black',
                markeredgewidth=0.3, markerfacecolor='white',
                label=f'Without u-KDD (Feedback Nums={len(original_data4)})')

line2, = ax.plot(iter_driginal_gcd, driginal_data4,
                color='#ff7f0e', linewidth=0.8,
                marker='s', markersize=2, markeredgecolor='black',
                markeredgewidth=0.3, markerfacecolor='white',
                label=f'With u-KDD (Feedback Nums={len(driginal_data4)})')

# 添加共识水平线
ax.axhline(y=0.99, color='red', linestyle='--', linewidth=0.8,
           alpha=0.7, label='Target GCD (0.99)')

# 标记达到目标共识的点
target_idx_original = np.where(original_data4 >= 0.99)[0]
if len(target_idx_original) > 0:
    ax.plot(target_idx_original[0] + 1, original_data4[target_idx_original[0]],
            'o', markersize=4, markeredgecolor='red',
            markerfacecolor='none', markeredgewidth=1)

target_idx_driginal = np.where(driginal_data4 >= 0.99)[0]
if len(target_idx_driginal) > 0:
    ax.plot(target_idx_driginal[0] + 1, driginal_data4[target_idx_driginal[0]],
            's', markersize=4, markeredgecolor='red',
            markerfacecolor='none', markeredgewidth=1)

# 添加标注
if len(target_idx_original) > 0:
    ax.annotate(f'Reached at feedback {target_idx_original[0]+1}',
                (target_idx_original[0]+1, original_data4[target_idx_original[0]]),
                textcoords="offset points", xytext=(-40, 8),
                ha='center', fontsize=base_fontsize-1, color='#1f77b4')

if len(target_idx_driginal) > 0:
    ax.annotate(f'Reached at feedback {target_idx_driginal[0]+1}',
                (target_idx_driginal[0]+1, driginal_data4[target_idx_driginal[0]]),
                textcoords="offset points", xytext=(-30, -35),
                ha='center', fontsize=base_fontsize-1, color='#ff7f0e')

# 设置坐标轴和标签
ax.set_xlabel('Feedback Nums', fontsize=base_fontsize, labelpad=8)
ax.set_ylabel('Group Consensus Degree (GCD)', fontsize=base_fontsize, labelpad=8)
ax.set_xlim([1, max(len(original_data4), len(driginal_data4))])
ax.set_ylim([min(np.min(original_data4), np.min(driginal_data4)) - 0.005, 1.0])

# 设置网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=base_fontsize-1)

# 添加图例
ax.legend(loc='lower right', fontsize=base_fontsize-2, frameon=True)

# 调整布局
plt.tight_layout()
save_path = r"E:\newManucript\manuscript2\image\12yue\Fig11_compareGCD.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"GCD comparison figure saved to: {save_path}")
plt.close()

print("All comparison figures have been generated successfully!")