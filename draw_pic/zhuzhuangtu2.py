import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
min_UL_loss = [0.16826556016597508, 0.16444007858546172, 0.16308482142857142, 0.17142857142857146, 0.16521758241758241]
max_GCD = [0.16818672199170126, 0.16449705304518666, 0.1632455357142857, 0.1710045351473923, 0.16521098901098902]
schemes = ['L1', 'L2', 'L3', 'L4', 'L5']

# 计算排名
min_UL_loss_rank = [sorted(min_UL_loss, reverse=True).index(x) + 1 for x in min_UL_loss]  # min UL-loss: 数值越小排名越好
max_GCD_rank = [sorted(max_GCD, reverse=True).index(x) + 1 for x in max_GCD]  # max GCD: 数值越大排名越好

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱状图位置
x = np.arange(len(schemes))
width = 0.35

# 绘制柱状图
bars1 = ax.bar(x - width/2, min_UL_loss, width, label='min UL-loss',
               color='#1f77b4', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, max_GCD, width, label='max GCD',
               color='#ff7f0e', edgecolor='black', linewidth=0.8)

# 在柱子顶部添加排名标注（使用白色边框避免重叠）
def add_rank_labels(bars, ranks, offset=0.01):
    for bar, rank in zip(bars, ranks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'({rank})', ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                         edgecolor='black', linewidth=0.5))

# 添加排名标注
add_rank_labels(bars1, min_UL_loss_rank)
add_rank_labels(bars2, max_GCD_rank)

# 设置坐标轴标签
ax.set_xlabel('Hydrogen Refueling Location', fontsize=12)
ax.set_ylabel('Score', fontsize=12)

# 设置x轴刻度
ax.set_xticks(x)
ax.set_xticklabels(schemes)

# 设置图例（放在右上角，用黑线包裹）
legend = ax.legend(loc='upper right', frameon=True, framealpha=1,
                  edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(1.0)

# 设置网格线（使图表更清晰）
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 调整y轴范围，为标注留出空间
y_max = max(max(min_UL_loss), max(max_GCD))
ax.set_ylim(0, y_max + 0.1)

# 设置图表边框
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# 调整布局
plt.tight_layout()

# 保存图片
save_path = r"E:\newManucript\manuscript2\image\选之地排序柱状图.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='black')
plt.close()

print(f"图片已保存至: {save_path}")