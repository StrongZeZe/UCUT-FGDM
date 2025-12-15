import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 数据
min_UL_loss = [0.142, 0.265, 0.216, 0.137, 0.106, 0.135]
schemes = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']

# 计算排名 - 按照从大到小排序（因为min_UL_loss越小越好）
min_UL_loss_rank = [sorted(min_UL_loss, reverse=True).index(x) + 1 for x in min_UL_loss]

# 创建雷达图（显示排名）
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# 设置角度
angles = np.linspace(0, 2 * np.pi, len(schemes), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 将排名归一化（排名1最好，显示在最外层）
# 注意：排名1对应雷达图的最外层，排名越大越靠内
max_rank = max(min_UL_loss_rank)
rank_norm = [(max_rank - rank + 1) / max_rank for rank in min_UL_loss_rank]
rank_norm += rank_norm[:1]

# 绘制排名雷达图
ax.plot(angles, rank_norm, 'o-', linewidth=3, label='Ranking (1=Best)', color='#2ca02c', markersize=8)
ax.fill(angles, rank_norm, alpha=0.25, color='#2ca02c')

# 设置角度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])  # 如果已经有xticks，用这行隐藏标签
for angle, label in zip(angles[:-1], schemes):
    # 将角度转换为文本位置
    # 使用text函数创建带框的标签
    ax.text(angle, 1.05,  # 稍微向外移动，避免与图形重叠
            label,
            ha='center', va='center', fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor='black', linewidth=1, alpha=0.8))

# 设置径向标签（显示排名）
ax.set_rlabel_position(30)
rank_ticks = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
rank_labels = [' ', ' ', ' ', ' ', ' ', ' ']
plt.yticks(rank_ticks, rank_labels, color="grey", size=10)
plt.ylim(0, 1)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=20)

# 添加标题
#plt.title('Ranking of Hydrogen Refueling Locations (Based on min UL-loss)',
          #size=15, y=1.05, weight='bold')

# 在每个数据点上添加实际值和排名
for i, (angle, ul_val, ul_rank) in enumerate(zip(
        angles[:-1], min_UL_loss, min_UL_loss_rank)):
    #y_position = rank_norm[i] + 10.58
    ax.text(angle, rank_norm[i] - 0.20,
            f'V: {ul_val:.3f}\nR: {ul_rank}',
            ha='center', va='center', fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor='#2ca02c', linewidth=1))

# 保存图片
save_path = r"E:\newManucript\manuscript2\image\12yue\fig3.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='black')
plt.show()

print(f"排名雷达图已保存至: {save_path}")
print("\n详细数据:")
print("方案 | min_UL_loss | 排名")
print("-" * 25)
for scheme, value, rank in zip(schemes, min_UL_loss, min_UL_loss_rank):
    print(f"{scheme:^4} | {value:^11.3f} | {rank:^4}")