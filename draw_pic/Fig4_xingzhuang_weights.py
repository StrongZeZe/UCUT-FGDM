import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D

# 设置后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也设置为Times风格

base_fontsize=15             #统一字体大小

# 数据准备（这里使用示例数据，您需要替换为实际的list_data）
# list_data应该是(200, 6)的numpy数组
position3 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility1"

ariginal_file = os.path.join(position3, f"group_weight_1.pkl")
with open(ariginal_file, 'rb') as f:
    list_data1 = pickle.load(f)      #group_weight_3
list_data=np.array(list_data1)

num_points=len(list_data)

# 这里生成示例数据
num_indicators = 6

# 生成形状参数值 (0.01到10)
shape_params = np.linspace(0.01, 10, num_points)


# 创建网格数据
X, Y = np.meshgrid(shape_params, range(num_indicators))
Z = list_data.T  # 转置以匹配网格形状

# 颜色设置
indicator_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
indicator_names = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']

# 创建图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制半透明表面
surf = ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis',
                      linewidth=0.5, antialiased=True, edgecolor='gray',
                      vmin=0, vmax=0.35)

# 为每个指标绘制单独的线
for i in range(num_indicators):
    ax.plot(shape_params, np.full_like(shape_params, i), list_data[:, i],
           color=indicator_colors[i], linewidth=2.5, label=indicator_names[i])

# 设置坐标轴标签
ax.set_xlabel('Shape Parameter', fontsize=16, labelpad=20)
ax.set_ylabel('Indicator', fontsize=16, labelpad=20)
ax.set_zlabel('Weight', fontsize=16, labelpad=20)

# 设置y轴刻度
ax.set_yticks(range(num_indicators))
ax.set_yticklabels(indicator_names)

ax.tick_params(axis='x', labelsize=base_fontsize-2)
ax.tick_params(axis='y', labelsize=base_fontsize-2)
ax.tick_params(axis='z', labelsize=base_fontsize-2)

# 设置视角
ax.view_init(elev=25, azim=45)


# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Weight', fontsize=16)
cbar.ax.tick_params(labelsize=base_fontsize-2)

# 调整颜色条位置，下移0.05（图形高度单位）
cbar_pos = cbar.ax.get_position()
new_cbar_pos = [cbar_pos.x0, cbar_pos.y0 - 0.13, cbar_pos.width, cbar_pos.height]
cbar.ax.set_position(new_cbar_pos)

# 添加图例
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16, frameon=True)

# 设置坐标轴范围
ax.set_xlim([0, 10])
ax.set_ylim([-0.5, 5.5])
ax.set_zlim([0, 0.35])

# 设置网格
ax.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图像
save_path = r"E:\newManucript\manuscript2\image\12yue\Fig4.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path}")

plt.show()