import numpy as np
import matplotlib

# 设置后端和字体
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma  # 伽马函数

# 设置绘图样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 1. 定义变量范围
mu = np.linspace(0, 1, 100)  # 均值μ∈(0,1)
lambda_ = np.linspace(0.1, 5, 100)  # 精度λ∈(0.1,5)，避免λ=0
mu, lambda_ = np.meshgrid(mu, lambda_)  # 生成网格

# 2. 正态-伽马分布参数（共轭先验参数）
alpha = 2       # 伽马分布形状参数
beta = 1        # 伽马分布率参数
mu0 = 0.5       # 正态部分均值
lambda0 = 1     # 正态部分精度

# 3. 计算联合概率密度（正态-伽马分布公式）
# f(μ,λ) = (√λ / √(2π)) * (β^α / Γ(α)) * λ^(α-1) * exp(-βλ - (λλ0/2)(μ - μ0)²)
term1 = np.sqrt(lambda_) / np.sqrt(2 * np.pi)
term2 = (beta ** alpha) / gamma(alpha)
term3 = lambda_ ** (alpha - 1)
term4 = np.exp(-beta * lambda_ - (lambda_ * lambda0 / 2) * (mu - mu0) ** 2)
joint_pdf = term1 * term2 * term3 * term4

# 4. 绘制3D图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图（使用viridis配色，清晰展示密度梯度）
surf = ax.plot_surface(mu, lambda_, joint_pdf, cmap='viridis',
                       alpha=0.9, edgecolor='k', linewidth=0.3)

# 标注与美化
ax.set_xlabel('μ (0 ≤ μ ≤ 1)', fontweight='bold', labelpad=10)
ax.set_ylabel('λ', fontweight='bold', labelpad=10)
ax.set_zlabel('f(μ,λ)', fontweight='bold', labelpad=10)
#ax.set_title('正态-伽马共轭分布（μ∈(0,1)）\n参数：α=2, β=1, μ₀=0.5, λ₀=1',
            # fontweight='bold', pad=20)

# 添加颜色条（指示密度大小）
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('', rotation=270, labelpad=15)

# 调整视角（更清晰展示分布形态）
ax.view_init(elev=30, azim=45)

# 保存为PNG
plt.tight_layout()
plt.savefig('normal_gamma_conjugate_3d.png', dpi=300, bbox_inches='tight')
plt.close()

print("正态-伽马共轭分布3D图已生成：normal_gamma_conjugate_3d.png")