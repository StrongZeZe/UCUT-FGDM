import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
from matplotlib.lines import Line2D

# 设置SCI期刊常用的字体（Times New Roman）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 16
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义二维联合正态分布的参数
mu1 = 0.1  # mean of x1
sigma1 = 0.2  # standard deviation of x1
mu2 = 0.5  # mean of x2
sigma2 = 0.2  # standard deviation of x2
rho = 0.0  # correlation coefficient


# 定义二维联合正态分布的概率密度函数
def joint_normal(x1, x2):
    """Calculate the probability density of bivariate normal distribution"""
    if abs(rho) == 1.0:
        # Perfect correlation case
        expected_x2 = mu2 + rho * (sigma2 / sigma1) * (x1 - mu1)
        if np.isclose(x2, expected_x2, atol=1e-3):
            return np.exp(-0.5 * ((x1 - mu1) / sigma1) ** 2) / (np.sqrt(2 * np.pi) * sigma1)
        else:
            return 0.0
    else:
        # General case
        z1 = (x1 - mu1) / sigma1
        z2 = (x2 - mu2) / sigma2
        denominator = 2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho ** 2)
        exponent = - (z1 ** 2 - 2 * rho * z1 * z2 + z2 ** 2) / (2 * (1 - rho ** 2))
        return np.exp(exponent) / denominator


# Create grid data
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)

# Calculate probability density at each point
Z = np.vectorize(joint_normal)(X1, X2)

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create color array based on x1 and x2 relationship
color_array = np.zeros((X1.shape[0], X1.shape[1], 3))
color_array[X1 > X2] = [0, 0, 1]  # Blue for x1 > x2
color_array[X1 <= X2] = [1, 0, 0]  # Red for x1 ≤ x2

# Plot surface
surf = ax.plot_surface(X1, X2, Z, facecolors=color_array, alpha=0.8,
                      linewidth=0.5, edgecolor='k', antialiased=True)

# Set axis labels and title
ax.set_xlabel('x1', fontsize=14, labelpad=12)
ax.set_ylabel('x2', fontsize=14, labelpad=12)
ax.set_zlabel('Probability Density', fontsize=14, labelpad=12)
plt.title(f'Bivariate Normal Distribution\n(μ₁={mu1}, σ₁={sigma1}, μ₂={mu2}, σ₂={sigma2}, ρ={rho})',
          fontsize=16, pad=20)

# Add custom legend
legend_elements = [
    Line2D([0], [0], color='blue', lw=4, label='Confidence Region (x₁ > x₂)'),
    Line2D([0], [0], color='red', lw=4, label='Trust Region (x₁ ≤ x₂)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Add caption text
# plt.figtext(0.5, 0.01,
#             "Figure description: The 3D surface height represents probability density. "
#             "Colors distinguish the relationship between x₁ and x₂, "
#             "where red regions represent trust and blue regions represent confidence.",
#             ha="center", fontsize=11,
#             bbox={"facecolor":"white", "alpha":0.8, "pad":5})

# Adjust viewing angle
ax.view_init(elev=45, azim=-125)

# Handle extreme values for perfect correlation case
if abs(rho) == 1.0:
    ax.set_zlim(0, np.percentile(Z[Z > 0], 95))

# Improve layout
plt.tight_layout()

# Save figure in high resolution for publication
save_path = r"E:\newManucript\manuscript2\image\Confidence_and_Trust_3D_Plot.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Figure saved to: {save_path}")

plt.show()

# Calculate cumulative probabilities
def integrand(x2, x1):
    """Integrand function for double integration"""
    return joint_normal(x1, x2)

# Calculate probability for x1 > x2 region (integration region: 0 ≤ x2 < x1 ≤ 1)
prob_x1_greater, error1 = dblquad(integrand, 0, 1, lambda x: 0, lambda x: x)

# Calculate total probability over [0,1]x[0,1] region
total_prob, error_total = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)

# Probability for x1 ≤ x2 region = total probability - x1 > x2 probability
prob_x1_less_or_equal = total_prob - prob_x1_greater

# Output results
print(f"Parameters: μ₁={mu1}, σ₁={sigma1}, μ₂={mu2}, σ₂={sigma2}, ρ={rho}")
print(f"Cumulative probability for x₁ > x₂ region: {prob_x1_greater:.6f} (error estimate: {error1:.6e})")
print(f"Cumulative probability for x₁ ≤ x₂ region: {prob_x1_less_or_equal:.6f}")
print(f"Total probability over [0,1]×[0,1] region: {total_prob:.6f} (error estimate: {error_total:.6e})")