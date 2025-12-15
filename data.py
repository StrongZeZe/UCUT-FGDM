
#计算相关性程度
import numpy as np
from scipy.stats import pearsonr, t

# 定义数据
a = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
b = [0.1811339419346629,0.18107455277426862, 0.180512274778436,0.18002120057316376,0.17962856597560542,0.1793141547669507,0.17905849623007192,0.17884711933221073,0.17866965519939693,0.17851863509476393, 0.1784007780125266,]

# 计算皮尔逊相关系数
correlation_coefficient, p_value = pearsonr(a, b)

print(f"皮尔逊相关系数: {correlation_coefficient:.6f}")
print(f"P值: {p_value:.10f}")

# 设置置信水平为0.1（显著性水平α=0.1）
alpha = 0.1

# 判断显著性
if p_value < alpha:
    significance = "显著"
else:
    significance = "不显著"

print(f"在{alpha*100}%的置信水平下，相关性{significance}")

# 计算置信区间（使用Fisher Z变换）
n = len(a)
z = np.arctanh(correlation_coefficient)  # Fisher Z变换
se = 1 / np.sqrt(n - 3)  # 标准误
z_lower = z - t.ppf(1 - alpha/2, n-2) * se
z_upper = z + t.ppf(1 - alpha/2, n-2) * se

# 将Z值转换回相关系数
corr_lower = np.tanh(z_lower)
corr_upper = np.tanh(z_upper)

print(f"相关系数的{100*(1-alpha)}%置信区间: [{corr_lower:.6f}, {corr_upper:.6f}]")

# 输出相关性强度判断
if abs(correlation_coefficient) >= 0.8:
    strength = "强"
elif abs(correlation_coefficient) >= 0.5:
    strength = "中等"
elif abs(correlation_coefficient) >= 0.3:
    strength = "弱"
else:
    strength = "极弱"

print(f"相关性强度: {strength}相关")
print(f"相关性方向: {'负' if correlation_coefficient < 0 else '正'}相关")

# 计算t统计量进行验证
t_statistic = correlation_coefficient * np.sqrt(n-2) / np.sqrt(1 - correlation_coefficient**2)
t_critical = t.ppf(1 - alpha/2, n-2)  # 双尾检验的临界值

print(f"t统计量: {t_statistic:.6f}")
print(f"t临界值(α={alpha}, 双尾): ±{t_critical:.6f}")

if abs(t_statistic) > t_critical:
    print("t检验结果: 拒绝原假设，相关性显著")
else:
    print("t检验结果: 不能拒绝原假设，相关性不显著")