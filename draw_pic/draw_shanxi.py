import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import matplotlib

# --------------------------
# 基础配置
# --------------------------
matplotlib.use('Agg')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'legend.frameon': True,
    'legend.edgecolor': 'black'
})

# --------------------------
# 1. 数据准备
# --------------------------
shp_path = r"E:\newManucript\manuscript2\script2data\shanxi_pic\shanxi_city.shp"
shanxi_map = gpd.read_file(shp_path)

# 氢能重卡数据
truck_data = {
    "City_Name": ["Taiyuan", "Datong", "Yangquan", "Changzhi", "Jincheng",
                  "Shuozhou", "Jinzhong", "Yuncheng", "Xinzhou", "Linfen", "Luliang"],
    "Hydrogen_Trucks": [100, 0, 5, 30, 0, 0, 0, 0, 0, 400, 550]
}
truck_df = pd.DataFrame(truck_data)

# 加氢站数据
station_data = {
    "City_Name": ["Taiyuan", "Datong", "Yangquan", "Changzhi", "Jincheng",
                  "Shuozhou", "Jinzhong", "Yuncheng", "Xinzhou", "Linfen", "Luliang"],
    "Completed_Hydrogen_Stations": [8, 1, 2, 3, 0, 1, 0, 2, 0, 5, 4]
}
station_df = pd.DataFrame(station_data)

# 中英文映射与数据合并
city_cn_en_map = {
    "太原市": "Taiyuan", "大同市": "Datong", "阳泉市": "Yangquan",
    "长治市": "Changzhi", "晋城市": "Jincheng", "朔州市": "Shuozhou",
    "晋中市": "Jinzhong", "运城市": "Yuncheng", "忻州市": "Xinzhou",
    "临汾市": "Linfen", "吕梁市": "Luliang"
}
shanxi_map["City_Name"] = shanxi_map["市"].map(city_cn_en_map)
merged = shanxi_map.merge(truck_df, on="City_Name", how="left")
merged = merged.merge(station_df, on="City_Name", how="left")
merged[["Hydrogen_Trucks", "Completed_Hydrogen_Stations"]] = merged[
    ["Hydrogen_Trucks", "Completed_Hydrogen_Stations"]
].fillna(0).astype(int)

# --------------------------
# 2. 保存目录
# --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
os.makedirs(save_dir, exist_ok=True)

# --------------------------
# 3. 绘制图1：氢能重卡分布（黑色边界线）
# --------------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 14))  # 进一步放大地图尺寸

# 绘制地图（黑色边界线，从灰色开始的紫色系）
merged.plot(
    column="Hydrogen_Trucks",
    ax=ax1,
    cmap="Purples",  # 紫色系，通过调整vmin使低值显示为灰色
    vmin=50,  # 最小值设为50，使0值区域显示为浅灰（非白色）
    edgecolor="black",  # 城市边界设为黑色
    linewidth=1.0,  # 边界线加粗，确保清晰
    legend=True,
    legend_kwds={
        "label": "Number of Hydrogen Heavy-Duty Trucks",
        "orientation": "horizontal",
        "pad": 0.05,  # 进一步缩小图例与地图间距
        "shrink": 0.9,
        "ticks": [0, 100, 200, 300, 400, 550]
    }
)

# 城市标注
for _, row in merged.iterrows():
    centroid = row["geometry"].centroid
    ax1.text(
        centroid.x, centroid.y,
        row["City_Name"],
        fontsize=11,  # 适当增大字体
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.15")
    )

# 指北针
north_arrow1 = mpatches.Arrow(
    x=merged.total_bounds[0] + 0.2,
    y=merged.total_bounds[3] - 0.2,
    dx=0,
    dy=0.8,  # 箭头长度增加，与大图匹配
    width=0.4,  # 箭头宽度增加
    facecolor="black"
)
ax1.add_patch(north_arrow1)
ax1.text(
    merged.total_bounds[0] + 0.2, merged.total_bounds[3] - 0.2 + 0.9,
    "N", fontsize=14, ha="center"
)

# 比例尺（放大以匹配地图尺寸）
scale_length = 50
scale_x_start = merged.total_bounds[0] + 1.0
scale_y_start = merged.total_bounds[1] + 0.5
scale_x_end = scale_x_start + scale_length / 111
ax1.plot([scale_x_start, scale_x_end], [scale_y_start, scale_y_start], color="black", linewidth=1.2)
ax1.plot([scale_x_start, scale_x_start], [scale_y_start - 0.08, scale_y_start + 0.08], color="black", linewidth=1.2)
ax1.plot([scale_x_end, scale_x_end], [scale_y_start - 0.08, scale_y_start + 0.08], color="black", linewidth=1.2)
ax1.text((scale_x_start + scale_x_end)/2, scale_y_start - 0.3, f"{scale_length} km", fontsize=10, ha="center")

# 标题与保存
ax1.set_title("Fig. 1 Distribution of Hydrogen-Powered Heavy-Duty Trucks in Shanxi",
              fontsize=14, pad=20, fontweight="bold")
ax1.axis("off")
fig1.tight_layout()
save_path1 = os.path.join(save_dir, "Hydrogen_Trucks_Distribution.png")
fig1.savefig(save_path1, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig1)

# --------------------------
# 4. 绘制图2：加氢站分布（黑色边界线）
# --------------------------
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 14))  # 与图1尺寸一致

# 绘制地图（黑色边界线）
station_plot = merged.plot(
    column="Completed_Hydrogen_Stations",
    ax=ax2,
    cmap="Reds",
    edgecolor="black",  # 城市边界设为黑色
    linewidth=1.0,  # 边界线加粗
    legend=True,
    legend_kwds={
        "label": "Number of Completed Hydrogen Refueling Stations",
        "orientation": "horizontal",
        "pad": 0.05,  # 缩小图例间距
        "shrink": 0.9,
        "ticks": [0, 2, 4, 6]
    }
)

# 城市标注
for _, row in merged.iterrows():
    centroid = row["geometry"].centroid
    ax2.text(
        centroid.x, centroid.y,
        row["City_Name"],
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.15")
    )

# 指北针
north_arrow2 = mpatches.Arrow(
    x=merged.total_bounds[0] + 0.2,
    y=merged.total_bounds[3] - 0.2,
    dx=0,
    dy=0.8,
    width=0.4,
    facecolor="black"
)
ax2.add_patch(north_arrow2)
ax2.text(
    merged.total_bounds[0] + 0.2, merged.total_bounds[3] - 0.2 + 0.9,
    "N", fontsize=14, ha="center"
)

# 比例尺（与图1一致）
ax2.plot([scale_x_start, scale_x_end], [scale_y_start, scale_y_start], color="black", linewidth=1.2)
ax2.plot([scale_x_start, scale_x_start], [scale_y_start - 0.08, scale_y_start + 0.08], color="black", linewidth=1.2)
ax2.plot([scale_x_end, scale_x_end], [scale_y_start - 0.08, scale_y_start + 0.08], color="black", linewidth=1.2)
ax2.text((scale_x_start + scale_x_end)/2, scale_y_start - 0.3, f"{scale_length} km", fontsize=10, ha="center")

# 标题与保存
ax2.set_title("Fig. 2 Distribution of Completed Hydrogen Refueling Stations in Shanxi",
              fontsize=14, pad=20, fontweight="bold")
ax2.axis("off")
fig2.tight_layout()
save_path2 = os.path.join(save_dir, "Hydrogen_Refueling_Stations_Distribution.png")
fig2.savefig(save_path2, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig2)

print(f"Figures saved successfully:\n1. {save_path1}\n2. {save_path2}")