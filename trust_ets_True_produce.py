import numpy as np
import pandas as pd
import os
import pickle
import random


def generate_combined_trust_data(position):
    """
    从trust_ets_random{i}.pkl与trust_ets_{i}.pkl中随机抽取信息
    生成决策者间随机相识与随机不相识的情况

    Parameters:
    position: 保存结果的路径
    """

    all_combined_results = []

    # 处理5个批次的数据
    for batch_idx in range(1, 6):
        print(f"Processing batch {batch_idx}...")

        # 加载原始数据和随机数据
        original_file = os.path.join(position, f"trust_ets_{batch_idx}.pkl")
        random_file = os.path.join(position, f"trust_ets_random{batch_idx}.pkl")

        if not os.path.exists(original_file):
            print(f"Warning: {original_file} not found")
            continue

        if not os.path.exists(random_file):
            print(f"Warning: {random_file} not found")
            continue

        # 加载数据
        with open(original_file, 'rb') as f:
            original_data = pickle.load(f)

        with open(random_file, 'rb') as f:
            random_data = pickle.load(f)

        # 存储当前批次的500次循环结果
        batch_results = []

        # 进行500次循环
        for loop_idx in range(500):
            loop_result = []

            # 处理每个决策者
            for i, dm_original in enumerate(original_data):
                dm_loop_result = []

                # 处理每个目标决策者
                for j, original_value_pair in enumerate(dm_original):
                    if i == j:
                        # 自信值和自信置信度：保留原始数据
                        dm_loop_result.append([original_value_pair[0], original_value_pair[1]])
                    else:
                        # 信任值和信任置信度：随机抽取
                        if random.uniform(0,1) < 0.7:
                            # 70%概率使用原始数据
                            dm_loop_result.append([original_value_pair[0], original_value_pair[1]])
                        else:
                            # 30%概率使用随机数据
                            random_value_pair = random_data[loop_idx][i][j]
                            dm_loop_result.append([random_value_pair[0], random_value_pair[1]])

                loop_result.append(dm_loop_result)

            batch_results.append(loop_result)

        all_combined_results.append(batch_results)

        # 保存为pickle文件
        pkl_path = os.path.join(position, f"trust_ets_True{batch_idx}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(batch_results, f)

        # 保存为Excel文件
        excel_path = os.path.join(position, f"trust_ets_True{batch_idx}.xlsx")
        save_combined_to_excel(batch_results, excel_path)

        print(f"Batch {batch_idx} saved to {pkl_path} and {excel_path}")

    return all_combined_results


def save_combined_to_excel(data, file_path):
    """
    将组合信任数据保存到Excel文件
    每个决策者一个工作表，每两次循环结果占两行
    """
    try:
        # 检查文件是否已存在，如果存在则删除
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed existing file: {file_path}")

        # 创建Excel写入器
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            n_dms = len(data[0])  # 决策者数量
            n_loops = len(data)  # 循环次数

            # 为每个决策者创建一个工作表
            for dm_idx in range(n_dms):
                # 创建DataFrame
                df_data = []

                # 添加表头
                header = ["num"] + [f"DM{j + 1}" for j in range(n_dms)]
                df_data.append(header)

                # 添加每次循环的数据
                for loop_idx in range(n_loops):
                    # 信任/自信数据行
                    trust_row = [f"{loop_idx + 1}"]
                    # 置信度数据行
                    confidence_row = [""]  # 空单元格，与信任行共享循环编号

                    for j in range(n_dms):
                        value_pair = data[loop_idx][dm_idx][j]
                        trust_row.append(f"{value_pair[0]:.3f}")
                        confidence_row.append(f"{value_pair[1]:.3f}")

                    df_data.append(trust_row)
                    df_data.append(confidence_row)

                # 创建DataFrame并保存
                df = pd.DataFrame(df_data)
                sheet_name = f"DM{dm_idx + 1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

                # 调整列宽
                worksheet = writer.sheets[sheet_name]
                for col_idx, col in enumerate(worksheet.columns):
                    max_length = 0
                    for cell in col:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    adjusted_width = min(max_length + 2, 50)  # 限制最大宽度
                    worksheet.column_dimensions[chr(65 + col_idx)].width = adjusted_width

        print(f"Combined trust Excel file saved to: {file_path}")
        return True

    except PermissionError as e:
        print(f"PermissionError: Cannot save to {file_path}. File might be open in another program.")
        print(f"Error details: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error while saving to Excel: {e}")
        return False


def main():
    """主函数"""
    position = r"E:\newManucript\python_code_rare\script2\data\simulation_data"

    # 确保目录存在
    if not os.path.exists(position):
        os.makedirs(position)
        print(f"Created directory: {position}")

    # 生成组合信任数据
    all_combined_results = generate_combined_trust_data(position)

    # 保存所有结果
    if all_combined_results:
        all_results_path = os.path.join(position, "all_combined_trust_results.pkl")
        with open(all_results_path, 'wb') as f:
            pickle.dump(all_combined_results, f)

        print(f"\nAll combined trust results saved to: {all_results_path}")

        # 打印示例结果
        print("\nExample results (first batch, first loop):")
        for i, dm_data in enumerate(all_combined_results[0][0]):
            print(f"DM{i + 1}: {dm_data}")

    return all_combined_results


if __name__ == "__main__":
    results = main()