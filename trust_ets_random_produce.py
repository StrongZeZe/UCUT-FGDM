import numpy as np
import pandas as pd
import os
import pickle
import random


def generate_random_trust_data(position):
    """
    生成带有随机信任值的自信-信任与置信度数组

    Parameters:
    position: 保存结果的路径
    """

    all_random_results = []

    # 处理5个批次的trust_ets文件
    for batch_idx in range(1, 6):
        print(f"Processing batch {batch_idx}...")

        # 加载原始的trust_ets数据
        trust_file = os.path.join(position, f"trust_ets_{batch_idx}.pkl")
        if not os.path.exists(trust_file):
            print(f"Warning: {trust_file} not found")
            continue

        with open(trust_file, 'rb') as f:
            original_data = pickle.load(f)

        # 存储当前批次的500次循环结果
        batch_results = []

        # 进行500次循环
        for loop_idx in range(500):
            loop_result = []

            # 复制原始数据，但只保留自信值（对角线元素）
            for i, dm_data in enumerate(original_data):
                dm_loop_result = []

                for j, value_pair in enumerate(dm_data):
                    if i == j:
                        # 保留原始的自信值和置信度
                        dm_loop_result.append([value_pair[0], value_pair[1]])
                    else:
                        # 随机生成信任值
                        trust_value = round(random.uniform(0.3, 0.95), 3)

                        # 使用正态分布生成置信度，标准差为1.5，确保在[0.3,0.95]范围内
                        confidence_value = generate_normal_confidence(0.625, 1.5)  # 均值为0.625，标准差为1.5

                        dm_loop_result.append([trust_value, confidence_value])

                loop_result.append(dm_loop_result)

            batch_results.append(loop_result)

        all_random_results.append(batch_results)

        # 保存随机信任数据为pickle文件
        pkl_path = os.path.join(position, f"trust_ets_random{batch_idx}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(batch_results, f)

        # 保存为Excel文件
        excel_path = os.path.join(position, f"trust_ets_random{batch_idx}.xlsx")
        save_random_to_excel(batch_results, excel_path)

        print(f"Batch {batch_idx} saved to {pkl_path} and {excel_path}")

    return all_random_results


def generate_normal_confidence(mean, std_dev):
    """
    使用正态分布生成置信度，确保在[0.3, 0.95]范围内
    """
    max_attempts = 100
    for _ in range(max_attempts):
        confidence = np.random.normal(mean, std_dev)
        if 0.3 <= confidence <= 0.95:
            return round(confidence, 3)

    # 如果多次尝试失败，使用截断正态分布
    confidence = np.random.normal(mean, std_dev)
    confidence = max(0.3, min(0.95, confidence))
    return round(confidence, 3)


def save_random_to_excel(data, file_path):
    """
    将随机信任数据保存到Excel文件
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

        print(f"Random trust Excel file saved to: {file_path}")
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

    # 生成随机信任数据
    all_random_results = generate_random_trust_data(position)

    # 保存所有结果
    if all_random_results:
        all_results_path = os.path.join(position, "all_random_trust_results.pkl")
        with open(all_results_path, 'wb') as f:
            pickle.dump(all_random_results, f)

        print(f"\nAll random trust results saved to: {all_results_path}")

        # 打印示例结果
        print("\nExample results (first batch, first loop):")
        for i, dm_data in enumerate(all_random_results[0][0]):
            print(f"DM{i + 1}: {dm_data}")

    return all_random_results


if __name__ == "__main__":
    results = main()