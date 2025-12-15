import numpy as np
import pandas as pd
import os
import pickle
import random


def generate_confidence_trust_data(k_values_list, position):
    """
    根据K值生成自信、信任和置信度数据

    Parameters:
    k_values_list: 包含5个批次K值的列表
    position: 保存结果的路径
    """

    all_results = []

    for batch_idx, k_values in enumerate(k_values_list):
        print(f"Processing batch {batch_idx + 1}...")

        # 步骤1: 读取K值
        n_dms = len(k_values)
        max_k = max(k_values)

        # 步骤2: 计算自信水平
        confidences = []
        for k in k_values:
            confidences.append(0.5)                 #设置自信都为0.5

        # 步骤3: 计算信任水平
        trust_matrix = np.zeros((n_dms, n_dms))
        for i in range(n_dms):
            for j in range(n_dms):
                if i == j:
                    # 自己对自己的信任就是自信
                    trust_matrix[i][j] = confidences[i]
                else:
                    trust = 0.5 + (k_values[j] - k_values[i]) / max_k

                    # 根据条件调整信任水平
                    if trust >= 0.9:
                        final_trust = 0.9+ random.uniform(-0.05, 0.05)
                    elif trust <= 0.3:
                        final_trust = 0.3+ random.uniform(0.01, 0.1)
                    else:
                        final_trust = trust

                    trust_matrix[i][j] = final_trust

        # 步骤4: 计算置信水平
        confidence_matrix = np.zeros((n_dms, n_dms))
        for i in range(n_dms):

            confidence_matrix[i] = 0.5                  #置信度也都设置为0.5

        # 构建结果数组
        batch_result = []
        for i in range(n_dms):
            dm_result = []
            for j in range(n_dms):
                # 自信和置信度
                if i == j:
                    dm_result.append([round(trust_matrix[i][j], 3), round(confidence_matrix[i][j], 3)])
                # 信任和置信度
                else:
                    dm_result.append([round(trust_matrix[i][j], 3), round(confidence_matrix[i][j], 3)])
            batch_result.append(dm_result)

        all_results.append(batch_result)

        # 保存Excel文件
        excel_path = os.path.join(position, f"trust_ets_none_confi_{batch_idx + 1}.xlsx")
        save_to_excel(batch_result, excel_path)

        # 保存pickle文件
        pkl_path = os.path.join(position, f"trust_ets_none_confi_{batch_idx + 1}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(batch_result, f)

        print(f"Batch {batch_idx + 1} saved to {excel_path} and {pkl_path}")

    return all_results


def save_to_excel(data, file_path):
    """将数据保存到Excel文件"""
    # 创建Excel写入器
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # 为每个决策者创建一个工作表
        for i, dm_data in enumerate(data):
            # 创建DataFrame
            df_data = []

            # 添加表头
            header =["Value Type"] + [f"DM{j + 1}" for j in range(len(dm_data))]
            df_data.append(header)

            # 添加自信/信任数据
            trust_row = [f"Trust/self-Confidence"]
            confidence_row = [f"Confidence Level"]

            for j, value_pair in enumerate(dm_data):
                trust_row.append(f"{value_pair[0]:.3f}")
                confidence_row.append(f"{value_pair[1]:.3f}")

            df_data.append(trust_row)
            df_data.append(confidence_row)

            # 创建DataFrame并保存
            df = pd.DataFrame(df_data)
            sheet_name = f"DM{i + 1}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

            # 调整列宽
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column].width = adjusted_width

    print(f"Excel file saved to: {file_path}")


def load_k_values(position):
    """从指定位置加载所有K值文件"""
    k_values_list = []

    for i in range(1, 6):
        k_file = os.path.join(position, f"k_values_{i}.pkl")
        if os.path.exists(k_file):
            with open(k_file, 'rb') as f:
                k_values = pickle.load(f)
                k_values_list.append(k_values)
                print(f"Loaded k_values_{i}: {k_values}")
        else:
            print(f"Warning: {k_file} not found")

    return k_values_list


def main():
    """主函数"""
    position = r"E:\newManucript\python_code_rare\script2\data\simulation_data"

    # 确保目录存在
    if not os.path.exists(position):
        os.makedirs(position)
        print(f"Created directory: {position}")

    # 加载K值
    k_values_list = load_k_values(position)

    if not k_values_list:
        print("No K values found. Please check the directory.")
        return

    # 生成自信、信任和置信度数据
    all_results = generate_confidence_trust_data(k_values_list, position)

    # 保存所有结果
    all_results_path = os.path.join(position, "all_trust_ets_noneConfi_results.pkl")
    with open(all_results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nAll results saved to: {all_results_path}")

    # 打印示例结果
    print("\nExample results (first batch):")
    for i, dm_data in enumerate(all_results[0]):
        print(f"DM{i + 1}: {dm_data}")

    return all_results


if __name__ == "__main__":
    results = main()