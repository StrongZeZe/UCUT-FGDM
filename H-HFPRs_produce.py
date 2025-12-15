import numpy as np
import pandas as pd
import random
import os
import pickle
from itertools import combinations


class HeterogeneousHesitantFuzzyPreferenceRelation:
    def __init__(self, k_value, weights):
        # 定义语义模糊集和模糊集
        self.linguistic_set = [0.06, 0.17, 0.28, 0.39, 0.5, 0.61, 0.72, 0.83, 0.94]
        self.fuzzy_set = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.unknown_value = 100  # 未知信息标记
        self.k_value = k_value  # 决策者能力值
        self.weights = weights  # 方案权重

    def get_preference_type_probability(self):
        """根据K值使用随机整数确定偏好类型"""
        if 50 <= self.k_value <= 100:
            # 四种类型等概率 (各25%)
            rand_val = random.randint(1, 4)
            if rand_val == 1:
                return 'HFPR'
            elif rand_val == 2:
                return 'HFLPR'
            elif rand_val == 3:
                return 'i-HFPR'
            else:  # rand_val == 4
                return 'i-HFLPR'

        elif 100 < self.k_value <= 150:
            # HFPR: 3/10=0.3, HFLPR: 3/10=0.3, i-HFPR: 2/10=0.2, i-HFLPR: 2/10=0.2
            rand_val = random.randint(1, 10)
            if rand_val <= 3:
                return 'HFPR'
            elif rand_val <= 6:
                return 'HFLPR'
            elif rand_val <= 8:
                return 'i-HFPR'
            else:  # rand_val <= 10
                return 'i-HFLPR'

        elif 150 < self.k_value <= 200:
            # HFPR: 7/20=0.35, HFLPR: 7/20=0.35, i-HFPR: 3/20=0.15, i-HFLPR: 3/20=0.15
            rand_val = random.randint(1, 20)
            if rand_val <= 7:
                return 'HFPR'
            elif rand_val <= 14:
                return 'HFLPR'
            elif rand_val <= 17:
                return 'i-HFPR'
            else:  # rand_val <= 20
                return 'i-HFLPR'

        elif 200 < self.k_value <= 250:
            # HFPR: 8/20=0.4, HFLPR: 8/20=0.4, i-HFPR: 2/20=0.1, i-HFLPR: 2/20=0.1
            rand_val = random.randint(1, 20)
            if rand_val <= 8:
                return 'HFPR'
            elif rand_val <= 16:
                return 'HFLPR'
            elif rand_val <= 18:
                return 'i-HFPR'
            else:  # rand_val <= 20
                return 'i-HFLPR'

        else:
            # 默认情况，等概率
            rand_val = random.randint(1, 4)
            if rand_val == 1:
                return 'HFPR'
            elif rand_val == 2:
                return 'HFLPR'
            elif rand_val == 3:
                return 'i-HFPR'
            else:  # rand_val == 4
                return 'i-HFLPR'

    def select_element_count(self):
        """根据K值使用随机整数确定元素数量"""
        if 50 <= self.k_value <= 150:
            # 1个元素: 4/10=0.4, 2个元素: 4/10=0.4, 3个元素: 2/10=0.2
            rand_val = random.randint(1, 10)
            if rand_val <= 5:
                return 1
            elif rand_val <= 9:
                return 2
            else:  # rand_val <= 10
                return 3

        elif 150 < self.k_value <= 250:
            # 1个元素: 9/20=0.45, 2个元素: 9/20=0.45, 3个元素: 2/20=0.1
            rand_val = random.randint(1, 30)
            if rand_val <= 14:
                return 1
            elif rand_val <= 28:
                return 2
            else:  # rand_val <= 20
                return 3

        else:
            # 默认情况
            rand_val = random.randint(1, 20)
            if rand_val <= 10:
                return 1
            elif rand_val <= 19:
                return 2
            else:  # rand_val <= 10
                return 3

    def find_closest_values(self, benchmark, value_set, count):
        """在值集中找到最接近基准值的count个值"""
        # 计算每个值与基准值的距离
        distances = [(abs(val - benchmark), val) for val in value_set]
        # 按距离排序
        distances.sort(key=lambda x: x[0])
        # 返回最接近的count个值
        return [val for _, val in distances[:count]]

    def generate_matrix_element(self, i, j, value_set):
        """生成矩阵元素，基于权重计算基准值并选择最接近的值"""
        # 计算基准值：方案i的权重 / (方案i的权重 + 方案j的权重)
        benchmark = self.weights[i] / (self.weights[i] + self.weights[j])

        # 选择元素数量
        element_count = self.select_element_count()

        # 找到最接近基准值的元素
        elements = self.find_closest_values(benchmark, value_set, element_count)
        elements.sort()

        return elements

    def generate_hfpr(self, n=6):
        """生成犹豫模糊偏好关系 (HFPR)"""
        matrix = np.full((n, n), None, dtype=object)

        # 对角线设为0.5
        for i in range(n):
            matrix[i][i] = [0.5]

        # 生成上三角元素
        for i, j in combinations(range(n), 2):
            if i < j:
                # 基于权重生成元素
                elements = self.generate_matrix_element(i, j, self.fuzzy_set)
                matrix[i][j] = elements

                # 下三角元素通过互补性得到
                complement = [round(1 - x, 2) for x in elements]
                complement.sort()
                matrix[j][i] = complement

        return matrix

    def generate_hflpr(self, n=6):
        """生成犹豫模糊语义偏好关系 (HFLPR)"""
        matrix = np.full((n, n), None, dtype=object)

        # 对角线设为0.5
        for i in range(n):
            matrix[i][i] = [0.5]

        # 生成上三角元素
        for i, j in combinations(range(n), 2):
            if i < j:
                # 基于权重生成元素
                elements = self.generate_matrix_element(i, j, self.linguistic_set)
                matrix[i][j] = elements

                # 下三角元素通过互补性得到
                complement = [round(1 - x, 2) for x in elements]
                complement.sort()
                matrix[j][i] = complement

        return matrix

    def is_valid_incomplete_matrix(self, matrix, n=6):
        """检查不完全矩阵是否满足条件：没有行或列全为未知信息"""
        for i in range(n):
            row_has_known = any(matrix[i][j] != [self.unknown_value] for j in range(n) if j != i)
            col_has_known = any(matrix[j][i] != [self.unknown_value] for j in range(n) if j != i)

            if not row_has_known or not col_has_known:
                return False
        return True

    def generate_incomplete_matrix(self, base_matrix, n=6):
        """生成不完全矩阵，确保满足特定约束条件"""
        matrix = [row.copy() for row in base_matrix]

        # 上三角元素位置
        upper_tri_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # 计算最大未知元素数量（小于上三角元素的1/5）
        max_unknown = max(1, len(upper_tri_indices) // 5 - 1)

        # 必须已知的位置（不能设为未知信息）
        must_known_positions = [(0, 1), (4, 5)]

        # 不能同时为未知信息的位置对
        mutually_exclusive_pairs = [
            [(0, 2), (1, 2)],  # 第0行2与第1行2不能同时为不完全信息
            [(3, 4), (3, 5)]  # 第3行4与5的位置不能同为不完全信息
        ]

        # 从可选位置中移除必须已知的位置
        available_positions = [pos for pos in upper_tri_indices if pos not in must_known_positions]

        max_attempts = 100
        for attempt in range(max_attempts):
            # 创建当前尝试的矩阵副本
            temp_matrix = [row.copy() for row in matrix]

            # 随机选择未知位置数量（1到max_unknown）
            num_unknown = random.randint(1, max_unknown)

            # 随机选择未知位置
            unknown_positions = random.sample(available_positions, num_unknown)

            # 检查互斥约束
            valid = True
            for pair in mutually_exclusive_pairs:
                if all(pos in unknown_positions for pos in pair):
                    valid = False
                    break

            if not valid:
                continue  # 违反互斥约束，重新选择

            # 设置未知值
            for i, j in unknown_positions:
                temp_matrix[i][j] = [self.unknown_value]
                temp_matrix[j][i] = [self.unknown_value]

            # 检查是否满足条件：没有行或列全为未知信息
            if self.is_valid_incomplete_matrix(temp_matrix, n):
                return temp_matrix

        # 如果多次尝试失败，返回原始矩阵
        return matrix

    def generate_i_hfpr(self, n=6):
        """生成不完全犹豫模糊偏好关系 (i-HFPR)"""
        base_matrix = self.generate_hfpr(n)
        return self.generate_incomplete_matrix(base_matrix, n)

    def generate_i_hflpr(self, n=6):
        """生成不完全犹豫模糊语义偏好关系 (i-HFLPR)"""
        base_matrix = self.generate_hflpr(n)
        return self.generate_incomplete_matrix(base_matrix, n)

    def extract_upper_triangle(self, matrix, n=6):
        """提取上三角元素（不包括对角线）"""
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(matrix[i][j])
        return upper_triangle

    def generate_heterogeneous_preference(self, n=6):
        """随机生成一种异质犹豫模糊偏好关系，使用随机整数决定类型"""
        # 直接使用随机整数决定的类型
        selected_type = self.get_preference_type_probability()

        # 生成对应的矩阵
        if selected_type == 'HFPR':
            matrix = self.generate_hfpr(n)
        elif selected_type == 'HFLPR':
            matrix = self.generate_hflpr(n)
        elif selected_type == 'i-HFPR':
            matrix = self.generate_i_hfpr(n)
        elif selected_type == 'i-HFLPR':
            matrix = self.generate_i_hflpr(n)

        upper_triangle = self.extract_upper_triangle(matrix, n)

        return selected_type, upper_triangle


def format_matrix_data(upper_triangle, unknown_value=100):
    """将上三角数据格式化为6x6矩阵格式"""
    n = 6
    matrix = [['' for _ in range(n)] for _ in range(n)]

    # 设置对角线
    for i in range(n):
        matrix[i][i] = '0.5'

    # 填充上三角
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # 将列表转换为字符串表示
            if upper_triangle[idx] == [unknown_value]:
                matrix[i][j] = '-'
            else:
                matrix[i][j] = ', '.join(f'{x:.2f}' for x in upper_triangle[idx])
            idx += 1

    return matrix


def save_to_excel(table_values, output_path):
    """将数据保存到Excel文件"""
    n_dms = len(table_values[0])  # 决策者数量
    n_simulations = len(table_values)  # 模拟次数

    # 创建Excel写入器
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for dm_idx in range(n_dms):
            # 创建DataFrame
            data = []

            # 添加表头
            header = ['Evaluation type'] + [f'Sheme{i + 1}' for i in range(6)]
            data.append(header)

            # 添加每个模拟的结果
            for sim_idx in range(n_simulations):
                pref_data = table_values[sim_idx][dm_idx]
                pref_type = pref_data[0]
                upper_triangle = pref_data[1:]  # 剩余15个元素是上三角数据

                # 格式化矩阵数据
                matrix_data = format_matrix_data(upper_triangle)

                # 添加类型行
                type_row = [pref_type] + [''] * 6
                data.append(type_row)

                # 添加矩阵数据
                for row in matrix_data:
                    data.append([''] + row)

            # 创建DataFrame并保存到sheet
            df = pd.DataFrame(data)
            sheet_name = f'DM{dm_idx + 1}'
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

    print(f"Data saved to: {output_path}")


def generate_table_values(dm_assessments, k_values):
    """生成tableValues格式的数据，考虑K值影响"""
    table_values = []

    n_simulations = len(dm_assessments)
    n_dms = len(dm_assessments[0])

    print(f"Generating {n_simulations} simulations for {n_dms} decision makers...")

    for sim_idx in range(n_simulations):
        if sim_idx % 100 == 0:
            print(f"Progress: {sim_idx}/{n_simulations}")

        dm_preferences = []

        for dm_idx in range(n_dms):
            # 获取当前决策者的K值和权重
            k_value = k_values[dm_idx]
            weights = dm_assessments[sim_idx, dm_idx]

            # 创建H-HFPR生成器，传入K值和权重
            hfpr_generator = HeterogeneousHesitantFuzzyPreferenceRelation(k_value, weights)

            # 为每个决策者生成异质犹豫模糊偏好关系
            pref_type, upper_triangle = hfpr_generator.generate_heterogeneous_preference()
            dm_preferences.append([pref_type] + upper_triangle)

        table_values.append(dm_preferences)

    return table_values


def save_table_values(table_values, file_path):
    """将table_values保存为Python数据文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(table_values, f)
    print(f"Table values saved to: {file_path}")


def load_table_values(file_path):
    """从文件加载table_values"""
    with open(file_path, 'rb') as f:
        table_values = pickle.load(f)
    return table_values


def process_batch_data():
    """处理批次数据的主函数"""
    # 定义数据路径
    data_path = r"E:\newManucript\python_code_rare\script2\data\simulation_data"

    # 确保目录存在
    if not os.path.exists(data_path):
        print(f"Directory does not exist: {data_path}")
        return

    # 存储所有批次的table_values
    all_table_values = []

    # 处理5个批次
    for i in range(1, 6):
        print(f"\n=== Processing Batch {i} ===")

        # 加载assessment_array
        assessment_file = os.path.join(data_path, f"assessment_array_{i}.pkl")
        k_values_file = os.path.join(data_path, f"k_values_{i}.pkl")

        if not os.path.exists(assessment_file):
            print(f"File not found: {assessment_file}")
            continue

        if not os.path.exists(k_values_file):
            print(f"File not found: {k_values_file}")
            continue

        try:
            # 加载assessment_array和k_values
            with open(assessment_file, 'rb') as f:
                assessment_array = pickle.load(f)

            with open(k_values_file, 'rb') as f:
                k_values = pickle.load(f)

            print(f"Loaded assessment_array_{i} with shape: {assessment_array.shape}")
            print(f"Loaded k_values_{i}: {k_values}")

            # 生成tableValues，传入K值
            table_values = generate_table_values(assessment_array, k_values)

            # 保存Excel文件
            excel_path = os.path.join(data_path, f"score_index_500_{i}.xlsx")
            save_to_excel(table_values, excel_path)

            # 保存table_values为pickle文件
            table_values_path = os.path.join(data_path, f"table_values_{i}.pkl")
            save_table_values(table_values, table_values_path)

            # 添加到总列表
            all_table_values.append(table_values)

            # 输出示例
            print(f"Example tableValues for batch {i} (first simulation):")
            for j, dm_data in enumerate(table_values[0]):
                print(f"DM{j + 1}: {dm_data}")

        except Exception as e:
            print(f"Error processing batch {i}: {e}")

    # 保存所有批次的table_values
    if all_table_values:
        all_table_values_path = os.path.join(data_path, "all_table_values.pkl")
        save_table_values(all_table_values, all_table_values_path)
        print(f"\nAll table values saved to: {all_table_values_path}")

    return all_table_values


# 使用示例
if __name__ == "__main__":
    # 处理批次数据
    all_table_values = process_batch_data()

    if all_table_values:
        print(f"\nSuccessfully processed {len(all_table_values)} batches")
    else:
        print("\nNo batches were processed successfully")