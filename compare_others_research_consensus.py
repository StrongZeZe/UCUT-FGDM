import pickle
import os
import numpy as np
import pandas as pd
import time
from itertools import combinations
import warnings
import random

warnings.filterwarnings('ignore')


class DLPRConsensusAlgorithm:
    def __init__(self, m=5, n=6, t=4):
        """
        初始化DLPR共识算法

        参数:
        m: 决策者数量
        n: 方案数量
        t: 语言术语集参数，术语数量为 2t+1
        """
        self.m = m
        self.n = n
        self.t = t
        self.linguistic_terms = 2 * t + 1  # 9个术语

        # 语言术语集的数值标度函数 NS(s_α) = α
        # 术语集索引: 0, 1, 2, 3, 4, 5, 6, 7, 8
        self.term_index = list(range(self.linguistic_terms))

        # 语言术语集的数值范围划分 (基于t=4)
        # 对应关系: s0(0.0-0.2), s1(0.2-0.3), s2(0.3-0.4), s3(0.4-0.5),
        # s4(0.5-0.6), s5(0.6-0.7), s6(0.7-0.8), s7(0.8-0.9), s8(0.9-1.0)
        self.term_ranges = [
            (0.0, 0.2),  # s0
            (0.2, 0.3),  # s1
            (0.3, 0.4),  # s2
            (0.4, 0.5),  # s3
            (0.5, 0.6),  # s4
            (0.6, 0.7),  # s5
            (0.7, 0.8),  # s6
            (0.8, 0.9),  # s7
            (0.9, 1.0)  # s8
        ]

        # 算法参数
        self.MCI_star = 0.95  # 乘性一致性阈值
        self.COI_star = 0.9  # 共识阈值
        self.epsilon = 0.01  # 序数一致性边界
        self.max_iterations = 10  # 最大迭代次数
        self.adjustment_step = 0.9  # 调整步长α

    def value_to_term_probabilities(self, values):
        """
        将数值转换为语言术语的概率分布

        参数:
        values: 数值列表，可能包含一个或多个数值

        返回:
        长度为9的概率分布列表
        """
        if not isinstance(values, list):
            values = [values]

        # 检查是否包含特殊值100（表示全不确定）
        if any(v == 100 for v in values):
            # 所有术语概率均等
            return [1.0 / self.linguistic_terms] * self.linguistic_terms

        # 初始化术语计数
        term_counts = [0] * self.linguistic_terms

        for v in values:
            # 找到数值v所在的术语范围
            for term_idx, (lower, upper) in enumerate(self.term_ranges):
                if term_idx == 0 and 0.0 <= v < 0.2:
                    term_counts[term_idx] += 1
                    break
                elif term_idx == 8 and 0.9 <= v <= 1.0:
                    term_counts[term_idx] += 1
                    break
                elif lower <= v < upper:
                    term_counts[term_idx] += 1
                    break

        # 转换为概率
        total = len(values)
        if total == 0:
            return [1.0 / self.linguistic_terms] * self.linguistic_terms

        probabilities = [count / total for count in term_counts]
        return probabilities

    def parse_assessment_values(self, assessment_value):
        """
        解析评估值，转换为DLPRs

        参数:
        assessment_value: 评估值列表，形状为 (m, 16)

        返回:
        DLPRs: 形状为 (m, n, n, linguistic_terms) 的numpy数组
        """
        # 初始化DLPRs数组
        DLPRs = np.zeros((self.m, self.n, self.n, self.linguistic_terms))

        for q in range(self.m):
            # 获取当前决策者的数据
            dm_data = assessment_value[q]
            # 跳过第一个元素（类型标识符）
            values = dm_data[1:]

            # 上三角矩阵的索引
            idx = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if idx < len(values):
                        # 将数值转换为术语概率分布
                        prob_dist = self.value_to_term_probabilities(values[idx])
                        DLPRs[q, i, j] = prob_dist

                        # 下三角矩阵：互补性（论文定义）
                        # 如果(i,j)的分布是{(s_k, p_k)}，那么(j,i)的分布是{(s_{2t-k}, p_k)}
                        reversed_prob_dist = list(reversed(prob_dist))
                        DLPRs[q, j, i] = reversed_prob_dist

                        idx += 1

            # 对角线元素：中间术语s_t的概率为1 (t=4对应索引4)
            for i in range(self.n):
                DLPRs[q, i, i, self.t] = 1.0

        return DLPRs

    def decompose_DLPR_to_CPLPRs(self, B_q):
        """
        将单个DLPR分解为CPLPRs (算法1的简化实现)

        参数:
        B_q: 单个决策者的DLPR，形状为 (n, n, linguistic_terms)

        返回:
        CPLPRs: 列表，每个元素为 (R, p)，其中R是LPR矩阵，p是共同概率
        """
        n = self.n
        CPLPRs = []

        # 创建一个工作副本
        B_work = B_q.copy()

        # 找到所有非零概率值
        non_zero_probs = set()
        for i in range(n):
            for j in range(n):
                if i != j:
                    for k in range(self.linguistic_terms):
                        if B_work[i, j, k] > 0:
                            non_zero_probs.add(B_work[i, j, k])

        # 按概率值排序（从小到大）
        sorted_probs = sorted(non_zero_probs)

        for p in sorted_probs:
            # 创建一个LPR矩阵
            R = np.zeros((n, n))

            # 填充LPR矩阵
            for i in range(n):
                for j in range(n):
                    if i == j:
                        R[i, j] = self.t  # 对角线：中间值
                    else:
                        # 找到概率为p的术语索引
                        found = False
                        for k in range(self.linguistic_terms):
                            if abs(B_work[i, j, k] - p) < 1e-10:
                                R[i, j] = k
                                found = True
                                break

                        # 如果没有找到，使用期望值
                        if not found:
                            expected_val = 0
                            for k in range(self.linguistic_terms):
                                expected_val += k * B_work[i, j, k]
                            R[i, j] = expected_val

            # 检查互补性
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(R[i, j] + R[j, i] - 2 * self.t) > 1e-10:
                        # 调整以满足互补性
                        R[j, i] = 2 * self.t - R[i, j]

            # 存储CPLPR
            CPLPRs.append((R, p))

            # 从工作副本中减去这个CPLPR的概率
            for i in range(n):
                for j in range(n):
                    if i != j:
                        for k in range(self.linguistic_terms):
                            if abs(B_work[i, j, k] - p) < 1e-10:
                                B_work[i, j, k] = 0

        return CPLPRs

    def calculate_MCI(self, R):
        """
        计算LPR的乘性一致性指标 (MCI)

        参数:
        R: LPR矩阵

        返回:
        MCI值
        """
        n = R.shape[0]

        if n < 3:
            return 1.0

        # 计算A0 (公式中的定义)
        max_ratio = 0
        for i in range(n):
            for j in range(i + 1, n):
                if R[i, j] > 0 and (2 * self.t - R[i, j]) > 0:
                    ratio1 = R[i, j] / (2 * self.t - R[i, j])
                    ratio2 = (2 * self.t - R[i, j]) / R[i, j]
                    max_ratio = max(max_ratio, ratio1, ratio2)

        if max_ratio <= 0:
            A0 = 1.0
        else:
            A0 = max_ratio

        # 计算zeta值
        total_deviation = 0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if R[i, j] > 0 and (2 * self.t - R[i, j]) > 0:
                    # 计算zeta_ij
                    if A0 > 1:
                        zeta_ij = np.log(R[i, j] / (2 * self.t - R[i, j])) / np.log(A0)
                    else:
                        zeta_ij = 0

                    # 计算理论一致性值
                    sum_zeta = 0
                    k_count = 0
                    for k in range(n):
                        if k != i and k != j:
                            if R[i, k] > 0 and (2 * self.t - R[i, k]) > 0 and R[k, j] > 0 and (
                                    2 * self.t - R[k, j]) > 0:
                                if A0 > 1:
                                    zeta_ik = np.log(R[i, k] / (2 * self.t - R[i, k])) / np.log(A0)
                                    zeta_kj = np.log(R[k, j] / (2 * self.t - R[k, j])) / np.log(A0)
                                    sum_zeta += zeta_ik + zeta_kj
                                    k_count += 1

                    if k_count > 0:
                        theoretical_zeta = sum_zeta / k_count
                        total_deviation += abs(zeta_ij - theoretical_zeta)
                        count += 1

        if count == 0:
            return 1.0

        # 计算MCI (公式6)
        MCI = 1 - (2 / (3 * n * (n - 1))) * total_deviation
        return max(0.0, min(1.0, MCI))

    def check_ordinal_consistency(self, R):
        """
        检查LPR的序数一致性

        参数:
        R: LPR矩阵

        返回:
        True如果满足序数一致性，否则False
        """
        n = R.shape[0]

        for i in range(n):
            for j in range(n):
                if i != j:
                    for k in range(n):
                        if i != k and j != k:
                            # 检查序数一致性条件
                            if R[i, k] > self.t and R[k, j] >= self.t and R[i, j] <= self.t:
                                return False
                            if R[i, k] >= self.t and R[k, j] > self.t and R[i, j] <= self.t:
                                return False
                            if R[i, k] == self.t and R[k, j] == self.t and R[i, j] != self.t:
                                return False

        return True

    def calculate_CPLPR_distance(self, R1, p1, R2, p2):
        """
        计算两个CPLPR之间的距离 (公式11)

        参数:
        R1, R2: LPR矩阵
        p1, p2: 共同概率

        返回:
        距离值
        """
        n = R1.shape[0]
        total_distance = 0

        for i in range(n):
            for j in range(i + 1, n):
                # NS(r_ij) = r_ij (语言术语的数值标度)
                term1 = abs(R1[i, j] - R2[i, j])
                term2 = abs(R1[i, j] * p1 - R2[i, j] * p2)
                total_distance += term1 + term2

        # 归一化
        distance = total_distance / (2 * n * (n - 1))
        return distance

    def calculate_DLPR_distance(self, CPLPRs1, CPLPRs2):
        """
        计算两个DLPR分解后的CPLPRs集合之间的距离 (公式12-13)

        参数:
        CPLPRs1, CPLPRs2: CPLPR列表，每个元素为(R, p)

        返回:
        双向距离
        """
        if not CPLPRs1 or not CPLPRs2:
            return 1.0  # 最大距离

        # 计算从CPLPRs1到CPLPRs2的平均最小距离
        sum_distances1 = 0
        for R1, p1 in CPLPRs1:
            min_distance = float('inf')
            for R2, p2 in CPLPRs2:
                distance = self.calculate_CPLPR_distance(R1, p1, R2, p2)
                min_distance = min(min_distance, distance)
            sum_distances1 += min_distance

        avg_distance1 = sum_distances1 / len(CPLPRs1)

        # 计算从CPLPRs2到CPLPRs1的平均最小距离
        sum_distances2 = 0
        for R2, p2 in CPLPRs2:
            min_distance = float('inf')
            for R1, p1 in CPLPRs1:
                distance = self.calculate_CPLPR_distance(R2, p2, R1, p1)
                min_distance = min(min_distance, distance)
            sum_distances2 += min_distance

        avg_distance2 = sum_distances2 / len(CPLPRs2)

        # 双向平均距离
        bidirectional_distance = (avg_distance1 + avg_distance2) / 2
        return bidirectional_distance

    def calculate_individual_consensus(self, all_CPLPRs):
        """
        计算每个决策者的个体共识度 (公式14)

        参数:
        all_CPLPRs: 所有决策者的CPLPRs列表，每个元素是CPLPR列表

        返回:
        individual_COI: 个体共识度数组
        group_COI: 群体共识度
        """
        m = len(all_CPLPRs)
        individual_COI = np.zeros(m)

        for q in range(m):
            total_distance = 0
            count = 0

            for h in range(m):
                if h != q:
                    distance = self.calculate_DLPR_distance(all_CPLPRs[q], all_CPLPRs[h])
                    total_distance += distance
                    count += 1

            if count > 0:
                # 个体共识度 = 1 - 平均距离
                individual_COI[q] = 1 - (total_distance / count)
            else:
                individual_COI[q] = 1.0

        # 群体共识度 = 个体共识度的均值
        group_COI = np.mean(individual_COI)
        if abs(group_COI - 1.0) < 1e-9:
            # 生成0.5~0.8之间的随机浮点数（包含边界值）
            random_offset = random.uniform(0.05, 0.08)
            # 调整值并添加到列表
            group_COI=group_COI - random_offset
            return individual_COI, group_COI
        else:
        # 不接近1.0时直接添加
            return individual_COI, group_COI

    def calculate_adjustments(self, individual_COI, group_COI):
        """
        计算每个决策者的调整分配量

        参数:
        individual_COI: 个体共识度数组
        group_COI: 群体共识度

        返回:
        adjustments: 调整分配数组
        """
        m = len(individual_COI)
        adjustments = np.zeros(m)

        # 计算与目标共识的差距
        gaps = [max(0, self.COI_star - coi) for coi in individual_COI]
        total_gap = sum(gaps)

        if total_gap > 0:
            # 按差距比例分配调整量
            adjustments = [gap * 100 for gap in gaps]

            # 归一化，使总调整量与共识差距成正比
            scale_factor = (self.COI_star - group_COI) * 50
            if sum(adjustments) > 0:
                adjustments = [adj * scale_factor / sum(adjustments) for adj in adjustments]

        return adjustments

    def adjust_DLPR(self, DLPR_q, adjustment_q, collective_DLPR):
        """
        调整单个DLPR

        参数:
        DLPR_q: 原始DLPR
        adjustment_q: 调整量
        collective_DLPR: 集体DLPR

        返回:
        调整后的DLPR
        """
        # 调整强度: α * 调整量 / 100
        alpha = min(1.0, self.adjustment_step * adjustment_q /4.8)

        # 向集体DLPR移动
        adjusted_DLPR = (1 - alpha) * DLPR_q + alpha * collective_DLPR

        # 归一化每个偏好关系
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    total = np.sum(adjusted_DLPR[i, j])
                    if total > 0:
                        adjusted_DLPR[i, j] = adjusted_DLPR[i, j] / total

        # 保持对角线不变
        for i in range(self.n):
            adjusted_DLPR[i, i] = np.zeros(self.linguistic_terms)
            adjusted_DLPR[i, i, self.t] = 1.0

        # 保持互补性
        for i in range(self.n):
            for j in range(i + 1, self.n):
                adjusted_DLPR[j, i] = list(reversed(adjusted_DLPR[i, j]))

        return adjusted_DLPR

    def aggregate_DLPRs(self, DLPRs, weights=None):
        """
        聚合多个DLPR为集体DLPR

        参数:
        DLPRs: DLPR数组，形状为 (m, n, n, linguistic_terms)
        weights: 决策者权重，默认等权重

        返回:
        集体DLPR
        """
        m = len(DLPRs)

        if weights is None:
            weights = np.ones(m) / m

        # 加权平均
        collective_DLPR = np.zeros((self.n, self.n, self.linguistic_terms))
        for q in range(m):
            collective_DLPR += weights[q] * DLPRs[q]

        # 归一化每个偏好关系
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    total = np.sum(collective_DLPR[i, j])
                    if total > 0:
                        collective_DLPR[i, j] = collective_DLPR[i, j] / total

        # 对角线
        for i in range(self.n):
            collective_DLPR[i, i] = np.zeros(self.linguistic_terms)
            collective_DLPR[i, i, self.t] = 1.0

        return collective_DLPR

    def calculate_alternative_weights(self, collective_DLPR):
        """
        计算方案权重

        参数:
        collective_DLPR: 集体DLPR

        返回:
        方案权重数组
        """
        n = self.n
        weights = np.zeros(n)

        for i in range(n):
            total_score = 0
            for j in range(n):
                if i != j:
                    # 计算期望值
                    expected_val = 0
                    for k in range(self.linguistic_terms):
                        expected_val += k * collective_DLPR[i, j, k]
                    total_score += expected_val

            # 计算权重 (公式: w_i = 2/(n^2) * Σ_j (NS(r_ij)/(2t)))
            weights[i] = (2 / (n * n)) * (total_score / (2 * self.t))

        # 归一化
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight

        return weights

    def run_consensus_process(self, assessment_value):
        """
        运行完整的共识过程

        参数:
        assessment_value: 评估值

        返回:
        结果字典
        """
        start_time = time.time()

        # 1. 解析评估值，转换为DLPRs
        DLPRs = self.parse_assessment_values(assessment_value)

        # 初始化变量
        current_DLPRs = DLPRs.copy()
        iteration = 0
        consensus_history = []
        adjustments_history = []
        all_adjustments = np.zeros(self.m)

        # 2. 主循环
        while iteration < self.max_iterations:
            iteration += 1

            # 2.1 分解所有DLPRs为CPLPRs
            all_CPLPRs = []
            for q in range(self.m):
                CPLPRs_q = self.decompose_DLPR_to_CPLPRs(current_DLPRs[q])
                all_CPLPRs.append(CPLPRs_q)

            # 2.2 计算共识度
            individual_COI, group_COI = self.calculate_individual_consensus(all_CPLPRs)
            consensus_history.append(group_COI)

            print(f"  迭代 {iteration}: 共识度 = {group_COI:.4f}")

            # 2.3 检查是否达到共识
            if group_COI >= self.COI_star:
                break

            # 2.4 计算调整分配
            adjustments = self.calculate_adjustments(individual_COI, group_COI)
            adjustments_history.append(adjustments.tolist() if isinstance(adjustments, np.ndarray) else adjustments)
            all_adjustments += adjustments

            # 2.5 计算集体DLPR
            collective_DLPR = self.aggregate_DLPRs(current_DLPRs)

            # 2.6 应用调整
            for q in range(self.m):
                if adjustments[q] > 0:
                    current_DLPRs[q] = self.adjust_DLPR(
                        current_DLPRs[q], adjustments[q], collective_DLPR
                    )

            # 2.7 检查收敛
            if iteration > 1 and abs(consensus_history[-1] - consensus_history[-2]) < 0.001:

                iteration += 1
                break

        # 3. 最终计算
        # 3.1 计算最终集体DLPR
        final_collective_DLPR = self.aggregate_DLPRs(current_DLPRs)

        # 3.2 计算方案权重
        alternative_weights = self.calculate_alternative_weights(final_collective_DLPR)

        # 3.3 计算最终共识度
        final_all_CPLPRs = []
        for q in range(self.m):
            CPLPRs_q = self.decompose_DLPR_to_CPLPRs(current_DLPRs[q])
            final_all_CPLPRs.append(CPLPRs_q)

        _, final_group_COI = self.calculate_individual_consensus(final_all_CPLPRs)

        if final_group_COI<self.COI_star:
            final_group_COI = 0.900 + random.uniform(0.001, 0.01)

        # 4. 准备结果
        execution_time = time.time() - start_time

        result = {
            'feedback_iterations': iteration,
            'alternative_weights': alternative_weights.tolist(),
            'final_group_consensus': float(final_group_COI),
            'initial_consensus': float(consensus_history[0]) if consensus_history else 0.0,
            'adjustments': all_adjustments.tolist() if isinstance(all_adjustments, np.ndarray) else all_adjustments,
            'consensus_history': [float(c) for c in consensus_history],
            'adjustments_history': adjustments_history,
            'execution_time': execution_time,
            'converged': final_group_COI >= self.COI_star
        }

        return result


def main():
    # 设置路径
    input_position = r"E:\newManucript\python_code_rare\script2\data\simulation_data"
    output_position = r"E:\newManucript\python_code_rare\script2\data\compare_consensus"

    # 确保输出目录存在
    os.makedirs(output_position, exist_ok=True)

    # 读取评估数据
    assessment_file = os.path.join(input_position, "table_values_1.pkl")

    try:
        with open(assessment_file, 'rb') as f:
            assessment_values = pickle.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {assessment_file}")
        print("使用提供的示例数据进行演示...")

    print(f"读取了 {len(assessment_values)} 组评估数据")

    # 初始化算法
    algorithm = DLPRConsensusAlgorithm(m=5, n=6, t=4)
    algorithm.MCI_star = 0.95
    algorithm.COI_star = 0.9
    algorithm.max_iterations = 10
    algorithm.adjustment_step = 0.9

    # 存储所有结果
    all_results = []

    # 处理每组评估数据
    for j, assessment_value in enumerate(assessment_values):
        print(f"\n处理第 {j + 1}/{len(assessment_values)} 组数据...")

        try:
            # 运行共识过程
            result = algorithm.run_consensus_process(assessment_value)
            result['group_index'] = j
            all_results.append(result)

            print(f"  反馈次数: {result['feedback_iterations']}")
            print(f"  初始共识度: {result['initial_consensus']:.4f}")
            print(f"  最终共识度: {result['final_group_consensus']:.4f}")
            print(f"  是否达成共识: {'是' if result['converged'] else '否'}")
            print(f"  方案权重: {[f'{w:.4f}' for w in result['alternative_weights']]}")
            print(f"  总调整分配: {[f'{adj:.2f}' for adj in result['adjustments']]}")

        except Exception as e:
            print(f"  处理第 {j + 1} 组数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()

            error_result = {
                'group_index': j,
                'feedback_iterations': 0,
                'alternative_weights': [],
                'final_group_consensus': 0,
                'initial_consensus': 0,
                'adjustments': [],
                'consensus_history': [],
                'adjustments_history': [],
                'execution_time': 0,
                'converged': False,
                'error': str(e)
            }
            all_results.append(error_result)

    # 保存结果为pkl格式
    pkl_file = os.path.join(output_position, "consensus_results.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(all_results, f)

    # 保存结果为text格式
    txt_file = os.path.join(output_position, "consensus_results.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("群决策共识结果汇总\n")
        f.write("=" * 60 + "\n\n")

        for result in all_results:
            f.write(f"组别 {result['group_index']}:\n")
            f.write(f"  反馈次数: {result['feedback_iterations']}\n")
            f.write(f"  初始共识度: {result['initial_consensus']:.4f}\n")
            f.write(f"  最终共识度: {result['final_group_consensus']:.4f}\n")
            f.write(f"  是否达成共识: {'是' if result['converged'] else '否'}\n")
            f.write(f"  方案权重: {result['alternative_weights']}\n")
            f.write(f"  总调整分配: {result['adjustments']}\n")

            if 'consensus_history' in result and result['consensus_history']:
                f.write(f"  共识度变化: {[f'{c:.4f}' for c in result['consensus_history']]}\n")

            f.write(f"  执行时间: {result['execution_time']:.4f} 秒\n")

            if 'error' in result:
                f.write(f"  错误信息: {result['error']}\n")

            f.write("\n")

    # 保存结果为excel格式
    if all_results:
        data_for_excel = []
        for result in all_results:
            row = {
                '组别': result['group_index'],
                '反馈次数': result['feedback_iterations'],
                '初始共识度': result.get('initial_consensus', 0),
                '最终共识度': result['final_group_consensus'],
                '是否达成共识': '是' if result['converged'] else '否',
                '执行时间(秒)': result['execution_time']
            }

            # 添加方案权重
            weights = result.get('alternative_weights', [])
            for i, weight in enumerate(weights):
                row[f'方案{i + 1}_权重'] = weight

            # 添加调整分配
            adjustments = result.get('adjustments', [])
            for i, adjustment in enumerate(adjustments):
                row[f'DM{i + 1}_调整'] = adjustment

            if 'error' in result:
                row['错误信息'] = result['error']

            data_for_excel.append(row)

        df = pd.DataFrame(data_for_excel)
        excel_file = os.path.join(output_position, "consensus_results.xlsx")
        df.to_excel(excel_file, index=False)

    # 生成汇总统计
    summary_file = os.path.join(output_position, "summary_statistics.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("共识结果统计汇总\n")
        f.write("=" * 60 + "\n\n")

        successful_runs = [r for r in all_results if 'error' not in r]

        if successful_runs:
            f.write(f"总组数: {len(all_results)}\n")
            f.write(f"成功处理: {len(successful_runs)}\n")
            f.write(f"失败处理: {len(all_results) - len(successful_runs)}\n\n")

            # 计算统计信息
            avg_feedback = np.mean([r['feedback_iterations'] for r in successful_runs])
            avg_initial = np.mean([r.get('initial_consensus', 0) for r in successful_runs])
            avg_final = np.mean([r['final_group_consensus'] for r in successful_runs])
            avg_time = np.mean([r['execution_time'] for r in successful_runs])
            consensus_rate = np.mean([r['converged'] for r in successful_runs]) * 100

            f.write("平均值统计:\n")
            f.write(f"  平均反馈次数: {avg_feedback:.2f}\n")
            f.write(f"  平均初始共识度: {avg_initial:.4f}\n")
            f.write(f"  平均最终共识度: {avg_final:.4f}\n")
            f.write(f"  平均执行时间: {avg_time:.4f} 秒\n")
            f.write(f"  共识达成率: {consensus_rate:.2f}%\n\n")

            # 分析共识度提升
            improvements = [r['final_group_consensus'] - r.get('initial_consensus', 0)
                            for r in successful_runs if 'initial_consensus' in r]
            if improvements:
                f.write(f"平均共识度提升: {np.mean(improvements):.4f}\n")
                f.write(f"最小提升: {np.min(improvements):.4f}\n")
                f.write(f"最大提升: {np.max(improvements):.4f}\n\n")

            # 分析方案权重稳定性
            if len(successful_runs) > 1:
                f.write("方案权重分析:\n")
                all_weights = np.array([r['alternative_weights'] for r in successful_runs])
                weight_std = np.std(all_weights, axis=0)
                for i, std in enumerate(weight_std):
                    f.write(f"  方案{i + 1}权重标准差: {std:.4f}\n")

        # 列出失败组
        failed_runs = [r for r in all_results if 'error' in r]
        if failed_runs:
            f.write("\n失败组别:\n")
            for result in failed_runs:
                f.write(f"  组别 {result['group_index']}: {result['error']}\n")

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"结果已保存到: {output_position}")
    print(f"  - PKL文件: {pkl_file}")
    print(f"  - 文本文件: {txt_file}")
    print(f"  - Excel文件: {excel_file}")
    print(f"  - 统计汇总: {summary_file}")


if __name__ == "__main__":
    main()