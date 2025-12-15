import numpy as np
import pandas as pd
import pickle
import os
from scipy.optimize import linprog
from itertools import combinations
import warnings
from lindo import *
import lindo
import math

warnings.filterwarnings('ignore')


class PersonalizedTrustConsensus:
    def __init__(self, q=5, n=6, epsilon=0.9, max_iter=50,
                 chi1=1.0, chi2=0.5, r=0.8):
        """
        初始化个性化信任共识模型

        参数:
        q: 决策者数量
        n: 准则数量
        epsilon: 共识阈值
        max_iter: 最大迭代次数
        chi1: 信任奖励因子
        chi2: 信任惩罚因子
        r: 信任衰减趋势参数
        """
        self.q = q
        self.n = n
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.chi1 = chi1
        self.chi2 = chi2
        self.r = r

        # 存储中间结果
        self.results = {}

    def parse_preference_matrix(self, sure_Hs_single):
        """
        解析单个决策者的偏好矩阵结构

        参数:
        sure_Hs_single: 单个决策者的偏好矩阵结构

        返回:
        P_k: 完整的n×n偏好矩阵
        """
        # 初始化矩阵，对角线为0.5
        P_k = np.zeros((self.n, self.n))
        np.fill_diagonal(P_k, 0.5)

        # 解析上三角矩阵结构
        # sure_Hs_single的结构: [第一行, 第二行, 第三行, 第四行, 第五行]
        for i in range(self.n - 1):  # 0到4行
            row_data = sure_Hs_single[0][i]
            # 每行的列数: n - i - 1
            for idx, value_list in enumerate(row_data):
                j = i + idx + 1  # 列索引
                # 提取数值 (可能有嵌套列表)
                if isinstance(value_list, list):
                    if len(value_list) > 0:
                        value = float(value_list[0])
                    else:
                        value = 0.5
                else:
                    value = float(value_list)
                P_k[i, j] = value
                # 下三角通过互补性得到
                P_k[j, i] = 1 - value

        return P_k

    def load_preference_matrices(self, sure_Hs):
        """
        加载偏好矩阵（从sure_Hs结构）

        参数:
        sure_Hs: 包含所有决策者偏好矩阵的列表
        """
        self.P = []
        for k in range(self.q):
            P_k = self.parse_preference_matrix(sure_Hs[k])
            self.P.append(P_k)

        self.P = np.array(self.P)  # shape: (q, n, n)

    def load_trust_matrix(self, trust_values):
        """
        加载信任矩阵

        参数:
        trust_values: 三维数组，shape为(q, q, 2)
                     trust_values[i][j] = [信任值, 置信度]
        """
        self.trust_raw = np.array(trust_values)
        self.trust_values = self.trust_raw[:, :, 0].copy()  # 信任值
        self.confidence = self.trust_raw[:, :, 1].copy()  # 置信度

        # 确保自身信任为1，置信度为1
        for i in range(self.q):
            self.trust_values[i, i] = 1.0
            self.confidence[i, i] = 1.0

    def classify_decision_makers(self):
        """
        根据置信度对决策者进行分类
        """
        self.Con = []  # 保守型
        self.Neu = []  # 中立型
        self.Rad = []  # 激进型

        for k in range(self.q):
            # 获取决策者k对所有其他决策者的置信度
            conf_k = self.confidence[k]

            # 排除自身（对角线）
            mask = np.ones(self.q, dtype=bool)
            mask[k] = False
            conf_others = conf_k[mask]

            if np.any(conf_others < 0.5):
                self.Con.append(k)
            elif np.all(conf_others < 0.85):
                self.Neu.append(k)
            else:
                self.Rad.append(k)

        return self.Con, self.Neu, self.Rad

    def compute_trust_score(self, t, d):
        """
        计算信任得分 (公式1)

        TS(w) = (t - d)/2 + 0.5
        """
        return (t - d) / 2 + 0.5

    def propagate_trust(self, t1, d1, t2, d2):
        """
        信任传播算子 (公式2)

        参数:
        t1, d1: 第一段信任的信任度与不信任度
        t2, d2: 第二段信任的信任度与不信任度

        返回:
        t_kl, d_kl: 传播后的信任度与不信任度
        """
        t_kl = (t1 * t2) / (1 + (1 - t1) * (1 - t2) + 1e-10)
        d_kl = (d1 + d2) / (1 + d1 * d2 + 1e-10)
        return t_kl, d_kl

    def complete_trust_network(self):
        """
        构建完整的信任网络
        """
        # 初始化信任矩阵 (信任度，不信任度)
        T_complete = np.zeros((self.q, self.q, 2))

        # 复制直接信任
        for i in range(self.q):
            for j in range(self.q):
                if i != j:
                    T_complete[i, j, 0] = self.trust_values[i, j]  # 信任度
                    # 计算不信任度
                    T_complete[i, j, 1] = 1 - self.trust_values[i, j]  # 简化为1-信任值

        # 使用最短路径传播信任 (简化实现)
        # 这里使用Floyd-Warshall算法的思想
        for k in range(self.q):
            for i in range(self.q):
                for j in range(self.q):
                    if i != j and i != k and j != k:
                        # 如果i->k和k->j都有信任关系
                        if T_complete[i, k, 0] > 0 and T_complete[k, j, 0] > 0:
                            t_ik, d_ik = T_complete[i, k, 0], T_complete[i, k, 1]
                            t_kj, d_kj = T_complete[k, j, 0], T_complete[k, j, 1]
                            t_ij, d_ij = self.propagate_trust(t_ik, d_ik, t_kj, d_kj)

                            # 计算当前信任得分和新信任得分
                            current_ts = self.compute_trust_score(T_complete[i, j, 0], T_complete[i, j, 1])
                            new_ts = self.compute_trust_score(t_ij, d_ij)

                            # 如果当前没有直接信任或新信任更好，则更新
                            if T_complete[i, j, 0] == 0 or new_ts > current_ts:
                                T_complete[i, j, 0] = t_ij
                                T_complete[i, j, 1] = d_ij

        self.T_complete = T_complete

        # 计算信任得分矩阵
        self.TSM = np.zeros((self.q, self.q))
        for i in range(self.q):
            for j in range(self.q):
                if i != j:
                    self.TSM[i, j] = self.compute_trust_score(
                        T_complete[i, j, 0], T_complete[i, j, 1]
                    )

    def compute_decision_maker_weights(self):
        """
        计算决策者权重 (公式5,6)
        """
        # 计算入度
        self.ID = np.zeros(self.q)
        for l in range(self.q):
            for k in range(self.q):
                if k != l:
                    self.ID[l] += self.TSM[k, l]

        # 归一化权重
        total_ID = np.sum(self.ID)
        if total_ID > 0:
            self.NID = self.ID / total_ID
        else:
            self.NID = np.ones(self.q) / self.q

        return self.NID

    def compute_consensus_levels(self, P_current):
        """
        计算共识度

        参数:
        P_current: 当前偏好矩阵，shape为(q, n, n)

        返回:
        CL: 每个决策者的共识度
        CLE: 每个决策者每个准则对的元素共识度
        """
        # 计算元素共识度 (只考虑上三角，i < j)
        CLE = np.zeros((self.q, self.n, self.n))

        for k in range(self.q):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    sum_diff = 0
                    count = 0
                    for l in range(self.q):
                        if l != k:
                            sum_diff += 1 - abs(P_current[k, i, j] - P_current[l, i, j])
                            count += 1
                    if count > 0:
                        CLE[k, i, j] = sum_diff / count
                        CLE[k, j, i] = CLE[k, i, j]  # 对称

        # 计算决策者共识度
        CL = np.zeros(self.q)
        for k in range(self.q):
            # 只考虑上三角元素
            sum_cle = 0
            count = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    sum_cle += CLE[k, i, j]
                    count += 1
            if count > 0:
                CL[k] = sum_cle / count

        return CL, CLE

    def identify_feedback_elements(self, CL, CLE):
        """
        识别反馈元素

        返回:
        IS: 需要调整的准则对集合
        """
        # 识别不一致的决策者
        IK = [k for k in range(self.q) if CL[k] < self.epsilon]

        # 识别不一致的准则对
        IE = []
        for k in IK:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if CLE[k, i, j] < self.epsilon:
                        IE.append((k, i, j))

        # 提取需要调整的准则对
        IS = set()
        for k, i, j in IE:
            IS.add((i, j))

        return list(IS)

    def compute_group_opinion_and_median(self, P_current, IS):
        """
        计算群体意见和中位数

        参数:
        P_current: 当前偏好矩阵
        IS: 需要调整的准则对集合

        返回:
        sigma_c: 群体意见
        Med: 中位数
        phi_sup, phi_inf: 上下激励控制线
        """
        sigma_c = np.zeros((self.n, self.n))
        Med = np.zeros((self.n, self.n))
        phi_sup = np.zeros((self.n, self.n))
        phi_inf = np.zeros((self.n, self.n))

        # 计算每个准则对的群体意见
        for (i, j) in IS:
            values = []
            weighted_sum = 0

            for l in range(self.q):
                value = P_current[l, i, j]
                values.append(value)
                weighted_sum += self.NID[l] * value

            sigma_c[i, j] = weighted_sum
            sigma_c[j, i] = 1 - weighted_sum  # 互补性

            # 计算中位数
            sorted_values = sorted(values)
            if self.q % 2 == 1:
                Med[i, j] = sorted_values[self.q // 2]
            else:
                Med[i, j] = (sorted_values[self.q // 2 - 1] + sorted_values[self.q // 2]) / 2
            Med[j, i] = 1 - Med[i, j]  # 互补性

            # 计算上下激励控制线
            phi_sup[i, j] = max(sigma_c[i, j], Med[i, j])
            phi_inf[i, j] = min(sigma_c[i, j], Med[i, j])
            phi_sup[j, i] = 1 - phi_inf[i, j]  # 互补性
            phi_inf[j, i] = 1 - phi_sup[i, j]  # 互补性

        return sigma_c, Med, phi_sup, phi_inf

    def compute_incentive_amount(self, p_value, phi_sup_ij, phi_inf_ij, lambda_k):
        """
        计算激励量 (公式9,10)
        """
        mu_sup = 0
        mu_inf = 0

        if p_value > phi_sup_ij:
            mu_sup = lambda_k * (p_value - phi_sup_ij)
        elif p_value < phi_inf_ij:
            mu_inf = lambda_k * (phi_inf_ij - p_value)

        return mu_sup, mu_inf

    def build_linear_model(self, P_current, IS, phi_sup, phi_inf):
        """
        构建并求解线性规划模型

        返回:
        lambda_vals: 激励因子
        P_adjusted: 调整后的偏好矩阵
        """
        # 初始化结果
        lambda_vals = np.zeros(self.q)
        P_adjusted = P_current.copy()

        # 对每个决策者设置lambda
        for k in range(self.q):
            if k in self.Con:
                # 保守型: lambda ∈ [0, 1/3)
                lambda_vals[k] = np.random.uniform(0, 1 / 3 - 0.01)
            elif k in self.Neu:
                # 中立型: lambda ∈ [1/3, 2/3]
                lambda_vals[k] = np.random.uniform(1 / 3, 2 / 3)
            else:  # k in self.Rad
                # 激进型: lambda ∈ (2/3, 1]
                lambda_vals[k] = np.random.uniform(2 / 3 + 0.01, 1.0)

        # 应用激励规则
        for (i, j) in IS:
            for k in range(self.q):
                p_val = P_current[k, i, j]
                mu_sup, mu_inf = self.compute_incentive_amount(
                    p_val, phi_sup[i, j], phi_inf[i, j], lambda_vals[k]
                )

                # 调整偏好值
                P_adjusted[k, i, j] = p_val - mu_sup + mu_inf

                # 确保在[0,1]范围内
                P_adjusted[k, i, j] = np.clip(P_adjusted[k, i, j], 0, 1)

                # 更新互补值
                P_adjusted[k, j, i] = 1 - P_adjusted[k, i, j]

        return lambda_vals, P_adjusted

    def update_trust_evolution(self, P_old, P_new):
        """
        更新信任演化 (公式14-16调整)
        """
        TSM_new = self.TSM.copy()

        # 计算冲突度
        CD_old = np.zeros((self.q, self.q))
        CD_new = np.zeros((self.q, self.q))

        for k in range(self.q):
            for l in range(k + 1, self.q):
                # 计算旧冲突度
                diff_sum_old = 0
                count = 0
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        diff_sum_old += abs(P_old[k, i, j] - P_old[l, i, j])
                        count += 1
                if count > 0:
                    CD_old[k, l] = CD_old[l, k] = diff_sum_old / count

                # 计算新冲突度
                diff_sum_new = 0
                count = 0
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        diff_sum_new += abs(P_new[k, i, j] - P_new[l, i, j])
                        count += 1
                if count > 0:
                    CD_new[k, l] = CD_new[l, k] = diff_sum_new / count

        # 更新信任得分
        for k in range(self.q):
            for l in range(self.q):
                if k != l:
                    delta_CD = abs(CD_new[k, l] - CD_old[k, l])

                    if k in self.Con:
                        # 保守型: 信任不变
                        pass
                    elif k in self.Neu:
                        # 中立型: 信任增强
                        TSM_new[k, l] = min(1.0, TSM_new[k, l] + self.chi1 * (delta_CD ** self.r))
                    else:  # k in self.Rad
                        # 激进型: 信任减弱
                        TSM_new[k, l] = max(0.0, TSM_new[k, l] - self.chi2 * (delta_CD ** (1 - self.r)))

        self.TSM = TSM_new
        return self.TSM

    def compute_criteria_weights(self, P_final):
        """
        计算准则权重 (行平均法)

        参数:
        P_final: 最终偏好矩阵

        返回:
        w_group: 群体准则权重
        w_individual: 个体准则权重
        """
        w_individual = np.zeros((self.q, self.n))

        # 计算每个决策者的准则权重
        for k in range(self.q):
            for i in range(self.n):
                # 行平均法
                w_individual[k, i] = np.mean(P_final[k, i, :])

            # 归一化
            sum_w = np.sum(w_individual[k, :])
            if sum_w > 0:
                w_individual[k, :] = w_individual[k, :] / sum_w

        # 计算群体权重
        w_group = np.zeros(self.n)
        for i in range(self.n):
            for k in range(self.q):
                w_group[i] += self.NID[k] * w_individual[k, i]

        # 归一化群体权重
        sum_w_group = np.sum(w_group)
        if sum_w_group > 0:
            w_group = w_group / sum_w_group

        return w_group, w_individual

    def run_consensus_process(self, verbose=False):
        """
        运行共识达成过程

        参数:
        verbose: 是否打印详细信息

        返回:
        result_dict: 包含所有结果的字典
        """
        P_current = self.P.copy()
        iteration = 0
        converged = False

        # 存储每轮结果
        iteration_results = []

        while iteration < self.max_iter and not converged:
            if verbose:
                print(f"\n=== 第 {iteration + 1} 轮迭代 ===")

            # 1. 构建完整信任网络
            self.complete_trust_network()

            # 2. 计算决策者权重
            self.compute_decision_maker_weights()

            # 3. 计算共识度
            CL, CLE = self.compute_consensus_levels(P_current)
            avg_CL = np.mean(CL)

            if verbose:
                print(f"决策者共识度: {CL}")
                print(f"平均共识度: {avg_CL:.4f}")

            # 4. 检查是否达到共识
            if np.all(CL >= self.epsilon):
                if verbose:
                    print(f"达到共识! 共识阈值: {self.epsilon}")
                converged = True
                break

            # 5. 识别反馈元素
            IS = self.identify_feedback_elements(CL, CLE)
            if verbose:
                print(f"反馈元素数量: {len(IS)}")

            # 6. 计算群体意见和中位数
            sigma_c, Med, phi_sup, phi_inf = self.compute_group_opinion_and_median(P_current, IS)

            # 7. 构建并求解线性模型
            lambda_vals, P_adjusted = self.build_linear_model(P_current, IS, phi_sup, phi_inf)

            # 8. 信任演化
            self.update_trust_evolution(P_current, P_adjusted)

            # 9. 更新当前偏好矩阵
            P_old = P_current.copy()
            P_current = P_adjusted.copy()

            # 存储本轮结果
            iter_result = {
                'iteration': iteration + 1,
                'CL': CL.copy(),
                'avg_CL': avg_CL,
                'lambda': lambda_vals.copy(),
                'P': P_current.copy(),
                'TSM': self.TSM.copy(),
                'NID': self.NID.copy(),
                'IS_count': len(IS)
            }
            iteration_results.append(iter_result)

            iteration += 1

        # 最终结果
        final_iteration = iteration
        final_CL, _ = self.compute_consensus_levels(P_current)
        final_avg_CL = np.mean(final_CL)
        final_P = P_current

        # 计算准则权重
        final_weights_group, final_weights_individual = self.compute_criteria_weights(P_current)

        # 构建结果字典
        result_dict = {
            'converged': converged,
            'feedback_iterations': final_iteration,
            'final_avg_CL': final_avg_CL,
            'final_CL': final_CL,
            'final_weights_group': final_weights_group,
            'final_weights_individual': final_weights_individual,
            'final_P': final_P,
            'final_TSM': self.TSM,
            'iteration_results': iteration_results,
            'decision_maker_types': {
                'Con': self.Con,
                'Neu': self.Neu,
                'Rad': self.Rad
            }
        }

        return result_dict


def run_multiple_simulations(assessment_value, trust_value, w_true=None, epsilon=0.9, max_iter=50):
    """
    运行多次仿真

    参数:
    assessment_value: 评估值列表，每个元素包含一次仿真的数据
    trust_value: 信任矩阵
    w_true: 真实权重（可选）
    epsilon: 共识阈值
    max_iter: 最大迭代次数

    返回:
    all_results: 所有仿真结果的列表
    """
    all_results = []

    print(f"开始运行 {len(assessment_value)} 次仿真...")

    for sim_idx, assessment_data in enumerate(assessment_value):
        print(f"\n=== 第 {sim_idx + 1} 次仿真 ===")

        try:
            # 初始化模型
            model = PersonalizedTrustConsensus(
                q=5,  # 5个决策者
                n=6,  # 6个准则
                epsilon=epsilon,
                max_iter=max_iter,
                chi1=1.0,
                chi2=0.5,
                r=0.8
            )

            # 加载偏好矩阵
            cl_index, needModify, GCD, groupIndexweight, Qvalue, solves_Message, index_weight, sure_Hs, group_Hs = groupdecision(
                assessment_data, len(w_true), 0.9)
            #sure_Hs = assessment_data  # 假设assessment_data就是sure_Hs
            model.load_preference_matrices(sure_Hs)

            # 加载信任矩阵
            #model.load_trust_matrix(trust_value)
            #如果是实验2
            model.load_trust_matrix(trust_value[sim_idx])

            # 分类决策者
            model.classify_decision_makers()
            print(f"决策者分类: 保守型={model.Con}, 中立型={model.Neu}, 激进型={model.Rad}")

            # 运行共识过程
            result = model.run_consensus_process(verbose=True)

            # 添加仿真索引
            result['simulation_index'] = sim_idx + 1

            # 如果有真实权重，计算误差
            if w_true is not None:
                w_pred = result['final_weights_group']
                # 计算均方误差
                mse = sum(abs((np.array(w_true) - w_pred) ))
                result['mse_vs_true'] = mse
                print(f"权重均方误差: {mse:.6f}")

            all_results.append(result)

            print(f"仿真 {sim_idx + 1} 完成: 迭代次数={result['feedback_iterations']}, "
                  f"平均共识度={result['final_avg_CL']:.4f}")

        except Exception as e:
            print(f"仿真 {sim_idx + 1} 出错: {str(e)}")
            # 添加错误信息
            all_results.append({
                'simulation_index': sim_idx + 1,
                'error': str(e),
                'success': False
            })

    print(
        f"\n所有仿真完成。成功: {sum(1 for r in all_results if 'success' not in r or r.get('success', True))}/{len(all_results)}")

    return all_results


def save_all_results(all_results, output_dir, w_true=None):
    """
    保存所有仿真结果

    参数:
    all_results: 所有仿真结果的列表
    output_dir: 输出目录路径
    w_true: 真实权重（可选）
    """
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)

    # 准备汇总数据
    summary_data = []
    detailed_results = []

    for i, result in enumerate(all_results):
        sim_idx = i + 1

        # 如果仿真失败，跳过
        if 'error' in result:
            summary_data.append({
                'simulation': sim_idx,
                'success': False,
                'error': result['error']
            })
            continue

        # 提取关键信息
        summary = {
            'simulation': sim_idx,
            'success': True,
            'feedback_iterations': result['feedback_iterations'],
            'converged': result['converged'],
            'final_avg_CL': result['final_avg_CL'],
            'decision_maker_types': str(result['decision_maker_types'])
        }

        # 添加准则权重
        weights = result['final_weights_group']
        for j, w in enumerate(weights):
            summary[f'weight_criterion_{j + 1}'] = w

        # 如果有真实权重，计算误差
        if w_true is not None and 'mse_vs_true' in result:
            summary['mse_vs_true'] = result['mse_vs_true']

        summary_data.append(summary)
        detailed_results.append(result)

    # 1. 保存为pkl文件
    pkl_path = os.path.join(output_dir, 'all_simulation_results_random.pkl')
    save_data = {
        'all_results': detailed_results,
        'summary_data': summary_data,
        'w_true': w_true
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"所有结果已保存为pkl文件: {pkl_path}")

    # 2. 保存为text文件
    txt_path = os.path.join(output_dir, 'simulation_summary_random.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== 个性化信任共识模型 - 多次仿真汇总 ===\n\n")

        f.write(f"仿真总数: {len(all_results)}\n")
        f.write(f"成功仿真数: {sum(1 for s in summary_data if s.get('success', False))}\n\n")

        f.write("各仿真关键指标:\n")
        f.write("仿真编号 | 反馈次数 | 是否收敛 | 平均共识度 | 决策者分类\n")
        f.write("-" * 80 + "\n")

        for summary in summary_data:
            if summary.get('success', False):
                f.write(f"{summary['simulation']:4d} | {summary['feedback_iterations']:6d} | "
                        f"{'是' if summary['converged'] else '否':^8} | "
                        f"{summary['final_avg_CL']:8.4f} | "
                        f"{summary['decision_maker_types']}\n")
            else:
                f.write(f"{summary['simulation']:4d} | 失败: {summary.get('error', '未知错误')}\n")

        if w_true is not None:
            f.write("\n真实权重:\n")
            for i, w in enumerate(w_true):
                f.write(f"准则{i + 1}: {w:.4f}\n")

            # 计算平均权重和误差
            successful_results = [r for r in detailed_results if 'error' not in r]
            if successful_results:
                avg_weights = np.mean([r['final_weights_group'] for r in successful_results], axis=0)
                f.write("\n平均估计权重:\n")
                for i, w in enumerate(avg_weights):
                    f.write(f"准则{i + 1}: {w:.4f}\n")

                # 计算平均MSE
                if 'mse_vs_true' in successful_results[0]:
                    avg_mse = np.mean([r['mse_vs_true'] for r in successful_results])
                    f.write(f"\n平均均方误差(MSE): {avg_mse:.6f}\n")

    print(f"汇总结果已保存为text文件: {txt_path}")

    # 3. 保存为excel文件
    excel_path = os.path.join(output_dir, 'simulation_results_random.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 汇总表
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='仿真汇总', index=False)

        # 权重详细表
        weights_data = []
        for result in detailed_results:
            if 'error' in result:
                continue
            weights = result['final_weights_group']
            row = {'simulation': result.get('simulation_index', 0)}
            for i, w in enumerate(weights):
                row[f'criterion_{i + 1}'] = w
            weights_data.append(row)

        if weights_data:
            weights_df = pd.DataFrame(weights_data)
            weights_df.to_excel(writer, sheet_name='准则权重', index=False)

        # 共识度表
        consensus_data = []
        for result in detailed_results:
            if 'error' in result:
                continue
            cl_values = result['final_CL']
            row = {'simulation': result.get('simulation_index', 0)}
            for i, cl in enumerate(cl_values):
                row[f'DM_{i + 1}'] = cl
            row['avg_CL'] = result['final_avg_CL']
            consensus_data.append(row)

        if consensus_data:
            consensus_df = pd.DataFrame(consensus_data)
            consensus_df.to_excel(writer, sheet_name='共识度', index=False)

        # 如果有真实权重，添加比较表
        if w_true is not None:
            true_weights_df = pd.DataFrame({
                'criterion': [f'准则{i + 1}' for i in range(len(w_true))],
                'true_weight': w_true
            })
            true_weights_df.to_excel(writer, sheet_name='真实权重', index=False)

    print(f"详细结果已保存为excel文件: {excel_path}")

    return save_data

def copy_3d_structure(original):
    return [
        [[] for _ in sublist]  # 第三维：为每个子列表创建对应长度的空列表
        for sublist in original  # 第二维：遍历原始列表的每个子列表
    ]

def solveModel(tableValue,len_planses):
    #tableValue即调用的storeTableValue()函数，输出结果为linguisticList_ets。
    #groupdata是一个空数列，用于存储该函数的输出结果，[评分数据，决策者的一致性水平，需要修改的的内容构建的字典，群共识读度以及得到的各方案权重]
    #len_plans存储的是方案个数，用于遍历数据并生成上三角矩阵形式

    groupdata=[]                        #用于存储所有信息

    for i in range(len(tableValue)):  # 确定决策者个数，len(tableValue)个
        len_plans=len_planses
        # EtScoreValue=StableValue[i]                #[[评分类型][上三角评分]]
        #首先构建上三角形式的assess的数组形式，并定义变量数和初始可选元素个数
        assess=[]
        keyflag=0
        #nM, nN, objsense, objconst,reward, rhs, contype,Anz, Abegcol, Alencol, A, Arowndx,lb, ub
        #注意，由于lindo使用的是numpy，因此，后续这些数组之类的都要转换成array
        nN = len_plans  # 变量数
        Anz = len_plans  # 整个行列式中存在的变量总数（只考虑一次变量，不考虑相乘的和二次），不包括单一变量的限制（如x1>0之类的）
        #conglist = []  # 存储各个权重变量有多少个，初始为1，即必定存在w1+w2+w3=1
        #wvarialist=[]                                    #存储各权重变量的常数，以及Di的常数
        #=[]                                       # 存储各个有信息的个数，以及变量所在行数

        Zvarlist=[]                                     #用于存储模糊元权重变量所在行数
        Znumlist = []                                   #用于存储模糊元权重变量的系数
        Dvarlist=[]                                     #用于存储Di变量所在行数
        Dnumlist = []                                   #于存储Di变量的系数
        for j in range(len_plans - 1):                  # 上三角只有评估矩阵的行数-1行，故要-1
            assess_son = []
            for x in range(len_plans - j - 1):
                if (len_plans - j - 1):
                    assess_son.append(tableValue[i][x + keyflag + 1])
            keyflag = keyflag + (len_plans - j - 1)
            assess.append(assess_son)
        #list_zanwei=[]                                          #用于占位的空list

        Cvarlist = [[] for _ in range(len_plans)]               #生成方案个数的子数组的二维数组，且数组间相互独立           #用于存储各方案权重变量所在行数
        Cnumlist = [[] for _ in range(len_plans)]               #存储行列式中方案权重的系数

        #构建一个矩阵，用来存储矩阵模糊元个数
        hesist_eli_num=copy_3d_structure(assess)
        #输出只与assess相同结构的矩阵



        nM = 0  # 记录整个行列式的行数
        objsense = lindo.LS_MIN  # 代表优化方向，是寻最大值还是最小值   寻最大值为-1，寻最小值为1，LS_MIN和LS_MAX是预定义宏，可以直接用于求解最大值或是最小值
        objconst = 0  # 目标函数中的常数项存储在一个双标量中
        reward = [0.0 for _ in range(len_plans)]  # 最优化函数的参数系数，即目标函数系数       需要将所有变量考虑在内,首先方案权重系数为0.0
        rhs = []                # 行列式的的右侧约束条件
        contype = []  # E代表=，G代表大于等于，L代表小于等于
        Abegcol = [0]  # 代表模糊元权重的变量都为+1，代表权重的变量，如果是犹豫模糊集，则行+2，如果不是，则行列+2，最后所有权重变量皆+1，代表D的变量都为+2  ,最后一个数值一定等于ANZ，用于记录每列的变量数，只是用累加记录
        #用来计算索引的，从0开始算，比如第一列有2个非0系数变量，第二列有3个，第三列有2个，第四列有2个，则Abegcol=[0,2,5,7,9]
        Alencol = np.asarray(None)                           # 定义了约束矩阵中每一列的长度。在本例中，这被设置为None，因为在矩阵中没有留下空白

        # 前面模糊元个数个的元素都为1.0，后面根据其行上格列所在元素是否为犹豫模糊集判定，最后一个为1，所有D的变量都为-1.0
        A = []                                   #记录行列式中各个变量的系数，同样是一列一列开始加入
        Arowndx = []                             # 记录第几行存在变量，依然从第一列开始数（从9开始计数），第一列数完数第二列，即行列式中非零系数对应的行索引
        #lb为单变量的约束条件下界，Ub为上界
        lb = [0.0 for _ in range(len_plans)]                     # 除了D变量为-LSconst.LS_INFINITY，其它都为0.0,现在是先将各个方案的权重限制加入，已知方案个数
        ub = [1.0 for _ in range(len_plans)]                     # 除了D变量为-LSconst.LS_INFINITY，其它都为1.0,现在是先将各个方案的权重限制加入，已知方案个数
        lb_Z = []                                               #Di以及各模糊元权重的上下限
        ub_Z = []
        lb_D = []
        ub_D = []

        # 确定信息
        identify_DATA = [[[0] for _ in range(len_plans-1)],[[0] for _ in range(len_plans-1)]]                  # 我这里用来观察矩阵中是否会出现某一行或某一列全为不确定信息
        hesistantitem = 0  # 存储犹豫模糊元的个数
        pricise = 0  # 存储精确信息个数
        noneInf = 0  # 存储未知信息个数
        congindex = 0

        # 二次矩阵，因为存在模糊元权重与方案权重的乘法，故存在二次矩阵需要处理
        qNZ = 0  # qNZ代表限定语句中二次变量存在的变量个数，注意，它只算上三角矩阵中的变量个数
        # （即由行列元素分别为[w1,w2,w3,...,z1,z2,z3,...,D1,D2,...]构建的矩阵中的上三角变量，包括对角线上的元素，同时，因为它是对称的，如果存在比如2w1^21,则其系数为4，而2w1*w2系数依然为2）
        qNZV = []  # 存储各二次变量的系数，从每行开始算，一次读行后连接，这就是模糊元个数乘4中，前2*num（模糊元）为value（模糊元），后2*num（模糊元）为-value（模糊元）

        qRowX = []  # 存储二次矩阵中变量所在的行索引;  行索引

        qColumnX = []  # 存储二次矩阵中每行第几列存在变量；然后每行约束函数可以生成一个二次矩阵   列索引
        qCI = []            # 存储约束函数中，确定这是第几个二次矩阵，-1代表目标函数的二次矩阵，依次往下数，我们的函数中，所有包含犹豫模糊集的行都是一个二次矩阵，依次往下数即可。其实也就是所有包含二次项的行索引

        z_num=len_plans                #用于查看z权重变量到了哪里

        #下面代码构建的模型排序方法是：先每行遍历上三角矩阵，再遍历所有模糊元权重之和=1，再加上最后的方案权重之和=1
        if tableValue[i][0] == 'HFPR' or tableValue[i][0] == 'HFLPR':
            for j in range(len(assess)):  # 遍历上三角矩阵                 按行遍历上三角矩阵
                congindex = congindex + 1
                for hesisitem in range(len(assess[j])):  # 进入各个犹豫模糊集

                    hesist_eli_num[j][hesisitem].append(len(assess[j][hesisitem]))

                    if len(assess[j][hesisitem]) > 1:  # 若为犹豫模糊集，存在二次项
                        nM = nM + 2  # 如果是犹豫模糊集，则行数会多3，即本来的两行，加上由于模糊元权重那一行
                        nN = nN + len(assess[j][hesisitem])+1  # 加上模糊元权重变量和Di变量
                        Anz = Anz + 4 + len(assess[j][hesisitem])  # 因为存在的模糊元权重都与方案权重变量相乘变为二次项了，因此这里犹豫模糊集只+4，即两个方案权重+两个Di,再加上模糊元变量，这里加的是后面SUM(zi)=1的变量！！
                        Cvarlist[j].append(nM-2)
                        Cvarlist[j].append(nM-1)                      # 存储权重变量所在的行数
                        Dvarlist.append([nM-2,nM-1])                 #存储Di变量所在行
                        Cnumlist[j].append(-1.0)                                  #添加w1的系数
                        Cnumlist[j].append(1.0)                                   #添加w1的系数
                        Dnumlist.append([-1.0,-1.0])                                #添加D1的系数
                        Zlist=[]                                                    #用于存储在该犹豫模糊集中的各个模糊元在模型中的系数
                        for Znum in range(len(assess[j][hesisitem])):
                            reward.append(0.0)                                      #所有模糊元权重对目标函数的影响为0，故系数为0.0
                            Zlist.append(1.0)
                            lb_Z.append(0.0)                                        #用于存储在该犹豫模糊集中的各个模糊元的下限
                            ub_Z.append(1.0)                                        #用于存储在该犹豫模糊集中的各个模糊元的上线
                        Znumlist.append(Zlist)                                  #添加模糊元的系数
                        for i in range(2):
                            rhs.append(0.0)
                            contype.append('L')
                        lb_D.append(-LSconst.LS_INFINITY)                                            #Di的上限
                        ub_D.append(LSconst.LS_INFINITY)                                            #Di的下限

                        #构建二次项需要的数据，首先明确是第几行,这里的行就是nM-2与nM-1，注意每一行都会构建一个二次矩阵，n为方案权重+模糊元变量+Di变量的合
                        #先添加((c1*z1+c2*z2+...)-1)*w1
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(j)                          #二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num+product)     #二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(assess[j][hesisitem][product])              #系数
                            qCI.append(nM-2)                                        #二次矩阵所在行
                            qNZ=qNZ+1                                               #增一个二次项个数
                        #后添加((c1*z1+c2*z2+...)-1)*wn
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(hesisitem+j+1)  # 二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num + product)  # 二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(assess[j][hesisitem][product])  # 系数
                            qCI.append(nM - 2)  # 二次矩阵所在行
                            qNZ = qNZ + 1  # 增一个二次项个数
                        #添加第二行的-((c1*z1+c2*z2+...)-1)*w1
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(j)                          #二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num+product)     #二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(-assess[j][hesisitem][product])              #系数
                            qCI.append(nM-1)                                        #二次矩阵所在行
                            qNZ = qNZ + 1  # 增一个二次项个数
                        #后添加((c1*z1+c2*z2+...)-1)*wn
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(hesisitem+j+1)  # 二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num + product)  # 二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(-assess[j][hesisitem][product])  # 系数
                            qCI.append(nM - 1)  # 二次矩阵所在行
                            qNZ = qNZ + 1  # 增一个二次项个数

                        z_num=z_num+len(assess[j][hesisitem])         #此模糊集的权重变量已经全部索引完毕

                    else:                           #若不为犹豫模糊集
                        nM = nM + 2
                        nN = nN + 1                 #只有Di变量
                        Anz = Anz+6                 #两行，每行两个方案权重变量+一个Di变量
                        # 存储行权重变量所在的行数
                        Cvarlist[j].append(nM - 2)
                        Cvarlist[j].append(nM - 1)
                        #存储列权重变量所在的行数
                        Cvarlist[hesisitem+j+1].append(nM - 2)
                        Cvarlist[hesisitem+j+1].append(nM - 1)
                        # 存储行权重变量的系数
                        Cnumlist[j].append(round(assess[j][hesisitem][0]-1,3))
                        Cnumlist[j].append(round(-assess[j][hesisitem][0]+1,3))
                        #存储列权重变量所在的行数
                        Cnumlist[hesisitem+j+1].append(round(assess[j][hesisitem][0],3))              #防止偏差，保留三位小数
                        Cnumlist[hesisitem+j+1].append(round(-assess[j][hesisitem][0],3))

                        Dvarlist.append([nM - 2, nM - 1])  # 存储Di变量所在行
                        Dnumlist.append([-1.0, -1.0])  # 添加D1的系数
                        lb_D.append(-LSconst.LS_INFINITY)                                            #Di的上限
                        ub_D.append(LSconst.LS_INFINITY)                                            #Di的下限
                        for i in range(2):
                            rhs.append(0.0)
                            contype.append('L')

            #全部遍历完成再遍历一次，找到各个模糊元所在行数
            for j in range(len(assess)):  # 遍历上三角矩阵                 按行遍历上三角矩阵
                for hesisitem in range(len(assess[j])):  # 进入各个犹豫模糊集
                    if len(assess[j][hesisitem]) > 1:  # 若为犹豫模糊集
                        for Znum in range(len(assess[j][hesisitem])):
                            Zvarlist.append(nM)                         #因为lindo的行数是从0开始算的
                        nM=nM+1                                         #加上该犹豫模糊集中的权重之和=1那一行
                        rhs.append(1.0)
                        contype.append('E')

            #最后添加各方案权重之和=1
            for j in range(len_plans):
                Cnumlist[j].append(1.0)
                Cvarlist[j].append(nM)

            #各方案权重之和=1，即行列式构成的矩阵的最后一行
            nM=nM+1
            rhs.append(1.0)
            contype.append('E')

            #汇总Abegcol，Arowndx以及A,构建lb,ub以及reward
            #排列形式为[方案权重集，模糊元权重集，Di]

            #lb_Z,ub_Z
            for znum in range(len(lb_Z)):
                lb.append(lb_Z[znum])
                ub.append(ub_Z[znum])

            #lb_D,ub_D
            for dnum in range(len(lb_D)):
                lb.append(lb_D[dnum])
                ub.append(ub_D[dnum])

            #Abegcol与Arowndx,A矩阵（即所有变量的系数），reward目标函数
            #第一步：整合方案权重变量的行数，并计算它的Abegcol，并整合方案权重变量的系数
            for j in range(len(Cvarlist)):
                for Cweight in range(len(Cvarlist[j])):
                    Arowndx.append(Cvarlist[j][Cweight])
                    A.append(Cnumlist[j][Cweight])
                Abegcol.append(Abegcol[len(Abegcol)-1]+len(Cvarlist[j]))
            #第二步：整合模糊元权重变量的行数，并计算它的Abegcol,Zvarlist是以一维变量的存储形式存储的，但Znumlist以二维
            #***********************************************************************
            for j in range(len(Zvarlist)):
                Arowndx.append(Zvarlist[j])
                Abegcol.append(Abegcol[len(Abegcol) - 1] + 1)  # 因为模糊元权重变量所属的每一列只有一个变量
            for j in range(len(Znumlist)):
                for Zweight in range(len(Znumlist[j])):
                    A.append(Znumlist[j][Zweight])
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #第三步：整合方案权重变量的行数，并计算它的Abegcol,且len(Dvarlist)表示构建了多少个Di变量,Dvarlist则是以二维变量的形式存储，整合Di权重变量的系数进入A，并构建reward目标函数
            for j in range(len(Dvarlist)):
                for Dweight in range(len(Dvarlist[j])):
                    Arowndx.append(Dvarlist[j][Dweight])
                    A.append(Dnumlist[j][Dweight])
                Abegcol.append(Abegcol[len(Abegcol)-1]+len(Dvarlist[j]))
                #由于最终目标函数为sum(Di)，故可以构建reward
                reward.append(1.0)



            #构建由二次变量组成的矩阵：行向量：[w1,w2,w3,w4...,z1,z2,z3,z4,...,D1,D2,D3,....]
            #print(A)
        #tableValue[i][0] == 'i-HFPR' or tableValue[i][0] == 'i-HFLPR':,这时候最主要就是D变量只剩下一个了，且存在不需要处理的无信息区域，即拿100代替的部分
        else:

            for j in range(len(assess)):  # 遍历上三角矩阵                 按行遍历上三角矩阵
                congindex = congindex + 1
                for hesisitem in range(len(assess[j])):  # 进入各个犹豫模糊集

                    hesist_eli_num[j][hesisitem].append(len(assess[j][hesisitem]))

                    if len(assess[j][hesisitem]) > 1:  # 若为犹豫模糊集
                        nM = nM + 2  # 如果是犹豫模糊集，则行数会多3，即本来的两行，加上由于模糊元权重那一行
                        nN = nN + len(assess[j][hesisitem])  # 加上模糊元权重变量，唯一的偏差变量放在最后面加
                        Anz = Anz + 4 + len(assess[j][hesisitem])  # 因为存在的模糊元权重都与方案权重变量相乘变为二次项了，因此这里犹豫模糊集只+4，即两个方案权重+两个Di,再加上模糊元变量，这里加的是后面SUM(zi)=1的变量！！
                        Cvarlist[j].append(nM-2)
                        Cvarlist[j].append(nM-1)                      # 存储权重变量所在的行数
                        Dvarlist.append([nM-2,nM-1])                 #存储D变量所在行
                        Cnumlist[j].append(-1.0)                                  #添加w1的系数
                        Cnumlist[j].append(1.0)                                   #添加w1的系数
                        Dnumlist.append([-1.0,-1.0])                                #添加D的系数
                        Zlist=[]                                                    #用于存储在该犹豫模糊集中的各个模糊元在模型中的系数
                        for Znum in range(len(assess[j][hesisitem])):
                            reward.append(0.0)                                      # 所有模糊元权重对目标函数的影响为0，故系数为0.0
                            Zlist.append(1.0)
                            lb_Z.append(0.0)                                        #用于存储在该犹豫模糊集中的各个模糊元的下限
                            ub_Z.append(1.0)                                        #用于存储在该犹豫模糊集中的各个模糊元的上线
                        Znumlist.append(Zlist)                                  #添加模糊元的系数
                        for i in range(2):
                            rhs.append(0.0)
                            contype.append('L')
                        identify_DATA[0][j][0]=identify_DATA[0][j][0]+1                         #若不为inf。则对应行+1
                        identify_DATA[1][hesisitem+j][0]=identify_DATA[1][hesisitem+j][0]+1     ##若不为inf。则对应列+1

                        #构建二次项需要的数据，首先明确是第几行,这里的行就是nM-2与nM-1，注意每一行都会构建一个二次矩阵，n为方案权重+模糊元变量+Di变量的合
                        #先添加((c1*z1+c2*z2+...)-1)*w1
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(j)                          #二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num+product)     #二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(assess[j][hesisitem][product])              #系数
                            qCI.append(nM-2)                                        #二次矩阵所在行
                            qNZ=qNZ+1                                               #增一个二次项个数
                        #后添加((c1*z1+c2*z2+...)-1)*wn
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(hesisitem+j+1)  # 二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num + product)  # 二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(assess[j][hesisitem][product])  # 系数
                            qCI.append(nM - 2)  # 二次矩阵所在行
                            qNZ = qNZ + 1  # 增一个二次项个数
                        #添加第二行的-((c1*z1+c2*z2+...)-1)*w1
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(j)                          #二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num+product)     #二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(-assess[j][hesisitem][product])              #系数
                            qCI.append(nM-1)                                        #二次矩阵所在行
                            qNZ = qNZ + 1  # 增一个二次项个数
                        #后添加((c1*z1+c2*z2+...)-1)*wn
                        for product in range(len(assess[j][hesisitem])):
                            qRowX.append(hesisitem+j+1)  # 二次矩阵中对应该方案权重所在的行
                            qColumnX.append(z_num + product)  # 二次矩阵中对应该模糊元权重所在的列
                            qNZV.append(-assess[j][hesisitem][product])  # 系数
                            qCI.append(nM - 1)  # 二次矩阵所在行
                            qNZ = qNZ + 1  # 增一个二次项个数

                        z_num=z_num+len(assess[j][hesisitem])         #此模糊集的权重变量已经全部索引完毕

                    elif assess[j][hesisitem][0]==100:                  #assess[j][hesisitem]是个list
                        continue
                    else:                           #若不为犹豫模糊集
                        nM = nM + 2
                        Anz = Anz+6                 #两行，每行两个方案权重变量+一个Di变量
                        # 存储行权重变量所在的行数
                        Cvarlist[j].append(nM - 2)
                        Cvarlist[j].append(nM - 1)
                        #存储列权重变量所在的行数,非模糊集部分的权重变量未与方案权重变量结合成二次变量
                        Cvarlist[hesisitem+j+1].append(nM - 2)
                        Cvarlist[hesisitem+j+1].append(nM - 1)
                        # 存储行权重变量的系数
                        Cnumlist[j].append(round(assess[j][hesisitem][0]-1,3))
                        Cnumlist[j].append(round(-assess[j][hesisitem][0]+1,3))
                        #存储列权重变量所在的行数,非模糊集部分的权重变量未与方案权重变量结合成二次变量
                        Cnumlist[hesisitem+j+1].append(round(assess[j][hesisitem][0],3))              #防止偏差，保留三位小数
                        Cnumlist[hesisitem+j+1].append(round(-assess[j][hesisitem][0],3))
                        Dvarlist.append([nM - 2, nM - 1])  # 存储Di变量所在行
                        Dnumlist.append([-1.0, -1.0])  # 添加D1的系数
                        identify_DATA[0][j][0]=identify_DATA[0][j][0]+1                         #若不为inf。则对应行+1
                        identify_DATA[1][hesisitem+j][0]=identify_DATA[1][hesisitem+j][0]+1     ##若不为inf。则对应列+1
                        for i in range(2):
                            rhs.append(0.0)
                            contype.append('L')

            #验证矩阵中是否存在某行或某列都为不完全信息，若都为，则强行终止，提示让决策者重新评分
            for j in range(len(identify_DATA)):
                for ident in range(len(identify_DATA[j])):
                    if identify_DATA[j][ident][0]==0:
                        if j==0:
                            print('评分格式错误，第{0}行全为空，无法获得准确结果'.format(ident+1))
                        else:
                            print('评分格式错误，第{0}列全为空，无法获得准确结果'.format(ident + 1))
                        sys.exit(1)

            #全部遍历完成再遍历一次，找到各个模糊元所在行数
            for j in range(len(assess)):  # 遍历上三角矩阵                 按行遍历上三角矩阵
                for hesisitem in range(len(assess[j])):  # 进入各个犹豫模糊集
                    if len(assess[j][hesisitem]) > 1:  # 若为犹豫模糊集
                        for Znum in range(len(assess[j][hesisitem])):
                            Zvarlist.append(nM)                         #因为lindo的行数是从0开始算的
                        nM=nM+1                                         #加上该犹豫模糊集中的权重之和=1那一行
                        rhs.append(1.0)
                        contype.append('E')

            #最后添加各方案权重之和=1
            for j in range(len_plans):
                Cnumlist[j].append(1.0)     #系数
                Cvarlist[j].append(nM)

            # 各方案权重之和=1，即行列式构成的矩阵的最后一行
            nM = nM + 1
            rhs.append(1.0)                 #右边约束条件
            contype.append('E')
            nN=nN+1                         #加上唯一的偏差变量

            reward.append(1.0)  # 目标函数就为唯一偏差变量D

            #汇总Abegcol，Arowndx以及A,构建lb,ub以及reward
            #排列形式为[方案权重集，模糊元权重集，Di]

            #lb_z,ub_z
            for znum in range(len(lb_Z)):
                lb.append(lb_Z[znum])
                ub.append(ub_Z[znum])

            #D
            lb.append(-LSconst.LS_INFINITY)
            ub.append(LSconst.LS_INFINITY)

            #Abegcol与Arowndx,A矩阵（即所有变量的系数），reward目标函数
            #第一步：整合方案权重变量的行数，并计算它的Abegcol，并整合方案权重变量的系数
            for j in range(len(Cvarlist)):
                for Cweight in range(len(Cvarlist[j])):
                    Arowndx.append(Cvarlist[j][Cweight])
                    A.append(Cnumlist[j][Cweight])
                Abegcol.append(Abegcol[len(Abegcol)-1]+len(Cvarlist[j]))
            #第二步：整合模糊元权重变量的行数，并计算它的Abegcol,Zvarlist是以一维变量的存储形式存储的，但Znumlist以二维
            #***********************************************************************
            for j in range(len(Zvarlist)):
                Arowndx.append(Zvarlist[j])
                Abegcol.append(Abegcol[len(Abegcol) - 1] + 1)  # 因为模糊元权重变量所属的每一列只有一个变量
            for j in range(len(Znumlist)):
                for Zweight in range(len(Znumlist[j])):
                    A.append(Znumlist[j][Zweight])
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #第三步：整合方案权重变量的行数，并计算它的Abegcol,,Dvarlist则是以二维变量的形式存储，整合Di权重变量的系数进入A，并构建reward目标函数
            for j in range(len(Dvarlist)):
                for Dweight in range(len(Dvarlist[j])):
                    Arowndx.append(Dvarlist[j][Dweight])
                    A.append(Dnumlist[j][Dweight])
            Abegcol.append(Anz)                         #只有一列

        #将数组将换位numpy
        rhs = np.array(rhs, dtype=np.double)  # 可被lindo执行的语句
        reward = np.array(reward, dtype=np.double)  # 可被lindo执行的语句
        contype = np.array(contype, dtype=np.character)       #转换成了字节型

        Abegcol = np.array(Abegcol, dtype=np.int32)

        A=np.array(A, dtype=np.double)

        Arowndx = np.array(Arowndx, dtype=np.int32)
        lb = np.array(lb, dtype=np.double)
        ub = np.array(ub, dtype=np.double)

        # 二次矩阵

        qCI = np.array(qCI, dtype=np.int32)
        qNZV = np.array(qNZV, dtype=np.double)
        qRowX = np.array(qRowX, dtype=np.int32)
        qColumnX = np.array(qColumnX, dtype=np.int32)
        # create LINDO environment and model objects
        LicenseKey = np.array('', dtype='S1024')
        LicenseFile = os.getenv("LINDOAPI_LICENSE_FILE")                #'C:\\Lindoapi\\license\\lndapi90.lic'
        if LicenseFile == None:
            print('Error: Environment variable LINDOAPI_LICENSE_FILE is not set')
            sys.exit(1)

        lindo.pyLSloadLicenseString(LicenseFile, LicenseKey)
        pnErrorCode = np.array([-1], dtype=np.int32)
        pEnv = lindo.pyLScreateEnv(pnErrorCode, LicenseKey)

        pModel = lindo.pyLScreateModel(pEnv, pnErrorCode)
        geterrormessage(pEnv, pnErrorCode[0])

        # load data into the model
        #print("Loading LP data...")
        # 针对不同的，这个LP是解线性的，NLP解非线性
        # 通过调用LSloadLPData（），将问题结构和线性数据加载到模型结构中。            #contype
        lindo.pyLSloadLPData(pModel, nM, nN, objsense, objconst,
                                         reward, rhs, contype,
                                         Anz, Abegcol, Alencol, A, Arowndx,
                                         lb, ub)
        #geterrormessage(pEnv, errorcode)  # 检测

        lindo.pyLSloadQCData(pModel, qNZ, qCI, qRowX, qColumnX, qNZV)  # 二次
        #geterrormessage(pEnv, errorcode)  # 检测

        # solve the model
        #print("Solving the model...")
        #print("Solving the model...")
        pnStatus = np.array([-1], dtype=np.int32)
        lindo.pyLSoptimize(pModel, LSconst.LS_METHOD_FREE,
                                       pnStatus)  # 通过调用LSoptimize（）（或如果有整数变量的LSsolveMIP（））来解决这个问题。     使用障碍求解器
        # errorcode = lindo.pyLSoptimize(pModel,lindo.LS_METHOD_NLP,pnStatus)
        #geterrormessage(pEnv, errorcode)  # 检测

        # retrieve the objective value       获取最优结果
        dObj = np.array([-1.0], dtype=np.double)
        lindo.pyLSgetInfo(pModel, LSconst.LS_DINFO_POBJ,
                                      dObj)  # 通过调用LSgetInfo（）、LSget初级解决方案（）和LSget双元解决方案（）来检索解决方案。
        #geterrormessage(pEnv, errorcode)
        #print("Objective is: %.5f" % dObj[0])
       # print("")

        # retrieve the primal solution       获取变量数值
        padPrimal = np.empty((nN), dtype=np.double)
        lindo.pyLSgetPrimalSolution(pModel, padPrimal)
        #geterrormessage(pEnv, errorcode)
        #print("Primal solution is: ")
        #for x in padPrimal: print("%.5f" % x)

        # delete LINDO model pointer
        lindo.pyLSdeleteModel(pModel)  # 通过调用LSdeleteModel，LSdeleteEnv（）来删除模型和环境。
        #geterrormessage(pEnv, errorcode)

        # delete LINDO environment pointer
        lindo.pyLSdeleteEnv(pEnv)
        # geterrormessage(pEnv, errorcode)

        groupdata_son=[]
        for j in range(len_plans):
            groupdata_son.append(padPrimal[j])

        # 获取各个犹豫模糊集的权重，获取各个方案的权重，获取总偏差f,以及通过权重修正后的矩阵,初始矩阵中每个偏好评分的模糊元个数，矩阵中的不完全信息个数
        groupdata_son=[[] for _ in range(6)]

        sure_D= copy_3d_structure(assess)

        for j in range(len_plans):                 #
            groupdata_son[1].append(padPrimal[j])                           #获取各个方案的权重

        unINf=0                                                             #存储矩阵中的不完全信息个数
        index_=0
        for index1 in range(len(hesist_eli_num)):
            for index2 in range(len(hesist_eli_num[index1])):
                grouphesstindex = 0
                if hesist_eli_num[index1][index2][0]>1:                    #如果是模糊集
                    sons=[]
                    for index3 in range(hesist_eli_num[index1][index2][0]):
                        sons.append(padPrimal[len_plans+index_])
                        grouphesstindex=grouphesstindex+assess[index1][index2][index3]*padPrimal[len_plans+index_]                  #通过模糊元权重向量将模糊集去模糊
                        index_ = index_ + 1
                    sure_D[index1][index2].append(grouphesstindex)
                    groupdata_son[0].append(sons)                                   #获取各个犹豫模糊集的权重
                else:
                    if assess[index1][index2][0]==100:                              #如果为空信息，通过乘性算法求解结果
                        sure_D[index1][index2].append(padPrimal[index1]/(padPrimal[index1]+padPrimal[index2]))
                        unINf=unINf+1
                    else:
                        sure_D[index1][index2].append(assess[index1][index2][0])


        groupdata_son[2].append(dObj[0])                                        #总偏差f
        groupdata_son[3].append(sure_D)                                         #去犹豫的矩阵
        groupdata_son[4].append(hesist_eli_num)                                 #初始矩阵中每个偏好评分的模糊元个数
        groupdata_son[5].append(unINf)                                          #存储矩阵中的不完全信息个数
        groupdata.append(groupdata_son)

    return groupdata                                                            #用于传输修改后矩阵的权重，获取各个变量的权重，获取总偏差f

#初步解决group的H-HFPRs
def groupdecision(tableValues,len_planses,GCI_yvzhi):                             #存储表格数据的数列
    #tableValue=['i-HFLPR', [0.61, 0.72], [0.5], [0.06, 0.28], [100], [0.28], [100], [0.72], [0.28, 0.39, 0.61], [0.72, 0.83], [0.94]]
    #self.tableValue=['i-HFLPR', [[0.61, 0.72], [0.5], [0.06, 0.28], [100], [0.28], [100], [0.72], [0.28, 0.39, 0.61], [0.72, 0.83], [0.94]]]
    #self.groupdata=[[决策者1：[模糊元权重],[方案权重],[偏差，即object]],[决策者2],...]=solveModel(tableValues, len_planses)的第一维每组的前三小组
    #确定矩阵
    # 用于存储S的所有评分[[（专家1评分）评价形式，[上半矩阵评分],[下半矩阵评分]],[专家2评分],...[专家n评分]]
    # 用于存储S的所有专家偏差信息[[[犹豫模糊集1各模糊元权重，犹豫模糊集2各模糊元权重..],[各指标权重]，[总偏差]],[专家2],...[专家n]]
    # 创建输出抑制上下文管理器

    try:
        solves_Message = solveModel(tableValues, len_planses)                   #存储处理后的信息                  #solve_Message的长度应和tableValues一样，除非报错

        #恢复原有代码
        tableValue=[]                                                                #存储需要的信息
        for i in range(len(tableValues)):
            tablev1=[]
            tablev1.append(tableValues[i][0])
            tablev2=[]
            for j in range(1,len(tableValues[i])):
                tablev2.append(tableValues[i][j])
            tablev1.append(tablev2)
            tableValue.append(tablev1)
        grouptable=[]                                                           #群共识矩阵
        cl=[]                                                                   #一致水平
        #misinfoNumdist= {}                                                      #存储第几个矩阵不完全，且不完全个数
        indexNum = len_planses                                       #按正常解决方案来说，一次处理，indexNum不会变化
        for i in range(len(tableValue)):
            hesistCl=0                                                                              #存储犹豫模糊集中的不一致水平
            grouptableitem=[tableValue[i][0],solves_Message[i][3]]                                 ##使其类似于tablevalue的评价形式  ['评分形式',[第一个专家的确切评分]]

            #计算hesistCL
            hesistCllist=[]
            for j in solves_Message[i][3][0]:
                for m in j:
                    hesistCllist.append(m[0])
            #[0.17, 0.39, 0.3059884545130745, 0.61, 0.7656601880483732, 0.5, 0.94, 0.39, 0.5018879693579759, 0.94]
            list_dif=[]
            for j in range(len(hesistCllist)):
                val1=hesistCllist[j]
                val2=tableValues[i][j+1]
                if 100 in val2:
                    diff_to_0 = abs(val1 - 0)
                    diff_to_1 = abs(val1 - 1)
                    min_diff = min(diff_to_0, diff_to_1)
                    list_dif.append(min_diff)
                else:
                    min_diff = min(abs(val1 - val) for val in val2)
                    list_dif.append(min_diff)
            hesistCl=sum(list_dif)

            #solves_Message[i][3]的形式是上三角矩阵的形式

            grouptable.append(grouptableitem)                                          #添加各专家的评分矩阵


            if solves_Message[i][5][0]==0:                                              #不存在不完全信息
                cl.append([i,1-2*(hesistCl+solves_Message[i][2][0])/(indexNum*(indexNum-1))])
            else:                                                                               #solves_Message[i][5][0]为第i个决策者矩阵中的不完全信息个数
                cl.append([i, 1-solves_Message[i][2][0]-2 * (hesistCl ) / (indexNum * (indexNum - 1)-2*solves_Message[i][5][0])])              #i是为了记录这是哪一个决策者的，用于后续排序后依然能够确定

        #给cl排序
        Qvalue=[]                                                               #存储通过I-IOWA算子确定各专家权重
        CL=0                                                                    #存储总偏差
        for i in range(len(tableValue)):
            Qvalue.append(0)
            for j in range(len(tableValue)-i-1):
                if cl[j+1][1]>cl[j][1]:
                #if cl[j + 1][1] < cl[j][1]:                                     #如果不做群共识，用cl作为评权重是极好的，只不过要从小到大排序，但是若要做群共识，那么
                    middle=cl[j]
                    cl[j]=cl[j+1]
                    cl[j+1]=middle
        for i in range(len(tableValue)): CL=CL+cl[i][1]
        # 确定各专家权重
        PreQ = 0
        Precl=0
        for i in range(len(tableValue)):
            if i==0:
                Qvalue[cl[i][0]]=math.pow((cl[i][1]/CL),0.9)
                PreQ = PreQ + Qvalue[cl[i][0]]
                Precl=Precl+cl[i][1]
            elif i==len(tableValue)-1:
                Qvalue[cl[i][0]]=1-PreQ
            else:
                Qvalue[cl[i][0]]=math.pow((Precl+cl[i][1])/CL,0.9)-PreQ
                PreQ = PreQ + Qvalue[cl[i][0]]
                Precl = Precl + cl[i][1]

        # 并建立群体元素权重和群共识矩阵
        groupIndexweight=[]                                          #指标总体权重
        groupMatrix=[]                                               #群决策矩阵
        for i in range(indexNum):
            Indexitem=0
            for j in range(len(tableValue)):
                Indexitem=Indexitem+solves_Message[cl[j][0]][1][i]*Qvalue[cl[j][0]]
            groupIndexweight.append(Indexitem)

        for i in range(indexNum-1):                      #找到元素个数
            groupMatrix_son=[]
            for m in range(indexNum-1-i):
                Matrixitem = 0
                for j in range(len(tableValue)):
                    Matrixitem = Matrixitem + Qvalue[cl[j][0]] * grouptable[cl[j][0]][1][0][i][m][0]
                groupMatrix_son.append(Matrixitem)
            groupMatrix.append(groupMatrix_son)

        #确定群共识度
        GCI=[]                                                      #群共识度
        for i in range(len(tableValue)):
            # if tableValue[i][0]=='HFPR' or tableValue[i][0]=='HFLPR':               #完全信息的群共识度
            #     GCIitem=0
            #     for j in range(indexNum):
            #         GCIitem=GCIitem+math.pow((solves_Message[i][1][j]-groupIndexweight[j]),2)
            #     GCI.append(1-math.sqrt(GCIitem/indexNum))
            # else:
            GCIitem = 0
            for j in range(indexNum-1):
                for m in range(indexNum-1-j):
                    GCIitem=GCIitem+abs(grouptable[i][1][0][j][m][0]-groupMatrix[j][m])
            GCI.append(1-GCIitem/(indexNum*(indexNum-1)/2))

        #确定群共识度是否达标
        needModify= {}                                       #不达标的矩阵：应修改的位置
        for i in range(len(tableValue)):
            if GCI[i] <GCI_yvzhi:
                needModify[i]=[[],[]]                        #第一个是位置，第二个是区间,存储的都是列表形式
                needModifytable=[]
                for j in range(indexNum-1):
                    for m in range(indexNum - 1 - j):
                        needModifytable.append(abs(grouptable[i][1][0][j][m][0]-groupMatrix[j][m]))
                needModifynum=max(needModifytable)
                for j in range(indexNum-1):
                    for m in range(indexNum - 1 - j):
                        if needModifynum==needModifytable[j]:
                            needModify[i][0].append([j+1,m+1+j])
                            needModify[i][1].append([grouptable[i][1][0][j][m][0],groupMatrix[j][m]])

        #将cl按原本的顺序排列
        cl_index=[[] for _ in range(len(cl))]
        for i in range(len(cl)):
            cl_index[cl[i][0]]=cl[i][1]
        #输出各决策者的指标权重
        index_weights=[]
        # 输出各决策者的去模糊矩阵
        sure_H=[]
        for i in range(len(solves_Message)):
            index_weights.append(solves_Message[i][1])
            sure_H.append(solves_Message[i][3])

        #返回第一次处理评估矩阵后表格，Information Processing by DMs Evaluation Matrix
        # 各决策者一致水平，需要修改的位置，各决策者的群共识水平，群决策条件下的方案权重，决策者权重,solvemodel()函数输出的结果,各决策者的方案权重，去犹豫模糊的矩阵,群决策矩阵
        return cl_index,needModify,sum(GCI)/len(GCI),groupIndexweight,Qvalue,solves_Message,  index_weights,sure_H,groupMatrix

    except Exception as e:
        print(f"groupdecision出现未知错误: {e}")
        return None

def main():
    """
    主函数：运行多次仿真
    """
    # 真实权重
    w_true = [0.139, 0.267, 0.216, 0.133, 0.107, 0.138]

    # 设置随机种子以便重现
    np.random.seed(42)

    # 数据文件路径
    position = r"E:\newManucript\python_code_rare\script2\data\simulation_data"
    assessment_file = os.path.join(position, "table_values_1.pkl")
    #trust_file = os.path.join(position, "trust_ets_1.pkl")         #第一次实验
    trust_file = os.path.join(position, "trust_ets_random1.pkl")

    print(f"读取评估数据: {assessment_file}")
    print(f"读取信任数据: {trust_file}")

    # 读取数据
    try:
        with open(assessment_file, 'rb') as f:
            assessment_value = pickle.load(f)

        with open(trust_file, 'rb') as f:
            trust_value = pickle.load(f)

        print(f"读取成功: assessment_value长度={len(assessment_value)}")
        print(f"信任矩阵形状: {np.array(trust_value).shape}")

    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        return

        # 运行多次仿真
    all_results = run_multiple_simulations(
        assessment_value=assessment_value,
        trust_value=trust_value,
        w_true=w_true,
        epsilon=0.9,
        max_iter=50
    )

    # 保存结果
    output_dir = r"E:\newManucript\python_code_rare\script2\data\compare_personalTrust"
    save_all_results(all_results, output_dir, w_true)

    print(f"\n所有结果已保存到: {output_dir}")

    return all_results


if __name__ == "__main__":
    results = main()