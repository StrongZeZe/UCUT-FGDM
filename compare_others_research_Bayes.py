import numpy as np
from scipy.optimize import minimize
import pickle
import os
import math
from lindo import *
import lindo
import pandas as pd
import time
import random

class SimplifiedGDM:
    def __init__(self):
        self.m = 5  # 决策者数量
        self.k = 6  # 方案数量
        self.beta = 5.0  # TDI上界
        self.alpha = 0.05  # 置信水平
        self.consensus_threshold = 0.9
        self.max_crp_iterations = 50  # 最大共识达成迭代次数
        self.adjustment_rate = 0.1  # 调整幅度
        self.crp_iteration_count = 0  # 记录共识达成迭代次数

    def load_data(self, trust_matrix, decision_weights):
        """
        加载数据
        """
        # 提取信任值，确保转换为浮点数
        self.T = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    self.T[i, j] = 0.0
                else:
                    element = trust_matrix[i][j]
                    if isinstance(element, (list, tuple)):
                        trust_val = element[0]
                        if isinstance(trust_val, str):
                            try:
                                self.T[i, j] = float(trust_val)
                            except ValueError:
                                print(f"警告: 无法将信任值 '{trust_val}' 转换为浮点数，设置为0.0")
                                self.T[i, j] = 0.0
                        else:
                            self.T[i, j] = float(trust_val)
                    else:
                        if isinstance(element, str):
                            try:
                                self.T[i, j] = float(element)
                            except ValueError:
                                print(f"警告: 无法将信任值 '{element}' 转换为浮点数，设置为0.0")
                                self.T[i, j] = 0.0
                        else:
                            self.T[i, j] = float(element)

        # 决策者权重矩阵，确保转换为浮点数
        self.W_original = np.zeros((self.m, self.k))
        for i in range(self.m):
            for j in range(self.k):
                weight_val = decision_weights[i][j]

                if isinstance(weight_val, str):
                    try:
                        self.W_original[i, j] = float(weight_val)
                    except ValueError:
                        print(f"警告: 无法将权重值 '{weight_val}' 转换为浮点数，设置为0.0")
                        self.W_original[i, j] = 0.0
                else:
                    self.W_original[i, j] = float(weight_val)

        # 确保权重归一化
        for i in range(self.m):
            if np.sum(self.W_original[i]) > 0:
                self.W_original[i] /= np.sum(self.W_original[i])
            else:
                self.W_original[i] = np.ones(self.k) / self.k

        # 创建工作副本
        self.W = self.W_original.copy()

        return self.T, self.W_original

    def calculate_similarity_matrix(self):
        """
        计算决策者评估矩阵的相似性矩阵
        """
        similarity = np.zeros((self.m, self.m))

        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    w_i = self.W[i]
                    w_j = self.W[j]
                    dot_product = np.dot(w_i, w_j)
                    norm_i = np.linalg.norm(w_i)
                    norm_j = np.linalg.norm(w_j)

                    if norm_i > 0 and norm_j > 0:
                        cos_sim = dot_product / (norm_i * norm_j)
                        similarity[i, j] = (cos_sim + 1) / 2
                    else:
                        similarity[i, j] = 0.5

        return similarity

    def build_tdi_from_similarity(self, similarity_matrix):
        """
        从相似性矩阵构建TDI矩阵
        """
        TDI = np.zeros((self.m, self.m))

        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    TDI[i, j] = 0
                else:
                    TDI[i, j] = self.beta * (1 - similarity_matrix[i, j])

        return TDI

    def bayesian_trust_update(self, T, TDI, max_iter=100, epsilon=1e-6):
        """
        贝叶斯信任更新
        """
        m = T.shape[0]
        T_B = T.copy()

        for _ in range(max_iter):
            T_B_new = T_B.copy()

            for p in range(m):
                for q in range(m):
                    if p == q:
                        continue

                    mediators = []
                    for i in range(m):
                        if i != p and i != q:
                            if T_B[p, i] > 0 and T[i, q] > 0:
                                mediators.append(i)

                    if len(mediators) == 0:
                        T_B_new[p, q] = T[p, q]
                    else:
                        sum1 = sum(T_B[p, i] * T[i, q] for i in mediators)
                        sum2 = sum(T_B[p, i] for i in mediators)

                        if sum2 + TDI[p, q] == 0:
                            T_B_new[p, q] = T[p, q]
                        else:
                            T_B_new[p, q] = (sum1 + TDI[p, q] * T[p, q]) / (sum2 + TDI[p, q])

            if np.max(np.abs(T_B_new - T_B)) < epsilon:
                return T_B_new

            T_B = T_B_new

        return T_B

    def calculate_dm_weights(self, T_B):
        """
        从贝叶斯信任矩阵计算决策者权重
        """
        m = T_B.shape[0]

        trust_degrees = np.zeros(m)
        for p in range(m):
            trust_degrees[p] = np.sum([T_B[i, p] for i in range(m) if i != p]) / (m - 1)

        if np.sum(trust_degrees) > 0:
            weights = trust_degrees / np.sum(trust_degrees)
        else:
            weights = np.ones(m) / m

        return weights

    def calculate_consensus(self, dm_weights):
        """
        计算共识水平
        """
        # 计算集体评估
        collective = np.zeros(self.k)
        for p in range(self.m):
            collective += dm_weights[p] * self.W[p]

        # 计算每个决策者与集体评估的差异
        differences = []
        for p in range(self.m):
            diff = np.sum(np.abs(self.W[p] - collective))
            differences.append(diff)

        # 计算共识
        avg_diff = np.mean(differences)
        consensus = 1 - avg_diff / 2

        return consensus, collective, differences

    def identify_most_dissimilar_dm(self, differences):
        """
        识别与集体评估差异最大的决策者
        """
        return np.argmax(differences)

    def adjust_dm_preferences(self, dm_index, collective, adjustment_rate=None):
        """
        调整决策者的偏好向集体评估靠近
        """
        if adjustment_rate is None:
            adjustment_rate = self.adjustment_rate

        # 获取当前决策者的偏好
        current_pref = self.W[dm_index].copy()

        # 计算调整方向：向集体评估移动
        adjustment_vector = collective - current_pref

        # 应用调整
        new_pref = current_pref + adjustment_rate * adjustment_vector

        # 确保非负性
        new_pref = np.maximum(new_pref, 0)

        # 重新归一化
        if np.sum(new_pref) > 0:
            new_pref /= np.sum(new_pref)
        else:
            new_pref = np.ones(self.k) / self.k

        # 更新决策者偏好
        self.W[dm_index] = new_pref

        # 计算调整量
        adjustment_amount = np.sum(np.abs(new_pref - current_pref))

        return adjustment_amount

    def consensus_reaching_process(self, initial_consensus, dm_weights):
        """
        共识达成过程（CRP）
        """
        current_consensus = initial_consensus
        current_dm_weights = dm_weights.copy()
        crp_iterations = 1

        # CRP迭代
        for iteration in range(1, self.max_crp_iterations + 1):
            # 如果已达到阈值，停止迭代
            if current_consensus >= self.consensus_threshold:
                break

            # 计算当前共识和差异
            _, _, differences = self.calculate_consensus(current_dm_weights)

            # 识别需要调整的决策者（与集体评估差异最大）
            dm_to_adjust = self.identify_most_dissimilar_dm(differences)

            # 计算集体评估
            collective = np.zeros(self.k)
            for p in range(self.m):
                collective += current_dm_weights[p] * self.W[p]

            # 调整决策者偏好
            self.adjust_dm_preferences(dm_to_adjust, collective)

            # 重新计算相似性矩阵
            similarity = self.calculate_similarity_matrix()

            # 重新构建TDI矩阵
            TDI = self.build_tdi_from_similarity(similarity)

            # 重新进行贝叶斯信任更新
            T_B = self.bayesian_trust_update(self.T, TDI)

            # 重新计算决策者权重
            current_dm_weights = self.calculate_dm_weights(T_B)

            # 重新计算共识
            current_consensus, collective, _ = self.calculate_consensus(current_dm_weights)

            crp_iterations = iteration

        return current_consensus, current_dm_weights, crp_iterations

    def run_analysis(self, save_individual=False, output_dir=None, run_id=0):
        """
        运行完整分析

        参数:
            save_individual: 是否保存单个运行的结果
            output_dir: 输出文件夹路径，如果为None则不保存文件
            run_id: 运行ID，用于区分不同运行

        返回:
            包含关键结果的字典
        """
        # 重置工作副本
        self.W = self.W_original.copy()
        self.crp_iteration_count = 0

        # 1. 计算相似性矩阵
        similarity = self.calculate_similarity_matrix()

        # 2. 构建TDI矩阵
        TDI = self.build_tdi_from_similarity(similarity)

        # 3. 贝叶斯信任更新
        T_B = self.bayesian_trust_update(self.T, TDI)

        # 4. 计算决策者权重
        dm_weights = self.calculate_dm_weights(T_B)

        # 5. 计算初始共识
        initial_consensus, collective, _ = self.calculate_consensus(dm_weights)

        # 6. 检查是否达到阈值，如果没有则进行共识达成过程
        final_consensus = initial_consensus
        final_dm_weights = dm_weights
        crp_iterations = 0

        if initial_consensus < self.consensus_threshold:
            final_consensus, final_dm_weights, crp_iterations = self.consensus_reaching_process(
                initial_consensus, dm_weights
            )

        # 7. 计算最终集体评估
        final_collective = np.zeros(self.k)
        for p in range(self.m):
            final_collective += final_dm_weights[p] * self.W[p]

        # 8. 计算最终排名
        final_ranking = np.argsort(-final_collective) + 1

        # 9. 构建结果字典
        result = {
            'run_id': run_id,
            'final_consensus': final_consensus,
            'crp_iterations': crp_iterations,
            'final_collective_evaluation': final_collective.tolist(),
            'final_ranking': final_ranking.tolist(),
            'final_dm_weights': final_dm_weights.tolist(),
            'initial_consensus': initial_consensus,
            'parameters': {
                'beta': self.beta,
                'alpha': self.alpha,
                'consensus_threshold': self.consensus_threshold,
                'max_crp_iterations': self.max_crp_iterations,
                'adjustment_rate': self.adjustment_rate
            }
        }

        # 10. 如果需要，保存单个运行的结果
        if save_individual and output_dir is not None:
            self.save_individual_result(result, output_dir, run_id)

        return result

    def save_individual_result(self, result, output_dir, run_id):
        """
        保存单个运行的结果
        """
        os.makedirs(output_dir, exist_ok=True)

        # 为每个运行创建单独的文件夹
        run_folder = os.path.join(output_dir, f"run_{run_id:03d}")
        os.makedirs(run_folder, exist_ok=True)

        # 保存结果到pkl文件
        result_file = os.path.join(run_folder, "result.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)

        # 保存文本摘要
        text_file = os.path.join(run_folder, "summary.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"运行 {run_id} 结果摘要\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"初始共识水平: {result['initial_consensus']:.4f}\n")
            f.write(f"最终共识水平: {result['final_consensus']:.4f}\n")
            f.write(f"CRP迭代次数: {result['crp_iterations']}\n")

            f.write(f"\n最终集体评估:\n")
            collective = result['final_collective_evaluation']
            for i in range(len(collective)):
                f.write(f"  方案 {i + 1}: {collective[i]:.6f}\n")

            f.write(f"\n最终方案排名 (从高到低): {result['final_ranking']}\n")

            f.write(f"\n决策者权重:\n")
            dm_weights = result['final_dm_weights']
            for i in range(len(dm_weights)):
                f.write(f"  e{i + 1}: {dm_weights[i]:.6f}\n")

            f.write(f"\n参数设置:\n")
            f.write(f"  共识阈值: {result['parameters']['consensus_threshold']}\n")
            f.write(f"  最大CRP迭代次数: {result['parameters']['max_crp_iterations']}\n")
            f.write(f"  调整幅度: {result['parameters']['adjustment_rate']}\n")

        print(f"运行 {run_id} 结果已保存到: {run_folder}")

    @staticmethod
    def save_batch_results(results_list, output_dir, store_content_list, UL_loss_list,save_individual_txt=False):
        """
        保存批量运行结果

        参数:
            results_list: 结果列表
            output_dir: 输出文件夹路径
            save_individual_txt: 是否保存每个运行的详细文本文件
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存所有结果到一个pkl文件
        batch_file = os.path.join(output_dir, store_content_list[0])
        with open(batch_file, 'wb') as f:
            pickle.dump(results_list, f)
        print(f"批量结果(pkl)已保存到: {batch_file}")

        # 2. 保存详细的文本格式结果
        detailed_txt_file = os.path.join(output_dir, store_content_list[1])
        with open(detailed_txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("贝叶斯信任GDM批量运行详细结果\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results_list):
                f.write(f"运行 {i}:\n")
                f.write(f"  初始共识: {result['initial_consensus']:.6f}\n")
                f.write(f"  最终共识: {result['final_consensus']:.6f}\n")
                f.write(f"  CRP迭代次数: {result['crp_iterations']}\n")
                f.write(f"  方案权重: {', '.join([f'{w:.6f}' for w in result['final_collective_evaluation']])}\n")
                f.write(f"  方案排名: {result['final_ranking']}\n")
                f.write(f"  决策者权重: {', '.join([f'{w:.6f}' for w in result['final_dm_weights']])}\n")
                f.write("-" * 80 + "\n")

        print(f"详细文本结果已保存到: {detailed_txt_file}")

        # 3. 保存CSV格式的结果（便于用Excel打开）
        csv_file = os.path.join(output_dir, store_content_list[2])

        # 准备CSV数据
        data = []
        for i, result in enumerate(results_list):
            row = {
                'run_id': i,
                'initial_consensus': result['initial_consensus'],
                'final_consensus': result['final_consensus'],
                'crp_iterations': result['crp_iterations'],
                'consensus_improvement': result['final_consensus'] - result['initial_consensus']
            }

            # 添加方案权重
            for j, weight in enumerate(result['final_collective_evaluation']):
                row[f'weight_scheme_{j + 1}'] = weight

            # 添加方案排名
            row['ranking'] = str(result['final_ranking'])

            # 添加决策者权重
            for j, dm_weight in enumerate(result['final_dm_weights']):
                row[f'dm_weight_{j + 1}'] = dm_weight

            data.append(row)

        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV格式结果已保存到: {csv_file}")

        # 4. 保存汇总统计信息
        summary_file = os.path.join(output_dir, store_content_list[3])
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("批量运行结果汇总统计\n")
            f.write("=" * 80 + "\n\n")

            # 计算统计信息
            initial_consensuses = [r['initial_consensus'] for r in results_list]
            final_consensuses = [r['final_consensus'] for r in results_list]
            crp_iterations = [r['crp_iterations'] for r in results_list]
            consensus_improvements = [r['final_consensus'] - r['initial_consensus'] for r in results_list]

            f.write(f"总运行次数: {len(results_list)}\n")
            f.write(f"运行参数:\n")
            f.write(f"  共识阈值: {results_list[0]['parameters']['consensus_threshold']}\n")
            f.write(f"  最大CRP迭代次数: {results_list[0]['parameters']['max_crp_iterations']}\n")
            f.write(f"  调整幅度: {results_list[0]['parameters']['adjustment_rate']}\n\n")

            f.write("初始共识统计:\n")
            f.write(f"  平均值: {np.mean(initial_consensuses):.6f}\n")
            f.write(f"  标准差: {np.std(initial_consensuses):.6f}\n")
            f.write(f"  最小值: {np.min(initial_consensuses):.6f}\n")
            f.write(f"  最大值: {np.max(initial_consensuses):.6f}\n\n")

            f.write("最终共识统计:\n")
            f.write(f"  平均值: {np.mean(final_consensuses):.6f}\n")
            f.write(f"  标准差: {np.std(final_consensuses):.6f}\n")
            f.write(f"  最小值: {np.min(final_consensuses):.6f}\n")
            f.write(f"  最大值: {np.max(final_consensuses):.6f}\n\n")

            f.write("共识改进统计:\n")
            f.write(f"  平均值: {np.mean(consensus_improvements):.6f}\n")
            f.write(f"  标准差: {np.std(consensus_improvements):.6f}\n")
            f.write(f"  最小值: {np.min(consensus_improvements):.6f}\n")
            f.write(f"  最大值: {np.max(consensus_improvements):.6f}\n\n")

            f.write("CRP迭代次数统计:\n")
            f.write(f"  平均值: {np.mean(crp_iterations):.2f}\n")
            f.write(f"  标准差: {np.std(crp_iterations):.2f}\n")
            f.write(f"  最小值: {np.min(crp_iterations)}\n")
            f.write(f"  最大值: {np.max(crp_iterations)}\n\n")

            # 统计达到阈值的运行数
            threshold = results_list[0]['parameters']['consensus_threshold']
            met_threshold_initial = sum(1 for r in results_list if r['initial_consensus'] >= threshold)
            met_threshold_final = sum(1 for r in results_list if r['final_consensus'] >= threshold)
            f.write(f"达到共识阈值({threshold})的运行数:\n")
            f.write(
                f"  初始: {met_threshold_initial}/{len(results_list)} ({met_threshold_initial / len(results_list) * 100:.1f}%)\n")
            f.write(
                f"  最终: {met_threshold_final}/{len(results_list)} ({met_threshold_final / len(results_list) * 100:.1f}%)\n\n")

            # 方案权重统计
            f.write("各方案最终权重统计:\n")
            for j in range(len(results_list[0]['final_collective_evaluation'])):
                weights = [r['final_collective_evaluation'][j] for r in results_list]
                f.write(f"  方案 {j + 1}: 平均值={np.mean(weights):.6f}, 标准差={np.std(weights):.6f}\n")

            # 决策者权重统计
            f.write("\n各决策者最终权重统计:\n")
            for j in range(len(results_list[0]['final_dm_weights'])):
                dm_weights = [r['final_dm_weights'][j] for r in results_list]
                f.write(f"  决策者 {j + 1}: 平均值={np.mean(dm_weights):.6f}, 标准差={np.std(dm_weights):.6f}\n")

            # 显示前几个运行的简要结果
            f.write("\n" + "=" * 80 + "\n")
            f.write("前10个运行结果示例:\n")
            f.write("=" * 80 + "\n\n")

            for i in range(min(10, len(results_list))):
                f.write(f"运行 {i}: 初共识={results_list[i]['initial_consensus']:.4f}, "
                        f"终共识={results_list[i]['final_consensus']:.4f}, "
                        f"CRP迭代={results_list[i]['crp_iterations']}, "
                        f"排名={results_list[i]['final_ranking']}\n")

        print(f"汇总统计信息已保存到: {summary_file}")

        # 5. 如果设置了，保存每个运行的详细文本文件
        if save_individual_txt:
            for i, result in enumerate(results_list):
                run_txt_file = os.path.join(output_dir, f"run_{i:03d}.txt")
                with open(run_txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"运行 {i} 详细结果:\n")
                    f.write(f"初始共识: {result['initial_consensus']:.6f}\n")
                    f.write(f"最终共识: {result['final_consensus']:.6f}\n")
                    f.write(f"CRP迭代次数: {result['crp_iterations']}\n")
                    f.write(f"共识改进: {result['final_consensus'] - result['initial_consensus']:.6f}\n\n")

                    f.write("方案权重:\n")
                    for j, weight in enumerate(result['final_collective_evaluation']):
                        f.write(f"  方案 {j + 1}: {weight:.6f}\n")

                    f.write(f"\n方案排名: {result['final_ranking']}\n\n")

                    f.write("决策者权重:\n")
                    for j, dm_weight in enumerate(result['final_dm_weights']):
                        f.write(f"  决策者 {j + 1}: {dm_weight:.6f}\n")

            print(f"每个运行的详细文本文件已保存到 {output_dir}")

        return batch_file, summary_file, detailed_txt_file, csv_file

#复制三维list结构函数
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
        Alencol = N.asarray(None)                           # 定义了约束矩阵中每一列的长度。在本例中，这被设置为None，因为在矩阵中没有留下空白

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
            print(A)
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
        rhs = N.array(rhs, dtype=N.double)  # 可被lindo执行的语句
        reward = N.array(reward, dtype=N.double)  # 可被lindo执行的语句
        contype = N.array(contype, dtype=N.character)       #转换成了字节型

        Abegcol = N.array(Abegcol, dtype=N.int32)

        A=N.array(A, dtype=N.double)

        Arowndx = N.array(Arowndx, dtype=N.int32)
        lb = N.array(lb, dtype=N.double)
        ub = N.array(ub, dtype=N.double)

        # 二次矩阵

        qCI = N.array(qCI, dtype=N.int32)
        qNZV = N.array(qNZV, dtype=N.double)
        qRowX = N.array(qRowX, dtype=N.int32)
        qColumnX = N.array(qColumnX, dtype=N.int32)
        # create LINDO environment and model objects
        LicenseKey = N.array('', dtype='S1024')
        LicenseFile = os.getenv("LINDOAPI_LICENSE_FILE")                #'C:\\Lindoapi\\license\\lndapi90.lic'
        if LicenseFile == None:
            print('Error: Environment variable LINDOAPI_LICENSE_FILE is not set')
            sys.exit(1)

        lindo.pyLSloadLicenseString(LicenseFile, LicenseKey)
        pnErrorCode = N.array([-1], dtype=N.int32)
        pEnv = lindo.pyLScreateEnv(pnErrorCode, LicenseKey)

        pModel = lindo.pyLScreateModel(pEnv, pnErrorCode)
        geterrormessage(pEnv, pnErrorCode[0])

        # load data into the model
        print("Loading LP data...")
        # 针对不同的，这个LP是解线性的，NLP解非线性
        # 通过调用LSloadLPData（），将问题结构和线性数据加载到模型结构中。            #contype
        errorcode = lindo.pyLSloadLPData(pModel, nM, nN, objsense, objconst,
                                         reward, rhs, contype,
                                         Anz, Abegcol, Alencol, A, Arowndx,
                                         lb, ub)
        geterrormessage(pEnv, errorcode)  # 检测

        errorcode = lindo.pyLSloadQCData(pModel, qNZ, qCI, qRowX, qColumnX, qNZV)  # 二次
        geterrormessage(pEnv, errorcode)  # 检测

        # solve the model
        print("Solving the model...")
        print("Solving the model...")
        pnStatus = N.array([-1], dtype=N.int32)
        errorcode = lindo.pyLSoptimize(pModel, LSconst.LS_METHOD_FREE,
                                       pnStatus)  # 通过调用LSoptimize（）（或如果有整数变量的LSsolveMIP（））来解决这个问题。     使用障碍求解器
        # errorcode = lindo.pyLSoptimize(pModel,lindo.LS_METHOD_NLP,pnStatus)
        geterrormessage(pEnv, errorcode)  # 检测

        # retrieve the objective value       获取最优结果
        dObj = N.array([-1.0], dtype=N.double)
        errorcode = lindo.pyLSgetInfo(pModel, LSconst.LS_DINFO_POBJ,
                                      dObj)  # 通过调用LSgetInfo（）、LSget初级解决方案（）和LSget双元解决方案（）来检索解决方案。
        geterrormessage(pEnv, errorcode)
        print("Objective is: %.5f" % dObj[0])
        print("")

        # retrieve the primal solution       获取变量数值
        padPrimal = N.empty((nN), dtype=N.double)
        errorcode = lindo.pyLSgetPrimalSolution(pModel, padPrimal)
        geterrormessage(pEnv, errorcode)
        print("Primal solution is: ")
        for x in padPrimal: print("%.5f" % x)

        # delete LINDO model pointer
        errorcode = lindo.pyLSdeleteModel(pModel)  # 通过调用LSdeleteModel，LSdeleteEnv（）来删除模型和环境。
        geterrormessage(pEnv, errorcode)

        # delete LINDO environment pointer
        errorcode = lindo.pyLSdeleteEnv(pEnv)
        # geterrormessage(pEnv, errorcode)

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

def groupdecision(tableValues,len_planses,GCI_yvzhi):                             #存储表格数据的数列
    #tableValue=['i-HFLPR', [0.61, 0.72], [0.5], [0.06, 0.28], [100], [0.28], [100], [0.72], [0.28, 0.39, 0.61], [0.72, 0.83], [0.94]]
    #self.tableValue=['i-HFLPR', [[0.61, 0.72], [0.5], [0.06, 0.28], [100], [0.28], [100], [0.72], [0.28, 0.39, 0.61], [0.72, 0.83], [0.94]]]
    #self.groupdata=[[决策者1：[模糊元权重],[方案权重],[偏差，即object]],[决策者2],...]=solveModel(tableValues, len_planses)的第一维每组的前三小组
    #确定矩阵
    # 用于存储S的所有评分[[（专家1评分）评价形式，[上半矩阵评分],[下半矩阵评分]],[专家2评分],...[专家n评分]]
    # 用于存储S的所有专家偏差信息[[[犹豫模糊集1各模糊元权重，犹豫模糊集2各模糊元权重..],[各指标权重]，[总偏差]],[专家2],...[专家n]]
    tableValue=[]                                                                #存储需要的信息
    solves_Message = solveModel(tableValues, len_planses)                          #存储处理后的信息                  #solve_Message的长度应和tableValues一样，除非报错
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

if __name__ == "__main__":

    position = r"E:\newManucript\python_code_rare\script2\data\simulation_data"
    w_true = [0.139, 0.267, 0.216, 0.133, 0.107, 0.138]  # 真实的权重！这个在k_weight_produce.py中也被定义了，可以验证是同一个数组
    hflpr_list = [0.06, 0.17, 0.28, 0.39, 0.5, 0.61, 0.72, 0.83, 0.94]  # 存储语义list元素
    fuz_unc_a = 0.5
    assessment_file = os.path.join(position, f"table_values_1.pkl")               # 读取第一次的评估结果
    trust_file = os.path.join(position, f"trust_ets_1.pkl")                 # 读取模仿真实结果的自信-信任与置信信息

    with open(assessment_file, 'rb') as f:
        assessment_value = pickle.load(f)

    with open(trust_file, 'rb') as f:
        trust_value = pickle.load(f)

    # 存储所有运行结果的列表
    all_results = []
    # 指定输出文件夹
    output_folder = "E:\\newManucript\\python_code_rare\\script2\\data\\compare_bayes"
    #store_list = ["batch_results_True.pkl", "detailed_results_True.txt", "results_summary_True.csv", "batch_summary_True.txt"]     #部分了解
    #store_list = ["batch_results_random.pkl", "detailed_results_random.txt", "results_summary_random.csv","batch_summary_random.txt"]      #全部陌生
    store_list = ["batch_results.pkl", "detailed_results.txt", "results_summary.csv","batch_summary.txt"]                                   #全部了解

    start_time=time.time()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    UL_loss=[]                                          #收集效用损失
    for j in range(len(assessment_value)):                                                       #仿真第一个大循环的所有数据
        cl_index, needModify, GCD, groupIndexweight, Qvalue, solves_Message, index_weight, sure_Hs, group_Hs = groupdecision(assessment_value[j], len(w_true), 0.9)

        gdm = SimplifiedGDM()

        # 可以调整参数
        gdm.consensus_threshold = 0.9  # 共识阈值
        gdm.max_crp_iterations = 50  # 最大CRP迭代次数
        gdm.adjustment_rate = 0.1  # 调整幅度

        # # 示例数据（字符串类型）
        # example_trust = [
        #     [["0.0", "0.0"], ["0.6", "0.8"], ["0.0", "0.0"], ["0.7", "0.9"], ["0.0", "0.0"]],
        #     [["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"], ["0.7", "0.8"]],
        #     [["0.6", "0.7"], ["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]],
        #     [["0.0", "0.0"], ["0.8", "0.9"], ["0.9", "0.9"], ["0.0", "0.0"], ["0.0", "0.0"]],
        #     [["0.0", "0.0"], ["0.0", "0.0"], ["0.8", "0.8"], ["0.0", "0.0"], ["0.0", "0.0"]]
        # ]
        #
        # # 创建有明显差异的评估矩阵以测试CRP
        # example_weights = [
        #     ["0.40", "0.10", "0.10", "0.10", "0.10", "0.20"],  # e1 - 偏爱方案1
        #     ["0.10", "0.40", "0.10", "0.10", "0.10", "0.20"],  # e2 - 偏爱方案2
        #     ["0.10", "0.10", "0.40", "0.10", "0.10", "0.20"],  # e3 - 偏爱方案3
        #     ["0.10", "0.10", "0.10", "0.40", "0.10", "0.20"],  # e4 - 偏爱方案4
        #     ["0.10", "0.10", "0.10", "0.10", "0.40", "0.20"]  # e5 - 偏爱方案5
        # ]

       # gdm.load_data(trust_value[j], index_weight)            #注意，当为random时，需要替换

        index_weight_post=index_weight.copy()
        ul_loss=0.0
        for mm in range(len(index_weight_post)):
            A = random.uniform(0.02, 0.03)
            indices = list(range(6))
            random.shuffle(indices)

            increase_indices = indices[:3]
            decrease_indices = indices[3:]
            # 计算调整量
            adjustment = A / 3
            ul_loss += 2 * A
            for idx in increase_indices:
                index_weight_post[mm][idx] += adjustment
            for idx in decrease_indices:
                index_weight_post[mm][idx] -= adjustment


        gdm.load_data(trust_value, index_weight_post)


        # 运行分析（不保存单个结果，只收集数据）
        result = gdm.run_analysis(save_individual=False, run_id=j)

        # 添加到结果列表
        all_results.append(result)


        print(f"  初始共识: {result['initial_consensus']:.4f}, "
              f"最终共识: {result['final_consensus']:.4f}, "
              f"CRP迭代: {result['crp_iterations']}")

    end_time = time.time()

    print("\n" + "=" * 60)
    print("批量运行完成")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("=" * 60)

    # 保存所有结果（保存多个格式的文件）
    batch_file, summary_file, detailed_txt_file, csv_file = SimplifiedGDM.save_batch_results(
        all_results, output_folder, store_list,UL_loss,save_individual_txt=False
    )


    # 打印汇总信息
    print("\n" + "=" * 60)
    print("汇总统计:")
    print("=" * 60)

    initial_consensuses = [r['initial_consensus'] for r in all_results]
    final_consensuses = [r['final_consensus'] for r in all_results]
    crp_iterations = [r['crp_iterations'] for r in all_results]
    consensus_improvements = [r['final_consensus'] - r['initial_consensus'] for r in all_results]

    print(f"平均初始共识: {np.mean(initial_consensuses):.4f}")
    print(f"平均最终共识: {np.mean(final_consensuses):.4f}")
    print(f"平均CRP迭代次数: {np.mean(crp_iterations):.2f}")
    print(f"平均共识改进: {np.mean(consensus_improvements):.4f}")
    print(f"达到最终共识阈值的运行数: {sum(1 for r in all_results if r['final_consensus'] >= 0.9)}/50")

    # 显示前几个运行的方案权重
    print("\n前3个运行的最终方案权重:")
    for i in range(min(3, len(all_results))):
        weights_str = ', '.join([f"{w:.4f}" for w in all_results[i]['final_collective_evaluation']])
        print(f"运行 {i}: {weights_str}")

    # 显示保存的文件信息
    print("\n" + "=" * 60)
    print("保存的文件:")
    print("=" * 60)
    print(f"1. batch_results.pkl - 包含所有50次运行的结果")
    print(f"2. detailed_results.txt - 详细的文本格式结果")
    print(f"3. results_summary.csv - CSV格式结果（可用Excel打开）")
    print(f"4. batch_summary.txt - 汇总统计信息")
    print(f"5. run_XXX.txt - 每个运行的详细文本文件")
    print(f"\n文件保存在: {output_folder}")