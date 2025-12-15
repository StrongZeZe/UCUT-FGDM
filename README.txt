(1)	utility_lindo_sovel.py 文件，为使用lindo模型处理复杂模型的子文件，集成处理模型函数和发生内存报错5s后直接跳出的功能函数
(2)	k_weight_produce.py文件，集成生成决策者k值与k值下权重数据的能力的函数”def  run_expert_simulation()“
-根据五次大循环的要求，生成五次不同k值下对应的权重结果，每个大循环生成200个五决策者的权重数据
-excel文件名称：”simulation_results(i).xlsx“,分析图片名称：”simulation_analysis(i).png“,输出的assessment_array(i).pkl与”k_value(i).pkl“可直接在后面使用。

(3)	H-HFPRs_produce.py文件，基于k_weight_produce.py生成的weights，生成符合的H-HFPRs矩阵，五次循环的数据存储在“table_values_{i}.pkl”与“score_index
_200_{i}.xlsx”文件中

(4)	trust_ets_produce.py文件，基于k_value(i).pkl结果，生成固定的先验自信与信任、及决策的置信度，用于仿真决策者间相识情况较大的情况，数据存储在“trust_ets_{i}.pkl”与“trust_ets_{i}.xlsx”文件中。
(5)	trust_ets_produce_random.py文件，基于k_value(i).pkl结果，生成固定的先验自信与自信的置信度，信任随机生成，信任的置信度从更为分散的分布中随机获取，用于仿真决策者间互不相识的情况，数据存储在“trust_ets_random{i}.pkl”与“trust_ets_random{i}.xlsx”文件中。
(6)	trust_ets_produce_True.py文件，基于k_value(i).pkl结果，从trust_ets_{i}.xlsx与trust_ets_random{i}.xlsx中随机抽取信任与信任的置信度，仿真决策者间随机相识与随机不相识的情况,数据存储在“trust_ets_random{i}.pkl”与“trust_ets_random{i}.xlsx”文件中。
(7)	trust_ets_none_confi_produce.py文件用于生成不考虑自信的数据，检验若不考虑各决策者的敏感性，结果会发生如何变化。只是在④的基础上，让所有自信度与置信度都为0.9，生成的文件为“trust_ets_none_confi_{i}.pkl”与“trust_ets_none_confi_{i}.xlsx”文件中。
(8)	Run_simulation_datas.py文件，用于批量处理仿真数据，总共包含四个实验，分别用 flag_test变量表征， flag_test={1，2，3}分别表示：
实验1：依托于决策者能力，固定先验自信与信任、及决策的置信度。仿真决策者间相互了解。
实验2：假定决策者间互不相识，固定先验自信与自信置信度，但信任随机抽取，置信度与先验自信挂钩，但通过更分散的分布获取。仿真决策者间互不相识。
实验3：从实验1与实验2中随机抽取信任与信任的置信度，仿真决策者间随机相识与随机不相识的情况。
实验4：依托于实验1的结果数据，将自信与置信度全部调整为0.9。仿真不考虑自信调节，使决策者间敏感性相同的情况（类似于传统方法）。
由于考虑到lindoAPI的解调与内存的影响，因此，需手动设置flag_test=？来确定是第几个实验。保存的结果分别存储在simulation_data文件夹的test1，test2，test3文件夹中。
break_ij代表因为内存占用原因，而在第一次没有正确抛出答案的仿真数据集合；
group_weight_name = os.path.join(eval("position"+str(flag_test)),f"group_weight_{flag_test}.pkl")  # 存储最终群共识的权重集合
group_CI_name = os.path.join(eval("position"+str(flag_test)),f"group_CI_{flag_test}.pkl")  # 存储最终的群决策一致水平结果集合
group_utility_name = os.path.join(eval("position"+str(flag_test)),f"group_utility_loss_{flag_test}.pkl")  # 存储最终的效用降低水平集合
GCD_final_name = os.path.join(eval("position"+str(flag_test)),f"GCD_final_name_{flag_test}.pkl")  # 存储最终的群共识集合
GROUP_H_name = os.path.join(eval("position"+str(flag_test)),f"GROUP_H_name_{flag_test}.pkl")  # 存储最终的群共识矩阵的集合
DMs_H_name = os.path.join(eval("position"+str(flag_test)),f"DMs_H_{flag_test}.pkl")  # 存储最终各个决策者最终的决策矩阵
C_T_FINAL_name = os.path.join(eval("position"+str(flag_test)),f"C_T_FINAL_{flag_test}.pkl")  # 存储最终各个决策者间的自信与信任
break_ij_record_name = os.path.join(eval("position" + str(flag_test)),f"break_ij_{flag_test}.pkl")  # 存储最终各个决策者间的自信与信任
feedback_Num_{i}表征每次的反馈次数，用于计算该算法的平均耗时。
(9)	complete_data.py文件，用于补足在Run_simulation_datas.py中未正常跑出结果的数据。需要在Run_simulation_datas.py三个实验都跑完之后，才可以运行。
(10)	run_sensitive.datas.py是敏感性分析文件，①用于跑出形状参数∈(0,10)（存储在sensibility1文件夹中），②不同GCD阈值的达成时间（存储在sensibility2_blackBox与sensibility2_whiteBox文件夹中），其中，sensibility2_blackBox存储研究模型为达到阈值为0.99，而需要反馈的次数；sensibility2_whiteBox文件存储不考虑不可知信息（即外部奖励）的模型为达到阈值0.99所需要反馈的次数。
(11)	compare_others_research_Bayes.py是对比分析“A Bayesian Framework For Modelling The Trust  Relationships To Group Decision Making Problems”研究的代码，其主要功能有①将本研究的输入转变为该研究的输入②基于论文方法，针对两种实验背景，对每个实验背景都抽500次的仿真数据，跑出结果（方案权重、反馈次数，群共识度），未选择True，原因在于该文献使用了信任传递理论，对未知信任进行了处理。
(12)	compare_others_research_person.py是对比分析“Personalized trust incentive mechanisms with personality characteristics for minimum cost consensus in group decision making”研究的代码，其主要功能有①将本研究的输入转变为该研究的输入②基于论文方法，针对三种实验背景，对每个实验背景都抽500次的仿真数据，跑出结果（方案权重、反馈次数，群共识度），未选择True，原因在于该文献使用了信任传递理论，对未知信任进行了处理。
(13)compare_others_research_consensus.py是对比分析“Consensus adjustment mechanism in view of coalition structure-based cooperative game for group decision making with distribution linguistic preference relations”研究代码，存储在“compare_consensus”文件中，该论文并未考虑信任问题。
(14)draw_pic文件夹中的所有.py文件都是绘制本研究趋势图的代码


