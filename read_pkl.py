import pickle
import os
import gurobipy             #类似于lindo的求解器



def save_table_values(data, file_path):
    """将数据保存为Python数据文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to: {file_path}")

position1 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility2_whiteBox"
position2 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility2_blackBox"
position3 = r"E:\newManucript\python_code_rare\script2\data\simulation_data\sensibility1"
position4= r"E:\newManucript\python_code_rare\script2\data\compare_bayes"
position5= r"E:\newManucript\python_code_rare\script2\data\compare_consensus"
position6= r"E:\newManucript\python_code_rare\script2\data\compare_personalTrust"

priginal_file = os.path.join(position4, f"batch_results.pkl")
priginal_file_random = os.path.join(position4, f"batch_results_random.pkl")
priginal_file2 = os.path.join(position6, f"all_simulation_results.pkl")
priginal_file2_random = os.path.join(position6, f"all_simulation_results_random.pkl")
priginal_file3 = os.path.join(position5, f"consensus_results.pkl")
with open(priginal_file, 'rb') as f:
    priginal_data = pickle.load(f)
with open(priginal_file_random, 'rb') as f:
    priginal_data_random = pickle.load(f)
with open(priginal_file2, 'rb') as f:
    priginal_data2 = pickle.load(f)
with open(priginal_file2_random, 'rb') as f:
    priginal_data2_ = pickle.load(f)
with open(priginal_file3, 'rb') as f:
    priginal_data3 = pickle.load(f)


original_file = os.path.join(position1, f"group_weight_2.pkl")
original_file2 = os.path.join(position1, f"break_ij_2.pkl")
original_file3 = os.path.join(position1, f"feedback_Num_2.pkl")
#original_file4 = os.path.join(position1, f"GCD_2.pkl")
original_file4 = os.path.join(position1, f"GCD_final_name_2.pkl")
original_file5 = os.path.join(position1, f"no_know_infor_2.pkl")
original_file6 = os.path.join(position1, f"group_CI_2.pkl")                     #一致性水平
original_file7 = os.path.join(position1, f"group_utility_loss_2.pkl")           #效用损失


driginal_file = os.path.join(position2, f"group_weight_3.pkl")
driginal_file2 = os.path.join(position2, f"break_ij_3.pkl")
driginal_file3 = os.path.join(position2, f"feedback_Num_3.pkl")
#original_file4 = os.path.join(position1, f"GCD_2.pkl")
driginal_file4 = os.path.join(position2, f"GCD_final_name_3.pkl")
driginal_file5 = os.path.join(position2, f"no_know_infor_3.pkl")                #u-KDD中的FC
driginal_file6 = os.path.join(position2, f"group_CI_3.pkl")                     #一致性水平
driginal_file7 = os.path.join(position2, f"group_utility_loss_3.pkl")           #效用损失
driginal_file8 = os.path.join(position2, f"C_T_FINAL_3.pkl")           #效用损失
driginal_file9 = os.path.join(position2, f"GROUP_H_name_3.pkl")


with open(original_file, 'rb') as f:
    original_data = pickle.load(f)
with open(original_file2, 'rb') as f:
    original_data2 = pickle.load(f)
with open(original_file3, 'rb') as f:
    original_data3 = pickle.load(f)
with open(original_file4, 'rb') as f:
    original_data4 = pickle.load(f)
with open(original_file5, 'rb') as f:
    original_data5 = pickle.load(f)
with open(original_file6, 'rb') as f:
    original_data6 = pickle.load(f)
with open(original_file7, 'rb') as f:
    original_data7 = pickle.load(f)


with open(driginal_file, 'rb') as f:
    driginal_data = pickle.load(f)      #group_weight_3
with open(driginal_file2, 'rb') as f:
    driginal_data2 = pickle.load(f)     #break_ij
with open(driginal_file3, 'rb') as f:
    driginal_data3 = pickle.load(f)         #feedback
with open(driginal_file4, 'rb') as f:
    driginal_data4 = pickle.load(f)         #GCD
with open(driginal_file5, 'rb') as f:
    driginal_data5 = pickle.load(f)         #
with open(driginal_file6, 'rb') as f:
    driginal_data6 = pickle.load(f)     #一致性水平
with open(driginal_file7, 'rb') as f:
    driginal_data7 = pickle.load(f)         #效用损失
with open(driginal_file8, 'rb') as f:
    driginal_data8 = pickle.load(f)

with open(driginal_file9, 'rb') as f:
    driginal_data9 = pickle.load(f)
# # #original_data3.append(42)
# # driginal_data3.append(40)
# # original_data4.append(0.99)
# # driginal_data4.append(0.99)
#
# original_data6.append(original_data6[38])
# original_data7.append(original_data7[38])
#
# #save_table_values(original_data,original_file)
# # save_table_values(original_data4,original_file4)
# save_table_values(original_data6,original_file6)
# save_table_values(original_data7,original_file7)



ariginal_file = os.path.join(position3, f"group_weight_1.pkl")
ariginal_file2 = os.path.join(position3, f"break_ij_1.pkl")
ariginal_file3 = os.path.join(position3, f"feedback_Num_1.pkl")
#original_file4 = os.path.join(position1, f"GCD_2.pkl")
ariginal_file4 = os.path.join(position3, f"GCD_final_name_1.pkl")
#ariginal_file5 = os.path.join(position3, f"no_know_infor_1.pkl")                #u-KDD中的FC
ariginal_file6 = os.path.join(position3, f"group_CI_1.pkl")                     #一致性水平
ariginal_file7 = os.path.join(position3, f"group_utility_loss_1.pkl")           #效用损失
ariginal_file8 = os.path.join(position3, f"C_T_FINAL_1.pkl")           #效用损失
ariginal_file9 = os.path.join(position3, f"GROUP_H_name_1.pkl")

with open(ariginal_file, 'rb') as f:
    ariginal_data = pickle.load(f)      #group_weight_3
with open(ariginal_file2, 'rb') as f:
    ariginal_data2 = pickle.load(f)     #break_ij
with open(ariginal_file3, 'rb') as f:
    ariginal_data3 = pickle.load(f)         #feedback
with open(ariginal_file4, 'rb') as f:
    ariginal_data4 = pickle.load(f)         #GCD
# with open(ariginal_file5, 'rb') as f:
#     ariginal_data5 = pickle.load(f)         #
with open(ariginal_file6, 'rb') as f:
    ariginal_data6 = pickle.load(f)     #一致性水平
with open(ariginal_file7, 'rb') as f:
    ariginal_data7 = pickle.load(f)         #效用损失
with open(ariginal_file8, 'rb') as f:
    ariginal_data8 = pickle.load(f)

with open(ariginal_file9, 'rb') as f:
    ariginal_data9 = pickle.load(f)

print("sss")