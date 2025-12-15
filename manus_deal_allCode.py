import os
#为了防止线程冲突，强制禁用多线程
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from lindo import *
import lindo
import re
import numpy as N

import pandas as pd
import sys
import math
#lindo强绑定python版本为3.8.3，因此下载其他库的时候，要记得锁定python版本
from scipy.stats import invgamma, norm, pearsonr                #共轭正态伽马，正态分布，皮尔逊系数
import matplotlib.pyplot as plt
# -------------------------- 关键修改：指定非交互式后端 --------------------------
# 强制使用 Agg 后端（非交互式，支持批量保存图片，不依赖 GUI）
import matplotlib
matplotlib.use('Agg', force=True)  # force=True 确保覆盖现有后端设置
# 后端设置需在创建 figure 前执行，因此放在函数开头
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
from matplotlib.lines import Line2D
from utility_lindo_sovel import utility_best_safeAPI                #调用子程序的lindo处理模型



#注意，去除轮廓信息部分，既然已经给了先验信任，便假定决策者在给出时已经获取了先验信息
'''
现在是都能跑，就是最后两个基于lindoAPI的，可能是变量太多了（17个），也可能是约束条件太多了（21个&22个），免费的liciense无法跑通，要钱买。
第一个：save_pic_3d()函数里，保存图片的代码因为需要调试给注释了，后面加回来
第二个：utility_best()和MAX_gcd()
#需要返回obj，各fc值，并通过FC_upper_num-fc计算ft，同时得到群共识下的方案权重，以此评估各方案的重要程度
其他的到最后都是跑通的

'''


# 设置中文字体支持
plt.rcParams["font.family"] = ["SimSun", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def read_excel_to_2d_lists(file_path):
    """
    读取指定路径的Excel文件，按工作表分页转换为二维列表

    返回:
        字典，键为工作表名称，值为包含该工作表所有数据的二维列表
        若出错则返回None
    """
    # 定义文件完整路径


    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return None

        # 检查路径是否为文件
        if not os.path.isfile(file_path):
            print(f"错误: 不是有效的文件 - {file_path}")
            return None

        # 读取Excel文件
        xls = pd.ExcelFile(file_path)
        result = {}

        # 遍历所有工作表
        for sheet_name in xls.sheet_names:
            # 读取工作表数据
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                print(f"警告: 工作表 '{sheet_name}' 为空")
                result[sheet_name] = []
                continue

            # 处理表头为字符串列表
            sheet_data = [[str(col) for col in df.columns]]

            # 处理每行数据为字符串列表
            for _, row in df.iterrows():
                row_list = [str(value) if pd.notna(value) else '' for value in row]
                sheet_data.append(row_list)

            result[sheet_name] = sheet_data
        return result

    except PermissionError:
        print(f"错误: 没有权限访问文件 - {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"错误: 无法解析Excel文件，请检查文件格式是否正确")
        return None
    except Exception as e:
        print(f"处理Excel时出错: {str(e)}")
        return None

#构造函数，用于处理初始的HFPR，
def storeTableValue(file_path):

    data = read_excel_to_2d_lists(file_path)
    if data:
        score_REM = []
        sheets_key=[]
        for sheet, content in data.items():
            #print(f"工作表: {sheet}")
            #print(f"共 {len(content)} 行数据")
            if content:
                #for i in range(len(content)):
                    #print(f"第{i}行: {content[i]}")  # content[i]的形式是一行构成的[“”，“”]，content则是一页构成的二维list[[],[],[]]
                score_REM.append(content)                   #读取数据，以三维数组的形式存储结果。第一维为决策者，第二维为各行，第三维为各列。注意，每个评估矩阵的第一行为方案集，可以不要，而且我们只需要上三角矩阵数据即可。
                sheets_key.append(str(sheet))               #存储评估形式
            #print("---")
        '''
        score_REM形式为：
        [
        [['长期合同模式', '股权联合开发模式', '联合项目模式', '跨境全产业链与特色金融模式', 'EPC 联合体模式'], 
        ['/', '0.2', '0.3，0.4', '0.1', '0.8'], 
        ['', '/', '0.4', '0.6，0.7，0.8', '0.2'], 
        ['', '', '/', '0.5', '0.3'], 
        ['', '', '', '/', '0.5'], 
        ['', '', '', '', '/']], 
        [['长期合同模式', '股权联合开发模式', '联合项目模式', '跨境全产业链与特色金融模式', 'EPC 联合体模式'], 
        ['/', '好', '极好，好', '', '好得很'], 
        ['', '/', '极其差', '适中', '好'], 
        ['', '', '/', '很好', '有点差'], 
        ['', '', '', '/', '适中'], 
        ['', '', '', '', '/']], 
        [['长期合同模式', '股权联合开发模式', '联合项目模式', '跨境全产业链与特色金融模式', 'EPC 联合体模式'], 
        ['/', '好', '适中', '极其差', ''], 
        ['', '/', '极其差', '', '好'], 
        ['', '', '/', '很好', '适中'], 
        ['', '', '', '/', '适中'], 
        ['', '', '', '', '/']]]
        '''
    # 检查表格中内容是否符合偏好矩阵形式
    #按各决策者进行分类
        linguisticList_ets=[]
        for experts in range(len(score_REM)):
            columnNum=len(score_REM[experts][0])    #观察行数，看有几个方案

        # 将linguistic矩阵变为可以被运行的数据,并检验是否符合偏好矩阵形式,若符合存储在linguisticList中
            linguisticList = []  # 存储处理后的数据,分上三角和下三角处理
            #因为sheet表名不能重复，因此只看最前面名字是什么类型
            sheets=re.split('[,，;；、]',sheets_key[experts])

            linguisticList.append(sheets[0])          #将属于什么类型的hfpr放入第一位数组中
    #确定术语对应的数值
            if sheets[0] == 'HFLPR' or sheets[0] == 'i-HFLPR':
                linguistList = [['极其差', '很差', '差', '稍差', '适中', '稍好', '好', '很好', '极其好'],
                                [0.06, 0.17, 0.28, 0.39, 0.5, 0.61, 0.72, 0.83, 0.94]]
                for i in range(1,columnNum+1):
                    for j in range(i-1 , columnNum-1):
                        #这里用两个if语句是判定两种写不完全信息的形式，第一个代表不填写，第二个代表用空格表示
                        itemString = score_REM[experts][i][j+1]

                        itemlist = re.split('[,，;；、]',itemString)  # 使用逗号作为分隔符, 需提前导入re.py，即import re,输出的是str类型
                        itemlist2 = []
                        linguistkey = 1
                        for item in range(len(itemlist)):           #将语义信息转换为数字，同时，需要注意如果语义不对给出提示
                            for m in range(len(linguistList[0])):
                                if itemlist[item] == linguistList[0][m]:
                                    itemlist2.append(linguistList[1][m])
                                    #itemlist2.append(1-round(linguistList[1][m],2))
                                    linguistkey = 0
                                    break

                        if linguistkey:
                            linguisticList.append([100])                        #！！！！！！！！！！！！！！！！！！！！
                        else:
                            linguisticList.append(itemlist2)
                        # linguisticList[1].append(itemlist2)
            else:
                for i in range(1,columnNum+1):
                    for j in range(i-1, columnNum-1):
                        linguistkey = 1                     #用来观察数字是否负荷结果
                        itemString = score_REM[experts][i][j+1]

                        itemlist = re.split('[,，;；、]', itemString)  # 使用逗号作为分隔符，输出的是[string,string,...],如果只要数字，需要通过eval来将str变为float类型
                        itemlist2 = []
                        for m in range(len(itemlist)):
                            if bool(itemlist[m])==True:
                                if float(itemlist[m]) <= 1 and float(itemlist[m]) >= 0:
                                    itemlist2.append(float(itemlist[m]) )                    ## 将引号去除，变为原始的float型
                                    #itemlist2.append(1 - round(itemlist[m], 2))         #输出两位小数，这里的处理结果是下三角数据的处理方法，暂时不需要，先留着
                                    linguistkey = 0
                        if linguistkey:
                            linguisticList.append([100])
                        else:
                            linguisticList.append(itemlist2)
                        # linguisticList[1].append(itemlist2)
            linguisticList_ets.append(linguisticList)
        '''
        linguisticList_ets:
        [['犹豫模糊偏好关系类型，HFPR\HFLPR\i-HFPR\i-HFLPR',上三角构成的list，每个模糊集用一个list表征],[上三角构成的list，每个模糊集用一个list表征],[上三角构成的list，每个模糊集用一个list表征]]
        '''
        #print(linguisticList_ets)
        return linguisticList_ets, score_REM          #   score_REM[i][0]为第i页列数，因为行数多了方案集的标头，所以-1为方案个数，故也可用score_REM[0]-1
    else:
        print("未从excel中抓取到数据，程序终止")
        sys.exit(1)

#构造函数，用于处理信任矩阵
def storeConfidanceValue(file_path,num_et):
    data = read_excel_to_2d_lists(file_path)

    if data:
        score_trust = []
        et_T = []
        for sheet, content in data.items():
            # print(f"工作表: {sheet}")
            # print(f"共 {len(content)} 行数据")
            if content:
                # for i in range(len(content)):
                # print(f"第{i}行: {content[i]}")  # content[i]的形式是一行构成的[“”，“”]，content则是一页构成的二维list[[],[],[]]
                score_trust.append(
                    content)  # 读取数据，以三维数组的形式存储结果。第一维为决策者，第二维为各行，第三维为各列。注意，每个评估矩阵的第一行为方案集，可以不要，而且我们只需要上三角矩阵数据即可。

        for i in range(len(score_trust)):
            et_trust=[]
            if(len(score_trust[i][1])-1)==num_et and (len(score_trust[i][2])-1)==num_et:                     #当对所有决策者都评价后
                for j in range(1,num_et+1):
                    et_trust_son=[]
                    et_trust_son.append(score_trust[i][1][j])
                    et_trust_son.append(score_trust[i][2][j])
                    et_trust.append(et_trust_son)
            else:
                print("第{0}个决策者未评估完成".format(i + 1))
                sys.exit(1)
            '''
            [[['0.8', '0.8'], ['0.6', '0.7'], ['0.6', '0.7'], ['0.6', '0.7']], 
            [['0.5', '0.5'], ['0.9', '1.0'], ['0.5', '0.5'], ['0.5', '0.5']], 
            [['0.6', '0.5'], ['0.6', '0.5'], ['0.8', '0.9'], ['0.6', '0.5']], 
            [['0.8', '0.7'], ['0.5', '0.6'], ['0.5', '0.6'], ['0.95', '0.9']]]
            '''
            et_T.append(et_trust)
        return et_T
    else:
        print("未从excel中抓取到数据，程序终止，文件位于："+file_path)
        sys.exit(1)

#构造函数，用于处理背景信息
def storeBackfroundInfo(file_path,num_et):
    data = read_excel_to_2d_lists(file_path)
    etB = [[] for _ in range(num_et)]
    if data:
        if(len(data)-1)==num_et:
            flag_index=0
            born_place=[]                    #记录出生地
            educ=[]                             #教育
            job=[]                             #工作
            title=[]                             #头衔
            kig=0
            end_kig1=0
            end_kig2 = 0
            end_kig3=0
            for i in range(len(data["标准"])):
                if data["标准"][i][0]=='出生地':
                    kig=i
                if data["标准"][i][0]=='教育程度':
                    end_kig1=i
                if data["标准"][i][0] == '工作':
                    end_kig2=i
                if data["标准"][i][0] == '头衔':
                    end_kig3=i
            for i in range(kig,end_kig1):
                born_place.append(data["标准"][i][1])
            for i in range(end_kig1,end_kig2):
                educ.append(data["标准"][i][1])
            for i in range(end_kig2,end_kig3):
                job.append(data["标准"][i][1])
            for i in range(end_kig3,len(data["标准"])):
                title.append(data["标准"][i][1])

            del data["标准"]

            for sheet, content in data.items():
                #性别
                if content[0][1]=='男' or content[0][1]=="男性":
                    etB[flag_index].append(1)
                else:
                    etB[flag_index].append(0)
                #年龄
                if float(content[1][1])<20:
                    etB[flag_index].append(1)
                elif float(content[1][1])<30 and float(content[1][1])>=20:
                    etB[flag_index].append(2)
                elif float(content[1][1])<40 and float(content[1][1])>=30:
                    etB[flag_index].append(3)
                elif float(content[1][1])<50 and float(content[1][1])>=40:
                    etB[flag_index].append(4)
                elif float(content[1][1])<60 and float(content[1][1])>=50:
                    etB[flag_index].append(5)
                else:
                    etB[flag_index].append(6)
                #出生地
                flag_born=0
                for j in range(len(born_place)):
                    if content[2][1]==born_place[j]:
                        etB[flag_index].append(j+1)
                        flag_born=1
                if flag_born==0:
                    etB[flag_index].append(len(born_place))
                #教育
                flag_educ=0
                for j in range(len(educ)):
                    if content[3][1]==educ[j]:
                        etB[flag_index].append(j)
                        flag_educ=1
                if flag_educ == 0:
                    etB[flag_index].append(len(educ))
                #工作
                flag_job=0
                for j in range(len(job)):
                    if content[4][1]==job[j]:
                        etB[flag_index].append(j+1)
                        flag_job=1
                if flag_job==0:
                    etB[flag_index].append(len(job)+1)
                #头衔
                flag_title=0
                for j in range(len(title)):
                    if content[5][1]==title[j]:
                        etB[flag_index].append(j+1)
                        flag_title = 1
                if flag_title==0:
                    etB[flag_index].append(len(educ)+1)

                flag_index=flag_index+1
        else:
            print("专家背景信息个数与评估时的专家个数不符合")
            sys.exit(1)
    else:
        print("未从excel中抓取到数据，程序终止，文件位于："+file_path)
        sys.exit(1)
    print(etB)
    return etB

'''
#遍历生成上三角矩阵
assess_ets=[]
for i in range(len(tableValue)):  # 确定决策者个数，
    # EtScoreValue=StableValue[i]                #[[评分类型][上三角评分]]
    #首先构建上三角形式的assess的数组形式，并定义变量数和初始可选元素个数
    assess=[]
    keyflag=0
    for j in range(len_plans-1):                 #上三角只有评估矩阵的行数-1行，故要-1
        assess_son=[]
        for x in range(len_plans-j-1):
            if (len_plans-j-1):
                assess_son.append(tableValue[i][x+keyflag+1])
        keyflag=keyflag+(len_plans-j-1)
        assess.append(assess_son)
    assess_ets.append(assess)
    print(assess)
'''
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

#确定自信参数
def confidance_pmt(cl,tableValue,trust_z_i,HFPRLIST,HFLPRLIST,f_u_a):                                           #输入的是groupdecision()传输出的cl,solves_Message以及初始的决策者间信任-置信矩阵
    average_confidance=[[] for _ in range(len(cl))]                     #构建一个与cl的结构相同的数组
    standard_confidance=[[] for _ in range(len(cl))]

    #假定隐形自信服从正态分布，均值参数与一致性水平成正比，分布标准差与信息熵成正比
    for i in range(len(cl)):
        average_confidance[i]=cl[i]/max(cl)                             #确定每个决策者的自信均值

    #确定分布的标准差
    #首先计算各犹豫模糊集的犹豫模糊熵
    standard_confidance_H=[[] for _ in range(len(tableValue))]

    for i in range(len(tableValue)):
        for j in range(1,len(tableValue[i])):
            fuzzy_index=0.0                     #存储模糊度
            uncert_index=0.0                    #存储不确定度
            if len(tableValue[i][j])>1:
                sum_tab=0.0
                for m in range(len(tableValue[i][j])):
                    sum_tab=sum_tab+(1-2*abs(tableValue[i][j][m]-0.5))
                fuzzy_index=(1/len(tableValue[i][j]))*sum_tab
                uncert_index=max(tableValue[i][j])-min(tableValue[i][j])
            elif tableValue[i][j][0]==100:
                sum_tab = 0.0
                if tableValue[i][0]=="i-HFLPR" or tableValue[i][0]=="HFLPR":
                    fuzzy_index=1.0
                    uncert_index=max(HFLPRLIST)-min(HFLPRLIST)
                else:
                    fuzzy_index=1.0
                    uncert_index = max(HFPRLIST) - min(HFPRLIST)
            else:
                fuzzy_index=1-2*abs(tableValue[i][j][0]-0.5)

            Eh=(f_u_a*fuzzy_index+(1-f_u_a)*uncert_index)/(f_u_a+(1-f_u_a)*uncert_index)
            standard_confidance_H[i].append(Eh)


    #将确定性水平归一化standard_confidance_H[i].append(Eh)
    # standard_confidance_norm = [[] for _ in range(len(tableValue))]
    # for i in range(len(standard_confidance_H)):
    #     for j in range(len(standard_confidance_H[i])):
    #         standard_confidance_norm[i].append(standard_confidance_H[i][j]/sum(standard_confidance_H[i]))

    #计算并存储各个决策者的自信标准差参数
    standard_confidance=[[] for _ in range(len(tableValue))]
    for i in range(len(standard_confidance_H)):
        confidance_num=0.0
        for j in range(len(standard_confidance_H[i])):
            confidance_num=confidance_num+standard_confidance_H[i][j]

        standard_confidance_zta=(1/(2*math.pi*math.e))*math.exp(2*confidance_num/len(standard_confidance_H[i]))

        standard_confidance[i]=standard_confidance_zta                    #确定每个决策者的隐形自信的平方差
    '''
    [[['0.8', '0.8'], ['0.6', '0.7'], ['0.6', '0.7'], ['0.6', '0.7']], 
    [['0.5', '0.5'], ['0.9', '1.0'], ['0.5', '0.5'], ['0.5', '0.5']], 
    [['0.6', '0.5'], ['0.6', '0.5'], ['0.8', '0.9'], ['0.6', '0.5']], 
    [['0.8', '0.7'], ['0.5', '0.6'], ['0.5', '0.6'], ['0.95', '0.9']]]
    '''
    for i in range(len(trust_z_i)):
        if float(trust_z_i[i][i][1])>=0.5:
            average_confidance[i]=(average_confidance[i]+float(trust_z_i[i][i][0]))/2

    #修正自信的置信度：
    trust_modify=copy_3d_structure(trust_z_i)
    for i in range(len(trust_z_i)):
        for j in range(len(trust_z_i[i])):
            trust_modify[i][j].append(trust_z_i[i][j][0])
            if float(trust_z_i[i][j][1])<0.5:
                trust_modify[i][j].append("0.5")
            else:
                trust_modify[i][j].append(trust_z_i[i][j][1])
    #返回自信参数表格，Confidence Distribution Parameters of DMs
    return average_confidance,standard_confidance,trust_modify          #输出各决策者分布的均值、平方差参数以及修改置信度后的信任矩阵

def trust_func(trust_modify,standard_confidance,a_zj,NUm_ets,index_weight,sure_H,tablevalue):                #
    #filebcak_path为背景信息的excel表路径，standard_confidance正比于伽马分布的率参数， trust_modify为先验信任的均值参数 ,a_zj为逆伽马分布的形状参数，背景相似性，,初始矩阵
    # #构建信任的共轭伽马分布参数
    #计算相似性,并处理均值信息
    #sim_H=[[] for _ in range(NUm_ets)]                           #背景相似性
    ko = [[] for _ in range(NUm_ets)]                            #各决策者间信任的精度参数集
    s_Ave=[[] for _ in range(NUm_ets)]                           #样本均值
    s_stand = [[] for _ in range(NUm_ets)]                       #样本方差
    s_len = 0                                                           #一个样本集中的样本个数
    for i in range(NUm_ets):
        rd_ij = []                                                      #各决策者方案权重差

        for j in range(NUm_ets):             #决策者i对决策者z的相似性，根据
            if i!=j:
                '''
                gender=0.0
                birth=0.0
                job=0.0
                if back_info[i][0]!=back_info[j][0]:
                    gender=1.0
                if back_info[i][2]!=back_info[j][2]:
                    birth =1.0
                if back_info[i][4]!=back_info[j][4]:
                    job=1.0

                sim_Pij=1/6*(gender+birth+job+3-(abs(back_info[i][1]-back_info[j][1])+abs(back_info[i][5]-back_info[j][5]))/3-abs(back_info[i][3]-back_info[j][3])/2)'''

                trust_modify[i][j][0]=float(trust_modify[i][j][0])*float(trust_modify[i][j][1])

                #sim_H[i].append(sim_Pij)
                # 获取精度参数K0
                rd_ij.append(sum((1-abs(et1-et2)) for et1,et2 in zip(index_weight[i],index_weight[j])))                             #快速加和两list              与距离呈反比
                #获取样本数据S的均值与方差
                sd_ij = []  # 各决策者上三角矩阵各元素差值

                for m1 in range(1,len(tablevalue[i])):
                    sd_ijs=0.0
                    if(len(tablevalue[i][m1])!=len(tablevalue[j][m1])):
                        if tablevalue[i][m1][0]!=100 and tablevalue[j][m1][0]!=100:
                            for m2 in range(min(len(tablevalue[i][m1]),len(tablevalue[j][m1]))):
                                sd_ijs=sd_ijs+abs(tablevalue[i][m1][m2]-tablevalue[j][m1][m2])
                            if len(tablevalue[i][m1])>len(tablevalue[j][m1]):
                                for m3 in range(len(tablevalue[j][m1]),len(tablevalue[i][m1])):
                                    sd_ijs=sd_ijs+max(abs(tablevalue[i][m1][m3]-0),abs(tablevalue[i][m1][m3]-1))
                            else:
                                for m3 in range(len(tablevalue[i][m1]),len(tablevalue[j][m1])):
                                    sd_ijs=sd_ijs+max(abs(tablevalue[j][m1][m3]-0),abs(tablevalue[j][m1][m3]-1))
                        if tablevalue[i][m1][0]==100 and tablevalue[j][m1][0]!=100:
                            for m2 in range(len(tablevalue[j][m1])):
                                sd_ijs=sd_ijs+max(abs(tablevalue[j][m1][m2]-0),abs(tablevalue[j][m1][m2]-1))
                        if tablevalue[j][m1][0] == 100 and tablevalue[i][m1][0]!=100:
                            for m2 in range(len(tablevalue[i][m1])):
                                sd_ijs=sd_ijs+max(abs(tablevalue[i][m1][m2]-0),abs(tablevalue[i][m1][m2]-1))
                        if tablevalue[j][m1][0] == 100 and tablevalue[i][m1][0] == 100:
                            sd_ijs = 1
                    if (len(tablevalue[i][m1]) == len(tablevalue[j][m1])):
                        if tablevalue[i][m1][0]!=100 and tablevalue[j][m1][0]!=100:
                            for m2 in range(len(tablevalue[i][m1])):
                                sd_ijs=sd_ijs+abs(tablevalue[i][m1][m2]-tablevalue[j][m1][m2])
                        if tablevalue[i][m1][0]==100 and tablevalue[j][m1][0]!=100:
                            for m2 in range(len(tablevalue[j][m1])):
                                sd_ijs=sd_ijs+max(abs(tablevalue[j][m1][m2]-0),abs(tablevalue[j][m1][m2]-1))
                        if tablevalue[j][m1][0] == 100 and tablevalue[i][m1][0]!=100:
                            for m2 in range(len(tablevalue[i][m1])):
                                sd_ijs=sd_ijs+max(abs(tablevalue[i][m1][m2]-0),abs(tablevalue[i][m1][m2]-1))
                        if tablevalue[j][m1][0] == 100 and tablevalue[i][m1][0] == 100:
                            sd_ijs=1
                    sd_ij.append(1-(sd_ijs/max(len(tablevalue[i][m1]),len(tablevalue[j][m1]))))

                    #度量距离与信任        传统
                # for m1 in range(len(sure_H[i][0])):
                #     for m2 in range(len(sure_H[i][0][m1])):
                #         sd_ij.append(1-(abs(sure_H[i][0][m1][m2][0]-sure_H[j][0][m1][m2][0]))/11)

                    # 度量距离与信任        非线性，以x^2为例
                # for m1 in range(len(sure_H[i][0])):
                #     for m2 in range(len(sure_H[i][0][m1])):
                #         sd_ij.append((1-(abs(sure_H[i][0][m1][m2][0]-sure_H[j][0][m1][m2][0]))/11)**2)

                s_len=len(sd_ij)
                ave_num=sum(sd_ij) / len(sd_ij)
                s_Ave[i].append(ave_num)
                s_stand[i].append(sum(math.pow(x1-ave_num, 2) for x1 in sd_ij))
            else:
                #sim_H[i].append(1.0)
                trust_modify[i][j][0]=float(trust_modify[i][j][0])
                s_Ave[i].append("占位")
                s_stand[i].append("占位")
        for j in range(len(rd_ij)):
            if j==i:
                ko[i].append('占位')
                ko[i].append(NUm_ets*rd_ij[j]/sum(rd_ij))
            else:
                ko[i].append(NUm_ets * rd_ij[j] / sum(rd_ij))




    #因为在最后一个决策者时，不会跑i=j，因此，需要对ko[len(back_info)-1]加一个占位符
    ko[NUm_ets - 1].append('占位')


    #计算并存储后验结果
    #  这里可以搜索 ko（K值），s_Ave（样本均值），填写Prior Trust Parameters of Group表格
    trust_post=copy_3d_structure(trust_modify)
    trust_sets=[[] for _ in range(len(trust_modify))]           #信任样本集
    confi_sets=[[] for _ in range(len(trust_modify))]           #自信样本集
    for i in range(len(trust_modify)):
        for j in range(len(trust_modify)):
            if i==j:
                trust_post[i][j]=["占位","占位","占位","占位"]
            else:
                ave_post=(s_len*s_Ave[i][j]+ko[i][j]*trust_modify[i][j][0])/(s_len+ko[i][j])
                Ko_post=(s_len+ko[i][j])
                a_post=(a_zj+s_len/2)
                b_post=standard_confidance[i]+ko[i][j]*math.pow((s_Ave[i][j]-trust_modify[i][j][0]),2)/(2*(Ko_post))+s_stand[i][j]/2
                trust_post[i][j]=[ave_post,Ko_post,a_post,b_post]
                trust_sets[i].append(ave_post)
                confi_sets[i].append(trust_modify[i][i][0])
                '''
                w1=float(trust_modify[i][i][1])/(float(trust_modify[i][j][1])+float(trust_modify[i][i][1]))
                w2=1-w1
                confi_index=w1*trust_modify[i][i][0]+w2*(1-ave_post)
                if confi_index<=1-ave_post:
                    confi_sets[i].append(confi_index)
                else:
                    confi_sets[i].append(1-ave_post)
                '''
    #Posterior Trust Parameters of Group
    return trust_post,trust_sets,confi_sets                             #信任的共轭正态分布的四个必要参数集合，信任样本集，自信样本集

#获取各决策者的皮尔逊系数
def get_pierson(trust_sets,confidant_sets):
    pierson_et=[]
    for i in range(len(trust_sets)):
        trusts=trust_sets[i]
        confidants=confidant_sets[i]
        #将两数组变为N类
        trustNp=N.array(trusts)
        confidantsNp=N.array(confidants)
        if N.std(trustNp) > 0 and N.std(confidantsNp) > 0:
            pearson_scipy, p_value = pearsonr(trustNp,confidantsNp)                 #计算两数组的皮尔逊系数
        else:
            pearson_scipy, p_value = (N.nan, N.nan)  # 或设置为0
        pierson_et.append(pearson_scipy)

    return pierson_et

#绘制三维图，存储，并得到最终自信与信任值
# 定义二维联合正态分布的概率密度函数,rho为相关性系数，mu1，sigma1为自信的分布参数，mu2与sigma2为信任的分布参数
def joint_normal(x1, x2, rho, mu1, mu2, sigma1, sigma2):                #
    """计算二维联合正态分布的概率密度，处理ρ=±1的特殊情况"""
    if abs(rho) == 1.0:
        # 完全相关的情况，分布退化为一条直线
        # 计算x2应该的值（基于x1和完全相关性）
        expected_x2 = mu2 + rho * (sigma2 / sigma1) * (x1 - mu1)

        # 非常接近预期值时概率密度很大，否则为0
        # 使用一个小的阈值来模拟这种情况
        if N.isclose(x2, expected_x2, atol=1e-3):
            # 边缘分布的密度
            return N.exp(-0.5 * ((x1 - mu1) / sigma1) ** 2) / (N.sqrt(2 * N.pi) * sigma1)
        else:
            return 0.0
    else:
        # 普通情况，使用标准联合正态分布公式
        z1 = (x1 - mu1) / sigma1
        z2 = (x2 - mu2) / sigma2

        denominator = 2 * N.pi * sigma1 * sigma2 * N.sqrt(1 - rho ** 2)
        exponent = - (z1 ** 2 - 2 * rho * z1 * z2 + z2 ** 2) / (2 * (1 - rho ** 2))
        return N.exp(exponent) / denominator

def save_pic_3d(rho, mu1, mu2, sigma1, sigma2, et1, et2,path_pic):
    # 创建网格数据
    x1 = N.linspace(0, 1, 100)
    x2 = N.linspace(0, 1, 100)
    X1, X2 = N.meshgrid(x1, x2)

    # 计算每个点的概率密度，向量化调用
    vectorized_joint= N.vectorize(joint_normal)
    Z =vectorized_joint(X1, X2, rho, mu1, mu2, sigma1, sigma2)  # 使用vectorize处理标量函数

    # 绘制三维图像
    fig = plt.figure(figsize=(16, 14))                  #宽，高
    ax = fig.add_subplot(111, projection='3d')

    # 创建颜色数组，根据x1是否大于x2设置不同颜色
    color_array = N.zeros((X1.shape[0], X1.shape[1], 3))
    color_array[X1 > X2] = [0, 0, 1]  # 蓝色表示x1 > x2
    color_array[X1 <= X2] = [1, 0, 0]  # 红色表示x1 <= x2

    # 绘制表面
    surf = ax.plot_surface(X1, X2, Z, facecolors=color_array, alpha=0.8, linewidth=0.5, edgecolor='k')

    # 设置坐标轴标签和标题
    ax.set_xlabel('self-Confidence', fontsize=20, labelpad=15)
    ax.set_ylabel('Trust', fontsize=20, labelpad=15)
    ax.set_zlabel('Probability Density', fontsize=20, labelpad=15)

    # 调整三个坐标轴的刻度字体大小为15
    ax.tick_params(axis='x', labelsize=15)  # X轴刻度
    ax.tick_params(axis='y', labelsize=15)  # Y轴刻度
    ax.tick_params(axis='z', labelsize=15)  # Z轴刻度
    '''
    plt.title(f'3D-Dimensional Joint Normal Distribution of Confidence and Trust from DM{et1} to DM{et2}',
              fontsize=14, pad=20)
    '''
    # 添加自定义图例（解释颜色代表的含义）
    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='blue area: self-Confidence (Confidence > Trust)'),
        Line2D([0], [0], color='red', lw=4, label='red area: Trust (Confidence ≤ Trust)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

    # 添加额外的文本说明（可选）
    '''
    plt.figtext(0.5, 0.01,
                "图表说明：三维表面高度表示概率密度，颜色区分自信与信任所占空间，"
                "其中红色区域代表信任，蓝色区域代表自信。",
                ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    '''
    # 调整视角
    ax.view_init(elev=45, azim=-125)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)  # 为底部文本留出空间
    # 保存图片到指定路径
    save_path = path_pic
    # 检查路径是否存在，不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # 拼接完整文件名
    file_name = f"pic_et{et1}_et{et2}.png"
    full_path = os.path.join(save_path, file_name)
    # 保存图片，dpi设置为300以保证清晰度
                #为了调试，暂时先把保存图片给取消了
    #plt.savefig(full_path, dpi=300, bbox_inches='tight')                       #暂时不保存

    print(f"图片已保存至: {full_path}")

#计算积分
def integrand(x2, x1, rho, mu1, mu2, sigma1, sigma2):
    return joint_normal(x1, x2, rho, mu1, mu2, sigma1, sigma2)

#计算，返回最后的自信值与信任值
def confi_trust(av_c,st_c,trust_post_par,Trust_set,confidance_set,path_pic):
    # 得到各决策者的自信与信任的相关系数，基于皮尔逊系数
    et_piersons=get_pierson(Trust_set,confidance_set)
    et_piersons_N=N.array(et_piersons)
    et_piersons_filled = N.nan_to_num(et_piersons_N, nan=0.0)             #将nan替换成0.0
    et_piersons_list=et_piersons_filled.tolist()
    confi_trust_set=copy_3d_structure(trust_post_par)                       #储存自信与信任数据
    sigma_trustlist=[[] for _ in range(len(trust_post_par))]
    for i in range(len(trust_post_par)):
        for j in range(len(trust_post_par[i])):
            if i!=j:
                sigma_trust=(trust_post_par[i][j][3]/(trust_post_par[i][j][2]-1))/trust_post_par[i][j][1]
                prob_x1_greater, error1 = dblquad(integrand, 0, 1, lambda x: 0, lambda x: x, args=(et_piersons_list[i],av_c[i],trust_post_par[i][j][0],math.sqrt(st_c[i]),math.sqrt(sigma_trust),))           #注意是标准差参数
                total_prob, error_total = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1, args=(et_piersons_list[i],av_c[i],trust_post_par[i][j][0],math.sqrt(st_c[i]),math.sqrt(sigma_trust),))
                prob_x1_less_or_equal = total_prob - prob_x1_greater
                #保存绘制的图片
                save_pic_3d(et_piersons_list[i],av_c[i],trust_post_par[i][j][0],math.sqrt(st_c[i]),math.sqrt(sigma_trust), i+1, j+1, path_pic)          #跳过0
                # 输出结果
                print(f"自信区域的累积概率: {prob_x1_greater:.6f} (估计误差: {error1:.6e})")
                print(f"信任区域的累积概率: {prob_x1_less_or_equal:.6f}")
                print(f"[0,1]x[0,1] 区域的总概率: {total_prob:.6f} (估计误差: {error_total:.6e})")
                trust_num=prob_x1_less_or_equal/(prob_x1_greater+prob_x1_less_or_equal)
                confi_num=1-trust_num
                confi_trust_set[i][j]=[confi_num,trust_num]                                 #[自信值，信任值]
                sigma_trustlist[i].append(math.sqrt(sigma_trust))
            else:
                confi_trust_set[i][j]=[av_c[i],av_c[i]]
                sigma_trustlist[i].append('zhanwei')


    print(sigma_trustlist)                  #输出降维后的信任标准差
    #(Confidence, Trust) Among All DMs
    return confi_trust_set

#计算个体效用损失的最小化模型
def utility_best(sure_H,C_T_set,trust_modify,et_weight,GCD_,index_weights):
    #各决策者已经去模糊的犹豫模糊集,各决策者对其他决策者的[自信,信任]集合，信任与置信度的集合（置信度为str类），各决策者权重,群共识度阈值,各决策者对应的指标权重
    #确定基于自信与信任的各矩阵修改值
    modify_H_ct_f=[]
    modify_H_FC_FT_f=[]
    modify_H_FC_FT_noL=[]                                                       #储存sure_H
    FC_upper_num=copy_3d_structure(trust_modify)
    num_modify_H=0                                                             #用于计算modify_H_ct_f与modify_H_FC_FT_f的总长度
    lwrbnd=[]               #储存变量下限
    uprbnd=[]               #储存变量上限
    varval=[]               #储存变量起始点
    numval = []             #储存常量，并以numval[i]中的i作为常量索引
    vtype=[]                #储存变量的性质，C代表连续型，B代表二进制
    varindex=[]             #用来存储变量索引（此存储是以一维的形式存储）
    for i in range(len(C_T_set)):               #遍历每个决策者
        modify_H_ct= copy_3d_structure(sure_H[i][0])                           #用于存储约束条件中第一行的常数变量数值
        modify_H_FC_FT=copy_3d_structure(sure_H[i][0])
        modify_H_FC_FT_noL_son=copy_3d_structure(sure_H[i][0])
        # 开始遍历决策者i的去模糊矩阵
        # 已知所有决策着的评估矩阵的方案个数相同
        for m1 in range(len(sure_H[i][0])):
            for m2 in range(len(sure_H[i][0][m1])):
                modify_H_son=0.0
                for j in range(len(C_T_set[i])):        #遍历每个决策者对其他决策者的自信与信任
                    if i!=j:
                        #只与其他决策者共同处理
                        modify_H_son=modify_H_son+(sure_H[i][0][m1][m2][0]*C_T_set[i][j][0]+sure_H[j][0][m1][m2][0]*C_T_set[i][j][1])\
                                     *(float(trust_modify[i][j][1])+float(trust_modify[i][i][1])-float(trust_modify[i][j][1])*float(trust_modify[i][i][1]))
                modify_H_ct[m1][m2]=(modify_H_son+sure_H[i][0][m1][m2][0])/len(C_T_set)         #hij+sum(m≠z，L)(tc*hijz+tt*hijm)
                modify_H_FC_FT[m1][m2]=sure_H[i][0][m1][m2][0]/len(C_T_set)                 #hijz/L
                modify_H_FC_FT_noL_son[m1][m2]=sure_H[i][0][m1][m2][0]                      #hij
                num_modify_H=num_modify_H+1
        modify_H_ct_f.append(modify_H_ct)
        modify_H_FC_FT_f.append(modify_H_FC_FT)
        modify_H_FC_FT_noL.append(modify_H_FC_FT_noL_son)
        #存储FC的取值常量
        for j in range(len(C_T_set[i])):
            FC_upper_num[i][j]=1+float(trust_modify[i][j][1])*float(trust_modify[i][i][1])-float(trust_modify[i][j][1])-float(trust_modify[i][i][1])



    #开始构建模型，使用MPI风格接口，因为存在abs
    #定义约束条件行数：
    ncons=0                 #后续慢慢加
    #定义目标条件行数：
    nobjs=1
    #群决策指标个数
    num_plans=len(modify_H_ct_f[0])+1                                       #指标个数

    # 定义变量个数
    nvars=((len(C_T_set)-1)*len(C_T_set))+num_plans+1                         #即FC变量为（决策者个数-1）*决策者个数，再加上群决策指标权重的变量：即方案个数#FT变量可由FC变量代替,目标优化偏差变量ζ
    #定义常量个数
    nnums=num_modify_H*3+len(et_weight)+1+(len(FC_upper_num)-1)*len(FC_upper_num)+num_plans*len(et_weight) +1+1+1+2           #这是所有常数的个数
    #常量索引如下顺序所示
    #1       #2/(n*(n-1))               n为指标个数
    #len(modify_H_ct_f)  按行数
    #len(modify_H_FC_FT_f)  按行数
    #FC_upper_num中按行数,去除i=j的情况数据
    #len(et_weight)
    #len(modify_H_FC_FT_noL)                  按行数
    #各决策者针对各方案构建的权重个数           #各决策者针对各方案构建的权重集，长度=len(index_weights)*len(index_weights[0])                len(index_weights)为决策者个数len(et_weight)，len(index_weights[0])为方案个数num_plans
    #1       #GCD
    #1.0/num_plans
    #1.0       #常量"1.0"
    #多目标优化的权重常量，暂定0.8，0.2
    #定义变量索引和常量索引
    num_index=1                     #用于存储常量索引
    num_index_list=[[],[],[],[],[],[],[]]        #常量索引库           各个存储的是哪些常量具体看文档中说明

    numval.append(2/(num_plans*(num_plans-1)))
    for i in modify_H_ct_f:
        H_copy_son=copy_3d_structure(i)
        for j in range(len(i)):
            for m in range(len(i[j])):
                numval.append(i[j][m])
                H_copy_son[j][m]=num_index
                num_index=num_index+1
        num_index_list[0].append(H_copy_son)
    for i in modify_H_FC_FT_f:
        H_copy_son2 = copy_3d_structure(i)
        for j in range(len(i)):
            for m in range(len(i[j])):
                numval.append(i[j][m])
                H_copy_son2[j][m] = num_index
                num_index = num_index + 1
        num_index_list[1].append(H_copy_son2)
    H_copy_son3=copy_3d_structure(FC_upper_num)
    for i in range(len(FC_upper_num)):
        for j in range(len(FC_upper_num[i])):
            if i!=j:
                numval.append(FC_upper_num[i][j])
                lwrbnd.append(0.0)                  #FC变量下限
                uprbnd.append(FC_upper_num[i][j])   #FC变量上线
                varval.append(0.0)                  #FC变量起始搜寻点
                vtype.append('C')                   #连续型变量
                H_copy_son3[i][j]=num_index
                num_index = num_index + 1
    num_index_list[2]=H_copy_son3
    for i in et_weight:
        numval.append(i)                                ##存入常量集
        num_index_list[3].append(num_index)
        num_index=num_index+1
    for i in modify_H_FC_FT_noL:
        H_copy_son4=copy_3d_structure(i)
        for j in range(len(i)):
            for m in range(len(i[j])):
                numval.append(i[j][m])                  #存入常量集
                H_copy_son4[j][m]=num_index
                num_index=num_index+1
        num_index_list[4].append(H_copy_son4)
    index_weights_son = copy_3d_structure(index_weights)
    for i in range(len(et_weight)):
        for j in range(num_plans):
            index_weights_son[i][j]=num_index
            num_index = num_index + 1
            numval.append(index_weights[i][j])                  #存入常量集
    num_index_list[5]=index_weights_son

    numval.append(0.8)
    numval.append(0.2)
    num_index_list[6]=[num_index,num_index+1]                                 #存入多目标优化的权重变量
    num_index=num_index+2

    numval.append(GCD_)
    numval.append(1.0/(num_plans*len(et_weight)*(num_plans-1)))
    num_index = num_index + 1       #加入常数1.0/num_plans
    numval.append(1.0)
    num_index=num_index+1           #加入常数"1.0"
    #故，GCD_的索引为num_index-2，2/(n*(n-1))的索引为0,常数1.0/(num_plans*len(et_weights))的索引为Num_index-1，常数"1"的索引为num_index
    '''
    定义变量索引，[FC12,FC13,FC14,FC21,FC23,FC24,...,W1,W2,W3,W4,W5]
    变量索引自己知道就好，只是需要对应唯一变量而已，下面是对变量区间进行定义，并定义变量的起始点，好的起始点能更好找到最优值
    这里我直接在变量区间中任取一点
    '''
    for i in range(num_plans):
        lwrbnd.append(0.0)  # 群决策指标权重变量下限
        uprbnd.append(0.40)  # 群决策指标权重变量上限
        varval.append(0.20)  # 群决策指标权重变量起始搜寻点
        vtype.append('C')  # 连续型变量

    #ζ变量
    lwrbnd.append(-LSconst.LS_INFINITY)
    uprbnd.append(LSconst.LS_INFINITY)
    varval.append(0.0)  # 群决策指标权重变量起始搜寻点
    vtype.append('C')  # 连续型变量

    #建立一个变量索引库：
    var_list_index=[[],[],[]]              #存储FC变量和Wg变量
    var_index=0
    for i in range(len(et_weight)):         #FC变量
        var_list=[]
        for j in range(len(et_weight)):
            if i !=j:
                varindex.append(var_index)
                var_list.append(var_index)
                var_index=var_index+1
            else:
                var_list.append("占位")
        var_list_index[0].append(var_list)
    for i in range(num_plans):          #Wg变量
        varindex.append(var_index)
        var_list_index[1].append(var_index)
        var_index=var_index+1

    var_list_index[2].append(var_index)
    varindex.append(var_index)          #ζ

    #定义指令索引
    ikod = 0
    #定义目标行索引
    iobj = 0
    #定义约束行索引
    icon = 0

    #设置模型指令列表
    objsense = []
    objs_beg = N.empty((1), dtype=N.int32)
    objs_length = []
    cons_beg = N.empty((100), dtype=N.int32)
    cons_length = N.empty((100), dtype=N.int32)
    code = N.empty((50000),dtype=N.int32)            #注意看8000个够不够
    ctype = []                #存储各行约束条件是<=还是>=

    #定义目标函数
    objsense.append(lindo.LS_MIN)
    #设置目标起始位置
    objs_beg[iobj]=ikod
    #开始
    code[ikod]= lindo.EP_PUSH_NUM
    ikod=ikod+1
    code[ikod]=0                                #2/(n*(n-1))常量在numval数列的第一个位置
    ikod = ikod + 1
    J_FLAG = 0
    for z in range(len(et_weight)):                 #len(et_weight)为决策者个数
        for i in range(num_plans-1):
            for j in range(num_plans-1-i):
                #a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod]=lindo.EP_PUSH_NUM          #第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod]=lindo.EP_PUSH_NUM          #第二个常数b
                ikod = ikod + 1
                code[ikod]=num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag=0                                  #用来观察是否为第一个变量和
                for ets in range(len(et_weight)):       #(c12+c13+c14)
                    if ets != z:
                        if ets_flag==0:
                            code[ikod] = lindo.EP_PUSH_VAR    #添加变量c1
                            ikod = ikod + 1
                            code[ikod] =var_list_index[0][z][ets]
                            ikod = ikod + 1
                            ets_flag=1
                        else:
                            code[ikod] = lindo.EP_PUSH_VAR    #添加变量c1
                            ikod = ikod + 1
                            code[ikod] =var_list_index[0][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PLUS        #“+”
                            ikod=ikod+1
                #FC加和完成
                code[ikod] = lindo.EP_MULTIPLY                #b*(c12+c13+c14)
                ikod=ikod+1
                code[ikod] = lindo.EP_PLUS                    #a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        if ets_flag == 0:
                            code[ikod] = lindo.EP_PUSH_NUM                # 第三个常数d2
                            ikod = ikod + 1
                            code[ikod] =num_index_list[1][ets][i][j]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_NUM                #e12
                            ikod = ikod + 1
                            code[ikod] = num_index_list[2][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_VAR                # 添加变量c12
                            ikod = ikod + 1
                            code[ikod] = var_list_index[0][z][ets]
                            ikod = ikod + 1
                            code[ikod] =lindo.EP_MINUS                    #e12-c12  "-"
                            ikod = ikod + 1
                            code[ikod] =lindo.EP_MULTIPLY                 #d2(e12-c12)
                            ikod = ikod + 1
                            ets_flag=1
                        else:
                            code[ikod] = lindo.EP_PUSH_NUM                # d
                            ikod = ikod + 1
                            code[ikod] =num_index_list[1][ets][i][j]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_NUM                #e12
                            ikod = ikod + 1
                            code[ikod] = num_index_list[2][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_VAR                # 添加变量c
                            ikod = ikod + 1
                            code[ikod] = var_list_index[0][z][ets]
                            ikod = ikod + 1
                            code[ikod] =lindo.EP_MINUS                    #e-c
                            ikod = ikod + 1
                            code[ikod] =lindo.EP_MULTIPLY                 #d(e-c)
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PLUS                    #d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS        # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                #计算完成第一个后验hij_z’
                #abs|hij_z‘-hij_z|
                code[ikod] =lindo.EP_PUSH_NUM
                ikod = ikod + 1
                code[ikod]=num_index_list[4][z][i][j]           #hij_z
                ikod = ikod + 1
                code[ikod] = lindo.EP_MINUS               # hij_z‘-hij_z
                ikod = ikod + 1
                code[ikod] = lindo.EP_ABS
                ikod = ikod + 1
                if J_FLAG==1:
                    code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                    ikod = ikod + 1
                else:
                    J_FLAG=1                #再累加中，只有第一个不需要在后面加一个“加号”
    code[ikod] = lindo.EP_MULTIPLY            # (2/(n*(n-1)))*  sum|hijz'-hijz|
    ikod=ikod+1

    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] =num_index_list[6][1]
    ikod = ikod + 1
    code[ikod] = lindo.EP_MULTIPLY  # (2/(n*(n-1)))*  sum|hijz'-hijz|
    ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] = num_index_list[6][0]
    ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_VAR
    ikod = ikod + 1
    code[ikod] =var_index
    ikod=ikod+1
    code[ikod] =lindo.EP_MULTIPLY
    ikod = ikod + 1
    code[ikod] =lindo.EP_PLUS
    ikod = ikod + 1


    objs_length.append(ikod - objs_beg[iobj])   #目标函数的长度
    iobj = iobj + 1
    #下面一串共计 （num_plans-1）*num_plans/2行
    for i in range(num_plans-1):
        for j in range(num_plans-i-1):
            # (hijG-1)*Wi+hijG*wj<=0
            ncons=ncons+1               #加一行
            ctype.append('L')             #小于等于
            cons_beg[icon]=ikod         #索引，表示从此ikod开始，进入下一行
            Z_FLAG=0
            #hijG
            for z in range(len(et_weight)):
                #a
                code[ikod]=lindo.EP_PUSH_NUM     #添加决策者权重常量
                ikod=ikod+1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY            #Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG==0:
                    Z_FLAG=1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod]=lindo.EP_PUSH_NUM              #"1"
            ikod = ikod + 1
            code[ikod] =num_index                       #"1"的索引
            ikod = ikod + 1
            code[ikod]=lindo.EP_MINUS                 #(hijG-1)
            ikod = ikod + 1
            code[ikod] =lindo.EP_PUSH_VAR             #Wi
            ikod = ikod + 1
            code[ikod] =var_list_index[1][i]
            ikod =ikod+1
            code[ikod] = lindo.EP_MULTIPLY
            ikod = ikod + 1
            Z_FLAG=0
            #hijG
            for z in range(len(et_weight)):
                #a
                code[ikod]=lindo.EP_PUSH_NUM     #添加决策者权重常量
                ikod=ikod+1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY            #Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG==0:
                    Z_FLAG=1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR             # Wj
            ikod = ikod + 1
            code[ikod] = var_list_index[1][j+i+1]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY            #hijG*wj
            ikod = ikod + 1
            code[ikod] = lindo.EP_PLUS                #(hijG-1)*Wi+hijG*wj
            ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR            #ζ
            ikod = ikod + 1
            code[ikod] = var_index
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS
            ikod = ikod + 1
            cons_length[icon] = ikod - cons_beg[icon]  # 确定这一行总共包含多少ikod
            icon=icon+1                                 #   记到第几行约束

            #(1-hijG)*Wi-hijG*wj
            ncons = ncons + 1  # 加一行
            ctype.append('L')  # 小于等于
            cons_beg[icon] = ikod  # 索引，表示从此ikod开始，进入下一行
            Z_FLAG = 0
            code[ikod] = lindo.EP_PUSH_NUM  # "1"
            ikod = ikod + 1
            code[ikod] = num_index  # "1"的索引
            ikod = ikod + 1
            # hijG
            for z in range(len(et_weight)):
                # a
                code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                ikod = ikod + 1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG == 0:
                    Z_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS  # (1-hijG)
            ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR  # Wi
            ikod = ikod + 1
            code[ikod] = var_list_index[1][i]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY
            ikod = ikod + 1
            Z_FLAG = 0
            # hijG
            for z in range(len(et_weight)):
                # a
                code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                ikod = ikod + 1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG == 0:
                    Z_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR  # Wj
            ikod = ikod + 1
            code[ikod] = var_list_index[1][j+i+1]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY  # hijG*wj
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS  # (1-hijG)*Wi-hijG*wj
            ikod = ikod + 1

            code[ikod] = lindo.EP_PUSH_VAR            #ζ
            ikod = ikod + 1
            code[ikod] = var_index
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS
            ikod = ikod + 1
            cons_length[icon]=ikod-cons_beg[icon]       #确定这一行总共包含多少ikod
            icon = icon + 1

    #继续下面一行，对群决策阈值的约束
    ncons = ncons + 1  # 加一行
    ctype.append('L')   # 小于等于0
    cons_beg[icon] = ikod  # 索引，表示从此ikod开始，进入下一行
    code[ikod] =lindo.EP_PUSH_NUM                     #1/(num_plans*len(et_weight)*(num_plans-1))
    ikod = ikod + 1
    code[ikod]=num_index-1
    ikod = ikod + 1
    MM_FLAG = 0
    # | h12G - h12 |+| h13G - h13 |+...
    for i in range(num_plans - 1):
        for j in range(num_plans - i - 1):
            # |hijG-hij|
            for mm in range(len(et_weight)):
                Z_FLAG = 0
                # hijG
                for z in range(len(et_weight)):
                    # a
                    code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                    ikod = ikod + 1
                    code[ikod] = num_index_list[3][z]
                    ikod = ikod + 1
                    # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                    ikod = ikod + 1
                    code[ikod] = num_index_list[0][z][i][j]
                    ikod = ikod + 1
                    code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                    ikod = ikod + 1
                    code[ikod] = num_index_list[1][z][i][j]
                    ikod = ikod + 1
                    ets_flag = 0  # 用来观察是否为第一个变量和
                    for ets in range(len(et_weight)):  # (c12+c13+c14)
                        if ets != z:
                            code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                            ikod = ikod + 1
                            code[ikod] = var_list_index[0][z][ets]
                            ikod = ikod + 1
                            if ets_flag == 0:
                                ets_flag = 1
                            else:
                                code[ikod] = lindo.EP_PLUS  # “+”
                                ikod = ikod + 1
                    # FC加和完成
                    code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                    ikod = ikod + 1
                    code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                    ikod = ikod + 1
                    ets_flag = 0  # 用来观察是否为第一个变量和
                    for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                        if ets != z:
                            code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                            ikod = ikod + 1
                            code[ikod] = num_index_list[1][ets][i][j]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_NUM  # e12
                            ikod = ikod + 1
                            code[ikod] = num_index_list[2][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                            ikod = ikod + 1
                            code[ikod] = var_list_index[0][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                            ikod = ikod + 1
                            if ets_flag == 0:
                                ets_flag = 1
                            else:
                                code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                                ikod = ikod + 1
                    code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                    ikod = ikod + 1
                    code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                    ikod = ikod + 1
                    if Z_FLAG == 0:
                        Z_FLAG = 1
                    else:
                        code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                        ikod = ikod + 1
                # hij
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][mm][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][mm][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != mm:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][mm][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != mm:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][mm][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][mm][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MINUS  # hijg-hij  "-"
                ikod = ikod + 1
                code[ikod] = lindo.EP_ABS  # |hijg-hij|
                ikod = ikod + 1
                if MM_FLAG == 0:
                    MM_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                    ikod = ikod + 1
    code[ikod] = lindo.EP_MULTIPLY
    ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_NUM                #gcd_
    ikod = ikod + 1
    code[ikod] =num_index-2
    ikod = ikod + 1
    code[ikod] = lindo.EP_PLUS
    ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_NUM                #1.0
    ikod = ikod + 1
    code[ikod] =num_index
    ikod = ikod + 1
    code[ikod] = lindo.EP_MINUS
    ikod = ikod + 1
    cons_length[icon] = ikod - cons_beg[icon]  # 确定这一行总共包含多少ikod
    icon = icon + 1

    #群决策指标之和=1
    ncons = ncons + 1  # 加一行
    ctype.append('E')   # 小于等于0
    cons_beg[icon] = ikod  # 索引，表示从此ikod开始，进入下一行
    w_Flag=0
    for i in range(num_plans):
        code[ikod] = lindo.EP_PUSH_VAR
        ikod = ikod + 1
        code[ikod]=var_list_index[1][i]
        ikod = ikod + 1
        if w_Flag==0:
            w_Flag=1
        else:
            code[ikod] = lindo.EP_PLUS
            ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] =num_index               #1.0
    ikod = ikod + 1
    code[ikod] = lindo.EP_MINUS
    ikod = ikod + 1
    cons_length[icon] = ikod - cons_beg[icon]  # 确定这一行总共包含多少ikod
    icon = icon + 1

    #说明项目总数
    lsize=ikod

    #将数组变为N.array类型
    lwrbnd = N.array(lwrbnd, dtype=N.double)
    uprbnd = N.array(uprbnd, dtype=N.double)
    varval = N.array(varval, dtype=N.double)
    numval = N.array(numval, dtype=N.double)
    vtype = N.array(vtype, dtype='S1')
    objsense = N.array(objsense, dtype=N.int32)
    objs_length = N.array(objs_length, dtype=N.int32)
    ctype=N.array(ctype, dtype='S1')

    varindex = N.array(varindex,dtype=N.int32)                  #变量索引
    #varindex = N.asarray(None)



    # # create LINDO environment and model objects
    # LicenseKey = N.array('', dtype='S1024')
    # LicenseFile = os.getenv("LINDOAPI_LICENSE_FILE")
    # if LicenseFile == None:
    #     print('Error: Environment variable LINDOAPI_LICENSE_FILE is not set')
    #     sys.exit(1)
    #
    # lindo.pyLSloadLicenseString(LicenseFile, LicenseKey)
    # pnErrorCode = N.array([-1], dtype=N.int32)
    # pEnv = lindo.pyLScreateEnv(pnErrorCode, LicenseKey)
    # try:
    #     pModel = lindo.pyLScreateModel(pEnv, pnErrorCode)
    #     geterrormessage(pEnv, pnErrorCode[0])
    #
    #     #确定线性级别,这里关闭了线性化选项，并通过以下代码段将微分设置为自动模式。
    #     nLinearz = 1
    #
    #     lindo.pyLSsetModelIntParameter(pModel,lindo.LS_IPARAM_NLP_LINEARZ,nLinearz)
    #
    #
    #     # 在凸松弛中选择代数重构级别
    #     nAutoDeriv=1
    #     lindo.pyLSsetModelIntParameter(pModel,LSconst.LS_IPARAM_NLP_AUTODERIV,nAutoDeriv)
    #
    #     # Load instruction list
    #     print("Loading instruction list...")
    #     #约束条件 ncons 29个 ，变量33个
    #     lindo.pyLSloadInstruct(pModel, ncons, nobjs, nvars, nnums,
    #                                        objsense, ctype, vtype, code, lsize,
    #                                        varindex, numval, varval, objs_beg, objs_length,
    #                                        cons_beg, cons_length, lwrbnd, uprbnd)
    #
    #
    #     # lindo.pyLSsetModelIntParameter(pModel,
    #     #                                            LSconst.LS_IPARAM_LP_PRINTLEVEL, 0)
    #     #
    #     #
    #     # lindo.pyLSsetModelIntParameter(pModel,
    #     #                                            LSconst.LS_IPARAM_GOP_PRINTLEVEL, 1)
    #
    #
    #     # solve the model
    #     print("Solving the model...")
    #     pnStatus = N.array([-1], dtype=N.int32)
    #     lindo.pyLSsolveGOP(pModel, pnStatus)
    #
    #     print("Solution status: %d" % pnStatus[0])
    #     print("")
    #
    #     # retrieve the objective value
    #     dObj = N.array([-1.0], dtype=N.double)
    #     lindo.pyLSgetInfo(pModel, LSconst.LS_DINFO_POBJ, dObj)
    #
    #     print("Objective is: %.10f" % dObj[0])
    #     print("")
    #
    #     # retrieve the primal solution
    #     padPrimal = N.empty((nvars), dtype=N.double)
    #     lindo.pyLSgetPrimalSolution(pModel, padPrimal)
    #
    #     print("Primal solution is: ")
    #     for x in padPrimal: print("%.10f" % x)
    #
    #     # delete LINDO model pointer
    #     lindo.pyLSdeleteModel(pModel)  # 通过调用LSdeleteModel，LSdeleteEnv（）来删除模型和环境。
    #
    #     # delete LINDO environment pointer
    #     lindo.pyLSdeleteEnv(pEnv)
    #
    # except lindo.LINDO_Exception as e:
    #     lindo.geterrormessage(pEnv, e.args[1])
    # except Exception as e:
    #     print(f"Other Error => {e}")
    G_weight = []                                               #群决策权重
    c_t_set_mody = copy_3d_structure(C_T_set)                   #最终的自信与信任关系
    none_know_FC_List = copy_3d_structure(C_T_set)  # 储存黑盒信息中的自信水平
    Ncl=0.0
    Objec=0.0
    try:
        result=utility_best_safeAPI(ncons, nobjs, nvars, nnums,
                                            objsense, ctype, vtype, code, lsize,
                                            varindex, numval, varval, objs_beg, objs_length,
                                            cons_beg, cons_length, lwrbnd, uprbnd)
        dObj =result["objective"]
        padPrimal=result["primal"]
        # G_weight_pre=[]


        for j in range(len(var_list_index[1])):
            G_weight.append(padPrimal[var_list_index[1][j]])

        # for G_weight_pre_i in G_weight_pre:
        #     G_weight.append(G_weight_pre_i/sum(G_weight_pre))

        modi_flag = 0
        for i in range(len(C_T_set)):
            for j in range(len(C_T_set[i])):
                if i != j:
                    sure_zone = float(trust_modify[i][j][1]) + float(trust_modify[i][i][1]) - float(
                        trust_modify[i][j][1]) * float(trust_modify[i][i][1])
                    none_know_FC_List[i][j].append(
                        [padPrimal[modi_flag], 1 - sure_zone - padPrimal[modi_flag]])  # 黑盒中的自信与信任
                    trusT = C_T_set[i][j][0] * sure_zone + padPrimal[modi_flag]  # FC为自信，故生成的应该是自信水平
                    #c_t_set_mody[i][j].append(trusT)                   #不输出不信任
                    c_t_set_mody[i][j].append(1 - trusT)
                    modi_flag = modi_flag + 1
                else:
                    c_t_set_mody[i][j].append(C_T_set[i][j][0])
                    none_know_FC_List[i][j].append(
                        [1 - float(trust_modify[i][j][1]) + float(trust_modify[i][i][1]) - float(
                            trust_modify[i][j][1]) * float(trust_modify[i][i][1]), 0.0])  # 模糊自信
        # 不一致水平
        Ncl = padPrimal[len(padPrimal) - 1]

        # padPrimal
        # 计算最终的评估矩阵与群决策矩阵
        FC_H = [[] for _ in range(len(modify_H_ct_f))]
        flag_fc = 0
        for i in range(len(C_T_set)):
            for j in range(len(C_T_set[i])):
                if i != j:
                    FC_H[i].append(padPrimal[flag_fc])
                    flag_fc = flag_fc + 1
                else:
                    FC_H[i].append('占位')

        finalH = []
        for i in range(len(C_T_set)):  # 遍历每个决策者
            finalH_son = copy_3d_structure(sure_H[0][0])
            # 开始遍历决策者i的去模糊矩阵
            # 已知所有决策着的评估矩阵的方案个数相同
            for m1 in range(len(sure_H[i][0])):
                for m2 in range(len(sure_H[i][0][m1])):
                    finalH_H_son = 0.0
                    for j in range(len(C_T_set[i])):  # 遍历每个决策者对其他决策者的自信与信任
                        if i != j:
                            finalH_H_son = finalH_H_son + modify_H_FC_FT_noL[i][m1][m2] * FC_H[i][j] + \
                                           modify_H_FC_FT_noL[j][m1][m2] * (FC_upper_num[i][j] - FC_H[i][j])
                    finalH_H_son = finalH_H_son / len(C_T_set[i])
                    finalH_son[m1][m2] = finalH_H_son + modify_H_ct_f[i][m1][m2]
            finalH.append(finalH_son)

        # 群共识矩阵
        GCD_h = copy_3d_structure(sure_H[0][0])
        for m1 in range(len(sure_H[0][0])):
            for m2 in range(len(sure_H[0][0][m1])):
                sum_G = 0.0
                for i in range(len(finalH)):
                    sum_G = sum_G + et_weight[i] * finalH[i][m1][m2]
                GCD_h[m1][m2] = sum_G

        # 计算群共识度
        GCD_minUL = 0.0
        for i in range(len(C_T_set)):  # 遍历每个决策者
            for m1 in range(len(sure_H[i][0])):
                for m2 in range(len(sure_H[i][0][m1])):
                    GCD_minUL = GCD_minUL + abs(GCD_h[m1][m2] - finalH[i][m1][m2])

        GCD_minUL = 1 - GCD_minUL / (len(C_T_set) * (num_plans) * (num_plans - 1) / 2)

        # #群决策矩阵
        # GCD_H=[[] for _ in range(len(G_weight))]
        # for i in range(len(G_weight)):
        #     for j in range(len(G_weight)):
        #         GCD_H[i].append(G_weight[i]/(G_weight[i]+G_weight[j]))

        # 真实效用降低水平：
        Objec = (dObj - 0.8 * Ncl) / 0.2
        # print("输出结果")
        # print("各方案群决策权重：", G_weight)
        # print("目标1，决策者一致水平偏差：", 1 - Ncl)
        # print("目标2，决策者效用降低：", Objec)
        # print("群共识度", GCD_minUL)
        # print("群共识矩阵", GCD_h)
        # print("修改后的各决策者矩阵：", finalH)
        # print("最终的信任水平", c_t_set_mody)
        return G_weight, c_t_set_mody, 1 - Ncl, Objec,GCD_minUL,GCD_h,finalH,none_know_FC_List
    except Exception as e:
        print("❌ 求解失败:", e)
        return 0,0,0,0,0,0,0,0





#计算群体共识最大的模型
def MAX_gcd(sure_H,C_T_set,trust_modify,et_weight,GCD_,index_weights):
    #各决策者已经去模糊的犹豫模糊集,各决策者对其他决策者的[自信,信任]集合，信任与置信度的集合（置信度为str类），各决策者权重,群共识度阈值,各决策者对应的指标权重
    #确定基于自信与信任的各矩阵修改值
    # 各决策者已经去模糊的犹豫模糊集,各决策者对其他决策者的[自信,信任]集合，信任与置信度的集合（置信度为str类），各决策者权重,群共识度阈值,各决策者对应的指标权重
    # 确定基于自信与信任的各矩阵修改值
    modify_H_ct_f = []
    modify_H_FC_FT_f = []
    FC_upper_num = copy_3d_structure(trust_modify)
    num_modify_H = 0  # 用于计算modify_H_ct_f与modify_H_FC_FT_f的总长度
    lwrbnd = []  # 储存变量下限
    uprbnd = []  # 储存变量上限
    varval = []  # 储存变量起始点
    numval = []  # 储存常量，并以numval[i]中的i作为常量索引
    vtype = []  # 储存变量的性质，C代表连续型，B代表二进制
    varindex = []  # 用来存储变量索引（此存储是以一维的形式存储）
    for i in range(len(C_T_set)):  # 遍历每个决策者
        modify_H_ct = copy_3d_structure(sure_H[i][0])  # 用于存储约束条件中第一行的常数变量数值
        modify_H_FC_FT = copy_3d_structure(sure_H[i][0])
        # 开始遍历决策者i的去模糊矩阵
        # 已知所有决策着的评估矩阵的方案个数相同
        for m1 in range(len(sure_H[i][0])):
            for m2 in range(len(sure_H[i][0][m1])):
                modify_H_son = 0.0
                for j in range(len(C_T_set[i])):  # 遍历每个决策者对其他决策者的自信与信任
                    if i != j:
                        # 只与其他决策者共同处理
                        modify_H_son = modify_H_son + (
                                    sure_H[i][0][m1][m2][0] * C_T_set[i][j][0] + sure_H[j][0][m1][m2][0] *
                                    C_T_set[i][j][1]) \
                                       * (float(trust_modify[i][j][1]) + float(trust_modify[i][i][1]) - float(
                            trust_modify[i][j][1]) * float(trust_modify[i][i][1]))
                modify_H_ct[m1][m2] = (modify_H_son + sure_H[i][0][m1][m2][0]) / len(
                    C_T_set)  # hij+sum(m≠z，L)(tc*hijz+tt*hijm)
                modify_H_FC_FT[m1][m2] = sure_H[i][0][m1][m2][0] / len(C_T_set)  # hijz/L
                num_modify_H = num_modify_H + 1
        modify_H_ct_f.append(modify_H_ct)
        modify_H_FC_FT_f.append(modify_H_FC_FT)
        # 存储FC的取值常量
        for j in range(len(C_T_set[i])):
            FC_upper_num[i][j] = 1 + float(trust_modify[i][j][1]) * float(trust_modify[i][i][1]) - float(
                trust_modify[i][j][1]) - float(trust_modify[i][i][1])

    # 开始构建模型，使用MPI风格接口，因为存在abs
    # 定义约束条件行数：
    ncons = 0  # 后续慢慢加
    # 定义目标条件行数：
    nobjs = 1
    # 群决策指标个数
    num_plans = len(modify_H_ct_f[0]) + 1  # 指标个数

    # 定义变量个数
    nvars = ((len(C_T_set) - 1) * len(C_T_set)) + num_plans+1  # 即FC变量为（决策者个数-1）*决策者个数，再加上群决策指标权重的变量：即方案个数#FT变量可由FC变量代替,
    # 定义常量个数
    nnums = num_modify_H * 2 + len(et_weight)  + (len(FC_upper_num) - 1) * len(FC_upper_num) + num_plans * len(
        et_weight)  + 1 + 1 + 2  # 这是所有常数的个数
    # 常量索引如下顺序所示
    # len(modify_H_ct_f)  按行数
    # len(modify_H_FC_FT_f)  按行数
    # FC_upper_num中按行数,去除i=j的情况数据
    # len(et_weight)
    # 各决策者针对各方案构建的权重个数           #各决策者针对各方案构建的权重集，长度=len(index_weights)*len(index_weights[0])                len(index_weights)为决策者个数len(et_weight)，len(index_weights[0])为方案个数num_plans
    # 1.0/num_plans
    # 1.0       #常量"1.0"
    # 定义变量索引和常量索引
    num_index = 0  # 用于存储常量索引
    num_index_list = [[], [], [], [], [], []]  # 常量索引库           各个存储的是哪些常量具体看文档中说明

    for i in modify_H_ct_f:
        H_copy_son = copy_3d_structure(i)
        for j in range(len(i)):
            for m in range(len(i[j])):
                numval.append(i[j][m])
                H_copy_son[j][m] = num_index
                num_index = num_index + 1
        num_index_list[0].append(H_copy_son)
    for i in modify_H_FC_FT_f:
        H_copy_son2 = copy_3d_structure(i)
        for j in range(len(i)):
            for m in range(len(i[j])):
                numval.append(i[j][m])
                H_copy_son2[j][m] = num_index
                num_index = num_index + 1
        num_index_list[1].append(H_copy_son2)
    H_copy_son3 = copy_3d_structure(FC_upper_num)
    for i in range(len(FC_upper_num)):
        for j in range(len(FC_upper_num[i])):
            if i != j:
                numval.append(FC_upper_num[i][j])
                lwrbnd.append(0.0)  # FC变量下限
                uprbnd.append(FC_upper_num[i][j])  # FC变量上线
                varval.append(0.0)  # FC变量起始搜寻点
                vtype.append('C')  # 连续型变量
                H_copy_son3[i][j] = num_index
                num_index = num_index + 1
    num_index_list[2] = H_copy_son3
    for i in et_weight:
        numval.append(i)  ##存入常量集
        num_index_list[3].append(num_index)
        num_index = num_index + 1
    index_weights_son = copy_3d_structure(index_weights)
    for i in range(len(et_weight)):
        for j in range(num_plans):
            index_weights_son[i][j] = num_index
            num_index = num_index + 1
            numval.append(index_weights[i][j])  # 存入常量集
    num_index_list[4] = index_weights_son

    numval.append(0.51)
    numval.append(0.49)
    num_index_list[5]=[num_index,num_index+1]                                 #存入多目标优化的权重变量
    num_index=num_index+2


    numval.append(2.0 / ((num_plans)*(len(et_weight))*(num_plans-1)))
    num_index = num_index + 1  # 加入常数2.0/((num_plans)*(len(et_weight))*(num_plans-1))
    numval.append(1.0)   # 加入常数"1.0"
    # 故，GCD_的索引为num_index-2，2/(n*(n-1))的索引为0,常数2.0 / ((num_plans)*(len(et_weight))*(num_plans-1))的索引为Num_index-1，常数"1"的索引为num_index
    '''
    定义变量索引，[FC12,FC13,FC14,FC21,FC23,FC24,...,W1,W2,W3,W4,W5]
    变量索引自己知道就好，只是需要对应唯一变量而已，下面是对变量区间进行定义，并定义变量的起始点，好的起始点能更好找到最优值
    这里我直接在变量区间中任取一点
    '''
    for i in range(num_plans):
        lwrbnd.append(0.00)  # 群决策指标权重变量下限
        uprbnd.append(0.25)  # 群决策指标权重变量上限
        varval.append(0.1)  # 群决策指标权重变量起始搜寻点
        vtype.append('C')  # 连续型变量

    #ζ变量
    lwrbnd.append(-LSconst.LS_INFINITY)
    uprbnd.append(LSconst.LS_INFINITY)
    varval.append(0.0)  # 群决策指标权重变量起始搜寻点
    vtype.append('C')  # 连续型变量

    # 建立一个变量索引库：
    var_list_index = [[], [], []]  # 存储FC变量和Wg变量
    var_index = 0
    for i in range(len(et_weight)):  # FC变量
        var_list = []
        for j in range(len(et_weight)):
            if i != j:
                varindex.append(var_index)
                var_list.append(var_index)
                var_index = var_index + 1
            else:
                var_list.append("占位")
        var_list_index[0].append(var_list)
    for i in range(num_plans):  # Wg变量
        varindex.append(var_index)
        var_list_index[1].append(var_index)
        var_index = var_index + 1

    #ζ
    var_list_index[2].append(var_index)
    varindex.append(var_index)

    # 定义指令索引
    ikod = 0
    # 定义目标行索引
    iobj = 0
    # 定义约束行索引
    icon = 0

    # 设置模型指令列表
    objsense = []
    objs_beg = N.empty((1), dtype=N.int32)
    objs_length = []
    cons_beg = N.empty((100), dtype=N.int32)
    cons_length = N.empty((100), dtype=N.int32)
    code = N.empty((50000), dtype=N.int32)  # 注意看8000个够不够
    ctype = []  # 存储各行约束条件是<=还是>=

    # 定义目标函数
    objsense.append(lindo.LS_MAX)         #求最大值
    # 设置目标起始位置
    objs_beg[iobj] = ikod
    # 开始
    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] = num_index  # 1.0
    ikod = ikod + 1

    # 继续下面一行，对群决策阈值的约束

    code[ikod] = lindo.EP_PUSH_NUM  # 2/(num_plans*len(et_weight)*(num_plans-1))
    ikod = ikod + 1
    code[ikod] = num_index - 1
    ikod = ikod + 1
    MM_FLAG=0
    #| h12G - h12 |+| h13G - h13 |+...
    for i in range(num_plans - 1):
        for j in range(num_plans - i - 1):
            # |hijG-hij|
            for mm in range(len(et_weight)):
                Z_FLAG = 0
                # hijG
                for z in range(len(et_weight)):
                    # a
                    code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                    ikod = ikod + 1
                    code[ikod] = num_index_list[3][z]
                    ikod = ikod + 1
                    # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                    ikod = ikod + 1
                    code[ikod] = num_index_list[0][z][i][j]
                    ikod = ikod + 1
                    code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                    ikod = ikod + 1
                    code[ikod] = num_index_list[1][z][i][j]
                    ikod = ikod + 1
                    ets_flag = 0  # 用来观察是否为第一个变量和
                    for ets in range(len(et_weight)):  # (c12+c13+c14)
                        if ets != z:
                            code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                            ikod = ikod + 1
                            code[ikod] = var_list_index[0][z][ets]
                            ikod = ikod + 1
                            if ets_flag == 0:
                                ets_flag = 1
                            else:
                                code[ikod] = lindo.EP_PLUS  # “+”
                                ikod = ikod + 1
                    # FC加和完成
                    code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                    ikod = ikod + 1
                    code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                    ikod = ikod + 1
                    ets_flag = 0  # 用来观察是否为第一个变量和
                    for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                        if ets != z:
                            code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                            ikod = ikod + 1
                            code[ikod] = num_index_list[1][ets][i][j]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_NUM  # e12
                            ikod = ikod + 1
                            code[ikod] = num_index_list[2][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                            ikod = ikod + 1
                            code[ikod] = var_list_index[0][z][ets]
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                            ikod = ikod + 1
                            code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                            ikod = ikod + 1
                            if ets_flag == 0:
                                ets_flag = 1
                            else:
                                code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                                ikod = ikod + 1
                    code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                    ikod = ikod + 1
                    code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                    ikod = ikod + 1
                    if Z_FLAG == 0:
                        Z_FLAG = 1
                    else:
                        code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                        ikod = ikod + 1
                #hij
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][mm][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][mm][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != mm:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][mm][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != mm:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][mm][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][mm][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MINUS  # hijg-hij  "-"
                ikod = ikod + 1
                code[ikod] =lindo.EP_ABS        #|hijg-hij|
                ikod = ikod + 1
                if MM_FLAG==0:
                    MM_FLAG=1
                else:
                    code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                    ikod = ikod + 1
    code[ikod] = lindo.EP_MULTIPLY          #求均值
    ikod =ikod+1

    code[ikod] = lindo.EP_MINUS
    ikod = ikod + 1

    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] =num_index_list[5][1]                #0.2
    ikod = ikod + 1
    code[ikod] = lindo.EP_MULTIPLY                  #
    ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] = num_index_list[5][0]               #0.8
    ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_VAR
    ikod = ikod + 1
    code[ikod] =var_index
    ikod=ikod+1
    code[ikod] = lindo.EP_MULTIPLY
    ikod = ikod + 1
    code[ikod] =lindo.EP_MINUS
    ikod = ikod + 1

    objs_length.append(ikod - objs_beg[iobj])  # 目标函数的长度
    # iobj = iobj + 1
    # 下面一串共计 （num_plans-1）*num_plans/2行
    for i in range(num_plans - 1):
        for j in range(num_plans - i - 1):
            # (hijG-1)*Wi+hijG*wj-ζ<=0
            ncons = ncons + 1  # 加一行
            ctype.append('L')  # 小于等于
            cons_beg[icon] = ikod  # 索引，表示从此ikod开始，进入下一行
            Z_FLAG = 0
            # hijG
            for z in range(len(et_weight)):
                # a
                code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                ikod = ikod + 1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG == 0:
                    Z_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_NUM  # "1"
            ikod = ikod + 1
            code[ikod] = num_index  # "1"的索引
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS  # (hijG-1)
            ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR  # Wi
            ikod = ikod + 1
            code[ikod] = var_list_index[1][i]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY
            ikod = ikod + 1
            Z_FLAG = 0
            # hijG
            for z in range(len(et_weight)):
                # a
                code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                ikod = ikod + 1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG == 0:
                    Z_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR  # Wj
            ikod = ikod + 1
            code[ikod] = var_list_index[1][j+i+1]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY  # hijG*wj
            ikod = ikod + 1
            code[ikod] = lindo.EP_PLUS  # (hijG-1)*Wi+hijG*wj
            ikod = ikod + 1

            code[ikod] = lindo.EP_PUSH_VAR            #ζ
            ikod = ikod + 1
            code[ikod] = var_index
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS
            ikod = ikod + 1

            cons_length[icon] = ikod - cons_beg[icon]  # 确定这一行总共包含多少ikod

            icon = icon + 1  # 记到第几行约束

            # (1-hijG)*Wi-hijG*wj
            ncons = ncons + 1  # 加一行
            ctype.append('L')  # 小于等于
            cons_beg[icon] = ikod  # 索引，表示从此ikod开始，进入下一行
            Z_FLAG = 0
            code[ikod] = lindo.EP_PUSH_NUM  # "1"
            ikod = ikod + 1
            code[ikod] = num_index  # "1"的索引
            ikod = ikod + 1
            # hijG
            for z in range(len(et_weight)):
                # a
                code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                ikod = ikod + 1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG == 0:
                    Z_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS  # (1-hijG)
            ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR  # Wi
            ikod = ikod + 1
            code[ikod] = var_list_index[1][i]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY
            ikod = ikod + 1
            Z_FLAG = 0
            # hijG
            for z in range(len(et_weight)):
                # a
                code[ikod] = lindo.EP_PUSH_NUM  # 添加决策者权重常量
                ikod = ikod + 1
                code[ikod] = num_index_list[3][z]
                ikod = ikod + 1
                # a+b*(c12+c13+c14)+d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                code[ikod] = lindo.EP_PUSH_NUM  # 第一个常数a
                ikod = ikod + 1
                code[ikod] = num_index_list[0][z][i][j]
                ikod = ikod + 1
                code[ikod] = lindo.EP_PUSH_NUM  # 第二个常数b
                ikod = ikod + 1
                code[ikod] = num_index_list[1][z][i][j]
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # (c12+c13+c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c1
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # “+”
                            ikod = ikod + 1
                # FC加和完成
                code[ikod] = lindo.EP_MULTIPLY  # b*(c12+c13+c14)
                ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)
                ikod = ikod + 1
                ets_flag = 0  # 用来观察是否为第一个变量和
                for ets in range(len(et_weight)):  # d2(e12-c12)+d3*(e13-c13)+d4*(e14-c14)
                    if ets != z:
                        code[ikod] = lindo.EP_PUSH_NUM  # 第三个常数d2
                        ikod = ikod + 1
                        code[ikod] = num_index_list[1][ets][i][j]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_NUM  # e12
                        ikod = ikod + 1
                        code[ikod] = num_index_list[2][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_PUSH_VAR  # 添加变量c12
                        ikod = ikod + 1
                        code[ikod] = var_list_index[0][z][ets]
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MINUS  # e12-c12  "-"
                        ikod = ikod + 1
                        code[ikod] = lindo.EP_MULTIPLY  # d2(e12-c12)
                        ikod = ikod + 1
                        if ets_flag == 0:
                            ets_flag = 1
                        else:
                            code[ikod] = lindo.EP_PLUS  # d(e-c)+d(e-c)
                            ikod = ikod + 1
                code[ikod] = lindo.EP_PLUS  # a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）
                ikod = ikod + 1
                code[ikod] = lindo.EP_MULTIPLY  # Wz*(a+b*(c12+c13+c14)+（d(e-c)+d(e-c)）)
                ikod = ikod + 1
                if Z_FLAG == 0:
                    Z_FLAG = 1
                else:
                    code[ikod] = lindo.EP_PLUS  # Wz1*A1+Wz2*A2
                    ikod = ikod + 1
            code[ikod] = lindo.EP_PUSH_VAR  # Wj
            ikod = ikod + 1
            code[ikod] = var_list_index[1][j+i+1]
            ikod = ikod + 1
            code[ikod] = lindo.EP_MULTIPLY  # hijG*wj
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS  # (1-hijG)*Wi-hijG*wj
            ikod = ikod + 1

            code[ikod] = lindo.EP_PUSH_VAR            #ζ
            ikod = ikod + 1
            code[ikod] = var_index
            ikod = ikod + 1
            code[ikod] = lindo.EP_MINUS
            ikod = ikod + 1
            cons_length[icon] = ikod - cons_beg[icon]  # 确定这一行总共包含多少ikod
            icon = icon + 1

    # 群决策指标之和=1
    ncons = ncons + 1  # 加一行
    ctype.append('E')  # 等于0
    cons_beg[icon] = ikod  # 索引，表示从此ikod开始，进入下一行
    w_Flag = 0
    for i in range(num_plans):
        code[ikod] = lindo.EP_PUSH_VAR
        ikod = ikod + 1
        code[ikod] = var_list_index[1][i]
        ikod = ikod + 1
        if w_Flag == 0:
            w_Flag = 1
        else:
            code[ikod] = lindo.EP_PLUS
            ikod = ikod + 1
    code[ikod] = lindo.EP_PUSH_NUM
    ikod = ikod + 1
    code[ikod] = num_index  # 1.0
    ikod = ikod + 1
    code[ikod] = lindo.EP_MINUS
    ikod = ikod + 1
    cons_length[icon] = ikod - cons_beg[icon]  # 确定这一行总共包含多少ikod
    icon = icon + 1

    # 说明项目总数
    lsize = ikod

    # 将数组变为N.array类型
    lwrbnd = N.array(lwrbnd, dtype=N.double)
    uprbnd = N.array(uprbnd, dtype=N.double)
    varval = N.array(varval, dtype=N.double)
    numval = N.array(numval, dtype=N.double)
    vtype = N.array(vtype, dtype=N.character)
    objsense = N.array(objsense, dtype=N.int32)
    objs_length = N.array(objs_length, dtype=N.int32)
    ctype = N.array(ctype, dtype=N.character)

    varindex = N.array(varindex, dtype=N.int32)  # 变量索引
    # varindex = N.asarray(None)

    # create LINDO environment and model objects
    LicenseKey = N.array('', dtype='S1024')
    LicenseFile = os.getenv("LINDOAPI_LICENSE_FILE")
    if LicenseFile == None:
        print('Error: Environment variable LINDOAPI_LICENSE_FILE is not set')
        sys.exit(1)

    lindo.pyLSloadLicenseString(LicenseFile, LicenseKey)
    pnErrorCode = N.array([-1], dtype=N.int32)
    pEnv = lindo.pyLScreateEnv(pnErrorCode, LicenseKey)

    pModel = lindo.pyLScreateModel(pEnv, pnErrorCode)
    geterrormessage(pEnv, pnErrorCode[0])

    # 确定线性级别,这里关闭了线性化选项，并通过以下代码段将微分设置为自动模式。
    nLinearz = 1

    errorcode = lindo.pyLSsetModelIntParameter(pModel,
                                               lindo.LS_IPARAM_NLP_LINEARZ,
                                               nLinearz)

    geterrormessage(pEnv, errorcode)

    # 在凸松弛中选择代数重构级别
    '''
    nCRAlgReform = 1
    errorcode = lindo.pyLSsetModelIntParameter(pModel,
                                               lindo.LS_IPARAM_NLP_CR_ALG_REFORM,
                                               nCRAlgReform)
    geterrormessage(pEnv, errorcode)

    # 选择凸松弛级别
    nConvexRelax = 0
    errorcode = lindo.pyLSsetModelIntParameter(pModel,
                                               lindo.LS_IPARAM_NLP_CONVEXRELAX,
                                               nConvexRelax)
    geterrormessage(pEnv, errorcode)
    '''
    nAutoDeriv = 1
    errorcode = lindo.pyLSsetModelIntParameter(pModel,
                                               LSconst.LS_IPARAM_NLP_AUTODERIV,
                                               nAutoDeriv)
    geterrormessage(pEnv, errorcode)

    # Load instruction list
    print("Loading instruction list...")
    # 约束条件 ncons 22个 ，变量17个
    errorcode = lindo.pyLSloadInstruct(pModel, ncons, nobjs, nvars, nnums,
                                       objsense, ctype, vtype, code, lsize,
                                       varindex, numval, varval, objs_beg, objs_length,
                                       cons_beg, cons_length, lwrbnd, uprbnd)

    geterrormessage(pEnv, errorcode)
    errorcode = lindo.pyLSsetModelIntParameter(pModel,
                                               LSconst.LS_IPARAM_LP_PRINTLEVEL, 0)
    geterrormessage(pEnv, errorcode)

    errorcode = lindo.pyLSsetModelIntParameter(pModel,
                                               LSconst.LS_IPARAM_GOP_PRINTLEVEL, 1)
    geterrormessage(pEnv, errorcode)

    # solve the model
    print("Solving the model2...")
    pnStatus = N.array([-1], dtype=N.int32)
    errorcode = lindo.pyLSsolveGOP(pModel, pnStatus)  # 免费的liciense无法求解，难搞。
    geterrormessage(pEnv, errorcode)
    print("Solution status: %d" % pnStatus[0])
    print("")

    # retrieve the objective value
    dObj = N.array([-1.0], dtype=N.double)
    errorcode = lindo.pyLSgetInfo(pModel, LSconst.LS_DINFO_POBJ, dObj)
    geterrormessage(pEnv, errorcode)
    print("Objective is: %.5f" % dObj[0])
    print("")

    # retrieve the primal solution
    padPrimal = N.empty((nvars), dtype=N.double)
    errorcode = lindo.pyLSgetPrimalSolution(pModel, padPrimal)
    geterrormessage(pEnv, errorcode)
    print("Primal solution is: ")
    for x in padPrimal: print("%.5f" % x)

    errorcode = lindo.pyLSdeleteModel(pModel)  # 通过调用LSdeleteModel，LSdeleteEnv（）来删除模型和环境。
    geterrormessage(pEnv, errorcode)

    # delete LINDO environment pointer
    errorcode = lindo.pyLSdeleteEnv(pEnv)

    #需要返回obj，各fc值，并通过FC_upper_num-fc计算ft，同时得到群共识下的方案权重，以此评估各方案的重要程度
    G_weight_pre = []
    G_weight = []
    c_t_set_mody = copy_3d_structure(C_T_set)
    fcs=copy_3d_structure(C_T_set)
    for j in range(len(var_list_index[1])):
        G_weight_pre.append(1 / padPrimal[var_list_index[1][j]])

    for G_weight_pre_i in G_weight_pre:
        G_weight.append(G_weight_pre_i / sum(G_weight_pre))

    modi_flag = 0
    for i in range(len(C_T_set)):
        for j in range(len(C_T_set[i])):
            if i != j:
                sure_zone = float(trust_modify[i][j][1]) + float(trust_modify[i][i][1]) - float(
                    trust_modify[i][j][1]) * float(trust_modify[i][i][1])
                trusT = C_T_set[i][j][0] * sure_zone + padPrimal[modi_flag] * (1 - sure_zone)           #FC结果指为confidence
                fcs[i][j].append(padPrimal[modi_flag])                  #fc
                fcs[i][j].append(1-sure_zone-padPrimal[modi_flag])          #ft
                c_t_set_mody[i][j].append( trusT)
                c_t_set_mody[i][j].append(1-trusT)
                modi_flag = modi_flag + 1
            else:
                c_t_set_mody[i][j] = C_T_set[i][j]
    # 不一致水平
    Ncl = padPrimal[len(padPrimal) - 1]

    #
    h_d=[]
    h_alls=copy_3d_structure(modify_H_ct_f[0])
    for m1 in range(len(sure_H[0][0])):
        for m2 in range(len(sure_H[0][0][m1])):
            h_alls_1=0
            for i in range(len(C_T_set)):
                for l in range(len(C_T_set)):
                    if i != l:
                        h_alls_1=h_alls_1+sure_H[i][0][m1][m2][0]*fcs[i][l][0]+sure_H[l][0][m1][m2][0]*fcs[i][l][1]
            h_alls[m1][m2]=h_alls_1

    for i in range(len(C_T_set)):
        for m1 in range(len(sure_H[i][0])):
            for m2 in range(len(sure_H[i][0][m1])):
                h_pro=modify_H_ct_f[i][m1][m2]+modify_H_FC_FT_f[i][m1][m2]+h_alls[m1][m2]/len(C_T_set)
                h_d.append(abs(h_pro-sure_H[i][0][m1][m2][0]))

    #群决策矩阵
    GCD_H=[[] for _ in range(len(G_weight))]
    for i in range(len(G_weight)):
        for j in range(len(G_weight)):
            GCD_H[i].append(G_weight[i]/(G_weight[i]+G_weight[j]))


    # 真实效用降低水平：
    Objec = (dObj[0] + 0.51 * Ncl) / 0.49
    print("输出结果")
    print("各方案群决策权重：", G_weight)
    print("目标1，决策者一致水平偏差：", 1-Ncl)
    print("目标2，群共识最优：", Objec)
    print("损失效用：", sum(h_d) / (len(sure_H[0][0]) * (len(sure_H[0][0]) - 1)))
    print("最终的信任水平", c_t_set_mody)
    print("群共识矩阵", GCD_H)




    return G_weight,c_t_set_mody,1-Ncl,Objec,sum(h_d) / (len(sure_H[0][0]) * (len(sure_H[0][0]) - 1))

def funcConf(trust_mat):
    non_conf = [[] for _ in range(len(trust_mat))]
    for i in range(len(trust_mat)):
        for j in range(len(trust_mat)):
            nonc=1+float(trust_mat[i][i][1])*float(trust_mat[i][j][1])-float(trust_mat[i][i][1])-float(trust_mat[i][j][1])
            non_conf[i].append(nonc)
            print(nonc)

    return non_conf

#反馈函数，用于考量由于自身距离与群决策的差距，而降低的自信度与置信度
def feedback(final_Trus, FinalDMH, Final_GroupH, Final_GCD):
    """
    基于自信公式的信任矩阵反馈调节函数

    参数:
    final_Trus: 最终信任值矩阵 [[信任值], ...]
    FinalDMH: 各决策者的上三角矩阵集合
    Final_GroupH: 群决策上三角矩阵
    Final_GCD: 全局共识度

    返回:
    C_T_set: 修改后的信任-不信任度矩阵 [[不信任度, 信任度], ...]
    """

    # 1. 计算各决策者与群决策的距离 Di
    n_decisions = len(FinalDMH)  # 决策者数量
    Di_list = []  # 存储每个决策者的距离

    for i in range(n_decisions):
        # 获取决策者i的上三角矩阵
        dm_matrix = FinalDMH[i]

        # 计算绝对距离（平均绝对误差）
        total_diff = 0
        element_count = 0

        # 遍历上三角矩阵的每个元素
        for j in range(len(dm_matrix)):
            for k in range(len(dm_matrix[j])):
                # 注意：Final_GroupH的元素是单元素列表，需要加[0]访问
                diff = abs(dm_matrix[j][k] - Final_GroupH[j][k])
                total_diff += diff
                element_count += 1

        # 计算平均绝对距离
        Di = total_diff / element_count if element_count > 0 else 0
        Di_list.append(Di)

    # 2. 计算距离均值
    mean_Di_max = N.max(Di_list) if Di_list else 0
    mean_Di=mean_Di_max*3/4

    # 3. 根据final_Trus创建初始的C_T_set
    # 规则：C_T_set[i][j] = [不信任度, 信任度]
    # 当i=j时，C_T_set[i][i] = [自信度, 自信度]
    C_T_set = []
    for i in range(len(final_Trus)):
        dm_trust = []
        for j in range(len(final_Trus[i])):
            # 提取信任值
            trust_val = final_Trus[i][j][0]

            # 对于对角线元素（自信度）
            if i == j:
                # [自信度, 自信度]
                dm_trust.append([trust_val, trust_val])
            else:
                # [不信任度, 信任度]
                dm_trust.append([1 - trust_val, trust_val])
        C_T_set.append(dm_trust)

    # 4. 对距离大于3/4距离的决策者进行特殊处理
    mini_num=0          #少数派个数
    for m in range(n_decisions):
        Dm = Di_list[m]
        if Dm > mean_Di:
            mini_num+=1
    for m in range(n_decisions):
        Dm = Di_list[m]

        if Dm > mean_Di:
            # 获取决策者m的自我信任值（自信度）
            self_trust = final_Trus[m][m][0]

            # 避免除以零
            if self_trust == 0:
                self_trust = 1e-6  # 使用极小值避免除零

            # 计算调节因子
            adjustment_factor = math.exp(-(4/3)*(1 / self_trust) * (Dm / Final_GCD)*((n_decisions-mini_num)/(n_decisions)))

            # 修改自信度：C_T_set[m][m] = [自信度*调整因子, 自信度*调整因子]
            adjusted_self_trust = self_trust * adjustment_factor
            C_T_set[m][m] = [adjusted_self_trust, adjusted_self_trust]

            # 修改该决策者对所有其他决策者的信任度
            for j in range(len(C_T_set[m])):
                if j != m:
                    # 原始信任值
                    original_trust = final_Trus[m][j][0]

                    # 计算调整后的不信任度
                    adjusted_distrust = (1 - original_trust) * adjustment_factor

                    # 计算调整后的信任度
                    adjusted_trust = 1 - adjusted_distrust

                    # 更新C_T_set
                    C_T_set[m][j] = [adjusted_distrust, adjusted_trust]

    return C_T_set

filePath = r"./data/scores_site.xlsx"            #评分信息
filePath2 =r"./data/trust_ets.xlsx"             #初始信任矩阵
filePath4=r"./data/3d_pic"                            #自信与信任的图片保存地址

hfpr_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]                         #存储数值list元素
hflpr_list=[0.06, 0.17, 0.28, 0.39, 0.5, 0.61, 0.72, 0.83, 0.94]        #存储语义list元素

# 在计算犹豫模糊熵时，假定设置a为0.5

fuz_unc_a = 0.5


#设置逆伽马分布的形状参数为0.5
a_gama_T=0.5

#设置群共识阈值
#GCD=0.81
if __name__=='__main__':
    TabValue,Num_planses=storeTableValue(filePath)                          #读取excel数据并存储到构建的list中
    #确定决策者个数
    num_ets=len(Num_planses)

    len_index=len(Num_planses[0][0])                                        #必须保证，每次的指标是一样的
    # trust1_test = [[[0.91, 0.363], [0.695, 0.758], [0.604, 0.405], [0.615, 0.551], [0.482, 0.754]],
    #                [[0.444, 0.911], [0.87, 0.423], [0.9, 0.343], [0.534, 0.568], [0.789, 0.847]],
    #                [[0.745, 0.466], [0.59, 0.775], [0.837, 0.7], [0.784, 0.864], [0.341, 0.843]],
    #                [[0.93, 0.723], [0.559, 0.852], [0.382, 0.841], [0.406, 0.529], [0.791, 0.948]],
    #                [[0.82, 0.306], [0.349, 0.806], [0.577, 0.651], [0.413, 0.798], [0.697, 0.95]]]
    #
    # assessment1_test = [
    #     ['HFPR', [0.3, 0.4], [0.4], [0.4, 0.5], [0.5], [0.4], [0.5], [0.6, 0.7], [0.7], [0.5, 0.6], [0.6], [0.6, 0.7],
    #      [0.5], [0.5], [0.4], [0.4, 0.5]],
    #     ['HFPR', [0.3, 0.4], [0.3, 0.4], [0.5], [0.4, 0.5], [0.5, 0.6], [0.4, 0.5], [0.5, 0.6], [0.6], [0.6, 0.7],
    #      [0.5, 0.6, 0.7], [0.6, 0.7], [0.6, 0.7], [0.4, 0.5], [0.5, 0.6], [0.6]],
    #     ['HFPR', [0.4], [0.5], [0.6, 0.7], [0.7], [0.6, 0.7], [0.6, 0.7], [0.7], [0.7], [0.6, 0.7], [0.5, 0.6], [0.6],
    #      [0.5, 0.6], [0.5], [0.4, 0.5], [0.5]],
    #     ['HFPR', [0.3, 0.4], [0.2, 0.3], [0.3], [0.3, 0.4], [0.3, 0.4], [0.4, 0.5], [0.5], [0.5], [0.5],
    #      [0.5, 0.6, 0.7], [0.6], [0.6, 0.7], [0.4, 0.5, 0.6], [0.4, 0.5], [0.5]],
    #     ['i-HFLPR', [0.28, 0.39], [0.39], [0.5, 0.61], [0.5], [0.39, 0.5], [0.61, 0.72], [0.72], [0.72], [0.61], [0.61],
    #      [0.61], [0.5, 0.61], [0.5], [100], [0.39, 0.5]]]

    cl_index,needModify,GCD,groupIndexweight,Qvalue,solves_Message,index_weight,sure_Hs,group_Hs=groupdecision(TabValue,len_index,0.9)          #群共识至少要大于0.9                                #设置GCI的阈值为0.9
    #cl_index, needModify, GCD, groupIndexweight, Qvalue, solves_Message, index_weight, sure_Hs, group_Hs = groupdecision(assessment1_test, len_index, 0.9)  # 群共识至少要大于0.9                                #设置GCI的阈值为0.9

    #设置该群共识阈值为初始群共识

    trustH=storeConfidanceValue(filePath2,num_ets)              #num_ets用于进入擦看是否存在决策者未对某决策者做出评价的情况

    #back_info=storeBackfroundInfo(filePath3,num_ets)            #获取已经处理成数字的背景信息

    #自信分布
    av_conf,st_conf,trust_M=confidance_pmt(cl_index,TabValue,trustH,hfpr_list,hflpr_list,fuz_unc_a )            #fuz_unc_a为明确模糊度与不确定度权重的参数a
    #av_conf, st_conf, trust_M = confidance_pmt(cl_index, assessment1_test, trust1_test, hfpr_list, hflpr_list,fuz_unc_a)  # fuz_unc_a为明确模糊度与不确定度权重的参数a

    #不自信水平
    non_confi=funcConf(trust_M)

    #信任的共轭正态伽马分布
    trust_post_par,Trust_set,confidance_set=trust_func(trust_M,st_conf,a_gama_T,num_ets,index_weight,sure_Hs,TabValue)                 #对trust_M进行了修改
    #trust_post_par, Trust_set, confidance_set = trust_func(trust_M, st_conf, a_gama_T, num_ets, index_weight, sure_Hs,assessment1_test)  # 对trust_M进行了修改
    #trust_post_par：[[[“占位”，“占位”，“占位”，“占位”],[均值，精度系数，伽马分布的a，伽马分布的b],[决策者1对决策者3],..]，[决策者2],[决策者3],[]]

    confi_trust_sets=confi_trust(av_conf,st_conf,trust_post_par,Trust_set,confidance_set,filePath4)



    #个体效用损失最小
    #for i in range(10):
    #no_knowledge_infor时黑盒中的自信与信任信息
    minGw,min_c_t_set_mody,minGcl,minGObjec,GCD_minULs,GCD_HH,finalHH,no_knowledge_infor=utility_best(sure_Hs,confi_trust_sets,trust_M,Qvalue,GCD,index_weight)
    print("输出结果")
    print("各方案群决策权重：", minGw)
    print("目标1，决策者一致水平偏差：", minGcl)
    print("目标2，决策者效用降低：", minGObjec)
    print("群共识度", GCD_minULs)
    print("群共识矩阵", GCD_HH)
    print("修改后的各决策者矩阵：", finalHH)
    print("最终的信任水平", min_c_t_set_mody)
    print("黑盒中的自信与信任水平", no_knowledge_infor)
    print('_______________________________________')
    #群决策权重,信任水平，群一致水平，最小损失效用

    trust_modi=feedback(min_c_t_set_mody,finalHH,GCD_HH,GCD_minULs)

    print("反馈后的信任水平",trust_modi)








    #最大群共识
    #maxG_weight,maxc_t_set_mody,maxGcl,maxObjec,maxXY=MAX_gcd(sure_Hs,confi_trust_sets,trust_M,Qvalue,GCD,index_weight)
    #群决策权重,信任水平，群一致水平，最大群共识，损失效用
    #['i-HFLPR', [0.61, 0.72], [0.5], [0.06, 0.28], [100], [0.28], [100], [0.72], [0.28, 0.39, 0.61], [0.72, 0.83], [0.94]]


