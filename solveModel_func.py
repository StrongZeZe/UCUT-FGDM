from lindo import *
import lindo
import numpy as N
import os
import time
import multiprocessing
#为了防止线程冲突，强制禁用多线程
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def copy_3d_structure(original):
    return [
        [[] for _ in sublist]  # 第三维：为每个子列表创建对应长度的空列表
        for sublist in original  # 第二维：遍历原始列表的每个子列表
    ]


# 自定义异常类
class SolveModelError(Exception):
    """solveModel调用失败异常"""
    pass


def solveModel_worker(tableValue, len_planses, result_queue):
    """在工作进程中运行solveModel"""
    try:
        # 在工作进程中也需要设置环境变量
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        result = solveModel_original(tableValue, len_planses)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', str(e)))


def solveModel_with_timeout(tableValue, len_planses, timeout=5, max_retries=3):
    """带超时和重试机制的solveModel"""
    for attempt in range(max_retries):
        print(f"第 {attempt + 1} 次尝试调用solveModel...")

        # 创建进程通信队列
        result_queue = multiprocessing.Queue()

        # 创建工作进程
        process = multiprocessing.Process(
            target=solveModel_worker,
            args=(tableValue, len_planses, result_queue)
        )

        try:
            # 启动进程
            process.start()

            # 等待结果，设置超时
            start_time = time.time()
            while time.time() - start_time < timeout:
                if process.is_alive():
                    time.sleep(0.1)  # 短暂休眠避免过度占用CPU
                else:
                    break

            # 检查进程状态
            if process.is_alive():
                print(f"第 {attempt + 1} 次尝试超时，终止进程...")
                process.terminate()
                process.join(timeout=2)  # 等待进程终止
                if process.is_alive():
                    process.kill()  # 强制终止
                continue  # 重试

            # 尝试获取结果
            if not result_queue.empty():
                status, result = result_queue.get()
                if status == 'success':
                    print(f"第 {attempt + 1} 次尝试成功")
                    return result
                else:
                    print(f"第 {attempt + 1} 次尝试出错: {result}")
                    continue  # 重试
            else:
                print(f"第 {attempt + 1} 次尝试无结果")
                continue  # 重试

        except Exception as e:
            print(f"第 {attempt + 1} 次尝试出现异常: {e}")
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            continue  # 重试

        finally:
            # 确保进程被终止
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

    # 所有重试都失败，抛出特定异常
    raise SolveModelError(f"经过 {max_retries} 次尝试后仍无法成功调用solveModel")

def solveModel_original(tableValue,len_planses):
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
        pnStatus = N.array([-1], dtype=N.int32)
        lindo.pyLSoptimize(pModel, LSconst.LS_METHOD_FREE,
                                       pnStatus)  # 通过调用LSoptimize（）（或如果有整数变量的LSsolveMIP（））来解决这个问题。     使用障碍求解器
        # errorcode = lindo.pyLSoptimize(pModel,lindo.LS_METHOD_NLP,pnStatus)
        #geterrormessage(pEnv, errorcode)  # 检测

        # retrieve the objective value       获取最优结果
        dObj = N.array([-1.0], dtype=N.double)
        lindo.pyLSgetInfo(pModel, LSconst.LS_DINFO_POBJ,
                                      dObj)  # 通过调用LSgetInfo（）、LSget初级解决方案（）和LSget双元解决方案（）来检索解决方案。
        #geterrormessage(pEnv, errorcode)
        #print("Objective is: %.5f" % dObj[0])
       # print("")

        # retrieve the primal solution       获取变量数值
        padPrimal = N.empty((nN), dtype=N.double)
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

# 保持原有的函数名兼容性
def solveModel(tableValue, len_planses):
    """新的solveModel函数，包含超时和重试机制"""
    return solveModel_with_timeout(tableValue, len_planses, timeout=5, max_retries=5)