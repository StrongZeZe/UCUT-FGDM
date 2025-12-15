import os
#为了防止线程冲突，强制禁用多线程
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as N
from lindo import *
import lindo
# === 新增：安全调用接口 ===
import concurrent.futures
import multiprocessing as mp
import time

def solve_complex_lindo(args_dict):

    try:
        LicenseKey = N.array('', dtype='S1024')
        LicenseFile = os.getenv("LINDOAPI_LICENSE_FILE")
        if LicenseFile == None:
            print('Error: Environment variable LINDOAPI_LICENSE_FILE is not set')
            sys.exit(1)

        lindo.pyLSloadLicenseString(LicenseFile, LicenseKey)
        pnErrorCode = N.array([-1], dtype=N.int32)
        pEnv = lindo.pyLScreateEnv(pnErrorCode, LicenseKey)
        ncons = args_dict['ncons']
        nobjs = args_dict['nobjs']
        nvars = args_dict['nvars']
        nnums = args_dict['nnums']
        objsense = N.array(args_dict['objsense'], dtype=N.int32)

        ctype = N.array(args_dict['ctype'], dtype='S1')

        vtype = N.array(args_dict['vtype'], dtype='S1')

        code = N.array(args_dict['code'], dtype=N.int32)

        lsize =args_dict['lsize']

        varindex = N.array(args_dict['varindex'], dtype=N.int32)

        numval = N.array(args_dict['numval'], dtype=N.double)

        varval = N.array(args_dict['varval'], dtype=N.double)

        objs_beg = N.array(args_dict['objs_beg'], dtype=N.int32)

        objs_length = N.array(args_dict['objs_length'], dtype=N.int32)

        cons_beg = N.array(args_dict['cons_beg'], dtype=N.int32)

        cons_length = N.array(args_dict['cons_length'], dtype=N.int32)

        lwrbnd = N.array(args_dict['lwrbnd'], dtype=N.double)

        uprbnd = N.array(args_dict['uprbnd'], dtype=N.double)

        pModel = lindo.pyLScreateModel(pEnv, pnErrorCode)
        geterrormessage(pEnv, pnErrorCode[0])

        #确定线性级别,这里关闭了线性化选项，并通过以下代码段将微分设置为自动模式。
        nLinearz = 1

        lindo.pyLSsetModelIntParameter(pModel,lindo.LS_IPARAM_NLP_LINEARZ,nLinearz)


        # 在凸松弛中选择代数重构级别
        nAutoDeriv=1
        lindo.pyLSsetModelIntParameter(pModel,LSconst.LS_IPARAM_NLP_AUTODERIV,nAutoDeriv)

        # Load instruction list
        print("Loading instruction list...")
        #约束条件 ncons 29个 ，变量33个
        lindo.pyLSloadInstruct(pModel, ncons, nobjs, nvars, nnums,
                                           objsense, ctype, vtype, code, lsize,
                                           varindex, numval, varval, objs_beg, objs_length,
                                           cons_beg, cons_length, lwrbnd, uprbnd)
        # solve the model
        pnStatus = N.array([-1], dtype=N.int32)
        lindo.pyLSsolveGOP(pModel, pnStatus)

        # retrieve the objective value
        dObj = N.array([-1.0], dtype=N.double)
        lindo.pyLSgetInfo(pModel, LSconst.LS_DINFO_POBJ, dObj)


        padPrimal = N.empty((nvars), dtype=N.double)
        lindo.pyLSgetPrimalSolution(pModel, padPrimal)
        # delete LINDO model pointer
        lindo.pyLSdeleteModel(pModel)  # 通过调用LSdeleteModel，LSdeleteEnv（）来删除模型和环境。

        # delete LINDO environment pointer
        lindo.pyLSdeleteEnv(pEnv)
        print("LINDO已完成计算！！！！！！！")

        return{
            "objective":float(dObj[0]),"primal": padPrimal.tolist(),"success": True
        }
    except Exception as e:
        # 尽可能返回错误信息
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }



def utility_best_safeAPI(ncons, nobjs, nvars, nnums,
                                           objsense, ctype, vtype, code, lsize,
                                           varindex, numval, varval, objs_beg, objs_length,
                                           cons_beg, cons_length, lwrbnd, uprbnd):
    timesout = 5  # 设置最大求解时间为5s，若5s没有求解出来，直接跳过

    args_dict = {

        'ncons': ncons,

        'nobjs': nobjs,

        'nvars': nvars,

        'nnums': nnums,

        'objsense': objsense.tolist() if hasattr(objsense, 'tolist') else objsense,

        'ctype': ctype.tolist(),

        'vtype': vtype.tolist(),

        'code': code.tolist(),

        'lsize': lsize,

        'varindex': varindex.tolist(),

        'numval': numval.tolist(),

        'varval': varval.tolist(),

        'objs_beg': objs_beg.tolist(),

        'objs_length': objs_length.tolist(),

        'cons_beg': cons_beg.tolist(),

        'cons_length': cons_length.tolist(),

        'lwrbnd': lwrbnd.tolist(),

        'uprbnd': uprbnd.tolist(),

    }

    # 启动单进程求解（隔离内存）
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(solve_complex_lindo, args_dict)

            result = future.result(timeout=timesout)  # ⏱️ 关键：设置超时！
    except concurrent.futures.TimeoutError:
        return {
            "success": False,
            "error": f"Timeout after {timesout} seconds",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        # 捕获主进程异常（如 pickle 错误）
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    # 检查是否成功

    if not result.get("success", False):
        return result  # 已包含 error 信息

    print("!!!！！！！！！！")
    return result

