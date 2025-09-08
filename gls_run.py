# 导入time模块，用于处理时间相关操作
import time
# 导入importlib模块，用于动态导入模块
import importlib
# 导入numpy库并简写为np，用于数值计算
import numpy as np

# 从utils模块中导入utils，可能包含工具函数
from utils import utils
# 从gls模块中导入gls_evol，可能包含引导式局部搜索的进化相关函数
from gls import gls_evol

# 定义solve_instance函数，用于求解单个TSP实例，参数包括实例编号n、最优成本opt_cost、距离矩阵dis_matrix、坐标coord、时间限制time_limit、最大迭代次数ite_max、扰动步数perturbation_moves、启发式算法heuristic
def solve_instance(n,opt_cost,dis_matrix,coord,time_limit, ite_max, perturbation_moves,heuristic):

    # 以下是注释掉的参数设置，分别为最大时间限制、最大局部搜索次数、每次扰动的边移动次数
    # time_limit = 60 # maximum 10 seconds for each instance
    # ite_max = 1000 # maximum number of local searchs in GLS for each instance
    # perturbation_moves = 1 # movers of each edge in each perturbation

   
    # 程序暂停1秒
    time.sleep(1)
    # 记录当前时间，作为开始时间
    t = time.time()

    # 尝试执行以下求解逻辑，捕获可能出现的异常
    try:
        # 调用gls_evol中的nearest_neighbor_2End函数，以0为起点生成初始路径，并转换为整数类型
        init_tour = gls_evol.nearest_neighbor_2End(dis_matrix, 0).astype(int)
        # 调用utils中的tour_cost_2End函数计算初始路径的成本
        init_cost = utils.tour_cost_2End(dis_matrix, init_tour)
        # 设置最近邻节点的数量为100
        nb = 100
        # 对距离矩阵按行排序，取每行除自身外的前100个最近节点的索引，作为最近邻索引矩阵
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb+1].astype(int)

        # 调用gls_evol中的guided_local_search函数执行引导式局部搜索，传入相关参数，包括坐标、距离矩阵、最近邻索引、初始路径、初始成本、结束时间（当前时间+时间限制）、最大迭代次数、扰动步数、是否首次改进、启发式算法，返回最佳路径、最佳成本和迭代次数
        best_tour, best_cost, iter_i = gls_evol.guided_local_search(coord, dis_matrix, nearest_indices, init_tour, init_cost,
                                                        t + time_limit, ite_max, perturbation_moves,
                                                        first_improvement=False, guide_algorithm=heuristic)

        # 计算求解结果与最优成本的差距（百分比）
        gap = (best_cost / opt_cost - 1) * 100

    # 捕获异常，出现异常时将差距设为一个很大的值（10的10次方）
    except Exception as e:
        # 注释掉的打印错误信息的语句
        #print("Error:", str(e))  # Print the error message
        gap = 1E10
    
    # 注释掉的打印实例求解信息的语句，包括实例编号、成本、差距、迭代次数、耗时
    #print(f"instance {n+1}: cost = {best_cost:.3f}, gap = {gap:.3f}, n_it = {iter_i}, cost_t = {time.time()-t:.3f}")

    # 返回计算得到的差距
    return gap