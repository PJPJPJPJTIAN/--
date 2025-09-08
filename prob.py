# 导入所需模块：numpy用于数值计算，time用于时间相关操作，joblib的Parallel和delayed用于并行计算，
# os用于文件路径操作，types用于创建模块对象，warnings用于处理警告，sys用于系统相关操作
import numpy as np
import time
from joblib import Parallel, delayed
import os
import types
import warnings
import sys

# 从utils模块导入readTSPRandom用于读取TSP随机实例数据，从gls.gls_run模块导入solve_instance用于求解单个TSP实例
from utils import readTSPRandom
from gls.gls_run import solve_instance

# 定义TSPGLS类，用于封装TSP问题结合引导式局部搜索（GLS）的相关功能
class TSPGLS():
    # 类的初始化方法，设置各种参数和加载数据
    def __init__(self) -> None:
        self.n_inst_eva = 3  # 评估时使用的实例数量，这里是测试用的小值
        self.time_limit = 10  # 每个实例的最大求解时间（秒）
        self.ite_max = 1000  # 每个实例在GLS中局部搜索的最大迭代次数
        self.perturbation_moves = 1  # 每次扰动中每条边的移动次数
        path = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
        self.instance_path = path+'/TrainingData/TSPAEL64.pkl'  # 训练数据的路径，包含TSP实例
        self.debug_mode=False  # 是否启用调试模式，默认为False

        # 读取训练数据，获取坐标、实例（距离矩阵）和最优成本
        self.coords,self.instances,self.opt_costs = readTSPRandom.read_instance_all(self.instance_path)

        # 从prompts模块导入GetPrompts类并创建实例，用于生成提示词
        from prompts import GetPrompts
        self.prompts = GetPrompts()

    # 计算给定路径的成本，基于实例的坐标计算欧氏距离
    def tour_cost(self,instance, solution, problem_size):
        cost = 0  # 初始化成本为0
        # 计算路径中相邻节点之间的距离并累加
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        # 计算路径最后一个节点与第一个节点之间的距离（闭合路径）
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost  # 返回总路径成本

    # 生成邻域矩阵，每个节点的邻域按距离排序
    def generate_neighborhood_matrix(self,instance):
        instance = np.array(instance)  # 将实例转换为numpy数组
        n = len(instance)  # 获取节点数量
        neighborhood_matrix = np.zeros((n, n), dtype=int)  # 初始化邻域矩阵

        # 对每个节点，计算其与其他所有节点的距离并排序，得到邻域索引
        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)  # 计算节点i到所有节点的欧氏距离
            sorted_indices = np.argsort(distances)  # 按距离排序得到索引
            neighborhood_matrix[i] = sorted_indices  # 将排序后的索引存入邻域矩阵

        return neighborhood_matrix  # 返回邻域矩阵

    # 评估启发式算法在多个TSP实例上的性能，返回平均差距（与最优解的偏差百分比）
    def evaluateGLS(self,heuristic):

        gaps = np.zeros(self.n_inst_eva)  # 初始化存储每个实例差距的数组

        # 遍历每个评估实例，计算差距
        for i in range(self.n_inst_eva):
            # 求解第i个实例，得到差距
            gap = solve_instance(i,self.opt_costs[i],  
                                 self.instances[i], 
                                 self.coords[i],
                                 self.time_limit,
                                 self.ite_max,
                                 self.perturbation_moves,
                                 heuristic)
            gaps[i] = gap  # 存储当前实例的差距

        return np.mean(gaps)  # 返回平均差距


    # 注释掉的另一种evaluateGLS实现，使用并行计算处理64个实例
    # def evaluateGLS(self,heuristic):
    #
    #     nins = 64    
    #     gaps = np.zeros(nins)
    #
    #     print("Start evaluation ...")   
    #
    #     inputs = [(x,self.opt_costs[x],  self.instances[x], self.coords[x],self.time_limit,self.ite_max,self.perturbation_moves) for x in range(nins)]
    #     #gaps = Parallel(n_jobs=nins)(delayed(solve_instance)(*input) for input in inputs)
    #     try:
    #             gaps = Parallel(n_jobs= 4, timeout = self.time_limit*1.1)(delayed(solve_instance)(*input) for input in inputs)
    #     except:
    #             print("### timeout or other error, return a large fitness value ###")
    #             return 1E10
    #     return np.mean(gaps)


    # 注释掉的evaluate方法，可能是早期版本，不接收代码字符串参数
    # def evaluate(self):
    #     try:        
    #         fitness = self.evaluateGLS()
    #         return fitness
    #     except Exception as e:
    #         print("Error:", str(e))  # Print the error message
    #         return None

    # 评估给定的代码字符串（启发式算法代码）的性能，返回适应度（平均差距）
    def evaluate(self, code_string):
        try:
            # 抑制警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 创建一个新的模块对象，用于执行代码字符串
                heuristic_module = types.ModuleType("heuristic_module")
                
                # 在新模块的命名空间中执行代码字符串
                exec(code_string, heuristic_module.__dict__)

                # 将新模块添加到sys.modules中，使其可以被导入
                sys.modules[heuristic_module.__name__] = heuristic_module

                # 打印代码字符串（调试用，当前注释掉）
                #print(code_string)
                # 评估该启发式模块的性能，得到适应度
                fitness = self.evaluateGLS(heuristic_module)

                return fitness  # 返回适应度
            
        except Exception as e:
            # 打印错误信息（当前注释掉）
            #print("Error:", str(e))
            return None  # 发生错误时返回None