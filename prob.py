# 导入所需的库：numpy用于数值计算，importlib用于动态导入模块，time用于计时，numba的jit用于加速函数执行
import numpy as np
import importlib
import time
from numba import jit

# 导入随机数生成、类型检查、警告处理和系统相关的库
import random
import types
import warnings
import sys

# 导入numba相关的警告类，用于过滤特定警告
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# 过滤numba的弃用警告和待弃用警告
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# 过滤特定的用户警告（关于从.libs加载多个DLL）
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs", category=UserWarning)

# 使用numba的jit装饰器加速makespan函数，该函数计算调度方案的最大完工时间
@jit(nopython=True)
def makespan(order, tasks, machines_val):
    # 初始化每个机器的时间为0
    times = []
    for i in range(0, machines_val):
        times.append(0)
    # 遍历作业序列，更新每个机器的时间
    for j in order:
        # 第一个机器的时间直接累加当前作业的处理时间
        times[0] += tasks[j][0]
        # 后续机器的时间取前一个机器完成时间和当前机器已有时间的最大值，再累加当前作业处理时间
        for k in range(1, machines_val):
            if times[k] < times[k-1]:
                times[k] = times[k-1]
            times[k] += tasks[j][k]
    # 返回最大完工时间（所有机器中最大的时间）
    return max(times)

# 使用numba加速局部搜索函数，通过交换或移动作业位置优化调度方案
@jit(nopython=True)
def local_search(sequence, cmax_old,tasks,machines_val):
    # 复制当前序列作为新序列的初始值
    new_seq = sequence[:]
    # 遍历序列中的作业对，尝试交换位置并计算新的最大完工时间
    for i in range(len(new_seq)):
        for j in range(i+1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            # 如果新的最大完工时间更小，则更新序列和最优值
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    # 遍历序列，尝试将作业移动到其他位置并计算新的最大完工时间
    for i in range(1,len(new_seq)):
        for j in range(1,len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            # 如果新的最大完工时间更小，则更新序列和最优值
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    # 返回优化后的序列
    return new_seq

# 带扰动的局部搜索函数，仅对指定的作业进行交换和移动操作
@jit(nopython=True)
def local_search_perturb(sequence, cmax_old,tasks,machines_val,job):
    new_seq = sequence[:]
    # 仅对job列表中的作业进行交换操作
    for i in job:
        for j in range(i+1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    # 仅对job列表中的作业进行移动操作
    for i in job:
        for j in range(1,len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    # 返回优化后的序列
    return new_seq

# 定义JSSPGLS类，用于解决作业车间调度问题的广义局部搜索
class JSSPGLS():
    def __init__(self) -> None:
        self.n_inst_eva = 3  # 用于测试的实例数量（较小）
        self.iter_max = 1000  # 广义局部搜索的最大迭代次数
        self.time_max = 30  # 每个实例的最大运行时间（秒）
        # 读取实例数据，获取作业数、机器数和处理时间矩阵
        self.tasks_val, self.machines_val, self.tasks = self.read_instances()
        # 导入提示信息类，用于与LLM交互
        from prompts import GetPrompts
        self.prompts = GetPrompts()

    ############################################### 局部搜索 ####################################################
    def ls(self,tasks_val, tasks, machines_val):
        # 使用NEH算法生成初始序列和对应的最大完工时间
        pi0, cmax0 = self.neh(tasks, machines_val, tasks_val) 
        # 初始化当前序列和最优最大完工时间
        pi = pi0
        cmax_old = cmax0
        # 循环进行局部搜索，直到无法找到更优解
        while True:
            piprim = local_search(pi, cmax_old,tasks,machines_val)
            cmax = makespan(piprim, tasks, machines_val)
            # 如果新解不优于当前解，则跳出循环
            if (cmax>=cmax_old):
                break
            # 否则更新当前序列和最优值
            else:
                pi = piprim
                cmax_old = cmax
        # 返回最优序列和对应的最大完工时间
        return pi, cmax_old

    ############################################### 迭代局部搜索 ####################################################
    def gls(self,heuristic):
        # 初始化存储每个实例最优最大完工时间的数组
        cmax_best_list = np.zeros(self.n_inst_eva)
        
        n_inst = 0
        # 遍历每个实例的作业数、处理时间矩阵和机器数
        for tasks_val,tasks,machines_val in zip(self.tasks_val, self.tasks, self.machines_val):
            # 初始化最优最大完工时间为一个很大的值
            cmax_best = 1E10
            # 设置随机种子，保证结果可复现
            random.seed(2024)
            try:
                # 使用NEH算法生成初始序列和最大完工时间
                pi, cmax = self.neh(tasks, machines_val, tasks_val) 
                n = len(pi)
                
                # 初始化最优序列和最优最大完工时间
                pi_best = pi
                cmax_best = cmax
                n_itr = 0
                time_start = time.time()
                # 在最大时间和最大迭代次数内循环
                while time.time() - time_start < self.time_max and n_itr <self.iter_max:
                    # 对当前序列进行局部搜索
                    piprim = local_search(pi, cmax,tasks,machines_val)

                    pi = piprim
                    cmax = makespan(pi, tasks, machines_val)
                    
                    # 如果找到更优解，更新最优序列和最优值
                    if (cmax<cmax_best):
                        pi_best = pi
                        cmax_best = cmax

                    # 使用启发式算法（可能来自LLM）获取扰动后的处理时间矩阵和待扰动作业
                    tasks_perturb, jobs = heuristic.get_matrix_and_jobs(pi, tasks.copy(), machines_val, n)

                    # 检查待扰动作业列表的有效性，若不符合要求则返回大值
                    if ( len(jobs) <= 1):
                        print("jobs is not a list of size larger than 1")          
                        return 1E10   
                    # 如果作业数量超过5，取前5个
                    if  ( len(jobs) > 5):
                        jobs = jobs[:5]

                    # 计算扰动后的处理时间矩阵对应的最大完工时间
                    cmax = makespan(pi, tasks_perturb, machines_val)

                    # 对指定作业进行带扰动的局部搜索
                    pi = local_search_perturb(pi, cmax,tasks_perturb,machines_val,jobs)

                    # 迭代次数加1
                    n_itr +=1
                    # 每50次迭代，将当前序列重置为最优序列
                    if n_itr % 50 == 0:
                        pi = pi_best
                        cmax = cmax_best

            # 捕获异常，将最优值设为大值
            except Exception as e:
                cmax_best = 1E10
        
            # 存储当前实例的最优最大完工时间
            cmax_best_list[n_inst] = cmax_best
            n_inst += 1
            # 达到测试实例数量则停止
            if n_inst == self.n_inst_eva:
                break
        
        # 返回所有实例的平均最优最大完工时间
        return np.average(cmax_best_list)

    ###################################################################### NEH算法 ############################################
    # 计算每个作业的总处理时间并按降序排序，返回排序后的作业索引
    def sum_and_order(self,tasks_val, machines_val, tasks):
        tab = []
        tab1 = []
        # 初始化存储总处理时间的列表
        for i in range(0, tasks_val):
            tab.append(0)
            tab1.append(0)
        # 计算每个作业的总处理时间
        for j in range(0, tasks_val):
            for k in range(0, machines_val):
                tab[j] += tasks[j][k]
        tmp_tab = tab.copy()
        place = 0
        iter = 0
        # 按总处理时间降序排序作业
        while(iter != tasks_val):
            max_time = 1
            for i in range(0, tasks_val):
                if(max_time < tab[i]):
                    max_time = tab[i]
                    place = i
            tab[place] = 1
            tab1[iter] = place
            iter = iter + 1
        return tab1

    # 在序列的指定位置插入值，返回新序列
    def insertNEH(self,sequence, position, value):
        new_seq = sequence[:]
        new_seq.insert(position, value)
        return new_seq

    # NEH算法：生成初始调度序列并计算其最大完工时间
    def neh(self,tasks, machines_val, tasks_val):
        # 按总处理时间降序获取作业顺序
        order = self.sum_and_order(tasks_val, machines_val, tasks)
        current_seq = [order[0]]
        # 依次将每个作业插入到当前序列的最优位置
        for i in range(1, tasks_val):
            min_cmax = float("inf")
            for j in range(0, i + 1):
                tmp = self.insertNEH(current_seq, j, order[i])
                cmax_tmp = makespan(tmp, tasks, machines_val)
                # 记录最优插入位置
                if min_cmax > cmax_tmp:
                    best_seq = tmp
                    min_cmax = cmax_tmp
            current_seq = best_seq
        # 返回最优序列和对应的最大完工时间
        return current_seq, makespan(current_seq, tasks, machines_val)

    # 读取训练数据集中的实例，返回作业数、机器数和处理时间矩阵的列表
    def read_instances(self):
        tasks_val_list = [] 
        machines_val_list = [] 
        tasks_list = []

        # 读取1到64号实例文件
        for i in range(1,65):
            filename = "./TrainingData/"+ str(i) + ".txt"
            file = open(filename, "r")

            # 读取第一行的作业数和机器数
            tasks_val, machines_val = file.readline().split()
            tasks_val = int(tasks_val)
            machines_val = int(machines_val)

            # 初始化处理时间矩阵并读取数据
            tasks = np.zeros((tasks_val,machines_val))
            for i in range(tasks_val):
                tmp = file.readline().split()
                for j in range(machines_val):
                    tasks[i][j] = int(float(tmp[j*2+1]))  # 取每个操作的处理时间（假设格式为机器索引和时间交替）

            # 将实例数据添加到列表
            tasks_val_list.append(tasks_val)
            machines_val_list.append(machines_val)
            tasks_list.append(tasks)

            file.close()

        return tasks_val_list, machines_val_list, tasks_list

    # 评估函数：执行传入的代码字符串作为启发式算法，并返回平均最优最大完工时间
    def evaluate(self, code_string):
        try:
            # 抑制警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 创建一个新的模块对象
                heuristic_module = types.ModuleType("heuristic_module")
                
                # 在新模块的命名空间中执行代码字符串
                exec(code_string, heuristic_module.__dict__)

                # 将模块添加到sys.modules中以便导入
                sys.modules[heuristic_module.__name__] = heuristic_module

                # 使用该启发式算法执行广义局部搜索并获取适应度（平均最大完工时间）
                fitness = self.gls(heuristic_module)

                return fitness
            
        # 捕获异常，返回None
        except Exception as e:
            return None