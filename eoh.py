import numpy as np
import json
import random
import time

from .eoh_interface_EC import InterfaceEC
# main class for eoh (Evolutionary Optimization with Heuristics，启发式进化优化算法)
class EOH:

    # 初始化方法：加载配置参数和外部依赖
    def __init__(self, paras, problem, select, manage, **kwargs):
        # 绑定外部依赖组件
        self.prob = problem  # 问题定义接口（用于评估算法性能）
        self.select = select  # 选择策略（用于从种群中选择父代个体）
        self.manage = manage  # 种群管理策略（用于控制种群规模和筛选优质个体）
        
        # LLM（大语言模型）配置
        self.use_local_llm = paras.llm_use_local  # 是否使用本地LLM
        self.llm_local_url = paras.llm_local_url  # 本地LLM服务地址
        self.api_endpoint = paras.llm_api_endpoint  # 远程API端点（目前前仅支持API2D + GPT）
        self.api_key = paras.llm_api_key  # API密钥
        self.llm_model = paras.llm_model  # LLM模型名称

        # ------------------ 本地注释本地LLM的备用配置（已注释） ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, '使用本地LLM时必须提供"url"参数'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # 实验参数配置
        self.pop_size = paras.ec_pop_size  # 种群规模（每代保留的算法数量）
        self.n_pop = paras.ec_n_pop  # 进化迭代次数（总代数）

        self.operators = paras.ec_operators  # 进化算子列表（如i1, e1, m1等）
        self.operator_weights = paras.ec_operator_weights  # 算子选择权重（控制每个算子的使用概率）
        # 校验父代选择数量m的合理性（至少为2，且不超过种群规模）
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m值不应大于种群规模或小于2，已自动调整为m=2")
            paras.ec_m = 2
        self.m = paras.ec_m  # 每次代选择数量（用于交叉算子）

        self.debug_mode = paras.exp_debug_mode  # 调试模式（开启后打印详细日志）
        self.ndelay = 1  # 默认延迟时间（未实际使用）

        # 种群初始化相关配置
        self.use_seed = paras.exp_use_seed  # 是否从种子文件初始化种群
        self.seed_path = paras.exp_seed_path  # 种子文件路径
        self.load_pop = paras.exp_use_continue  # 是否从已有种群文件加载（断点续跑）
        self.load_pop_path = paras.exp_continue_path  # 待加载的种群文件路径
        self.load_pop_id = paras.exp_continue_id  # 断点续跑的起始代数

        self.output_path = paras.exp_output_path  # 结果输出目录路径

        self.exp_n_proc = paras.exp_n_proc  # 并行进程数（用于加速算法评估）
        self.timeout = paras.eva_timeout  # 算法评估超时时间
        self.use_numba = paras.eva_numba_decorator  # 是否使用numba装饰器加速代码执行

        print("- EoH参数加载完成 -")

        # 设置随机种子，保证实验可复现性
        random.seed(2024)

    # 向种群中添加新个体（子代），并做简单的重复检查
    def add2pop(self, population, offspring):
        for off in offspring:
            # 检查是否与种群中已有个体的目标值重复（仅打印提示，不阻止添加）
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("发现重复结果，重试中...")
            # 将子代添加到种群
            population.append(off)
    

    # 运行EOH算法的主方法
    def run(self):
        print("- 进化过程开始 -")
        time_start = time.time()  # 记录开始时间

        # 评估接口（绑定具体问题）
        interface_prob = self.prob

        # 初始化进化算子接口（连接LLM、评估器和选择策略）
        interface_ec = InterfaceEC(
            self.pop_size, self.m, 
            self.api_endpoint, self.api_key, self.llm_model, 
            self.use_local_llm, self.llm_local_url,
            self.debug_mode, interface_prob, 
            select=self.select, n_p=self.exp_n_proc,
            timeout=self.timeout, use_numba=self.use_numba
        )

        # 初始化种群
        population = []
        if self.use_seed:
            # 从种子文件加载初始种群
            with open(self.seed_path) as file:
                data = json.load(file)
            # 基于种子数据生成种群（包含算法描述、代码和评估结果）
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)
            # 保存初始种群
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0  # 起始代数为0
        else:
            if self.load_pop:  # 从已有种群文件加载（断点续跑）
                print("从" + self.load_pop_path + "加载初始种群")
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("初始种群加载完成！")
                n_start = self.load_pop_id  # 起始代数为断点代数
            else:  # 全新生成初始种群
                print("创建初始种群中：")
                # 调用进化算子接口生成初始种群（使用i1算子）
                population = interface_ec.population_generation()
                # 裁剪种群至设定规模
                population = self.manage.population_management(population, self.pop_size)
                
                # 打印初始种群的目标值（用于调试）
                print(f"初始种群目标值：")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("初始种群创建完成！")
                # 保存初始种群
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0  # 起始代数为0

        # 进化主循环（迭代n_pop代）
        n_op = len(self.operators)  # 算子数量
        for pop in range(n_start, self.n_pop):  
            # 遍历所有进化算子
            for i in range(n_op):
                op = self.operators[i]  # 当前算子（如e1, m1）
                print(f" 算子: {op}, [{i + 1} / {n_op}] ", end="|") 
                op_w = self.operator_weights[i]  # 算子选择权重
                # 根据权重随机决定是否执行当前算子
                if (np.random.rand() < op_w):
                    # 调用算子生成父代和子代（父代通过选择策略从种群中选出）
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                # 将子代添加到种群
                self.add2pop(population, offsprings)
                # 打印子代的目标值
                for off in offsprings:
                    print(" Obj: ", off['objective'], end="|")
                # 裁剪种群至设定规模（保留适应度更高的个体）
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()

            # 保存当前代的完整种群
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # 保存当前代的最优个体（默认种群已按适应度排序，第一个为最优）
            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            # 打印当前代的统计信息
            print(f"--- 第 {pop + 1}/{self.n_pop} 代完成，耗时: {((time.time()-time_start)/60):.1f} 分钟")
            print("种群目标值: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()