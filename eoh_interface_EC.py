# 导入必要的库：numpy用于数值计算，time用于时间控制，Evolution类用于进化算法逻辑，warnings用于警告处理，joblib用于并行计算，evaluator_accelerate中的add_numba_decorator用于加速代码，re用于正则表达式处理，concurrent.futures用于并发执行
import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures

# 定义进化计算接口类，用于连接进化算法与问题评估逻辑
class InterfaceEC():
    # 初始化方法，接收种群大小、父代数量、API端点、密钥、LLM模型、是否使用本地LLM、本地LLM地址、调试模式、问题接口、选择算子、并行数量、超时时间、是否使用numba加速等参数
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM相关设置
        self.pop_size = pop_size  # 种群大小
        self.interface_eval = interface_prob  # 问题评估接口（用于评估算法性能）
        prompts = interface_prob.prompts  # 获取提示词配置
        # 初始化进化算法实例，传入LLM配置和提示词
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.m = m  # 父代数量
        self.debug = debug_mode  # 调试模式标志

        # 如果不处于调试模式，忽略警告
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select  # 选择算子（用于选择父代）
        self.n_p = n_p  # 并行计算的进程数量
        
        self.timeout = timeout  # 评估超时时间
        self.use_numba = use_numba  # 是否使用numba加速代码
        
    # 将生成的代码写入文件（当前写入ael_alg.py）
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # 将代码写入文件
            file.write(code)
        return 
    
    # 将子代添加到种群中，检查是否存在目标值相同的个体，避免重复
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")  # 调试模式下打印重复提示
                return False  # 存在重复，返回False
        population.append(offspring)  # 无重复，添加到种群
        return True  # 返回True
    
    # 检查种群中是否存在相同代码的个体，避免代码重复
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True  # 存在重复代码，返回True
        return False  # 无重复，返回False

    # 生成初始种群（注释掉的方法为种群管理和父代选择的示例，未使用）
    # def population_management(self,pop):
    #     # 删除最差个体
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    # 生成初始种群（通过i1算子创建2个个体）
    def population_generation(self):
        
        n_create = 2  # 初始创建2个个体
        
        population = []  # 初始化种群列表

        # 循环创建n_create个个体
        for i in range(n_create):
            _,pop = self.get_algorithm([],'i1')  # 使用i1算子生成算法（i1为初始创建算子）
            for p in pop:
                population.append(p)  # 将生成的个体添加到种群
             
        return population  # 返回初始种群
    
    # 基于种子算法生成初始种群（种子算法为已有的算法）
    def population_generation_seed(self,seeds,n_p):

        population = []  # 初始化种群列表

        # 并行评估所有种子算法的性能
        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        # 遍历种子算法，构建种群个体
        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],  # 算法描述
                    'code': seeds[i]['code'],  # 算法代码
                    'objective': None,  # 目标值（性能指标）
                    'other_inf': None  # 其他信息
                }

                obj = np.array(fitness[i])  # 将评估结果转换为数组
                seed_alg['objective'] = np.round(obj, 5)  # 保留5位小数
                population.append(seed_alg)  # 添加到种群

            except Exception as e:
                print("Error in seed algorithm")  # 种子算法评估出错时打印提示
                exit()  # 退出程序

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")  # 打印初始化完成信息

        return population  # 返回初始种群
    

    # 内部方法：通过指定的进化算子生成子代算法
    def _get_alg(self,pop,operator):
        # 初始化子代字典，包含算法描述、代码、目标值等
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        # 根据不同的算子生成子代
        if operator == "i1":  # i1算子：初始创建，无需父代
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()  # 调用Evolution类的i1方法生成算法          
        elif operator == "e1":  # e1算子：基于多个父代生成全新算法
            parents = self.select.parent_selection(pop,self.m)  # 选择m个父代
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)  # 调用e1方法生成算法
        elif operator == "e2":  # e2算子：基于父代的共同思想生成新算法
            parents = self.select.parent_selection(pop,self.m)  # 选择m个父代
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents)  # 调用e2方法生成算法
        elif operator == "m1":  # m1算子：对单个父代进行修改生成新算法
            parents = self.select.parent_selection(pop,1)  # 选择1个父代
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])  # 调用m1方法生成算法
        elif operator == "m2":  # m2算子：修改单个父代的参数生成新算法
            parents = self.select.parent_selection(pop,1)  # 选择1个父代
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0])  # 调用m2方法生成算法
        elif operator == "m3":  # m3算子：简化单个父代的组件以增强泛化性
            parents = self.select.parent_selection(pop,1)  # 选择1个父代
            [offspring['code'],offspring['algorithm']] = self.evol.m3(parents[0])  # 调用m3方法生成算法
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")  # 算子未实现时打印提示

        return parents, offspring  # 返回父代和生成的子代

    # 获取子代并评估其性能
    def get_offspring(self, pop, operator):

        try:
            p, offspring = self._get_alg(pop, operator)  # 调用内部方法生成子代和父代
            
            # 如果使用numba加速，为生成的函数添加numba装饰器
            if self.use_numba:
                
                # 正则表达式匹配函数定义，提取函数名
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # 在代码中搜索函数定义
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)  # 获取函数名

                # 为函数添加numba装饰器
                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']  # 不使用加速，直接使用原代码

            n_retry= 1  # 重试计数器
            # 检查代码是否重复，重复则重试
            while self.check_duplicate(pop, offspring['code']):
                
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")  # 调试模式下打印重复提示
                    
                p, offspring = self._get_alg(pop, operator)  # 重新生成子代

                # 再次处理代码（添加numba装饰器）
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                    
                if n_retry > 1:  # 最多重试1次
                    break
                
                
            # 并发执行评估，设置超时时间
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)  # 提交评估任务
                fitness = future.result(timeout=self.timeout)  # 获取评估结果，超时则抛出异常
                offspring['objective'] = np.round(fitness, 5)  # 保留5位小数
                future.cancel()  # 取消任务
                # fitness = self.interface_eval.evaluate(code)  # 直接评估（注释掉的备用方式）
                

        except Exception as e:  # 捕获异常（如超时、代码错误等）

            # 异常时返回空的子代信息
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # 返回父代和子代（目标值已取整）
        return p, offspring
    # 处理任务的备用方法（使用并发执行get_offspring，设置超时，未使用）
    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()
                
    #     return result

    
    # 批量生成算法（并行生成pop_size个子代）
    def get_algorithm(self, pop, operator):
        results = []  # 存储结果
        try:
            # 并行执行get_offspring，生成pop_size个子代，设置超时时间
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")  # 调试模式下打印错误信息
            print("Parallel time out .")  # 打印并行超时提示
            
        time.sleep(2)  # 等待2秒，避免请求过于频繁


        out_p = []  # 存储父代列表
        out_off = []  # 存储子代列表

        # 分离父代和子代
        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")  # 调试模式下打印子代信息
        return out_p, out_off  # 返回父代和子代列表
    # 生成算法的备用方法（单个生成，包含重复检查和错误重试，未使用）
    # def get_algorithm(self,pop,operator, pop_size, n_p):
        
    #     # 执行pop_size次，使用n_p个进程并行
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring