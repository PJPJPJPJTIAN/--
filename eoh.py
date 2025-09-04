import random
from .utils import createFolders  # 工具类：创建输出文件夹
from .methods import methods      # 算法方法模块：包含各类进化算法实现
from .problems import problems    # 问题模块：定义待解决的任务（如机器学习、优化问题等）

class EVOL:
    def __init__(self, paras, prob=None, **kwargs):
        # 打印启动信息
        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        
        # 创建输出文件夹（如结果存储目录）
        createFolders.create_folders(paras.exp_output_path)
        print("- output folder created -")
        
        # 保存参数（如实验配置、算法参数等）
        self.paras = paras
        print("-  parameters loaded -")
        
        # 可选：直接传入预定义的问题（若未传入则后续通过问题生成器获取）
        self.prob = prob
        
        # 设置随机种子（确保实验可复现）
        random.seed(2024)

    def run(self):
        # 1. 获取问题实例（根据参数定义的任务类型，如自动算法设计、机器学习任务等）
        problemGenerator = problems.Probs(self.paras)
        problem = problemGenerator.get_problem()
        
        # 2. 获取算法实例（根据参数选择具体的进化方法，如之前代码中的 Local Search、EOH 等）
        methodGenerator = methods.Methods(self.paras, problem)
        method = methodGenerator.get_method()
        
        # 3. 启动算法运行（执行进化流程，如种群迭代、LLM 生成算法、评估优化等）
        method.run()
        
        # 4. 打印结束信息
        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")