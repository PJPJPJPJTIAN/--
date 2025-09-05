class Paras():  # 定义一个参数配置类Paras，用于管理整个算法流程中的各类参数
    def __init__(self):  # 初始化方法，设置各类参数的默认值
        #####################
        ### General settings  ###  # 通用设置部分
        #####################
        self.method = 'eoh'  # 默认使用的算法方法为'eoh'（可改为'ael'、'ls'等）
        self.problem = 'tsp_construct'  # 默认求解的问题为'tsp_construct'（旅行商构造问题，可改为'bp_online'等）
        self.selection = None  # 选择算子默认未设置，后续会根据算法自动配置
        self.management = None  # 种群管理策略默认未设置，后续会根据算法自动配置

        #####################
        ###  EC settings  ###  # 进化计算（Evolutionary Computation）相关设置
        #####################
        self.ec_pop_size = 5  # 每个种群中的算法数量，默认5（原文注释为10，可能是笔误）
        self.ec_n_pop = 5  # 种群的数量，默认5（原文注释为10，可能是笔误）
        self.ec_operators = None  # 进化算子列表，默认未设置，后续根据算法自动配置（如['e1','e2','m1','m2']）
        self.ec_m = 2  # 'e1'和'e2'算子使用的父代数量，默认2
        self.ec_operator_weights = None  # 算子的权重（即每次迭代中使用该算子的概率），默认未设置，后续自动设为等概率
        
        #####################
        ### LLM settings  ###  # 大语言模型（LLM）相关设置
        #####################
        self.llm_use_local = False  # 是否使用本地LLM模型，默认不使用（即使用远程模型）
        self.llm_local_url = None  # 本地LLM服务的URL，如'http://127.0.0.1:11012/completions'，默认未设置
        self.llm_api_endpoint = None  # 远程LLM的API端点，如api.deepseek.com，默认未设置
        self.llm_api_key = None  # 远程LLM的API密钥，如sk-xxxx，默认未设置
        self.llm_model = None  # 远程LLM的模型类型，如deepseek-chat，默认未设置

        #####################
        ###  Exp settings  ###  # 实验相关设置
        #####################
        self.exp_debug_mode = False  # 是否开启调试模式，默认关闭
        self.exp_output_path = "./"  # 实验输出文件的默认路径
        self.exp_use_seed = False  # 是否使用预设的随机种子，默认不使用
        self.exp_seed_path = "./seeds/seeds.json"  # 随机种子文件的路径，默认值
        self.exp_use_continue = False  # 是否从之前的实验结果继续运行，默认不继续
        self.exp_continue_id = 0  # 继续运行的起始ID，默认0
        self.exp_continue_path = "./results/pops/population_generation_0.json"  # 继续运行的结果文件路径，默认值
        self.exp_n_proc = 1  # 实验使用的进程数，默认1（并行计算）
        
        #####################
        ###  Evaluation settings  ###  # 评估相关设置
        #####################
        self.eva_timeout = 30  # 评估的超时时间（秒），默认30
        self.eva_numba_decorator = False  # 是否使用numba装饰器加速评估，默认不使用


    def set_parallel(self):  # 设置并行计算的进程数
        import multiprocessing  # 导入multiprocessing模块，用于获取CPU核心数
        num_processes = multiprocessing.cpu_count()  # 获取当前设备的CPU核心数
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:  # 如果设置的进程数为-1或超过CPU核心数
            self.exp_n_proc = num_processes  # 将进程数调整为CPU核心数
            print(f"Set the number of proc to {num_processes} .")  # 打印调整信息
    
    def set_ec(self):    # 配置进化计算（EC）相关的参数
        
        if self.management == None:  # 如果种群管理策略未设置
            if self.method in ['ael','eoh']:  # 若算法为'ael'或'eoh'
                self.management = 'pop_greedy'  # 使用贪心策略管理种群
            elif self.method == 'ls':  # 若算法为局部搜索（ls）
                self.management = 'ls_greedy'  # 使用局部搜索贪心策略
            elif self.method == 'sa':  # 若算法为模拟退火（sa）
                self.management = 'ls_sa'  # 使用模拟退火的种群管理策略
        
        if self.selection == None:  # 如果选择算子未设置
            self.selection = 'prob_rank'  # 默认使用基于概率排名的选择算子
            
        
        if self.ec_operators == None:  # 如果进化算子未设置
            if self.method == 'eoh':  # 若算法为'eoh'
                self.ec_operators  = ['e1','e2','m1','m2']  # 使用这4种进化算子
            elif self.method == 'ael':  # 若算法为'ael'
                self.ec_operators  = ['crossover','mutation']  # 使用交叉和变异算子
            elif self.method == 'ls':  # 若算法为局部搜索
                self.ec_operators  = ['m1']  # 仅使用m1算子
            elif self.method == 'sa':  # 若算法为模拟退火
                self.ec_operators  = ['m1']  # 仅使用m1算子

        if self.ec_operator_weights == None:  # 如果算子权重未设置
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]  # 为每个算子设置权重1（等概率）
        elif len(self.ec_operators) != len(self.ec_operator_weights):  # 如果算子数量与权重数量不匹配
            print("Warning! Lengths of ec_operator_weights and ec_operator shoud be the same.")  # 打印警告
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]  # 重新设置为等概率权重
                    
        if self.method in ['ls','sa'] and self.ec_pop_size >1:  # 若算法为局部搜索或模拟退火，且种群大小大于1
            self.ec_pop_size = 1  # 将种群大小强制设为1（单点搜索不需要多个个体）
            self.exp_n_proc = 1  # 进程数设为1
            print("> single-point-based, set pop size to 1. ")  # 打印调整信息
            
    def set_evaluation(self):  # 配置评估相关的参数
        # Initialize evaluation settings
        if self.problem == 'bp_online':  # 若问题为在线装箱问题
            self.eva_timeout = 20  # 评估超时时间设为20秒
            self.eva_numba_decorator  = True  # 启用numba装饰器加速评估
        elif self.problem == 'tsp_construct':  # 若问题为旅行商构造问题
            self.eva_timeout = 20  # 评估超时时间设为20秒
                
    def set_paras(self, *args, **kwargs):  # 用于外部设置参数的方法，支持关键字参数
        
        # Map paras
        for key, value in kwargs.items():  # 遍历传入的关键字参数
            if hasattr(self, key):  # 如果当前类有该属性
                setattr(self, key, value)  # 更新该属性的值
              
        # Identify and set parallel 
        self.set_parallel()  # 调用set_parallel方法配置并行进程数
        
        # Initialize method and ec settings
        self.set_ec()  # 调用set_ec方法配置进化计算参数
        
        # Initialize evaluation settings
        self.set_evaluation()  # 调用set_evaluation方法配置评估参数




if __name__ == "__main__":  # 当该脚本作为主程序运行时

    # Create an instance of the Paras class
    paras_instance = Paras()  # 创建Paras类的实例

    # Setting parameters using the set_paras method
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)  # 调用set_paras方法设置参数

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # 输出：True（打印更新后的本地LLM使用状态）
    print(paras_instance.llm_local_url)  # 输出：http://example.com（打印更新后的本地LLM URL）
    print(paras_instance.ec_pop_size)    # 输出：8（打印更新后的种群大小）
            
            
            