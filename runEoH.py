# 从eoh模块导入eoh（可能是进化优化启发式相关的主模块）和从eoh.utils.getParas模块导入Paras类（用于参数设置）
from eoh import eoh
from eoh.utils.getParas import Paras
# 从当前目录的prob模块导入TSPGLS类（TSP问题结合引导式局部搜索的核心问题定义类）
from prob import TSPGLS

# 参数初始化，创建Paras类的实例paras用于存储算法参数
paras = Paras() 

# 设置本地问题实例，创建TSPGLS类的对象problem_local，该对象封装了TSP问题的定义、数据加载和评估逻辑
problem_local = TSPGLS()

# 设置具体参数：method设为"eoh"表示使用进化优化启发式算法；problem设为前面创建的problem_local作为要解决的问题；
# llm_api_endpoint和llm_api_key分别设置LLM的接口端点和密钥；llm_model指定使用的LLM模型为"gpt-3.5-turbo"；
# ec_pop_size设置每个种群的样本数量为4；ec_n_pop设置种群数量为4；
# exp_n_proc设置4个多核并行进程；exp_debug_mode设为False表示不启用调试模式；
# eva_numba_decorator设为False表示不使用numba装饰器加速评估；
# eva_timeout设为60表示每个启发式的最大评估时间为60秒（注释中提到如果使用更多实例评估可能需要增加该值）
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = problem_local, # Set local problem, else use default problems
                llm_api_endpoint = "XXX", # set your LLM endpoint
                llm_api_key = "XXX",   # set your key
                llm_model = "gpt-3.5-turbo",
                ec_pop_size = 4, # number of samples in each population
                ec_n_pop = 4,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False,
                eva_numba_decorator = False,
                eva_timeout = 60  
                # Set the maximum evaluation time for each heuristic !
                # Increase it if more instances are used for evaluation !
                ) 

# 初始化进化优化框架，将设置好的参数paras传入eoh.EVOL类创建evolution实例
evolution = eoh.EVOL(paras)

# 运行进化优化过程，开始执行基于进化算法和LLM的启发式优化以求解TSP问题
evolution.run()