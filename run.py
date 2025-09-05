### Test Only ###  # 注释：表明此文件仅用于测试
# Set system path  # 注释：设置系统路径
import sys  # 导入sys模块，用于操作Python解释器的运行时环境（如路径设置）
import os  # 导入os模块，用于与操作系统交互（如文件路径处理）
ABS_PATH = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径的父目录，即当前脚本所在文件夹的路径
ROOT_PATH = os.path.join(ABS_PATH, "..", "..")  # 拼接路径，从当前文件夹向上两级（得到项目根目录）
sys.path.append(ROOT_PATH)  # 将项目根目录添加到Python的模块搜索路径中，确保能导入项目内的模块
sys.path.append(ABS_PATH)  # 将当前文件夹路径添加到模块搜索路径中
print(ABS_PATH)  # 打印当前文件所在文件夹的绝对路径
from eoh import eoh  # 从eoh模块导入eoh（包含EVOL类，是算法流程的入口）
from eoh.utils.getParas import Paras  # 从eoh.utils.getParas模块导入Paras类（用于参数配置）
# from evol.utils.createReport import ReportCreator  # 注释：未启用的报告生成器导入语句

# Parameter initilization # 注释：参数初始化
paras = Paras()  # 实例化Paras类，创建参数对象

# Set parameters # 注释：设置参数
paras.set_paras(method = "eoh",    # 指定使用的算法为"eoh"
                ec_operators  = ['e1','e2','m1','m2','m3'], # EoH算法中使用的进化算子列表
                problem = "bp_online", # 指定求解的问题为"bp_online"（在线装箱问题），可选其他问题如旅行商问题等
                llm_api_endpoint = "XXX", # 设置LLM（大语言模型）的API端点（需替换为实际地址）
                llm_api_key = "XXX",   # 设置LLM的API密钥（需替换为实际密钥）
                llm_model = "XXX", # 设置使用的LLM模型名称（需替换为实际模型名）
                ec_pop_size = 4,  # 进化算法的种群大小为4
                ec_n_pop = 2,     # 进化迭代的种群数量为2
                exp_n_proc = 4,   # 实验使用的进程数为4（并行计算）
                exp_debug_mode = False)  # 关闭调试模式（不打印调试信息）

# EoH initilization  # 注释：初始化EoH算法
evolution = eoh.EVOL(paras)  # 用参数对象paras实例化EVOL类，创建进化算法对象

# run EoH  # 注释：运行EoH算法
evolution.run()  # 调用EVOL对象的run方法，启动整个进化优化流程

# Generate EoH Report  # 注释：生成EoH报告（未启用）
# RC = ReportCreator(paras)  # 注释：实例化报告生成器
# RC.generate_doc_report()  # 注释：生成文档报告