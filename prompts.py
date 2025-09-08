# 定义GetPrompts类，用于管理生成提示词的相关属性和方法
class GetPrompts():
    # 类的初始化方法，设置各种提示词相关的属性
    def __init__(self):
        # 定义任务提示，说明需要设计策略来更新距离矩阵以避免陷入局部最优，最终目标是找到距离最小的路径
        self.prompt_task = "Task: Given an edge distance matrix and a local optimal route,
        please help me design a strategy to update the distance matrix to avoid being trapped in the local 
        optimum with the final goal of finding a tour with minimized distance. \
You should create a heuristic for me to update the edge distance matrix."
        # 定义函数名称提示，指定所需函数名为update_edge_distance
        self.prompt_func_name = "update_edge_distance"
        # 定义函数输入参数提示，包含edge_distance、local_opt_tour、edge_n_used三个输入
        self.prompt_func_inputs = ['edge_distance', 'local_opt_tour', 'edge_n_used']
        # 定义函数输出参数提示，输出为updated_edge_distance
        self.prompt_func_outputs = ['updated_edge_distance']
        # 定义输入输出信息提示，说明local_opt_tour包含局部最优路径的ID，edge_distance和edge_n_used是矩阵，edge_n_used包含每个边在排列过程中使用的次数
        self.prompt_inout_inf = "'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' 
        and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation."
        # 定义其他信息提示，说明所有参数都是Numpy数组
        self.prompt_other_inf = "All are Numpy arrays."

    # 定义获取任务提示的方法，返回任务提示内容
    def get_task(self):
        return self.prompt_task
    
    # 定义获取函数名称提示的方法，返回函数名称
    def get_func_name(self):
        return self.prompt_func_name
    
    # 定义获取函数输入参数提示的方法，返回输入参数列表
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    # 定义获取函数输出参数提示的方法，返回输出参数列表
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    # 定义获取输入输出信息提示的方法，返回输入输出信息
    def get_inout_inf(self):
        return self.prompt_inout_inf

    # 定义获取其他信息提示的方法，返回其他信息
    def get_other_inf(self):
        return self.prompt_other_inf
    
#     # 注释掉的获取创建提示的方法，用于生成创建新策略的提示内容，包含任务描述、函数要求、输入输出说明等
#     def get_prompt_create(self):
#         prompt_content = "Task: Given an edge distance matrix and a local optimal route, 
please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
# You should create a strategy for me to update the edge distance matrix. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 
'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of 
each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content
    

#     # 注释掉的获取交叉提示的方法，用于基于两个现有策略生成新策略的提示内容，包含两个策略的描述和代码，要求生成不同于它们但受其启发的新策略
#     def get_prompt_crossover(self,indiv1,indiv2):
#         prompt_content = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
# I have two strategies with their codes to update the distance matrix. \
# The first strategy and the corresponding code are: \n\
# Strategy description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# The second strategy and the corresponding code are: \n\
# Strategy description: "+indiv2['algorithm']+"\n\
# Code:\n\
# "+indiv2['code']+"\n\
# Please help me create a new strategy that is totally different from them but can be motivated from them. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content
    
#     # 注释掉的获取变异提示的方法，用于基于现有策略生成修改版本的提示内容，要求对提供的策略进行修改并生成新策略
#     def get_prompt_mutation(self,indiv1):
#         prompt_content = "Task: Given a set of nodes with their coordinates, \
# you need to find the shortest route that visits each node once and returns to the starting node. \
# The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
# I have a strategy with its code to select the next node in each step as follows. \
# Strategy description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# Please assist me in creating a modified version of the strategy provided. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content
