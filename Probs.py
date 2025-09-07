# 注释掉了从不同模块导入内容的语句，可能是暂时不需要或预留的导入
# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *

# 定义一个名为Probs的类，用于管理和加载不同的问题实例
class Probs():
    # 初始化方法，接收参数对象paras，根据paras中的problem信息加载对应的问题
    def __init__(self,paras):

        # 检查paras.problem是否不是字符串类型（可能是已实例化的问题对象）
        if not isinstance(paras.problem, str):
            # 直接将该对象赋值给self.prob
            self.prob = paras.problem
            # 打印提示信息，表明本地问题已加载
            print("- Prob local loaded ")
        # 如果问题是"tsp_construct"（TSP构造问题）
        elif paras.problem == "tsp_construct":
            # 从对应的优化模块中导入run模块
            from .optimization.tsp_greedy import run
            # 实例化TSPCONST类并赋值给self.prob
            self.prob = run.TSPCONST()
            # 打印提示信息，表明该问题已加载
            print("- Prob "+paras.problem+" loaded ")
        # 如果问题是"bp_online"（在线装箱问题）
        elif paras.problem == "bp_online":
            # 从对应的优化模块中导入run模块
            from .optimization.bp_online import run
            # 实例化BPONLINE类并赋值给self.prob
            self.prob = run.BPONLINE()
            # 打印提示信息，表明该问题已加载
            print("- Prob "+paras.problem+" loaded ")
        # 如果问题不存在于已知列表中
        else:
            # 打印错误提示信息
            print("problem "+paras.problem+" not found!")

    # 定义一个方法，用于获取当前加载的问题实例
    def get_problem(self):
        # 返回当前问题实例
        return self.prob