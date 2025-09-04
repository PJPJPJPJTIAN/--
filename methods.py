# 导入选择策略：用于从种群中选择父代个体的算法
from .selection import prob_rank, equal, roulette_wheel, tournament
# 导入种群管理策略：用于更新种群（如保留优质个体、淘汰劣质个体）
from .management import pop_greedy, ls_greedy, ls_sa

class Methods():
    def __init__(self, paras, problem) -> None:
        self.paras = paras  # 保存参数配置
        self.problem = problem  # 保存问题实例（如TSP、装箱问题等）
    
        # 选择个体选择策略（用于进化算法中选择父代）
        if paras.selection == "prob_rank":
            self.select = prob_rank  # 基于概率排名的选择
        elif paras.selection == "equal":
            self.select = equal  # 等概率随机选择
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel  # 轮盘赌选择（适应度越高被选中概率越大）
        elif paras.selection == 'tournament':
            self.select = tournament  # 锦标赛选择（随机抽取个体竞争）
        else:
            # 若参数指定的选择策略未实现，则报错退出
            print("selection method " + paras.selection + " has not been implemented !")
            exit()
    
        # 选择种群管理策略（用于更新种群，保留优质个体）
        if paras.management == "pop_greedy":
            self.manage = pop_greedy  # 贪心策略（只保留最优个体）
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy  # 局部搜索贪心策略
        elif paras.management == 'ls_sa':
            self.manage = ls_sa  # 基于模拟退火的局部搜索管理
        else:
            # 若参数指定的管理策略未实现，则报错退出
            print("management method " + paras.management + " has not been implemented !")
            exit()

        
    def get_method(self):
        if self.paras.method == "ael":
            from .ael.ael import AEL  # 导入AEL算法
            return AEL(self.paras, self.problem, self.select, self.manage)
        elif self.paras.method == "eoh":
            from .eoh.eoh import EOH  # 导入EOH算法
            return EOH(self.paras, self.problem, self.select, self.manage)
        elif self.paras.method in ['ls', 'sa']:
            from .localsearch.ls import LS  # 导入局部搜索（LS）算法
            return LS(self.paras, self.problem, self.select, self.manage)
        elif self.paras.method == "funsearch":
            from .funsearch.funsearch import FunSearch  # 导入FunSearch算法
            return FunSearch(self.paras, self.problem, self.select, self.manage)
        elif self.paras.method == "reevo":
            from .reevo.reevo import ReEVO  # 导入ReEVO算法
            return ReEVO(self.paras, self.problem, self.select, self.manage)
        else:
            # 若参数指定的算法未实现，则报错退出
            print("method " + self.method + " has not been implemented!")
            exit()