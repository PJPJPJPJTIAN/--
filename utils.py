# 从numba库导入jit装饰器，用于将函数编译为机器码以提高执行速度
from numba import jit

# 定义tour_to_edge_attribute函数，接收图G和路径tour作为参数，用于判断图中每条边是否在路径中
def tour_to_edge_attribute(G, tour):
    # 创建一个空字典in_tour，用于存储边是否在路径中的信息
    in_tour = {}
    # 将路径tour转换为边的列表，每条边由相邻两个节点组成（不包含最后一个节点与起点的闭环）
    tour_edges = list(zip(tour[:-1], tour[1:]))
    # 遍历图G中的所有边
    for e in G.edges:
        # 判断边e是否在路径边列表中，或其反向边是否在路径边列表中，将结果存入in_tour字典
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    # 返回存储边是否在路径中信息的字典
    return in_tour

# 注释掉的@jit装饰器，若启用可加速tour_cost函数，nopython=True表示在纯Python环境下编译
#@jit(nopython=True)
# 定义tour_cost函数，接收距离矩阵dis_m和路径tour，计算路径的总距离成本
def tour_cost(dis_m, tour):
    # 初始化成本c为0
    c = 0
    # 遍历路径中相邻节点组成的边
    for e in zip(tour[:-1], tour[1:]):
        # 将每条边的距离累加到成本c中
        c += dis_m[e]
    # 返回总距离成本
    return c

# 使用@jit装饰器加速tour_cost_2End函数，适用于2End格式的路径（每个节点记录前后连接节点）
@jit(nopython=True)
# 定义tour_cost_2End函数，接收距离矩阵dis_m和2End格式的路径tour2End，计算路径总距离
def tour_cost_2End(dis_m, tour2End):
    # 初始化成本c为0
    c=0
    # 从起点0开始，s为当前节点，e为下一个节点（取tour2End中起点0的后向连接节点）
    s = 0
    e = tour2End[0,1]
    # 遍历路径中的所有节点（次数为路径矩阵的行数）
    for i in range(tour2End.shape[0]):
        # 累加当前节点s到下一个节点e的距离
        c += dis_m[s,e]
        # 更新当前节点s为e，下一个节点e为当前节点s的后向连接节点
        s = e
        e = tour2End[s,1]
    # 返回总距离成本
    return c

# 定义is_equivalent_tour函数，判断两个路径tour_a和tour_b是否等价（相同路径或反向路径）
def is_equivalent_tour(tour_a, tour_b):
    # 若tour_a与tour_b的反向路径相同，返回True
    if tour_a == tour_b[::-1]:
        return True
    # 若tour_a与tour_b完全相同，返回True
    if tour_a == tour_b:
        return True
    # 否则返回False
    return False