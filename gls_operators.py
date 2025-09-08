import itertools  # 导入itertools模块，用于生成迭代器（虽然当前代码未直接使用，但可能为后续扩展预留）
import numpy as np  # 导入numpy库，用于处理数值计算和数组操作
from numba import jit  # 从numba导入jit装饰器，用于将Python函数编译为机器码，加速执行

@jit(nopython=True)  # 使用numba编译该函数，禁用Python解释器，提升运行速度
def two_opt(tour, i, j):  # 定义2-opt局部搜索算子，用于优化路径，参数为2End格式路径tour和两个节点i、j
    if i == j:  # 若i和j为同一节点，无需优化，直接返回原路径
        return tour
    a = tour[i,0]  # 获取节点i的前向连接节点a
    b = tour[j,0]  # 获取节点j的前向连接节点b
    tour[i,0] = tour[i,1]  # 更新节点i的前向连接为其原后向连接
    tour[i,1] = j  # 更新节点i的后向连接为j
    tour[j,0] = i  # 更新节点j的前向连接为i
    tour[a,1] = b  # 更新节点a的后向连接为b
    tour[b,1] = tour[b,0]  # 交换节点b的前后向连接（反转路径段）
    tour[b,0] = a  # 更新节点b的前向连接为a
    c = tour[b,1]  # 获取节点b的新后向连接节点c
    while tour[c,1] != j:  # 遍历从c到j的路径段，反转其中所有节点的前后向连接
        d = tour[c,0]  # 暂存节点c的原前向连接d
        tour[c,0] = tour[c,1]  # 交换节点c的前后向连接
        tour[c,1] = d
        c = d  # 移动到下一个节点d继续处理
    return tour  # 返回优化后的路径

@jit(nopython=True)  # 编译函数，提升速度
def two_opt_cost(tour, D, i, j):  # 计算2-opt操作的成本变化（delta），参数为路径、距离矩阵D、节点i和j
    if i == j:  # 若i和j相同，成本变化为0
        return 0
    a = tour[i,0]  # 节点i的前向节点a
    b = tour[j,0]  # 节点j的前向节点b
    # 计算delta：新路径（a-b和i-j）与原路径（a-i和b-j）的成本差
    delta = D[a, b] + D[i, j] - D[a, i] - D[b, j]
    return delta  # 返回成本变化值

@jit(nopython=True)
def two_opt_a2a(tour, D, N, first_improvement=False, set_delta=0):  # 全对全（all-to-all）2-opt搜索，在所有节点对中寻找最优改进
    best_move = None  # 记录最优移动（i,j）
    best_delta = set_delta  # 初始化最优成本变化为set_delta（默认为0）
    idxs = range(0, len(tour) - 1)  # 生成节点索引范围（排除最后一个节点，可能为起点）
    for i in idxs:  # 遍历每个节点i
        for j in N[i]:  # 遍历i的最近邻节点j（来自nearest_indices）
            if i in tour[j] or j in tour[i]:  # 若i和j已直接相连，跳过（避免无效操作）
                continue
            delta = two_opt_cost(tour, D, i, j)  # 计算当前i,j的成本变化
            if delta < best_delta and not np.isclose(0, delta):  # 若找到更优的改进（delta更小且非零）
                best_delta = delta  # 更新最优delta
                best_move = i, j  # 记录当前节点对
                if first_improvement:  # 若为首次改进策略，找到第一个改进即跳出循环
                    break
    if best_move is not None:  # 若找到最优移动，执行2-opt并返回结果
        return best_delta, two_opt(tour, *best_move)
    return 0, tour  # 未找到改进，返回0和原路径

@jit(nopython=True)
def two_opt_o2a(tour, D, i, first_improvement=False):  # 点对全（one-to-all）2-opt搜索，固定节点i，遍历其他节点j
    assert i > 0 and i < len(tour) - 1  # 确保i不是起点或终点（避免无效节点）
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)  # 节点索引范围（排除起点和终点）
    for j in idxs:  # 遍历每个节点j
        if abs(i - j) < 2:  # 若j与i相邻，跳过（2-opt对相邻节点无效）
            continue
        delta = two_opt_cost(tour, D, i, j)  # 计算成本变化
        if delta < best_delta and not np.isclose(0, delta):  # 找到更优改进
            best_delta = delta
            best_move = i, j
            if first_improvement:  # 首次改进策略则跳出
                break
    if best_move is not None:  # 执行最优移动并返回
        return best_delta, two_opt(tour, *best_move)
    return 0, tour  # 无改进返回原路径

@jit(nopython=True)
def two_opt_o2a_all(tour, D, N, i):  # 基于最近邻的点对全2-opt搜索，固定i，仅遍历其最近邻节点j
    best_move = None
    best_delta = 0
    idxs = N[i]  # i的最近邻节点列表
    for j in idxs:  # 遍历i的最近邻j
        if i in tour[j] or j in tour[i]:  # 若i和j已相连，跳过
            continue
        delta = two_opt_cost(tour, D, i, j)  # 计算delta
        if delta < best_delta and not np.isclose(0, delta):  # 找到更优改进
            best_delta = delta
            best_move = i, j
            tour = two_opt(tour, *best_move)  # 立即执行2-opt更新路径
    return best_delta, tour  # 返回总delta和更新后的路径

@jit(nopython=True)
def relocate(tour, i, j):  # 重定位（relocate）算子，将节点i移动到节点j之后，优化路径
    a = tour[i,0]  # 节点i的前向节点a
    b = tour[i,1]  # 节点i的后向节点b
    tour[a,1] = b  # 更新a的后向连接为b（移除i）
    tour[b,0] = a  # 更新b的前向连接为a（移除i）
    d = tour[j,1]  # 节点j的后向节点d
    tour[d,0] = i  # 更新d的前向连接为i
    tour[i,0] = j  # 更新i的前向连接为j
    tour[i,1] = d  # 更新i的后向连接为d
    tour[j,1] = i  # 更新j的后向连接为i（插入i）
    return tour  # 返回更新后的路径

@jit(nopython=True)
def relocate_cost(tour, D, i, j):  # 计算重定位操作的成本变化
    if i == j:  # 若i和j相同，成本变化为0
        return 0
    a = tour[i,0]  # i的前向节点a
    b = i  # 节点i自身
    c = tour[i,1]  # i的后向节点c
    d = j  # 节点j
    e = tour[j,1]  # j的后向节点e
    # 计算delta：移除i后的路径（a-c）和插入i后的路径（d-i和i-e）与原路径（a-i、i-c、d-e）的成本差
    delta = -D[a, b] - D[b, c] + D[a, c] - D[d, e] + D[d, b] + D[b, e]
    return delta  # 返回成本变化

@jit(nopython=True)
def relocate_o2a(tour, D, i, first_improvement=False):  # 点对全重定位搜索，固定i，遍历其他节点j
    assert i > 0 and i < len(tour) - 1  # 确保i为有效节点
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)  # 节点索引范围
    for j in idxs:  # 遍历每个j
        if i == j:  # 若j与i相同，跳过
            continue
        delta = relocate_cost(tour, D, i, j)  # 计算delta
        if delta < best_delta and not np.isclose(0, delta):  # 找到更优改进
            best_delta = delta
            best_move = i, j
            if first_improvement:  # 首次改进策略跳出
                break
    if best_move is not None:  # 执行重定位并返回
        return best_delta, relocate(tour, *best_move)
    return 0, tour  # 无改进返回原路径

@jit(nopython=True)
def relocate_o2a_all(tour, D, N, i):  # 基于最近邻的点对全重定位搜索，固定i，遍历其最近邻j
    best_move = None
    best_delta = 0
    for j in N[i]:  # 遍历i的最近邻j
        if tour[j,1] == i:  # 若j的后向连接是i，跳过（避免重复操作）
            continue
        delta = relocate_cost(tour, D, i, j)  # 计算delta
        if delta < best_delta and not np.isclose(0, delta):  # 找到更优改进
            best_delta = delta
            best_move = i, j
            tour = relocate(tour, *best_move)  # 立即执行重定位更新路径
    return best_delta, tour  # 返回总delta和更新后的路径

@jit(nopython=True)
def relocate_a2a(tour, D, N, first_improvement=False, set_delta=0):  # 全对全重定位搜索，遍历所有节点对
    best_move = None
    best_delta = set_delta  # 初始化最优delta
    idxs = range(0, len(tour) - 1)  # 节点索引范围
    for i in idxs:  # 遍历每个i
        for j in N[i]:  # 遍历i的最近邻j
            if tour[j,1] == i:  # 若j的后向是i，跳过
                continue
            delta = relocate_cost(tour, D, i, j)  # 计算delta
            if delta < best_delta and not np.isclose(0, delta):  # 找到更优改进
                best_delta = delta
                best_move = i, j
                if first_improvement:  # 首次改进策略跳出
                    break
    if best_move is not None:  # 执行重定位并返回
        return best_delta, relocate(tour, *best_move)
    return 0, tour  # 无改进返回原路径