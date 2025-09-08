# 导入time模块用于时间相关操作，numpy库用于数值计算并简写为np，numba的jit装饰器用于加速函数，
# gls_operators模块包含局部搜索算子，utils模块包含工具函数，random模块用于随机操作
import time
import numpy as np
from numba import jit
from gls import gls_operators
from utils import utils
import random

# 定义nearest_neighbor函数（被注释的@jit装饰器可用于加速，nopython=True表示纯Python环境编译），功能是基于距离矩阵和起点生成最近邻路径
#@jit(nopython=True) 
def nearest_neighbor(dis_matrix, depot):
    tour = [depot]  # 初始化路径列表，以起点depot开始
    n = len(dis_matrix)  # 获取节点数量
    nodes = np.arange(n)  # 生成节点索引数组
    while len(tour) < n:  # 当路径未包含所有节点时循环
        i = tour[-1]  # 当前节点为路径最后一个节点
        neighbours = [(j, dis_matrix[i,j]) for j in nodes if j not in tour]  # 找出未包含在路径中的邻居节点及其距离
        j, dist = min(neighbours, key=lambda e: e[1])  # 选择距离最近的邻居节点
        tour.append(j)  # 将该节点加入路径

    tour.append(depot)  # 路径最后回到起点，形成闭环

    return tour  # 返回生成的路径

# 定义nearest_neighbor_2End函数（类似nearest_neighbor，生成2End格式的路径，
# 即每个节点记录前后连接节点）
#@jit(nopython=True) 
def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]  # 初始化路径列表，以起点depot开始
    n = len(dis_matrix)  # 获取节点数量
    nodes = np.arange(n)  # 生成节点索引数组
    while len(tour) < n:  # 当路径未包含所有节点时循环
        i = tour[-1]  # 当前节点为路径最后一个节点
        neighbours = [(j, dis_matrix[i,j]) for j in nodes if j not in tour] 
        # 找出未包含在路径中的邻居节点及其距离
        j, dist = min(neighbours, key=lambda e: e[1])  # 选择距离最近的邻居节点
        tour.append(j)  # 将该节点加入路径

    tour.append(depot)  # 路径最后回到起点，形成闭环
    route2End = np.zeros((n,2))  # 创建n行2列的数组，存储每个节点的前向和后向连接节点
    route2End[0,0] = tour[-2]  # 起点的前向连接为路径倒数第二个节点
    route2End[0,1] = tour[1]  # 起点的后向连接为路径第二个节点
    for i in range(1,n):  # 遍历其他节点
        route2End[tour[i],0] = tour[i-1]  # 第i个节点的前向连接为前一个节点
        route2End[tour[i],1] = tour[i+1]  # 第i个节点的后向连接为后一个节点
    return route2End  # 返回2End格式的路径

# 使用@jit装饰器加速local_search函数，功能是对初始路径进行局部搜索优化，寻找更优路径
@jit(nopython=True) 
def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost  # 初始化当前路径和成本为初始路径和成本
    # search_progress = []  # 注释掉的用于记录搜索过程的列表

    improved = True  # 标记是否有改进
    while improved:  # 当有改进时循环

        improved = False  # 重置改进标记
        # for operator in [operators.two_opt_a2a, operators.relocate_a2a]:  # 注释掉的算子循环

        # 调用2-opt全对全算子寻找改进，返回成本变化量和新路径
        delta, new_tour = gls_operators.two_opt_a2a(cur_route, D,N, first_improvement)
        if delta < 0:  # 若成本降低（有改进）
            improved = True  # 更新改进标记
            cur_cost += delta  # 更新当前成本
            cur_route = new_tour  # 更新当前路径

            # search_progress.append({  # 注释掉的记录搜索进度的代码
            #     'time': time.time(),
            #     'cost': cur_cost
            # })

        # 调用relocate全对全算子寻找改进，返回成本变化量和新路径
        delta, new_tour = gls_operators.relocate_a2a(cur_route, D,N, first_improvement)
        if delta < 0:  # 若成本降低（有改进）
            improved = True  # 更新改进标记
            cur_cost += delta  # 更新当前成本
            cur_route = new_tour  # 更新当前路径

            # search_progress.append({  # 注释掉的记录搜索进度的代码
            #     'time': time.time(),
            #     'cost': cur_cost
            # })            

    return cur_route, cur_cost  # 返回优化后的路径和成本

# 使用@jit装饰器加速route2tour函数，功能是将2End格式的路径转换为普通路径列表
@jit(nopython=True) 
def route2tour(route):
    s = 0  # 从起点0开始
    tour=[]  # 初始化路径列表
    for i in range(len(route)):  # 遍历路径长度
        tour.append(route[s,1])  # 将当前节点的后向连接节点加入路径
        s = route[s,1]  # 更新当前节点为后向连接节点
    return tour  # 返回转换后的路径

# 使用@jit装饰器加速tour2route函数，功能是将普通路径列表转换为2End格式的路径
@jit(nopython=True) 
def tour2route(tour):
    n = len(tour)  # 获取路径长度
    route2End = np.zeros((n,2))  # 创建n行2列的数组
    route2End[tour[0],0] = tour[-1]  # 路径第一个节点的前向连接为最后一个节点
    route2End[tour[0],1] = tour[1]  # 路径第一个节点的后向连接为第二个节点
    for i in range(1,n-1):  # 遍历中间节点
        route2End[tour[i],0] = tour[i-1]  # 第i个节点的前向连接为前一个节点
        route2End[tour[i],1] = tour[i+1]  # 第i个节点的后向连接为后一个节点
    route2End[tour[n-1],0] = tour[n-2]  # 路径最后一个节点的前向连接为倒数第二个节点
    route2End[tour[n-1],1] = tour[0]  # 路径最后一个节点的后向连接为第一个节点，形成闭环
    return route2End  # 返回2End格式的路径

# 定义guided_local_search函数（被注释的@jit装饰器可用于加速），功能是执行引导式局部搜索，
# 结合启发式算法更新距离矩阵以跳出局部最优
# @jit(nopython=True) 
def guided_local_search(coords, edge_weight, nearest_indices,  init_tour, init_cost, t_lim,ite_max, perturbation_moves,
                        first_improvement=False,guide_algorithm=None):

    # 设置随机种子，保证结果可复现
    random.seed(2024)

    # 对初始路径进行局部搜索，得到当前路径和成本，并初始化最佳路径和成本
    cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight,nearest_indices, first_improvement)
    best_route, best_cost = cur_route, cur_cost

    # 获取距离矩阵的大小（节点数量）
    length = len(edge_weight[0])

    # 计算扰动的节点数量，取节点数量的1/10和20中的较小值
    n_pert = min(int(length/10),20)

    # 初始化迭代次数
    iter_i = 0

    # 初始化边惩罚矩阵，用于记录边的使用惩罚
    edge_penalty = np.zeros((length,length))

    # 当迭代次数小于最大迭代次数且未超过时间限制时循环
    while iter_i < ite_max and time.time()<t_lim:

        # 进行指定次数的扰动操作
        for move in range(perturbation_moves):

            # 将当前路径和最佳路径从2End格式转换为普通路径列表
            cur_tour, best_tour = route2tour(cur_route), route2tour(best_route)

            # 注释掉的使用引导算法获取引导距离矩阵和节点的代码
            #edge_weight_guided, node_guided = guide_algorithm.get_matrix_and_nodes(edge_weight, np.array(cur_tour),np.array(best_tour), edge_penalty)
            # print(node_guided.shape)
            # print(node_guided)
            # 使用引导算法更新距离矩阵，得到引导距离矩阵
            edge_weight_guided = guide_algorithm.update_edge_distance(edge_weight, np.array(cur_tour), edge_penalty)

            # 将引导距离矩阵转换为矩阵格式
            edge_weight_guided =  np.asmatrix(edge_weight_guided)
            
            # 计算引导距离矩阵与原距离矩阵的差值
            edge_weight_gap = edge_weight_guided - edge_weight

            # 处理前5个最大差值的边
            for topid in range(5):

                # 找到差值矩阵中最大元素的索引
                max_indices = np.argmin(-edge_weight_gap, axis=None)               

                # 将一维索引转换为二维行列索引
                rows, columns = np.unravel_index(max_indices, edge_weight_gap.shape)
                #print(rows,columns)  # 注释掉的打印索引的代码

                # 对找到的边及其反向边增加惩罚
                edge_penalty[rows,columns] += 1
                edge_penalty[columns,rows] += 1

                # 将已处理的边的差值设为0，避免重复处理
                edge_weight_gap[rows, columns] = 0
                edge_weight_gap[columns, rows] = 0

                # 对边的两个节点分别应用2-opt和relocate算子进行优化
                for id in [rows,columns]:
                    # 应用2-opt算子，返回成本变化量和新路径
                    delta, new_route = gls_operators.two_opt_o2a_all(cur_route, edge_weight_guided,nearest_indices, id)
                    if delta<0:  # 若成本降低
                        #print(delta)  # 注释掉的打印成本变化的代码
                        cur_cost = utils.tour_cost_2End(edge_weight,new_route)  # 更新当前成本
                        cur_route = new_route  # 更新当前路径
                    # 应用relocate算子，返回成本变化量和新路径
                    delta, new_route = gls_operators.relocate_o2a_all(cur_route, edge_weight_guided,nearest_indices, id)
                    if delta<0:  # 若成本降低
                        #print(delta)  # 注释掉的打印成本变化的代码
                        cur_cost = utils.tour_cost_2End(edge_weight,new_route)  # 更新当前成本
                        cur_route = new_route  # 更新当前路径

            #print(nodes_perturb)  # 注释掉的打印扰动节点的代码
            
            # 注释掉的另一种扰动方式的代码
            # for id in nodes_perturb:

            #     edge_penalty[id,cur_route[id][1]] += 1
            #     edge_penalty[cur_route[id][1],id] += 1
            #     edge_penalty[id,cur_route[id][0]] += 1
            #     edge_penalty[cur_route[id][0],id] += 1

            #     delta, new_route = gls_operators.two_opt_o2a_all(cur_route, edge_weight_guided,nearest_indices, id)
            #     if delta<0:
            #         #print(delta)
            #         cur_cost = utils.tour_cost_2End(edge_weight,new_route)
            #         cur_route = new_route
            #     delta, new_route = gls_operators.relocate_o2a_all(cur_route, edge_weight_guided,nearest_indices, id)
            #     if delta<0:
            #         #print(delta)
            #         cur_cost = utils.tour_cost_2End(edge_weight,new_route)
            #         cur_route = new_route

            # 注释掉的路径格式转换代码
            #cur_route_new = [int(element) for element in cur_route]
            #cur_route = tour2route(cur_route_new).astype(int)
                
        # 对当前路径再次进行局部搜索优化
        cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight, nearest_indices, first_improvement)
        # 重新计算当前路径的成本（基于原始距离矩阵）
        cur_cost = utils.tour_cost_2End(edge_weight,cur_route)

        # 若当前成本优于最佳成本，则更新最佳路径和成本
        if cur_cost < best_cost:
            best_route, best_cost = cur_route, cur_cost
        #print(str(iter_i)+" current cost = ",cur_cost)  # 注释掉的打印当前成本的代码
        # 迭代次数加1
        iter_i += 1

        # 每50次迭代，将当前路径和成本重置为最佳路径和成本
        if iter_i%50==0:
            cur_route, cur_cost = best_route, best_cost

    #print(str(iter_i)+" current cost = ",cur_cost)  # 注释掉的打印最终成本的代码
    # 返回最佳路径、最佳成本和迭代次数
    return best_route, best_cost, iter_i