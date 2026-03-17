# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import copy
import random
import geatpy as ea  # 导入geatpy库
from shutil import copy as fcopy
from collections import deque


from Operators import flight_circle_to_chorm
from Operators import init_Chrom
from Operators import evaluation
from Local_Search import Here_Insert
from Local_Search import Here_Cross
from Local_Search import Homo_Insert
from Local_Search import Homo_Cross
from Local_Search import Cancel_Flight


def get_cost(element):
    return element['Cost']

def get_exchanges(element):
    return element['Exchanges']

def is_dominated(sol_1, sol_2):
    """
    判断 sol_a 是否被 sol_b 支配
    sol_a 和 sol_b 是两个包含目标函数值的数组
    """
    sol_a = [sol_1['Cost'], sol_1['Exchanges']]
    sol_b = [sol_2['Cost'], sol_2['Exchanges']]
    sol_a = np.array(sol_a)
    sol_b = np.array(sol_b)
    if np.all(sol_b <= sol_a) and np.any(sol_b < sol_a):
        return True
    else:
        return False


def find_non_dominated_solutions(solutions):
    """
    提取非支配解
    solutions 是一个包含多解的 np.array，每行是一个解，每列是一个目标函数值
    """
    num_solutions = len(solutions)
    non_dominated_solutions = []
    non_dominated_solutions_tag = []

    for i in range(num_solutions):
        dominated = True
        if solutions[i]['Violation'][0, 0] == 0 and solutions[i]['Violation'][0, 1] == 0 and solutions[i]['Violation'][0, 2] == 0 and solutions[i]['Violation'][0, 3] == 0 and solutions[i]['Violation'][0, 4] == 0:
            dominated = False
            for j in range(num_solutions):
                if solutions[j]['Violation'][0, 0] == 0 and solutions[j]['Violation'][0, 1] == 0 and solutions[j]['Violation'][0, 2] == 0 and solutions[j]['Violation'][0, 3] == 0 and solutions[j]['Violation'][0, 4] == 0:
                    if i != j and is_dominated(solutions[i], solutions[j]):
                        dominated = True
                        break

            if not dominated:
                non_dominated_solutions.append(solutions[i])
                non_dominated_solutions_tag.append(i)

    return non_dominated_solutions, non_dominated_solutions_tag


def remove_duplicates(individuals):
    """
    从字典列表中去除 'Cost' 和 'Exchanges' 字段相同的重复项。
    :param individuals: 包含多个字典的列表
    :return: 去除重复项后的新列表
    """

    seen = set()  # 用于存储已出现的 (Cost, Exchanges) 组合
    unique_individuals = []

    for individual in range(0, len(individuals)):
        # 获取当前字典的 'Cost' 和 'Exchanges' 值
        cost_exchange_pair = (individuals[individual]['Cost'], individuals[individual]['Exchanges'])

        # 检查是否已经遇到过相同的 (Cost, Exchanges) 组合
        if cost_exchange_pair not in seen:
            seen.add(cost_exchange_pair)  # 记录新的组合
            unique_individuals.append(individuals[individual])  # 将独特的字典添加到结果列表中

    return unique_individuals




class SAMA_RL_templet(ea.MoeaAlgorithm):

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         dirName)
        if population.ChromNum == 1:
            raise RuntimeError('传入的种群对象必须是多染色体的种群类型。')
        self.name = 'psy-NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        # 由于有多个染色体，因此需要用多个重组和变异算子
        self.recOpers = []
        self.mutOpers = []
        self.co_mutOpers = []
        for i in range(population.ChromNum):
            if population.Encodings[i] == 'P':
                recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
                mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
                co_mutOper = ea.Mutinv(Pm=1)
            elif population.Encodings[i] == 'BG':
                recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
                mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
                co_mutOper = ea.Mutbin(Pm=None)
            elif population.Encodings[i] == 'RI':
                recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
                aircraft_assignment_mutation = self.problem.aircraft_assignment_mutation
                mutOper = ea.Mutpolyn(Pm=aircraft_assignment_mutation, DisI=20)  # 生成多项式变异算子对象
                co_mutOper = ea.Mutpolyn(Pm= 2 * aircraft_assignment_mutation, DisI=20)  # 生成多项式变异算子对象
            else:
                raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)
            self.co_mutOpers.append(co_mutOper)

        self.solution_set = []  # 解集存储
        self.off_solution_set = []  # 存储评估的子代解
        self.co_solution_set = []
        self.non_dominated_set = []  # 非支配解集
        self.current_iter = 0  # 当前迭代计数器
        self.evalsNum = 0 # 评价次数
        self.iter_max = 1


    def reinsertion(self, population, offspring, NUM, solution_set_size, off_NIND, reinsertion_tag):
        if reinsertion_tag == 'Pop':
            # 父子两代合并
            population = population + offspring
            self.solution_set.extend(self.off_solution_set)
            # 选择个体保留到下一代
            [levels, criLevel] = self.ndSort(population.ObjV, population.sizes, None, population.CV, self.problem.maxormins)  # 对NUM个个体进行非支配分层
            dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
            population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
            chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体


            non_dominated_tag = np.where(levels == 1)[0]
            NDSet = population[non_dominated_tag]
            non_dominated_solution_set_1 = []
            for tag_indi in non_dominated_tag:
                non_dominated_solution_set_1.append(self.solution_set[tag_indi])
            non_dominated_solution_set_2 = []
            # if NDSet.CV is not None:  # CV不为None说明有设置约束条件
            constraints_tag = np.where(np.all(NDSet.CV <= 0, 1))[0]
            select_NDSet = NDSet[constraints_tag]  # 最后要彻底排除非可行解
            for tag_indi in constraints_tag:
                non_dominated_solution_set_2.append(non_dominated_solution_set_1[tag_indi])

            non_dominated_solution_set_2 = remove_duplicates(non_dominated_solution_set_2)
            self.non_dominated_set = non_dominated_solution_set_2


            orig_solution_set = copy.deepcopy(self.solution_set)
            self.solution_set = []

            off_1 = 0
            off_2 = 0
            for i in range(len(chooseFlag)):
                num = chooseFlag[i]
                if num >= solution_set_size + off_NIND:
                    off_2 += 1
                elif num >= solution_set_size:
                    off_1 += 1

                if num >= solution_set_size:
                    num = num - solution_set_size
                    self.solution_set.append(self.off_solution_set[num])
                else:
                    self.solution_set.append(orig_solution_set[num])

            print('主种群选择: (主) %s , (辅) %s' %(off_1, off_2))

        elif reinsertion_tag == 'Co_Pop':
            # 父子两代合并
            pop_size = population.sizes
            population = population + offspring
            sort_population_CV = copy.deepcopy(population.CV)
            for indi in range(population.sizes):
                sort_population_CV[indi, 2] = 0

            # 选择个体保留到下一代
            [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, sort_population_CV, self.problem.maxormins)  # 对NUM个个体进行非支配分层
            dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
            population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
            chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体

            orig_solution_set = copy.deepcopy(self.co_solution_set)
            self.co_solution_set = []

            off_1 = 0
            off_2 = 0

            for i in range(len(chooseFlag)):
                num = chooseFlag[i]
                if num >= pop_size + off_NIND:
                    off_2 += 1
                elif num >= pop_size:
                    off_1 += 1
                if num >= pop_size:
                    num = num - pop_size
                    self.co_solution_set.append(self.off_solution_set[num])
                else:
                    self.co_solution_set.append(orig_solution_set[num])

            print('辅助种群选择: (主) %s , (辅) %s' % (off_1, off_2))

        return population[chooseFlag]

    def extract_cancelled_flights(self, solution, config):
        """
        提取取消的航班或航班环，计算取消成本
        """
        cancelled_set = {}
        Cabin = ['F', 'B', 'E']
        for group_id, group_info in solution.items():
            # 检查航班环内的航班是否取消
            for flight in group_info['Flights']:
                if flight['State'] == -1:
                    cancelled_set[group_id] = {
                        **group_info
                    }

        for group_id, group_info in cancelled_set.items():
            # 取消航班环内所有航班
            cancel_cost_sum = 0
            for flight in group_info['Flights']:
                flight_type = flight['Type']
                flight['State'] = -1
                for cabin_type in Cabin:
                    cancel_cost_tag = f'CancelInCost_' + cabin_type + '_' + flight_type
                    cancel_cost = config[cancel_cost_tag]
                    cancel_cost_sum += cancel_cost * flight[cabin_type]

            cancelled_set[group_id]['CancelCost'] = cancel_cost_sum

        return cancelled_set

    def group_active_flights(self, solution, config):
        """
        将未取消的航班按照执行飞机分组
        """
        active_flights = {}
        Cabin = ['F', 'B', 'E']
        # 查询未取消航班，按飞机分组存放
        for group_id, group_info in solution.items():
            check_cancel = 0
            for flight in group_info['Flights']:
                if flight['State'] == -1:
                    check_cancel = 1
            if check_cancel == 0:
                # 计算航班环取消成本
                cancel_cost_sum = 0
                for flight in group_info['Flights']:
                    flight_type = flight['Type']
                    for cabin_type in Cabin:
                        cancel_cost_tag = f'CancelInCost_' + cabin_type + '_' + flight_type
                        cancel_cost = config[cancel_cost_tag]
                        cancel_cost_sum += cancel_cost * flight[cabin_type]

                aircraft_id = group_info['Aircraft']
                if aircraft_id not in active_flights:
                    active_flights[aircraft_id] = {}
                    active_flights[aircraft_id]['FlightGroups'] = []
                active_flights[aircraft_id]['FlightGroups'].append({
                    'Group_ID': group_id,
                    'CancelCost': cancel_cost_sum,
                    **group_info
                })


        # 对集合里每个飞机的航班序列按照起飞先后顺序排序
        for aircraft_id, groups in active_flights.items():
            sorted(groups['FlightGroups'], key=lambda item: (item['Flights'][0]['New_DepTime']))

        return active_flights

    # 从恢复的飞机环中获取取消航班及飞机序列信息，并进行计算评估
    def extract_information(self, flight_circles, config, evaluation_record, constraints, Vars):
        # 提取取消的航班并排序
        cancelled_set = self.extract_cancelled_flights(flight_circles, config)

        # 对于临时取消的航班集，按成本从高到低排序
        sorted_cancelled_set = dict(sorted(
            cancelled_set.items(),
            key=lambda item: (item[1]['CancelCost']),
            reverse=True
        ))

        # 构建未取消的航班分组
        active_flights = self.group_active_flights(flight_circles, config)

        # 计算当前每个飞机的飞行时间
        for aircraft_id, groups in active_flights.items():
            flying_time = 0
            exchange = 0
            delay_time = 0
            for flight_group in groups['FlightGroups']:
                if flight_group['Aircraft'] != aircraft_id:
                    exchange_flag = True
                else:
                    exchange_flag = False
                for flight in flight_group['Flights']:
                    if flight['State'] != -1:
                        delay_time += flight['Delay']
                        flying_time += flight['Duration']
                    if exchange_flag:
                        exchange += 1

            active_flights[aircraft_id]['FlyingTime'] = flying_time
            active_flights[aircraft_id]['Exchanges'] = exchange
            active_flights[aircraft_id]['Delay'] = delay_time


        # 集成解
        solution_collection = {
            'CanceledFlights': sorted_cancelled_set,
            'AircraftRotations': active_flights,
            'FlightCircles': flight_circles,
            'Cost': evaluation_record['fitness_1'],
            'Exchanges': evaluation_record['fitness_2'],
            'Violation': constraints,
            'EvaluationRecord': evaluation_record,
            'Vars': Vars
        }
        return solution_collection

    def evaluate_solutions(self, problem, population, NIND, evaluate_tag):
        """
        第一步：初始化解集，在原始计划基础上随机取消或延迟航班。
        """
        # 获取初始航班信息及其他必要信息
        ini_set_size = NIND
        init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
        init_airports_info = copy.deepcopy(problem.airports_info)
        flights_info = copy.deepcopy(problem.flights_info)
        group_lookup_dic = copy.deepcopy(problem.group_lookup_dic)
        aircraft_info = problem.aircraft_info
        airports_info = copy.deepcopy(problem.airports_info)
        airports_capacity_total = copy.deepcopy(problem.airports_capacity_total)
        position_info = copy.deepcopy(problem.position_info)
        aircraft_category = problem.aircraft_category

        config = problem.config_info
        chorm_num = problem.chorm_size
        initialize_index = problem.initialize
        RVEA_init = problem.RVEA_init

        recovery_start_time = problem.recovery_start_time
        recovery_end_time = problem.recovery_end_time
        aircraft_info = problem.aircraft_info
        alt_flights_info = problem.alt_flights_info
        alt_airports_info = problem.alt_airports_info
        alt_aircraft_info = problem.alt_aircraft_info
        cancel_hour_dc = problem.cancel_hour_dc
        cancel_hour_i = problem.cancel_hour_i
        cancel_decision = problem.cancel_decision
        sort_str = problem.sort_str
        canceled_flights_swap = problem.canceled_flights_count
        max_delay = problem.max_delay
        init_aircraft_assignment = self.problem.init_aircraft_assignment
        init_cancel_flight_num = self.problem.init_cancel_flight_num

        num_groups = len(self.problem.flight_circles_info)

        if evaluate_tag == 'Pop':
            self.solution_set = []
        elif evaluate_tag == 'Init':
            self.solution_set = []
        elif evaluate_tag == 'Off':
            self.off_solution_set = []
        elif evaluate_tag == 'Co_Pop':
            self.co_solution_set = []

        # 获取种群中每个个体的染色体，并进行评估
        for i in range(NIND):
            self.evalsNum += 1
            Vars = np.append(population.Chroms[0][i], population.Chroms[1][i])

            init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
            if evaluate_tag == 'Init':
                solution = evaluation(Vars, config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                                        alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                                        init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total,
                                        flights_info, group_lookup_dic, position_info, aircraft_category, cancel_decision,
                                        canceled_flights_swap, max_delay, init_cancel_flight_num, 'init_re_repair')

            else:
                solution = evaluation(Vars, config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                                      alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                                      init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total,
                                      flights_info, group_lookup_dic, position_info, aircraft_category, cancel_decision,
                                      canceled_flights_swap, max_delay, init_cancel_flight_num, 'repair')

            evaluation_record = solution['EvaluationRecord']
            constraints = solution['Violation']
            flight_circles = solution['FlightCircles']
            Vars = solution['Vars']
            population.Chroms[0][i] = Vars[:num_groups]
            population.Chroms[1][i] = Vars[num_groups:]
            population.ObjV[i, 0] = solution['EvaluationRecord']['fitness_1'] # 目标函数1
            population.ObjV[i, 1] = solution['EvaluationRecord']['fitness_2'] # 目标函数2

            # 更新约束违反量
            population.CV[i, 0] = solution['Violation'][0][0]
            population.CV[i, 1] = solution['Violation'][0][1]
            population.CV[i, 2] = solution['Violation'][0][2]
            population.CV[i, 3] = solution['Violation'][0][3]
            population.CV[i, 4] = solution['Violation'][0][4]


            # 集成解存入解集
            solution_collection = self.extract_information(flight_circles, config, evaluation_record, constraints, Vars)
            if evaluate_tag == 'Pop':
                self.solution_set.append(solution_collection)
            elif evaluate_tag == 'Init':
                self.solution_set.append(solution_collection)
            elif evaluate_tag == 'Off':
                self.off_solution_set.append(solution_collection)
            elif evaluate_tag == 'Co_Pop':
                self.co_solution_set.append(solution_collection)

        return population

    def repair_solutions(self, problem, population, to_repair_population, repair_NIND):
        # 获取初始航班信息及其他必要信息
        ini_set_size = repair_NIND
        init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
        init_airports_info = copy.deepcopy(problem.airports_info)
        flights_info = copy.deepcopy(problem.flights_info)
        group_lookup_dic = copy.deepcopy(problem.group_lookup_dic)
        aircraft_info = problem.aircraft_info
        airports_info = copy.deepcopy(problem.airports_info)
        airports_capacity_total = copy.deepcopy(problem.airports_capacity_total)
        position_info = copy.deepcopy(problem.position_info)
        aircraft_category = problem.aircraft_category

        config = problem.config_info
        chorm_num = problem.chorm_size
        initialize_index = problem.initialize
        RVEA_init = problem.RVEA_init

        recovery_start_time = problem.recovery_start_time
        recovery_end_time = problem.recovery_end_time
        aircraft_info = problem.aircraft_info
        alt_flights_info = problem.alt_flights_info
        alt_airports_info = problem.alt_airports_info
        alt_aircraft_info = problem.alt_aircraft_info
        cancel_hour_dc = problem.cancel_hour_dc
        cancel_hour_i = problem.cancel_hour_i
        cancel_decision = problem.cancel_decision
        sort_str = problem.sort_str
        canceled_flights_swap = problem.canceled_flights_count
        max_delay = problem.max_delay
        init_aircraft_assignment = self.problem.init_aircraft_assignment
        init_cancel_flight_num = self.problem.init_cancel_flight_num

        num_groups = len(self.problem.flight_circles_info)


        # 获取种群中每个个体的染色体，并进行评估
        repair_solution_set = []
        for i in range(to_repair_population.sizes):
            if np.any(to_repair_population.CV[i][0:4]) > 0 and to_repair_population.CV[i][4] == 0:
                if random.random() < 2:
                # if random.random() < 0.5:
                    indi = i
                    self.evalsNum += 1
                    Vars = np.append(to_repair_population.Chroms[0][indi], to_repair_population.Chroms[1][indi])

                    init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
                    solution = evaluation(Vars, config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                                          alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                                          init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total,
                                          flights_info, group_lookup_dic, position_info, aircraft_category, cancel_decision,
                                          canceled_flights_swap, max_delay, init_cancel_flight_num, 'repair')

                    evaluation_record = solution['EvaluationRecord']
                    constraints = solution['Violation']
                    flight_circles = solution['FlightCircles']
                    Vars = solution['Vars']

                    # 集成解存入解集
                    solution_collection = self.extract_information(flight_circles, config, evaluation_record, constraints, Vars)
                    # if evaluate_tag == 'Pop':
                    repair_solution_set.append(solution_collection)

        if len(repair_solution_set) > 0:
            repair_population = ea.PsyPopulation(population.Encodings, population.Fields, len(repair_solution_set))
            repair_population.initChrom()
            repair_population.ObjV = np.zeros((repair_population.sizes, 2))
            repair_population.CV = np.zeros((repair_population.sizes, 5))
            for indi in range(len(repair_solution_set)):
                repair_population.Chroms[0][indi] = repair_solution_set[indi]['Vars'][:num_groups]
                repair_population.Chroms[1][indi] = repair_solution_set[indi]['Vars'][num_groups:]
                repair_population.ObjV[indi, 0] = repair_solution_set[indi]['EvaluationRecord']['fitness_1']  # 目标函数1
                repair_population.ObjV[indi, 1] = repair_solution_set[indi]['EvaluationRecord']['fitness_2']  # 目标函数2

                # 更新约束违反量
                repair_population.CV[indi, 0] = repair_solution_set[indi]['Violation'][0][0]
                repair_population.CV[indi, 1] = repair_solution_set[indi]['Violation'][0][1]
                repair_population.CV[indi, 2] = repair_solution_set[indi]['Violation'][0][2]
                repair_population.CV[indi, 3] = repair_solution_set[indi]['Violation'][0][3]
                repair_population.CV[indi, 4] = repair_solution_set[indi]['Violation'][0][4]

            self.solution_set.extend(repair_solution_set)
            population = population + repair_population

        return population

    def search_evaluation(self, cancelled_set, active_flights, config, Obj1, Obj2):
        self.evalsNum += 1
        sucess = 0
        problem = self.problem
        init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
        flights_info = copy.deepcopy(problem.flights_info)
        group_lookup_dic = copy.deepcopy(problem.group_lookup_dic)
        aircraft_info = problem.aircraft_info
        airports_info = copy.deepcopy(problem.airports_info)
        airports_capacity_total = copy.deepcopy(problem.airports_capacity_total)
        position_info = copy.deepcopy(problem.position_info)
        aircraft_category = problem.aircraft_category

        config = problem.config_info
        recovery_start_time = problem.recovery_start_time
        recovery_end_time = problem.recovery_end_time
        alt_flights_info = problem.alt_flights_info
        alt_airports_info = problem.alt_airports_info
        alt_aircraft_info = problem.alt_aircraft_info
        cancel_hour_dc = problem.cancel_hour_dc
        cancel_hour_i = problem.cancel_hour_i
        cancel_decision = problem.cancel_decision
        sort_str = problem.sort_str
        canceled_flights_swap = problem.canceled_flights_count
        max_delay = problem.max_delay
        chroms_change = problem.chroms_change
        init_cancel_flight_num = problem.init_cancel_flight_num

        new_flight_circles = {}
        # 将取消航班集合和飞机航班序列整合为flight_circle序列，再使用原有的方式判断
        for group_id, group_info in cancelled_set.items():
            new_flight_circles[group_id] = {
                'Group_ID': group_id,
                **group_info
            }

        for aircraft_id, groups in active_flights.items():
            for group_index in range(len(groups['FlightGroups'])):
                group_info = groups['FlightGroups'][group_index]
                group_id = group_info['Group_ID']
                new_flight_circles[group_id] = group_info
                new_flight_circles[group_id]['Aircraft'] = aircraft_id

        flight_circles = dict(sorted(
            new_flight_circles.items(),
            key=lambda item: int(item[0][6:])
        ))
        num_groups = len(self.problem.flight_circles_info)
        Vars = flight_circle_to_chorm(num_groups, num_groups, flight_circles, aircraft_info, cancel_hour_dc, cancel_hour_i, cancel_decision)
        init_aircraft_assignment = self.problem.init_aircraft_assignment

        solution = evaluation(Vars[0], config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                              alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                              init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total,
                              flights_info, group_lookup_dic, position_info, aircraft_category, cancel_decision,
                              canceled_flights_swap, max_delay, init_cancel_flight_num, 'repair')

        evaluation_record = solution['EvaluationRecord']
        constraints = solution['Violation']
        flight_circles = solution['FlightCircles']
        Vars = solution['Vars']

        # 若找到更优解，终止搜索，加入新解到集合
        New_Obj1 = evaluation_record['fitness_1']
        New_Obj2 = evaluation_record['fitness_2']
        if New_Obj1 < Obj1 or New_Obj2 < Obj2:
            if constraints[0, 0] + constraints[0, 1] + constraints[0, 2] + constraints[0, 3] + constraints[0, 4] == 0:
                sucess += 1
                # print('Better!')
            else:
                print(constraints)
        # 集合生成解
        new_solution_collection = self.extract_information(flight_circles, config, evaluation_record,
                                                       constraints, Vars)
        check_flag = True

        return new_solution_collection, check_flag, sucess

    def cross_operator(self, orig_parent_1, orig_parent_2, obj, config):
        # 获取航班环信息、取消航班信息和飞机序列信息
        problem = self.problem
        iter_max = self.iter_max
        Obj1 = orig_parent_1['Cost']
        Obj2 = orig_parent_1['Exchanges']
        flight_circles = copy.deepcopy(orig_parent_1['FlightCircles'])
        init_aircraft_assignment = self.problem.init_aircraft_assignment
        aircraft_info = copy.deepcopy(problem.aircraft_info)
        init_flight_circles = copy.deepcopy(problem.flight_circles_info)
        len_field1 = len(problem.flight_circles_info)
        len_field2 = len(problem.flight_circles_info)
        cancel_hour_dc = problem.cancel_hour_dc
        cancel_hour_i = problem.cancel_hour_i
        cancel_decision = problem.cancel_decision
        off_set = []

        # 提取取消的航班并排序
        cancelled_set = self.extract_cancelled_flights(flight_circles, config)

        # 对于临时取消的航班集，按成本从高到低排序
        sorted_cancelled_set = dict(sorted(
            cancelled_set.items(),
            key=lambda item: (item[1]['CancelCost']),
            reverse=True
        ))

        # 构建未取消的航班分组
        active_flights = self.group_active_flights(flight_circles, config)

        # 计算当前每个飞机的飞行时间
        for aircraft_id, groups in active_flights.items():
            flying_time = 0
            exchange = 0
            delay_time = 0
            cancel_cost = 0
            for flight_group in groups['FlightGroups']:
                cancel_cost += flight_group['CancelCost']
                if flight_group['Aircraft'] != aircraft_id:
                    exchange_flag = True
                else:
                    exchange_flag = False
                for flight in flight_group['Flights']:
                    if flight['State'] != -1:
                        delay_time += flight['Delay']
                        flying_time += flight['Duration']
                    if exchange_flag:
                        exchange += 1

            active_flights[aircraft_id]['FlyingTime'] = flying_time
            active_flights[aircraft_id]['CancelCost'] = cancel_cost
            active_flights[aircraft_id]['Exchanges'] = exchange
            active_flights[aircraft_id]['Delay'] = delay_time

        # 根据目标函数，确认交叉策略
        if obj == 'Cost':
            active_flights = dict(sorted(
                active_flights.items(),
                key=lambda item: (item[1]['FlyingTime'])
                # reverse=True
            ))

        elif obj == 'Exchanges':
            active_flights = dict(sorted(
                active_flights.items(),
                key=lambda item: (item[1]['Delay'])
            ))

        # 执行交叉，获得新解
        op_cancelled_set = copy.deepcopy(cancelled_set)
        op_active_flights = copy.deepcopy(active_flights)
        rest_flight_circles = copy.deepcopy(orig_parent_2['FlightCircles'])
        # 构建未取消的航班分组
        rest_active_flights = self.group_active_flights(rest_flight_circles, config)

        aircraft_num = len(aircraft_info)
        # 从主父代中传递的飞机航班序列数量
        pass_rate = 0.3 + random.random() * 0.4
        cross_aircraft_num = int(aircraft_num * pass_rate)

        total_aircraft_list = list(aircraft_info.keys())
        total_flight_circle_list = list(init_flight_circles.keys())

        select_aircraft_id = list(op_active_flights.keys())[:cross_aircraft_num]

        # 将主父代的相应飞机航班序列信息传递给子代，并删去已传递飞机、航班
        offspring_flight_circles = copy.deepcopy(init_flight_circles)
        for aircraft_id in select_aircraft_id:
            for group in op_active_flights[aircraft_id]['FlightGroups']:
                group_id = group['Group_ID']
                offspring_flight_circles[group_id] = group
                total_flight_circle_list.remove(group_id)

            total_aircraft_list.remove(aircraft_id)

        # 将次父代中的剩余飞机航班序列信息传递给子代，若有重复航班环，则删除此航班环
        for aircraft_id, groups in rest_active_flights.items():
            if aircraft_id in total_aircraft_list:
                for group in groups['FlightGroups']:
                    group_id = group['Group_ID']
                    if group_id in total_flight_circle_list:
                        offspring_flight_circles[group_id] = group
                        total_flight_circle_list.remove(group_id)

                total_aircraft_list.remove(aircraft_id)

        # 将剩余的航班全部取消
        # print(len(total_flight_circle_list))
        for group_id in total_flight_circle_list:
            for flight in offspring_flight_circles[group_id]['Flights']:
                flight['State'] = -1
                flight['Delay'] = 0


        # 获取航班环对应染色体
        offspring_chorm = flight_circle_to_chorm(len_field1, len_field2, offspring_flight_circles, aircraft_info,
                                                 cancel_hour_dc, cancel_hour_i, cancel_decision)

        return  offspring_chorm



    def neighborhood_search(self, solution, obj, config, Q_table, epsilon, action_list, search_state, chosen_tag):
        """
        对指定的解进行邻域搜索，以优化两个目标函数
        """
        # 获取航班环信息、取消航班信息和飞机序列信息
        problem = self.problem
        iter_max = self.iter_max
        Obj1 = solution['Cost']
        Obj2 = solution['Exchanges']
        flight_circles = solution['FlightCircles']
        if search_state == 'search-left':
            state = 1
        elif search_state == 'search-middle':
            state = 2
        elif search_state == 'search-right':
            state = 3
        elif search_state == 'dominated':
            state = 4

        if random.random() < epsilon:
            chosen_p = [1 / (tag + 1) for tag in chosen_tag[state]]
            sum_p = sum(chosen_p)
            chosen_p = [p / sum_p for p in chosen_p]
            action = random.choices(action_list, weights=chosen_p, k=1)[0]
            print('随机搜索概率：', state , chosen_p)
        else:
            if min(chosen_tag[state]) == 0:
                chosen_p = [1 / (tag + 1) for tag in chosen_tag[state]]
                sum_p = sum(chosen_p)
                chosen_p = [p / sum_p for p in chosen_p]
                action = random.choices(action_list, weights=chosen_p, k=1)[0]
                print('随机搜索概率：', state , chosen_p)
            else:
                max_value = np.max(Q_table[state])
                max_index = []
                for i in range(len(Q_table[state])):
                    if Q_table[state][i] == max_value:
                        max_index.append(i)
                chooose_index = random.choice(max_index)
                action = action_list[chooose_index]
                print('Q学习选择概率：', state , Q_table[state])
                print('Q学习选择算子：', action)

        action_index = action_list.index(action)
        chosen_tag[state][action_index] += 1

        init_aircraft_assignment = self.problem.init_aircraft_assignment
        off_set = []
        curr_obj = solution[obj]

        # 提取取消的航班并排序
        cancelled_set = self.extract_cancelled_flights(flight_circles, config)

        # 对于临时取消的航班集，按成本从高到低排序
        sorted_cancelled_set = dict(sorted(
            cancelled_set.items(),
            key=lambda item: (item[1]['CancelCost']),
            reverse=True
        ))

        # 构建未取消的航班分组
        active_flights = self.group_active_flights(flight_circles, config)

        # 计算当前每个飞机的飞行时间
        for aircraft_id, groups in active_flights.items():
            flying_time = 0
            exchange = 0
            delay_time = 0
            for flight_group in groups['FlightGroups']:
                if flight_group['Aircraft'] != aircraft_id:
                    exchange_flag = True
                else:
                    exchange_flag = False
                for flight in flight_group['Flights']:
                    if flight['State'] != -1:
                        delay_time += flight['Delay']
                        flying_time += flight['Duration']
                    if exchange_flag:
                        exchange += 1

            active_flights[aircraft_id]['FlyingTime'] = flying_time
            active_flights[aircraft_id]['Exchanges'] = exchange
            active_flights[aircraft_id]['Delay'] = delay_time

        # 根据目标函数，确认邻域搜索策略
        if obj == 'Cost':
            cancelled_set = dict(sorted(
                cancelled_set.items(),
                key=lambda item: (item[1]['CancelCost']),
                reverse=True
            ))
            active_flights = dict(sorted(
                active_flights.items(),
                key=lambda item: (item[1]['FlyingTime'])
            ))
        elif obj == 'Exchanges':
            cancelled_set = dict(sorted(
                cancelled_set.items(),
                key=lambda item: (item[1]['CancelCost']),
                reverse=True
            ))
            active_flights = dict(sorted(
                active_flights.items(),
                key=lambda item: (item[1]['Delay'])
            ))

        # 执行邻域搜索，获得一个新解
        op_cancelled_set = copy.deepcopy(cancelled_set)
        op_active_flights = copy.deepcopy(active_flights)
        iter_num = 1
        sucess_total_1 = 0
        sucess_total_2 = 0
        sucess_total_3 = 0
        sucess_total_4 = 0
        sucess_total_5 = 0
        while iter_num <= iter_max:
            cancelled_group_num = len(cancelled_set)
            if cancelled_group_num != 0:
                if action[0] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Here_Insert(problem, op_cancelled_set, op_active_flights)
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_1 += sucess
                    if check_flag:
                        off_set.append(new_solution)
                if action[1] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Here_Cross(problem, op_cancelled_set, op_active_flights)
                    # print('Here_Cross')
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_2 += sucess
                    if check_flag:
                        off_set.append(new_solution)
                if action[2] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Homo_Insert(problem, op_cancelled_set, op_active_flights)
                    # print('Homo_Insert')
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_3 += sucess
                    if check_flag:
                        off_set.append(new_solution)
                if action[3] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Homo_Cross(problem, op_cancelled_set, op_active_flights)
                    # print('Homo_Cross')
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_4 += sucess
                    if check_flag:
                        off_set.append(new_solution)
                if action[4] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Cancel_Flight(problem, op_cancelled_set, op_active_flights, obj)
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_5 += sucess
                    if check_flag:
                        off_set.append(new_solution)

            else:
                if action[2] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Homo_Insert(problem, op_cancelled_set, op_active_flights)
                    # print('2 Homo_Insert')
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_3 += sucess
                    if check_flag:
                        off_set.append(new_solution)
                if action[3] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Homo_Cross(problem, op_cancelled_set, op_active_flights)
                    # print('2 Homo_Cross')
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_4 += sucess
                    if check_flag:
                        off_set.append(new_solution)
                if action[4] == 1:
                    op_cancelled_set = copy.deepcopy(cancelled_set)
                    op_active_flights = copy.deepcopy(active_flights)
                    new_cancelled_set, new_active_flights = Cancel_Flight(problem, op_cancelled_set, op_active_flights, obj)
                    new_solution, check_flag, sucess = self.search_evaluation(new_cancelled_set, new_active_flights, config, Obj1, Obj2)
                    new_solution['Action'] = action_list.index(action)
                    sucess_total_5 += sucess
                    if check_flag:
                        off_set.append(new_solution)

            iter_num += 1

        print(sucess_total_1, sucess_total_2, sucess_total_3, sucess_total_4, sucess_total_5)

        return off_set, chosen_tag


    def run(self, MAXGEN, _output_path, _datapath, output_file_name, output_time, prophetPop=None):
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        config = self.problem.config_info
        num_groups = len(self.problem.flight_circles_info)

        self.initialization()  # 初始化算法类的一些动态参数

        # ===========================准备进化============================
        population.initChrom()  # 初始化种群染色体矩阵

        # 依据初始化设置修改Chrom信息
        problem = self.problem
        init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
        flights_info = copy.deepcopy(problem.flights_info)
        group_lookup_dic = copy.deepcopy(problem.group_lookup_dic)
        aircraft_info = problem.aircraft_info
        airports_info = copy.deepcopy(problem.airports_info)
        airports_capacity_total = copy.deepcopy(problem.airports_capacity_total)
        position_info = copy.deepcopy(problem.position_info)
        aircraft_category = problem.aircraft_category

        config = problem.config_info
        recovery_start_time = problem.recovery_start_time
        recovery_end_time = problem.recovery_end_time
        alt_flights_info = problem.alt_flights_info
        alt_airports_info = problem.alt_airports_info
        alt_aircraft_info = problem.alt_aircraft_info
        cancel_hour_dc = problem.cancel_hour_dc
        cancel_hour_i = problem.cancel_hour_i
        cancel_decision = problem.cancel_decision
        sort_str = problem.sort_str
        canceled_flights_swap = problem.canceled_flights_count
        max_delay = problem.max_delay
        chroms_change = problem.chroms_change

        # Q表初始化
        state_num = 4
        operator_num = 5
        action_num = operator_num
        time_mut_p = 1.0
        learning_rate = 0.3
        discount_factor = 0.9
        max_epsilon = 0.9
        min_epsilon = 0.5
        mut_rate = 1.0
        mut_num = 1.0 * NIND

        action_dic = {}
        action_list = []
        # 生成操作数位数的0，1指示元组
        action_list = [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
        chosen_tag = [[0 for action in range(action_num)] for state in range(state_num)]

        Q_table = {}
        for state in range(1, state_num + 1):
            Q_table[state] = np.zeros((action_num))

        population, init_cancel_flight_num, init_aircraft_assignment = init_Chrom(problem, population)
        problem.init_cancel_flight_num = init_cancel_flight_num
        problem.init_aircraft_assignment = init_aircraft_assignment
        self.problem = problem

        population.ObjV = np.zeros((population.sizes, 2))
        population.CV = np.zeros((population.sizes, 5))

        # 获得初始解集的目标及可行性，获取非支配解集
        population = self.evaluate_solutions(problem, population, population.sizes, 'Init')
        [levels, criLevel] = self.ndSort(population.ObjV, population.sizes, None, population.CV, self.problem.maxormins)  # 对NIND个个体进行非支配分层

        # 记录选入下一代种群的个体的状态
        for solution_index in range(len(levels)):
            # 选中的个体是前沿解
            if levels[solution_index] == 1:
                self.solution_set[solution_index]['Non_Dominated_State'] = 1
                self.solution_set[solution_index]['Source'] = 'init'
            else:
                self.solution_set[solution_index]['Non_Dominated_State'] = 0
                self.solution_set[solution_index]['Source'] = 'init'

        non_dominated_tag = np.where(levels == 1)[0]
        cross_tag = np.where(levels == 1)[0]
        SBX_tag = np.where(levels > 1)[0]
        NDSet = population[non_dominated_tag]
        non_dominated_solution_set_1 = []
        for tag_indi in non_dominated_tag:
            non_dominated_solution_set_1.append(self.solution_set[tag_indi])

        non_dominated_solution_set_2 = []
        if NDSet.CV is not None:  # CV不为None说明有设置约束条件
            constraints_tag = np.where(np.all(NDSet.CV <= 0, 1))[0]
            select_NDSet = NDSet[constraints_tag]  # 最后要彻底排除非可行解
            for tag_indi in constraints_tag:
                non_dominated_solution_set_2.append(non_dominated_solution_set_1[tag_indi])

        tag = non_dominated_tag
        non_dominated_solution_set_2 = remove_duplicates(non_dominated_solution_set_2)
        self.non_dominated_set = non_dominated_solution_set_2

        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        # 增加重复个体删除模块
        duplication_set = []
        for pop_indi in range(population.sizes):
            if dis[pop_indi] == 0:
                check_set = (population.ObjV[pop_indi, 0], population.ObjV[pop_indi, 1], levels[pop_indi])
                if check_set in duplication_set:
                    levels[pop_indi] = np.max(levels)
                else:
                    duplication_set.append(check_set)

        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度


        def_gen = 0
        time_def = 0
        records = []
        time_records = []
        Q_records = []
        pop_records = []
        time_pop_records = []
        test_tag = self.problem.test_tag
        time_max = self.problem.time_max
        time_interval = time_max / 40
        time_max = time_max + time_interval

        search_source_list = ['search-left', 'search-middle', 'search-right']

        search_times = 1
        objectives = ['Cost']
        # ===========================开始进化============================
        with open(_output_path + output_time + '_Revolution_Record.txt', 'a') as f:
            time_tip = time.time() - self.timeSlot
            # 记录进化结果
            curr_time_gen_records = []
            for solution in self.non_dominated_set:
                obj1 = solution['EvaluationRecord']['fitness_1']
                obj2 = solution['EvaluationRecord']['fitness_2']
                cv_num = solution['Violation']
                vars_rec = solution['Vars']

                total_flight_num = solution['EvaluationRecord']['total_flight_num']
                cancel_flight_num = solution['EvaluationRecord']['cancel_flight_num']
                delay_flight_num = solution['EvaluationRecord']['delay_flight_num']
                cancel_passenger_num = solution['EvaluationRecord']['cancel_passenger_num']
                delay_passenger_num = solution['EvaluationRecord']['delay_passenger_num']
                delay_flight_time = solution['EvaluationRecord']['delay_flight_time']
                delay_passenger_time = solution['EvaluationRecord']['delay_passenger_time']
                cancel_cost_sum = solution['EvaluationRecord']['cancel_cost_sum']
                delay_cost_sum = solution['EvaluationRecord']['delay_cost_sum']
                aircraft_exchanges = solution['EvaluationRecord']['aircraft_exchanges']
                cost1_1_total = solution['EvaluationRecord']['cost1_1_total']
                cost1_1 = solution['EvaluationRecord']['cost1_1']
                cost1_2 = solution['EvaluationRecord']['cost1_2']
                cost1_3 = solution['EvaluationRecord']['cost1_3']

                data = (
                    def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num, delay_flight_num,
                    cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                    cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2, cost1_3,
                    vars_rec)
                records.append(data)

                if time_tip > time_def * time_interval:
                    data = [
                        time_def, def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num,
                        delay_flight_num,
                        cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                        cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2,
                        cost1_3,
                        vars_rec]
                    # time_records.append(data)
                    curr_time_gen_records.append(data)

            # 记录当前种群
            curr_time_pop_gen_records = []
            for solution in self.solution_set:
                obj1 = solution['EvaluationRecord']['fitness_1']
                obj2 = solution['EvaluationRecord']['fitness_2']
                cv_num = solution['Violation']
                vars_rec = solution['Vars']

                total_flight_num = solution['EvaluationRecord']['total_flight_num']
                cancel_flight_num = solution['EvaluationRecord']['cancel_flight_num']
                delay_flight_num = solution['EvaluationRecord']['delay_flight_num']
                cancel_passenger_num = solution['EvaluationRecord']['cancel_passenger_num']
                delay_passenger_num = solution['EvaluationRecord']['delay_passenger_num']
                delay_flight_time = solution['EvaluationRecord']['delay_flight_time']
                delay_passenger_time = solution['EvaluationRecord']['delay_passenger_time']
                cancel_cost_sum = solution['EvaluationRecord']['cancel_cost_sum']
                delay_cost_sum = solution['EvaluationRecord']['delay_cost_sum']
                aircraft_exchanges = solution['EvaluationRecord']['aircraft_exchanges']
                cost1_1_total = solution['EvaluationRecord']['cost1_1_total']
                cost1_1 = solution['EvaluationRecord']['cost1_1']
                cost1_2 = solution['EvaluationRecord']['cost1_2']
                cost1_3 = solution['EvaluationRecord']['cost1_3']

                pop_data = (
                    def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num, delay_flight_num,
                    cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                    cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2, cost1_3,
                    vars_rec)
                pop_records.append(pop_data)

                if time_tip > time_def * time_interval:
                    data = [
                        time_def, def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num,
                        delay_flight_num,
                        cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                        cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2,
                        cost1_3,
                        vars_rec]
                    curr_time_pop_gen_records.append(data)

            if time_tip > time_def * time_interval:
                curr_data = copy.deepcopy(curr_time_gen_records)
                curr_pop_data = copy.deepcopy(curr_time_pop_gen_records)
                time_records.extend(curr_data)
                time_pop_records.extend(curr_pop_data)
                time_def += 1
            while (time_tip > time_def * time_interval and time_def < 41):
                curr_data = copy.deepcopy(curr_time_gen_records)
                curr_pop_data = copy.deepcopy(curr_time_pop_gen_records)
                for data_info in curr_data:
                    data_info[0] = time_def
                for data_info in curr_pop_data:
                    data_info[0] = time_def
                time_records.extend(curr_data)
                time_pop_records.extend(curr_pop_data)
                time_def += 1

            self.non_dominated_set.sort(key=get_cost)
            best_indi = self.non_dominated_set[0]
            min_cost = self.non_dominated_set[0]['Cost']
            max_cost = self.non_dominated_set[-1]['Cost']
            obj1 = best_indi['EvaluationRecord']['fitness_1']
            obj2 = best_indi['EvaluationRecord']['fitness_2']
            cv_num = best_indi['Violation']
            vars_rec = best_indi['Vars'][0]
            print('## current generation is:##', def_gen, '目标1最小：', obj1, obj2, cv_num[0],
                  len(self.non_dominated_set))

            f.write('## current generation is:## %s 目标1最小：%s %s %s ' % (
                def_gen, obj1, obj2, len(self.non_dominated_set)))
            f.writelines(str(cv_num.tolist()) + '\n')

            self.non_dominated_set.sort(key=get_exchanges)
            best_indi = self.non_dominated_set[0]
            min_exchanges = self.non_dominated_set[0]['Exchanges']
            max_exchanges = self.non_dominated_set[-1]['Exchanges']
            obj1 = best_indi['EvaluationRecord']['fitness_1']
            obj2 = best_indi['EvaluationRecord']['fitness_2']
            cv_num = best_indi['Violation']
            vars_rec = best_indi['Vars'][0]
            print('## current generation is:##', def_gen, '目标2最小：', obj1, obj2, cv_num[0],
                  len(self.non_dominated_set))

            f.write('## current generation is:## %s 目标2最小：%s %s %s ' % (
                def_gen, obj1, obj2, len(self.non_dominated_set)))
            f.writelines(str(cv_num.tolist()) + '\n')

            def_gen += 1


            # ===========================正式进化============================
            while time_tip <= time_max:
                epsilon = max_epsilon - (max_epsilon - min_epsilon) * time_tip / time_max

                [levels, criLevel] = self.ndSort(population.ObjV, population.sizes, None, population.CV, self.problem.maxormins)  # 对NIND个个体进行非支配分层
                cross_tag = np.where(levels == 1)[0]
                add_tag = np.where(levels == 2)[0]
                add_size = len(add_tag)

                off_size = int(NIND / 2)
                select_num = 0
                if len(cross_tag) % 2 == 1:
                    cross_size = len(cross_tag) + 1
                    remove_num = random.randint(0, add_size-1)
                    append_num = add_tag[remove_num]
                    cross_tag = np.append(cross_tag, remove_num)
                else:
                    cross_size = len(cross_tag)

                if cross_size != 0:
                    offspring_2_base = population[cross_tag]
                    select_tag = cross_tag
                    np.random.shuffle(select_tag)
                    if len(select_tag) > off_size:
                        select_tag = select_tag[:off_size]
                        cross_size = off_size
                    # off_set = []
                    offspring = ea.PsyPopulation(population.Encodings, population.Fields, cross_size)
                    offspring.initChrom()
                    offspring.ObjV = np.zeros((offspring.sizes, 2))
                    offspring.CV = np.zeros((offspring.sizes, 5))

                    while select_num < cross_size:
                        if random.random() < 1.1:
                            orig_parent_1 = copy.deepcopy(self.solution_set[select_tag[select_num]])
                            orig_parent_2 = copy.deepcopy(self.solution_set[select_tag[select_num + 1]])

                            if orig_parent_1['Cost'] < orig_parent_2['Cost']:
                                off_chrom = self.cross_operator(orig_parent_1, orig_parent_2, 'Cost', config)
                                offspring.Chroms[0][select_num] = off_chrom[0][:num_groups]
                                offspring.Chroms[1][select_num] = off_chrom[0][num_groups:]
                                select_num += 1
                            else:
                                off_chrom = self.cross_operator(orig_parent_2, orig_parent_1, 'Cost', config)
                                offspring.Chroms[0][select_num] = off_chrom[0][:num_groups]
                                offspring.Chroms[1][select_num] = off_chrom[0][num_groups:]
                                select_num += 1

                            if orig_parent_1['Exchanges'] < orig_parent_2['Exchanges']:
                                off_chrom = self.cross_operator(orig_parent_1, orig_parent_2, 'Exchanges', config)
                                offspring.Chroms[0][select_num] = off_chrom[0][:num_groups]
                                offspring.Chroms[1][select_num] = off_chrom[0][num_groups:]
                                select_num += 1
                            else:
                                off_chrom = self.cross_operator(orig_parent_2, orig_parent_1, 'Exchanges', config)
                                offspring.Chroms[0][select_num] = off_chrom[0][:num_groups]
                                offspring.Chroms[1][select_num] = off_chrom[0][num_groups:]
                                select_num += 1
                        else:
                            offspring.Chroms[0][select_num] = population.Chroms[0][select_tag[select_num]]
                            offspring.Chroms[1][select_num] = population.Chroms[1][select_tag[select_num]]
                            offspring.Chroms[0][select_num + 1] = population.Chroms[0][select_tag[select_num + 1]]
                            offspring.Chroms[1][select_num + 1] = population.Chroms[1][select_tag[select_num + 1]]
                            select_num += 2

                    for i in range(offspring.ChromNum):
                        if i == 1:
                            if random.random() < 0.5:
                                ind = np.arange(offspring.sizes)
                                ind = random.choices(ind, k=int(offspring.sizes))
                                operator_selection = random.randint(1, 2)
                                print(operator_selection)
                                if operator_selection == 1:
                                    for change_num in ind:
                                        offspring.Chroms[i][change_num] = np.zeros(num_groups)
                                elif operator_selection == 2:
                                    for change_num in ind:
                                        offspring.Chroms[i][change_num] = [max(int(a / 2), 0) for a in
                                                                           offspring.Chroms[i][change_num]]
                            else:
                                operator_selection = 0

                    offspring_size = offspring.sizes
                else:
                    offspring_size = 0

                if def_gen > 10:
                    time_mut_p = 0
                SBX_size = max(population.sizes -cross_size, off_size)
                if SBX_size != 0:
                    off1_tag = ea.selecting(self.selFunc, population.FitnV, SBX_size)
                    offspring_1 = population[off1_tag]
                    for i in range(population.ChromNum):
                        offspring_1.Chroms[i] = self.recOpers[i].do(offspring_1.Chroms[i])  # 重组
                        offspring_1.Chroms[i] = self.mutOpers[i].do(offspring_1.Encodings[i], offspring_1.Chroms[i],
                                                                    offspring_1.Fields[i])  # 变异
                    offspring_1_size = offspring_1.sizes
                else:
                    offspring_1_size = 0

                if offspring_1_size == 0:
                    total_offspring = offspring
                if offspring_size == 0:
                    total_offspring = offspring_1
                if offspring_1_size > 0 and offspring_size > 0:
                    total_offspring = offspring + offspring_1
                total_offspring = self.evaluate_solutions(problem, total_offspring, total_offspring.sizes, 'Off')

                if def_gen > 0:
                    off_set = []
                    n = len(self.non_dominated_set)
                    self.non_dominated_set.sort(key=lambda x: (x['Cost']))
                    front_part = self.non_dominated_set[:n // 3]  # 前 1/3
                    middle_part = self.non_dominated_set[n // 3: 2 * n // 3]  # 中间 1/3
                    rear_part = self.non_dominated_set[2 * n // 3:]  # 后 1/3

                    # 分别随机选择
                    search_cato = ['front', 'middle', 'rear']

                    search_num = min(n // 3, int(mut_num)//3)
                    for search_index in search_cato:
                        if search_index == 'front':
                            search_set = front_part
                            search_tag = 'search-left'
                            selected_indices = random.sample(range(len(search_set)), search_num)
                        elif search_index == 'middle':
                            search_set = middle_part
                            search_tag = 'search-middle'
                            if n == 2:
                                search_num = 1
                            selected_indices = random.sample(range(len(search_set)), search_num)
                        elif search_index == 'rear':
                            search_set = rear_part
                            search_tag = 'search-right'
                            if n == 1:
                                search_num = 1
                            selected_indices = random.sample(range(len(search_set)), search_num)

                        if len(search_set) > 0:
                            for i in selected_indices:
                                off_solution = search_set[i]
                                if random.random() < mut_rate:
                                    # 对解执行邻域搜索，优化两个目标
                                    for obj in objectives:
                                        for j in range(search_times):
                                            orig_solution = copy.deepcopy(off_solution)
                                            single_offset, chosen_tag = self.neighborhood_search(orig_solution, obj, config, Q_table, epsilon, action_list, search_tag, chosen_tag)
                                            # 添加原解
                                            single_offspring = ea.PsyPopulation(population.Encodings, population.Fields,len(single_offset))  # 实例化一个种群对象用于存储进化的后代
                                            for chrom_tag in range(single_offspring.ChromNum):
                                                single_offspring.Chroms[chrom_tag] = np.zeros((single_offspring.sizes, num_groups))
                                            single_offspring.ObjV = np.zeros((single_offspring.sizes, 2))
                                            single_offspring.CV = np.zeros((single_offspring.sizes, 5))
                                            for index in range(len(single_offset)):
                                                single_offspring.Chroms[0][index] = single_offset[index]['Vars'][:num_groups]
                                                single_offspring.Chroms[1][index] = single_offset[index]['Vars'][num_groups:]
                                                # 更新约束违反量
                                                single_offspring.CV[index, 0] = single_offset[index]['Violation'][0][0]
                                                single_offspring.CV[index, 1] = single_offset[index]['Violation'][0][1]
                                                single_offspring.CV[index, 2] = single_offset[index]['Violation'][0][2]
                                                single_offspring.CV[index, 3] = single_offset[index]['Violation'][0][3]
                                                single_offspring.CV[index, 4] = single_offset[index]['Violation'][0][4]

                                                # 更新目标函数
                                                # print(fitness_1)
                                                single_offspring.ObjV[index, 0] = single_offset[index]['EvaluationRecord']['fitness_1']  # 目标函数1
                                                single_offspring.ObjV[index, 1] = single_offset[index]['EvaluationRecord']['fitness_2']  # 目标函数2

                                            [levels, criLevel] = self.ndSort(single_offspring.ObjV, single_offspring.sizes, None, single_offspring.CV, self.problem.maxormins)  # 对NIND个个体进行非支配分层
                                            select_tag = np.where(levels > 0)[0]
                                            storage_set = []
                                            for select in select_tag:
                                                single_offset[select]['Parent_Cost'] = off_solution['Cost']
                                                single_offset[select]['Parent_Exchanges'] = off_solution['Exchanges']
                                                if off_solution['Cost'] > single_offset[select]['Cost'] or off_solution['Exchanges'] > single_offset[select]['Exchanges']:
                                                    storage_set.append(single_offset[select])

                                            storage_set.sort(key=get_cost)

                                            for storage_indi in storage_set:
                                                storage_indi['Source'] = search_tag
                                                off_set.append(storage_indi)


                    # 把搜到的个体合并到子种群中
                    new_off_solution_set = off_set
                    new_off_solution_set = remove_duplicates(new_off_solution_set)

                    if len(new_off_solution_set) != 0:
                        new_offspring = ea.PsyPopulation(population.Encodings, population.Fields, len(new_off_solution_set))  # 实例化一个种群对象用于存储进化的后代
                        new_offspring.initChrom()
                        new_offspring.ObjV = np.zeros((new_offspring.sizes, 2))
                        new_offspring.CV = np.zeros((new_offspring.sizes, 5))
                        for index in range(len(new_off_solution_set)):
                            new_offspring.Chroms[0][index] = new_off_solution_set[index]['Vars'][:num_groups]
                            new_offspring.Chroms[1][index] = new_off_solution_set[index]['Vars'][num_groups:]
                            # 更新约束违反量
                            new_offspring.CV[index, 0] = new_off_solution_set[index]['Violation'][0][0]
                            new_offspring.CV[index, 1] = new_off_solution_set[index]['Violation'][0][1]
                            new_offspring.CV[index, 2] = new_off_solution_set[index]['Violation'][0][2]
                            new_offspring.CV[index, 3] = new_off_solution_set[index]['Violation'][0][3]
                            new_offspring.CV[index, 4] = new_off_solution_set[index]['Violation'][0][4]

                            # 更新目标函数
                            # print(fitness_1)
                            new_offspring.ObjV[index, 0] = new_off_solution_set[index]['EvaluationRecord']['fitness_1']  # 目标函数1
                            new_offspring.ObjV[index, 1] = new_off_solution_set[index]['EvaluationRecord']['fitness_2']  # 目标函数2

                        population = population + new_offspring
                        self.solution_set.extend(new_off_solution_set)

                population = population + total_offspring

                # 记录交叉子代的来源为cross
                cross_count = 0
                for off_solution in self.off_solution_set:
                    if cross_count < cross_size:
                        off_solution['Source'] = 'cross'
                        cross_count += 1
                    else:
                        off_solution['Source'] = 'SBX'
                        cross_count += 1

                self.solution_set.extend(self.off_solution_set)

                # 更新种群及外部存档
                [levels, criLevel] = self.ndSort(population.ObjV, population.sizes, None, population.CV, self.problem.maxormins)  # 对NIND个个体进行非支配分层
                dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离

                # 增加重复个体删除模块
                duplication_set = []
                for pop_indi in range(population.sizes):
                    if dis[pop_indi] == 0:
                        check_set = (population.ObjV[pop_indi, 0], population.ObjV[pop_indi, 1], levels[pop_indi])
                        if check_set in duplication_set:
                            levels[pop_indi] = np.max(levels)
                        else:
                            duplication_set.append(check_set)

                population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
                chooseFlag = ea.selecting('dup', population.FitnV, NIND)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体

                non_dominated_tag = np.where(levels == 1)[0]

                NDSet = population[non_dominated_tag]
                non_dominated_solution_set_1 = []
                record_index = 0
                for tag_indi in non_dominated_tag:
                    self.solution_set[tag_indi]['Solution_index'] = tag_indi
                    non_dominated_solution_set_1.append(self.solution_set[tag_indi])
                non_dominated_solution_set_2 = []
                if NDSet.CV is not None:  # CV不为None说明有设置约束条件
                    constraints_tag = np.where(np.all(NDSet.CV <= 0, 1))[0]
                    select_NDSet = NDSet[constraints_tag]  # 最后要彻底排除非可行解
                    for tag_indi in constraints_tag:
                        non_dominated_solution_set_2.append(non_dominated_solution_set_1[tag_indi])

                non_dominated_solution_set_2 = remove_duplicates(non_dominated_solution_set_2)
                self.non_dominated_set = non_dominated_solution_set_2

                # 对非支配解排序，获取左中右个体
                self.non_dominated_set.sort(key=get_exchanges)
                curr_min_exchanges = self.non_dominated_set[0]['Exchanges']
                curr_max_exchanges = self.non_dominated_set[-1]['Exchanges']
                if curr_min_exchanges < min_exchanges:
                    min_exchanges = curr_min_exchanges
                if curr_max_exchanges > max_exchanges:
                    max_exchanges = curr_max_exchanges
                best_indi = self.non_dominated_set[0]
                obj21 = best_indi['EvaluationRecord']['fitness_1']
                obj22 = best_indi['EvaluationRecord']['fitness_2']
                cv_num2 = best_indi['Violation']
                vars_rec = best_indi['Vars'][0]

                self.non_dominated_set.sort(key=get_cost)
                curr_min_cost = self.non_dominated_set[0]['Cost']
                curr_max_cost = self.non_dominated_set[-1]['Cost']
                if curr_min_cost < min_cost:
                    min_cost = curr_min_cost
                if curr_max_cost > max_cost:
                    max_cost = curr_max_cost
                best_indi = self.non_dominated_set[0]
                obj1 = best_indi['EvaluationRecord']['fitness_1']
                obj2 = best_indi['EvaluationRecord']['fitness_2']
                cv_num = best_indi['Violation']
                vars_rec = best_indi['Vars'][0]
                print(epsilon)
                print('## current generation is:##', def_gen, '目标1最小：', obj1, obj2, cv_num[0],
                      len(self.non_dominated_set))

                f.write('## current generation is:## %s 目标1最小：%s %s %s ' % (
                    def_gen, obj1, obj2, len(self.non_dominated_set)))
                f.writelines(str(cv_num.tolist()) + '\n')

                print('## current generation is:##', def_gen, '目标2最小：', obj21, obj22, cv_num2[0],
                      len(self.non_dominated_set))

                f.write('## current generation is:## %s 目标2最小：%s %s %s ' % (
                    def_gen, obj21, obj22, len(self.non_dominated_set)))
                f.writelines(str(cv_num2.tolist()) + '\n')


                # 记录选入下一代种群的个体的状态
                for solution_index in chooseFlag:
                    # 选中的个体是前沿解
                    if levels[solution_index] == 1:
                        self.solution_set[solution_index]['Non_Dominated_State'] = 1
                    else:
                        self.solution_set[solution_index]['Non_Dominated_State'] = 0
                        self.solution_set[solution_index]['State'] = 4

                        if self.solution_set[solution_index]['Source'] in search_source_list:
                            # 判断解的来源
                            curr_state = 4
                            source_action = self.solution_set[solution_index]['Action']
                            # 查看解的来源，若为search，更新Q表
                            if self.solution_set[solution_index]['Source'] == 'search-left':
                                source_state = 1
                            elif self.solution_set[solution_index]['Source'] == 'search-middle':
                                source_state = 2
                            elif self.solution_set[solution_index]['Source'] == 'search-right':
                                source_state = 3

                            # 更新Q表
                            reward = 0
                            parent_cost = self.solution_set[solution_index]['Parent_Cost']
                            parent_exchanges = self.solution_set[solution_index]['Parent_Exchanges']
                            off_cost = self.solution_set[solution_index]['Cost']
                            off_exchanges = self.solution_set[solution_index]['Exchanges']
                            if max_cost - min_cost != 0:
                                reward += (max_cost - off_cost) / (max_cost - min_cost)
                            if max_exchanges - min_exchanges != 0:
                                reward += (max_exchanges - off_exchanges) / (max_exchanges - min_exchanges)

                            times_check = 0
                            if parent_cost - off_cost > 0:
                                times_check += 1
                            if parent_exchanges - off_exchanges > 0:
                                times_check += 1
                            if times_check == 2:
                                reward += 2
                            elif times_check == 1:
                                reward += 1
                            else:
                                reward = 0
                            Q_table[source_state][source_action] = (1 - learning_rate) * Q_table[source_state][source_action] + learning_rate * (reward + discount_factor * np.max(Q_table[curr_state]))
                            print(reward, max_cost - min_cost, parent_cost - off_cost, max_exchanges - min_exchanges, parent_exchanges - off_exchanges)

                n = len(self.non_dominated_set)
                front_part = self.non_dominated_set[:n // 3]  # 前 1/3
                middle_part = self.non_dominated_set[n // 3: 2 * n // 3]  # 中间 1/3
                rear_part = self.non_dominated_set[2 * n // 3:]  # 后 1/3
                for solution in front_part:
                    solution_index = solution['Solution_index']
                    self.solution_set[solution_index]['State'] = 1

                    if self.solution_set[solution_index]['Source'] in search_source_list:
                        curr_state = 1
                        source_action = self.solution_set[solution_index]['Action']
                        if self.solution_set[solution_index]['Source'] == 'search-left':
                            source_state = 1
                            print('Front, left')
                            f.write('Front, left' + '\n')
                        elif self.solution_set[solution_index]['Source'] == 'search-middle':
                            source_state = 2
                            print('Front, middle')
                            f.write('Front, middle' + '\n')
                        elif self.solution_set[solution_index]['Source'] == 'search-right':
                            source_state = 3
                            print('Front, right')
                            f.write('Front, right' + '\n')

                        # 更新Q表
                        reward = 0
                        parent_cost = self.solution_set[solution_index]['Parent_Cost']
                        parent_exchanges = self.solution_set[solution_index]['Parent_Exchanges']
                        off_cost = self.solution_set[solution_index]['Cost']
                        off_exchanges = self.solution_set[solution_index]['Exchanges']

                        if max_cost - min_cost != 0:
                            reward += (max_cost - off_cost) / (max_cost - min_cost)
                        if max_exchanges - min_exchanges != 0:
                            reward += (max_exchanges - off_exchanges) / (max_exchanges - min_exchanges)

                        times_check = 0
                        if parent_cost - off_cost > 0:
                            times_check += 1
                        if parent_exchanges - off_exchanges > 0:
                            times_check += 1
                        if times_check == 2:
                            reward += 2
                        elif times_check == 1:
                            reward += 1
                        else:
                            reward = 0

                        Q_table[source_state][source_action] = (1-learning_rate) * Q_table[source_state][source_action] + learning_rate * (reward + discount_factor * np.max(Q_table[curr_state]))
                        print(reward)

                for solution in middle_part:
                    solution_index = solution['Solution_index']
                    self.solution_set[solution_index]['State'] = 2

                    if self.solution_set[solution_index]['Source'] in search_source_list:
                        curr_state = 2
                        source_action = self.solution_set[solution_index]['Action']
                        if self.solution_set[solution_index]['Source'] == 'search-left':
                            source_state = 1
                            print('Middle, left', )
                            f.write('Middle, left' + '\n')
                        elif self.solution_set[solution_index]['Source'] == 'search-middle':
                            source_state = 2
                            print('Middle, middle')
                            f.write('Middle, middle' + '\n')
                        elif self.solution_set[solution_index]['Source'] == 'search-right':
                            source_state = 3
                            print('Middle, right')
                            f.write('Middle, right' + '\n')

                        # 更新Q表
                        reward = 0
                        parent_cost = self.solution_set[solution_index]['Parent_Cost']
                        parent_exchanges = self.solution_set[solution_index]['Parent_Exchanges']
                        off_cost = self.solution_set[solution_index]['Cost']
                        off_exchanges = self.solution_set[solution_index]['Exchanges']

                        if max_cost - min_cost != 0:
                            reward += (max_cost - off_cost) / (max_cost - min_cost)
                        if max_exchanges - min_exchanges != 0:
                            reward += (max_exchanges - off_exchanges) / (max_exchanges - min_exchanges)

                        times_check = 0
                        if parent_cost - off_cost > 0:
                            times_check += 1
                        if parent_exchanges - off_exchanges > 0:
                            times_check += 1
                        if times_check == 2:
                            reward += 2
                        elif times_check == 1:
                            reward += 1
                        else:
                            reward = 0

                        Q_table[source_state][source_action] = (1 - learning_rate) * Q_table[source_state][source_action] + learning_rate * (reward + discount_factor * np.max(Q_table[curr_state]))
                        print(reward)

                for solution in rear_part:
                    # print('Rear')
                    solution_index = solution['Solution_index']
                    self.solution_set[solution_index]['State'] = 3

                    if self.solution_set[solution_index]['Source'] in search_source_list:
                        curr_state = 3
                        source_action = self.solution_set[solution_index]['Action']
                        if self.solution_set[solution_index]['Source'] == 'search-left':
                            source_state = 1
                            print('Rear, left')
                            f.write('Rear, left' + '\n')
                        elif self.solution_set[solution_index]['Source'] == 'search-middle':
                            source_state = 2
                            print('Rear, middle')
                            f.write('Rear, middle' + '\n')
                        elif self.solution_set[solution_index]['Source'] == 'search-right':
                            source_state = 3
                            print('Rear, right')
                            f.write('Rear, right' + '\n')

                        # 更新Q表
                        reward = 0
                        parent_cost = self.solution_set[solution_index]['Parent_Cost']
                        parent_exchanges = self.solution_set[solution_index]['Parent_Exchanges']
                        off_cost = self.solution_set[solution_index]['Cost']
                        off_exchanges = self.solution_set[solution_index]['Exchanges']

                        if max_cost - min_cost != 0:
                            reward += (max_cost - off_cost) / (max_cost - min_cost)
                        if max_exchanges - min_exchanges != 0:
                            reward += (max_exchanges - off_exchanges) / (max_exchanges - min_exchanges)

                        times_check = 0
                        if parent_cost - off_cost > 0:
                            times_check += 1
                        if parent_exchanges - off_exchanges > 0:
                            times_check += 1
                        if times_check == 2:
                            reward += 2
                        elif times_check == 1:
                            reward += 1
                        else:
                            reward = 0

                        Q_table[source_state][source_action] = (1 - learning_rate) * Q_table[source_state][source_action] + learning_rate * (reward + discount_factor * np.max(Q_table[curr_state]))
                        print(reward)

                chooseFlag.sort()
                population = population[chooseFlag]
                solution_set = []
                for index_num in chooseFlag:
                    solution_set.append(self.solution_set[index_num])
                self.solution_set = solution_set

                # 记录进化结果
                time_tip = time.time() - self.timeSlot


                # 记录当前Q表状态
                record_action = []
                for action in range(action_num):
                    record_Q = []
                    for state in range(1, state_num + 1):
                        record_Q.append(Q_table[state][action])
                    record_action.append(record_Q)

                record_action = tuple(record_action)
                Q_records.append(record_action)

                curr_time_gen_records = []
                non_domi_component = [0, 0, 0, 0]
                for solution in self.non_dominated_set:
                    obj1 = solution['EvaluationRecord']['fitness_1']
                    obj2 = solution['EvaluationRecord']['fitness_2']
                    cv_num = solution['Violation']
                    vars_rec = solution['Vars']
                    if solution['Source'] == 'cross':
                        non_domi_component[0] += 1
                    elif solution['Source'] == 'SBX':
                        non_domi_component[1] += 1
                    elif solution['Source'] in search_source_list:
                        non_domi_component[2] += 1
                    elif solution['Source'] == 'Parent':
                        non_domi_component[3] += 1

                    total_flight_num = solution['EvaluationRecord']['total_flight_num']
                    cancel_flight_num = solution['EvaluationRecord']['cancel_flight_num']
                    delay_flight_num = solution['EvaluationRecord']['delay_flight_num']
                    cancel_passenger_num = solution['EvaluationRecord']['cancel_passenger_num']
                    delay_passenger_num = solution['EvaluationRecord']['delay_passenger_num']
                    delay_flight_time = solution['EvaluationRecord']['delay_flight_time']
                    delay_passenger_time = solution['EvaluationRecord']['delay_passenger_time']
                    cancel_cost_sum = solution['EvaluationRecord']['cancel_cost_sum']
                    delay_cost_sum = solution['EvaluationRecord']['delay_cost_sum']
                    aircraft_exchanges = solution['EvaluationRecord']['aircraft_exchanges']
                    cost1_1_total = solution['EvaluationRecord']['cost1_1_total']
                    cost1_1 = solution['EvaluationRecord']['cost1_1']
                    cost1_2 = solution['EvaluationRecord']['cost1_2']
                    cost1_3 = solution['EvaluationRecord']['cost1_3']

                    data = (
                        def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num, delay_flight_num,
                        cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                        cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2, cost1_3,
                        vars_rec)
                    records.append(data)

                    if time_tip > time_def * time_interval:
                        data = [
                            time_def, def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num,
                            delay_flight_num,
                            cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                            cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2,
                            cost1_3,
                            vars_rec]
                        curr_time_gen_records.append(data)

                print('非支配解集的来源统计：', non_domi_component, operator_selection)
                f.write('非支配解集的来源统计：' + str(non_domi_component) + ',' + str(operator_selection) + '\n')
                # 记录当前种群
                curr_time_pop_gen_records = []
                population_component = [0, 0, 0, 0]
                for solution in self.solution_set:
                    if solution['Source'] == 'cross':
                        population_component[0] += 1
                    elif solution['Source'] == 'SBX':
                        population_component[1] += 1
                    elif solution['Source'] in search_source_list:
                        population_component[2] += 1
                    elif solution['Source'] == 'Parent':
                        population_component[3] += 1
                    solution['Source'] = 'Parent'
                    obj1 = solution['EvaluationRecord']['fitness_1']
                    obj2 = solution['EvaluationRecord']['fitness_2']
                    cv_num = solution['Violation']
                    vars_rec = solution['Vars']


                    total_flight_num = solution['EvaluationRecord']['total_flight_num']
                    cancel_flight_num = solution['EvaluationRecord']['cancel_flight_num']
                    delay_flight_num = solution['EvaluationRecord']['delay_flight_num']
                    cancel_passenger_num = solution['EvaluationRecord']['cancel_passenger_num']
                    delay_passenger_num = solution['EvaluationRecord']['delay_passenger_num']
                    delay_flight_time = solution['EvaluationRecord']['delay_flight_time']
                    delay_passenger_time = solution['EvaluationRecord']['delay_passenger_time']
                    cancel_cost_sum = solution['EvaluationRecord']['cancel_cost_sum']
                    delay_cost_sum = solution['EvaluationRecord']['delay_cost_sum']
                    aircraft_exchanges = solution['EvaluationRecord']['aircraft_exchanges']
                    cost1_1_total = solution['EvaluationRecord']['cost1_1_total']
                    cost1_1 = solution['EvaluationRecord']['cost1_1']
                    cost1_2 = solution['EvaluationRecord']['cost1_2']
                    cost1_3 = solution['EvaluationRecord']['cost1_3']

                    pop_data = (
                        def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num, delay_flight_num,
                        cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                        cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2, cost1_3,
                        vars_rec)
                    pop_records.append(pop_data)

                    if time_tip > time_def * time_interval:
                        data = [
                            time_def, def_gen, time_tip, obj1, obj2, cv_num, total_flight_num, cancel_flight_num,
                            delay_flight_num,
                            cancel_passenger_num, delay_passenger_num, delay_flight_time, delay_passenger_time,
                            cancel_cost_sum, delay_cost_sum, aircraft_exchanges, cost1_1_total, cost1_1, cost1_2,
                            cost1_3,
                            vars_rec]
                        curr_time_pop_gen_records.append(data)

                if time_tip > time_def * time_interval:
                    curr_data = copy.deepcopy(curr_time_gen_records)
                    curr_pop_data = copy.deepcopy(curr_time_pop_gen_records)
                    time_records.extend(curr_data)
                    time_pop_records.extend(curr_pop_data)
                    time_def += 1
                while (time_tip > time_def * time_interval and time_def < 41):
                    curr_data = copy.deepcopy(curr_time_gen_records)
                    curr_pop_data = copy.deepcopy(curr_time_pop_gen_records)
                    for data_info in curr_data:
                        data_info[0] = time_def
                    for data_info in curr_pop_data:
                        data_info[0] = time_def
                    time_records.extend(curr_data)
                    time_pop_records.extend(curr_pop_data)
                    time_def += 1

                print('当前种群的来源统计：', population_component, operator_selection)
                f.write('当前种群的来源统计：' + str(population_component) + ',' + str(operator_selection) + '\n')

                def_gen += 1

        # 创建一个包含数据的DataFrame
        df = pd.DataFrame(records, columns=['Generation', 'Time', 'Obj1', 'Obj2', 'CV', 'Total_Flight', 'Cancel_Flight',
                                            'Delay_Flight', 'Cancel_Passenger', 'Delay_Passenger', 'Delay_FlightTime',
                                            'Delay_PassengerTime', 'Cancel_Cost', 'Delay_Cost', 'Aircraft_Exchanges',
                                            'Total_Operating_Cost', 'Operating_Cost', 'Flight_Recovery_Cost',
                                            'Aircraft_Position_Cost', 'Phen'])

        # 将DataFrame输出到Excel文件
        df.to_csv(_output_path + output_time + output_file_name, index=False)
        fcopy(_output_path + output_time + output_file_name, _datapath + output_file_name)

        # 创建一个包含数据的DataFrame
        time_df = pd.DataFrame(time_records,
                               columns=['Generation', 'Actural_Gene', 'Time', 'Obj1', 'Obj2', 'CV', 'Total_Flight',
                                        'Cancel_Flight',
                                        'Delay_Flight', 'Cancel_Passenger', 'Delay_Passenger', 'Delay_FlightTime',
                                        'Delay_PassengerTime', 'Cancel_Cost', 'Delay_Cost', 'Aircraft_Exchanges',
                                        'Total_Operating_Cost', 'Operating_Cost', 'Flight_Recovery_Cost',
                                        'Aircraft_Position_Cost', 'Phen'])

        # 将DataFrame输出到Excel文件
        time_df.to_csv(_output_path + output_time + '_Time_Output' + '.csv', index=False)
        fcopy(_output_path + output_time + '_Time_Output' + '.csv', _datapath + '_Time_Output' + '.csv')

        Q_df = pd.DataFrame(Q_records,
                            columns=['Action1', 'Action2', 'Action3', 'Action4', 'Action5'])

        # 将DataFrame输出到Excel文件
        Q_df.to_csv(_output_path + output_time + '_Q_Output' + '.csv', index=False)
        fcopy(_output_path + output_time + '_Q_Output' + '.csv', _datapath + '_Q_Output' + '.csv')


        # 创建一个包含数据的DataFrame
        pop_df = pd.DataFrame(pop_records, columns=['Generation', 'Time', 'Obj1', 'Obj2', 'CV', 'Total_Flight', 'Cancel_Flight',
                                                    'Delay_Flight', 'Cancel_Passenger', 'Delay_Passenger', 'Delay_FlightTime',
                                                    'Delay_PassengerTime', 'Cancel_Cost', 'Delay_Cost', 'Aircraft_Exchanges',
                                                    'Total_Operating_Cost', 'Operating_Cost', 'Flight_Recovery_Cost',
                                                    'Aircraft_Position_Cost', 'Phen'])
        # 将DataFrame输出到Excel文件
        pop_df.to_csv(_output_path + output_time + '_Pop_Output' + '.csv', index=False)
        fcopy(_output_path + output_time + '_Pop_Output' + '.csv', _datapath + '_Pop_Output' + '.csv')

        # 创建一个包含数据的DataFrame
        time_pop_df = pd.DataFrame(time_pop_records,
                                   columns=['Generation', 'Actural_Gene', 'Time', 'Obj1', 'Obj2', 'CV', 'Total_Flight',
                                            'Cancel_Flight',
                                            'Delay_Flight', 'Cancel_Passenger', 'Delay_Passenger', 'Delay_FlightTime',
                                            'Delay_PassengerTime', 'Cancel_Cost', 'Delay_Cost', 'Aircraft_Exchanges',
                                            'Total_Operating_Cost', 'Operating_Cost', 'Flight_Recovery_Cost',
                                            'Aircraft_Position_Cost', 'Phen'])

        # 将DataFrame输出到Excel文件
        time_pop_df.to_csv(_output_path + output_time + '_Time_Pop_Output' + '.csv', index=False)
        fcopy(_output_path + output_time + '_Time_Pop_Output' + '.csv', _datapath + '_Time_Pop_Output' + '.csv')

        print(Q_table)

        self.passTime += time.time() - self.timeSlot  # 更新用时记录

        return self.non_dominated_set, self.solution_set



