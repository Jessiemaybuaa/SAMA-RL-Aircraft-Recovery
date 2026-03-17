# -*- coding: utf-8 -*-

import geatpy as ea
import numpy as np
import time
import copy
import json
import os
import random
from datetime import datetime, timedelta
from Operators import load_dict_from_pickle
from Operators import get_max_delay
from Operators import flight_circle_to_schedule
from Operators import flight_circle_to_chorm
from Operators import evaluation
from Operators import get_airports_capacity
from Operators import get_group_id
from SAMA_RL import SAMA_RL_templet


def get_exchanges(element):
    return element['Exchanges']

class FlightRecoveryProblem(ea.Problem):
    def __init__(self, test_tag, time_max, chorm_num, aircraft_mutation, canceled_flights_swap, initialize_tag, random_init,
                 RVEA_init, sort_str, chroms_change, cancel_decision, cancel_hour_dc, cancel_hour_i, config_dic,
                 flight_circles_dic, aircraft_dic, passenger_orders, alt_flights_info, alt_airports_info, alt_aircraft_info,
                 position_info, aircraft_category, airports_info, airports_capacity_total, flights_info, same_space_flights_info,
                 flight_related_passengers, flight_ring_index, started_rotations_info, group_lookup_dic):
        name = 'FlightScheduleRecovery'
        M = 2  # 目标维数，包括航司恢复成本、场面运行复杂度
        maxormins = [1, 1]  # 最小化所有目标
        Dim = len(flight_circles_dic) * 2

        # 变量类型
        varTypes = [1] * Dim  # 离散型决策变量

        # 决策变量的上下界
        if cancel_decision == 0:
            lb = [0] * len(flight_circles_dic) + [0] * len(flight_circles_dic)
        elif cancel_decision == 1:
            lb = [0] * len(flight_circles_dic) + [-1] * len(flight_circles_dic)

        ub = get_max_delay(aircraft_dic, flight_circles_dic, cancel_hour_dc, cancel_hour_i, cancel_decision)
        lbinitial = [1] * Dim
        ubinitial = [1] * Dim

        recovery_start_time = datetime.strptime(config_dic['Recovery_StartDate'] + ' ' + config_dic['Recovery_StartTime'], '%d/%m/%y %H:%M')
        recovery_end_time = datetime.strptime(config_dic['Recovery_EndDate'] + ' ' + config_dic['Recovery_EndTime'], '%d/%m/%y %H:%M')

        self.test_tag = test_tag
        self.time_max = time_max
        self.recovery_start_time = recovery_start_time
        self.recovery_end_time = recovery_end_time
        self.flight_circles_info = flight_circles_dic
        self.aircraft_info = aircraft_dic
        self.alt_flights_info = alt_flights_info
        self.alt_airports_info = alt_airports_info
        self.alt_aircraft_info = alt_aircraft_info
        self.init_schedule = flight_circle_to_schedule(self.flight_circles_info)
        self.init_passenger_orders = passenger_orders
        self.position_info = position_info
        self.aircraft_category = aircraft_category
        self.airports_info = airports_info
        self.airports_capacity_total = airports_capacity_total
        self.flights_info = flights_info
        self.same_space_flights_info = same_space_flights_info
        self.flight_related_passengers = flight_related_passengers
        self.flight_ring_index = flight_ring_index
        self.started_rotations = started_rotations_info
        self.group_lookup_dic = group_lookup_dic
        self.sort_str = sort_str
        self.cancel_decision = cancel_decision
        self.cancel_hour_dc = cancel_hour_dc
        self.cancel_hour_i = cancel_hour_i
        self.max_delay = ub
        self.chroms_change = chroms_change
        self.config_info = config_dic
        self.chorm_size = chorm_num
        self.aircraft_assignment_mutation = aircraft_mutation
        self.canceled_flights_count = canceled_flights_swap
        self.initialize = initialize_tag
        self.random_init = random_init
        self.RVEA_init = RVEA_init
        self.chroms_change = chroms_change
        
        self.flag = 0
        self.pop_record = 0

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbinitial, ubinitial)


    def aimFunc(self, pop):
        num_groups = len(self.flight_circles_info)
        recovery_start_time = self.recovery_start_time
        recovery_end_time = self.recovery_end_time
        position_info = copy.deepcopy(self.position_info)
        aircraft_info = self.aircraft_info
        aircraft_category = self.aircraft_category
        alt_flights_info = self.alt_flights_info
        alt_airports_info = self.alt_airports_info
        alt_aircraft_info = self.alt_aircraft_info
        airports_info = copy.deepcopy(self.airports_info)
        airports_capacity_total = copy.deepcopy(self.airports_capacity_total)
        flights_info = copy.deepcopy(self.flights_info)
        group_lookup_dic = copy.deepcopy(self.group_lookup_dic)
        flight_circles_info = copy.deepcopy(self.flight_circles_info)
        canceled_flights_swap = self.canceled_flights_count
        chorm_size_num = self.chorm_size
        initialize_index = self.initialize
        RVEA_init = self.RVEA_init
        chroms_change = self.chroms_change

        sort_str = self.sort_str
        cancel_decision = self.cancel_decision
        cancel_hour_dc = self.cancel_hour_dc
        cancel_hour_i = self.cancel_hour_i
        max_delay = self.max_delay
        config = self.config_info

        init_flight_circles_info = copy.deepcopy(flight_circles_info)
        init_airports_info = copy.deepcopy(airports_info)

        pop.ObjV = np.zeros((pop.sizes, 2))
        pop.CV = np.zeros((pop.sizes, 5))

        init_schedule_vars = flight_circle_to_chorm(num_groups, num_groups, flight_circles_info, aircraft_info, cancel_hour_dc, cancel_hour_i, cancel_decision)
        init_aircraft_assignment = init_schedule_vars[0, :num_groups]

        total_flight_num_list = []
        cancel_flight_num_list = []
        delay_flight_num_list = []
        cancel_passenger_num_list = []
        delay_passenger_num_list = []
        delay_flight_time_list = []
        delay_passenger_time_list = []
        cancel_cost_sum_list = []
        delay_cost_sum_list = []
        aircraft_exchanges_list = []
        fitness_1_list = []
        fitness_2_list = []
        cost1_1_total_list = []
        cost1_1_list = []
        cost1_2_list = []
        cost1_3_list = []
        init_cancel_flight_num = self.init_cancel_flight_num

        for i in range(pop.sizes):
            Chroms = pop.Chroms
            Vars = np.append(pop.Chroms[0][i], pop.Chroms[1][i])  # 获取个体的表现型矩阵
            flight_circles = copy.deepcopy(self.flight_circles_info)

            evaluate_state = 0
            if pop.sizes != 1:  # 第一次是pop，之后是offspring
                # if self.pop_record == 0 and i == 0:
                solution = evaluation(Vars, config, recovery_start_time, recovery_end_time, num_groups,
                                      alt_airports_info, alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str,
                                      init_flight_circles_info, init_aircraft_assignment, aircraft_info, airports_info,
                                      airports_capacity_total, flights_info, group_lookup_dic, position_info, aircraft_category,
                                      cancel_decision, canceled_flights_swap, max_delay, init_cancel_flight_num, 'repair')
                evaluate_state = 1

            if evaluate_state == 0:
                solution = evaluation(Vars, config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                                      alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                                      init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total,
                                      flights_info, group_lookup_dic, position_info, aircraft_category, cancel_decision,
                                      canceled_flights_swap, max_delay, init_cancel_flight_num, 'repair')

            Chroms[0][i, :] = solution['Vars'][:num_groups]
            Chroms[1][i, :] = solution['Vars'][num_groups:]

            total_flight_num_list.append(solution['EvaluationRecord']['total_flight_num'])
            cancel_flight_num_list.append(solution['EvaluationRecord']['cancel_flight_num'])
            delay_flight_num_list.append(solution['EvaluationRecord']['delay_flight_num'])
            cancel_passenger_num_list.append(solution['EvaluationRecord']['cancel_passenger_num'])
            delay_passenger_num_list.append(solution['EvaluationRecord']['delay_passenger_num'])
            delay_flight_time_list.append(solution['EvaluationRecord']['delay_flight_time'])
            delay_passenger_time_list.append(solution['EvaluationRecord']['delay_passenger_time'])
            cancel_cost_sum_list.append(solution['EvaluationRecord']['cancel_cost_sum'])
            delay_cost_sum_list.append(solution['EvaluationRecord']['delay_cost_sum'])
            aircraft_exchanges_list.append(solution['EvaluationRecord']['aircraft_exchanges'])
            fitness_1_list.append(solution['EvaluationRecord']['fitness_1'])
            fitness_2_list.append(solution['EvaluationRecord']['fitness_2'])
            cost1_1_total_list.append(solution['EvaluationRecord']['cost1_1_total'])
            cost1_1_list.append(solution['EvaluationRecord']['cost1_1'])
            cost1_2_list.append(solution['EvaluationRecord']['cost1_2'])
            cost1_3_list.append(solution['EvaluationRecord']['cost1_3'])

            # 更新约束违反量
            pop.CV[i, 0] = solution['Violation'][0][0]
            pop.CV[i, 1] = solution['Violation'][0][1]
            pop.CV[i, 2] = solution['Violation'][0][2]
            pop.CV[i, 3] = solution['Violation'][0][3]
            pop.CV[i, 4] = solution['Violation'][0][4]


            # 更新目标函数
            pop.ObjV[i, 0] = solution['EvaluationRecord']['fitness_1'] # 目标函数1
            pop.ObjV[i, 1] = solution['EvaluationRecord']['fitness_2']  # 目标函数2

        if pop.sizes != 1: # 第一次是pop，之后是offspring
            if self.pop_record == 0:
                self.total_flight_num_list = total_flight_num_list
                self.cancel_flight_num_list = cancel_flight_num_list
                self.delay_flight_num_list = delay_flight_num_list
                self.cancel_passenger_num_list = cancel_passenger_num_list
                self.delay_passenger_num_list = delay_passenger_num_list
                self.delay_flight_time_list = delay_flight_time_list
                self.delay_passenger_time_list = delay_passenger_time_list
                self.cancel_cost_sum_list = cancel_cost_sum_list
                self.delay_cost_sum_list = delay_cost_sum_list
                self.aircraft_exchanges_list = aircraft_exchanges_list
                self.fitness_1_list = fitness_1_list
                self.fitness_2_list = fitness_2_list
                self.cost1_1_total_list = cost1_1_total_list
                self.cost1_1_list = cost1_1_list
                self.cost1_2_list = cost1_2_list
                self.cost1_3_list = cost1_3_list
                self.pop_record += 1
            else:
                self.off_total_flight_num_list = total_flight_num_list
                self.off_cancel_flight_num_list = cancel_flight_num_list
                self.off_delay_flight_num_list = delay_flight_num_list
                self.off_cancel_passenger_num_list = cancel_passenger_num_list
                self.off_delay_passenger_num_list = delay_passenger_num_list
                self.off_delay_flight_time_list = delay_flight_time_list
                self.off_delay_passenger_time_list = delay_passenger_time_list
                self.off_cancel_cost_sum_list = cancel_cost_sum_list
                self.off_delay_cost_sum_list = delay_cost_sum_list
                self.off_aircraft_exchanges_list = aircraft_exchanges_list
                self.off_fitness_1_list = fitness_1_list
                self.off_fitness_2_list = fitness_2_list
                self.off_cost1_1_total_list = cost1_1_total_list
                self.off_cost1_1_list = cost1_1_list
                self.off_cost1_2_list = cost1_2_list
                self.off_cost1_3_list = cost1_3_list
        else:
            self.off_total_flight_num_list = total_flight_num_list
            self.off_cancel_flight_num_list = cancel_flight_num_list
            self.off_delay_flight_num_list = delay_flight_num_list
            self.off_cancel_passenger_num_list = cancel_passenger_num_list
            self.off_delay_passenger_num_list = delay_passenger_num_list
            self.off_delay_flight_time_list = delay_flight_time_list
            self.off_delay_passenger_time_list = delay_passenger_time_list
            self.off_cancel_cost_sum_list = cancel_cost_sum_list
            self.off_delay_cost_sum_list = delay_cost_sum_list
            self.off_aircraft_exchanges_list = aircraft_exchanges_list
            self.off_fitness_1_list = fitness_1_list
            self.off_fitness_2_list = fitness_2_list
            self.off_cost1_1_total_list = cost1_1_total_list
            self.off_cost1_1_list = cost1_1_list
            self.off_cost1_2_list = cost1_2_list
            self.off_cost1_3_list = cost1_3_list

        if chroms_change == 1:
            pop.Chroms = Chroms


def run_optimization(test_tag, time_max, init, max_generation, size, repeat, aircraft_mutation, canceled_flights_swap, initialize_index, random_tag,
                     RVEA_init, sort_str, chroms_change, algorithm, cancel_decision, cancel_hour_dc, cancel_hour_i, config,
                     processed_flights_info, loaded_aircraft_info, passenger_orders, alt_flights_info, alt_airports_info,
                     alt_aircraft_info, position_info, aircraft_category, airports_info, airports_capacity_total, flights_info, same_space_flights_info,
                     flight_related_passengers, flight_ring_dic, started_rotations_info, _path, _datapath, _output_path_init,
                     clocktime, group_lookup_dic):
        
    problem = FlightRecoveryProblem (test_tag, time_max, size, aircraft_mutation, canceled_flights_swap, initialize_index, random_tag,
                                     RVEA_init, sort_str, chroms_change, cancel_decision, cancel_hour_dc, cancel_hour_i, config,
                                     processed_flights_info, loaded_aircraft_info, passenger_orders, alt_flights_info,
                                     alt_airports_info, alt_aircraft_info, position_info, aircraft_category, airports_info,
                                     airports_capacity_total, flights_info, same_space_flights_info, flight_related_passengers,
                                     flight_ring_dic, started_rotations_info, group_lookup_dic)

    # 定义种群和算法
    Encodings = ['RI', 'RI']  # 多染色体编码：飞机分配染色体 + 航班调整染色体
    len_field1 = len(processed_flights_info)
    len_field2 = len(processed_flights_info)
    Field1 = ea.crtfld(Encodings[0], problem.varTypes[:len_field1], problem.ranges[:,:len_field1], problem.borders[:,:len_field1])
    Field2 = ea.crtfld(Encodings[1], problem.varTypes[len_field1:], problem.ranges[:,len_field1:], problem.borders[:,len_field1:])
    Fields = [Field1, Field2]
    population = ea.PsyPopulation(Encodings, Fields, size)

    if init == 0:
        output_file_path = ('Chorm_' + str(size) + '_MaxGen_' + str(max_generation) + '_Init_' + str(initialize_index)
                            + '_Sort_' + sort_str + '_Test_' + str(repeat + 1) + '/' + 'Algorithm_' + algorithm + '/' )
        _output_path = _output_path_init + output_file_path
        output_time = clocktime
        output_file_name = '_Output' + '.csv'
        if not os.path.exists(_output_path):
            os.makedirs(_output_path)
    if init == 1:
        output_file_path = ('Chorm_' + str(size) + '_MaxGen_' + str(max_generation)
                            + '_Init_' + str(initialize_index) + '_Sort_' + sort_str + '_Test_' + str(repeat + 1)
                            + '_InitRecord' + '/' + 'Algorithm_' + algorithm + '/' )
        _output_path = _output_path_init + output_file_path
        output_time = clocktime
        output_file_name = '_Output' + '.csv'
        if not os.path.exists(_output_path):
            os.makedirs(_output_path)


    # 初始化算法
    if algorithm == 'SAMA_RL':
        myAlgorithm = SAMA_RL_templet(
            problem,
            population,
            max_generation,
        )



    start_time = time.time()
    [BestIndi, _] = myAlgorithm.run(max_generation, _output_path, _datapath, output_file_name, output_time)
    end_time = time.time()
    algorithm_time = end_time - start_time
    print("求解时间为："+str(algorithm_time)+"秒")
    algorithm_time = int(algorithm_time)

    record_phen = []

    if len(BestIndi)!= 0:
        for i in range(len(BestIndi)):
            print('最优的目标函数值为：%s  %s' % (BestIndi[i]['Cost'], BestIndi[i]['Exchanges']), BestIndi[i]['Violation'])
        if init != 0:
            BestIndi.sort(key=get_exchanges)
            record_phen = BestIndi[0]['Vars']
            with open(_output_path + clocktime + '_RVEA_Record.txt', 'a') as f:
                f.write("求解时间为：" + str(algorithm_time) + "秒" + '\n')
                f.write('评价次数：%d 次' % myAlgorithm.evalsNum + '\n')
                if BestIndi.sizes != 0:
                    for i in range(len(BestIndi.ObjV)):
                        f.write('最优的目标函数值为：%s  %s' % (BestIndi[i]['Cost'], BestIndi[i]['Exchanges']) + '\n')
                        f.write('约束违反情况为：' + '\n')
                        f.writelines(str(BestIndi[i]['Violation'][0].tolist()) + '\n')
                        f.write('最优的控制变量值为：' + '\n')
                        f.writelines(str(BestIndi[i]['Vars'].tolist()))
                        f.write('\n')
        else:
            record_phen = []

        count = 0

    else:
        record_phen = []
        print('没找到可行解。')

    if init == 0:
        with open(_output_path + clocktime + '_Record.txt', 'a') as f:
            f.write("求解时间为：" + str(algorithm_time) + "秒" + '\n')
            f.write('评价次数：%d 次' % myAlgorithm.evalsNum + '\n')
            if len(BestIndi) != 0:
                for i in range(len(BestIndi)):
                    f.write('变异概率为：%s' % (aircraft_mutation) + '\n')
                    f.write('最优的目标函数值为：%s  %s' % (BestIndi[i]['Cost'], BestIndi[i]['Exchanges']) + '\n')
                    f.write('约束违反情况为：' + '\n')
                    f.writelines(str(BestIndi[i]['Violation'][0].tolist()) + '\n')
                    f.write('最优的控制变量值为：' + '\n')
                    f.writelines(str(BestIndi[i]['Vars'].tolist()))
                    f.write('\n')
            else:
                f.write('没找到可行解。' + '\n')

    # 输出 
    print('用时：%f 秒' % myAlgorithm.passTime)
    print('评价次数：%d 次' % myAlgorithm.evalsNum)

    return record_phen


if __name__=='__main__':
    _path = 'D:/CodeRun/'
    init_output_path_root = _path + 'Results/'
    _datapath_root = _path + 'CurrTest/'

    test_list = ['A01/', 'A02/', 'A03/', 'A04/', 'A05/', 'A06/', 'A07/', 'A08/', 'A09/', 'A10/', 'XA01/', 'XA02/', 'XA03/', 'XA04/', 'B01/', 'B02/', 'B03/', 'B04/', 'B05/', 'B06/', 'B07/', 'B08/', 'B09/', 'B10/', 'XB01/', 'XB02/', 'XB03/', 'XB04/']
    for test_tag in test_list:
        print(test_tag)
        _datapath = _datapath_root + test_tag
        init_output_path = init_output_path_root + test_tag
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime())

        MAXGEN = 1000  # 最大遗传代数
        chorm_size_ini = 60  # 种群规模
        repeat_times = 5  # 重复测试次数
        start_time = 4
        initialize_list = [1]
        canceled_flights_count = 0
        chroms_change_list = [1]
        chroms_change = 1
        algorithms_list = ['SAMA_RL']
        slot_tag = 1
        random_setting = 0

        sort_str = 'Dep'
        cancel_hour_dc = 18  
        cancel_hour_i = 36 

        if test_tag in ['A01/', 'A02/', 'A03/', 'A06/', 'A07/', 'A08/', 'XA01/', 'XA03/']:
            time_max = 800
        elif test_tag in ['A04/', 'A09/']:
            time_max = 1200
            
        elif test_tag in ['A05/', 'A10/', 'XA02/', 'XA04/', 'B01/', 'B02/', 'B03/', 'B04/', 'B05/', 'B06/', 'B07/', 'B08/', 'B09/', 'B10/', 'XB01/', 'XB02/', 'XB03/', 'XB04/']:
            time_max = 1800
        else:
            time_max = 60

        # 数据读取
        config_info = load_dict_from_pickle(_datapath + 'config_info.pkl')
        aircraft_info = load_dict_from_pickle(_datapath + 'aircraft_info.pkl')
        aircraft_type = load_dict_from_pickle(_datapath + 'aircraft_category.pkl')
        airports = load_dict_from_pickle(_datapath + 'airports_info.pkl')
        alt_flights = load_dict_from_pickle(_datapath + 'alt_flights_info.pkl')
        alt_airports = load_dict_from_pickle(_datapath + 'alt_airports_info.pkl')
        alt_aircraft = load_dict_from_pickle(_datapath + 'alt_aircraft_info.pkl')
        itineraries = load_dict_from_pickle(_datapath + 'itineraries_info.pkl')
        position = load_dict_from_pickle(_datapath + 'position_info.pkl')

        flights = load_dict_from_pickle(_datapath + 'flights_info.pkl')
        flight_circles_info = load_dict_from_pickle(_datapath + 'processed_flights_info.pkl')
        same_space_flights = load_dict_from_pickle(_datapath + 'same_space_flights_info.pkl')
        flight_passengers = load_dict_from_pickle(_datapath + 'flight_related_passengers.pkl')
        flight_ring = load_dict_from_pickle(_datapath + 'flight_ring_index.pkl')
        started_rotations_info = load_dict_from_pickle(_datapath + 'started_rotations_info.pkl')

        Dim = len(flight_circles_info) * 2
        aircraft_assignment_mutation = round(1 / Dim, 3)

        for aircraft_id, aircraft in aircraft_info.items():
            if aircraft['Available_time'] != None:
                aircraft['Available_time'] = datetime.strptime(aircraft['Available_date'] + ' ' + aircraft['Available_time'], '%d/%m/%y %H:%M')

        for flight_id, flight in flights.items():
            flight['DepTime'] = datetime.strptime(flight['DepDate'] + ' ' + flight['DepTime'], '%d/%m/%y %H:%M')
            flight['ArrTime'] = datetime.strptime(flight['DepDate'] + ' ' + flight['ArrTime'], '%d/%m/%y %H:%M')
            flight['New_DepTime'] = flight['DepTime']
            flight['New_ArrTime'] = flight['ArrTime']
            if flight['Overnight']:
                flight['ArrTime'] = flight['ArrTime'] + timedelta(days=1)
                flight['New_ArrTime'] = flight['ArrTime']
            if flight['Delay'] != 0:
                flight['New_DepTime'] = flight['New_DepTime'] + timedelta(minutes=flight['Delay'])
                flight['New_ArrTime'] = flight['New_ArrTime'] + timedelta(minutes=flight['Delay'])

        for group_id, group_info in flight_circles_info.items():
            for flight in group_info['Flights']:
                flight['DepTime'] = datetime.strptime(flight['DepDate'] + ' ' + flight['DepTime'], '%d/%m/%y %H:%M')
                flight['ArrTime'] = datetime.strptime(flight['DepDate'] + ' ' + flight['ArrTime'], '%d/%m/%y %H:%M')
                flight['New_DepTime'] = flight['DepTime']
                flight['New_ArrTime'] = flight['ArrTime']
                if flight['Overnight']:
                    flight['ArrTime'] = flight['ArrTime'] + timedelta(days=1)
                    flight['New_ArrTime'] = flight['ArrTime']
                if flight['Delay'] != 0:
                    flight['New_DepTime'] = flight['New_DepTime'] + timedelta(minutes=flight['Delay'])
                    flight['New_ArrTime'] = flight['New_ArrTime'] + timedelta(minutes=flight['Delay'])

        for airport_id, airport in alt_airports.items():
            for alt_info_index in range(len(airport)):
                airport[alt_info_index]['StartTime'] = datetime.strptime(airport[alt_info_index]['StartDate'] + ' ' + airport[alt_info_index]['StartTime'], '%d/%m/%y %H:%M')
                airport[alt_info_index]['EndTime'] = datetime.strptime(airport[alt_info_index]['EndDate'] + ' ' + airport[alt_info_index]['EndTime'], '%d/%m/%y %H:%M')

        airports_capacity_total = get_airports_capacity(config_info, airports, alt_airports, flights, slot_tag)
        group_lookup_dic = get_group_id(flights, flight_circles_info)


        for initialize in initialize_list:
            for repeat in range(start_time, repeat_times):
                init = 0
                RVEA_init = []
                for algorithm in algorithms_list:
                    chorm_size = chorm_size_ini
                    if algorithm == 'TVCSO_RE':
                        cancel_decision = 0
                    else:
                        cancel_decision = 0
                    # 运行优化
                    record = run_optimization(test_tag, time_max, init, MAXGEN, chorm_size, repeat, aircraft_assignment_mutation,
                                              canceled_flights_count, initialize, random_setting, RVEA_init, sort_str,
                                              chroms_change, algorithm, cancel_decision, cancel_hour_dc, cancel_hour_i,
                                              config_info, flight_circles_info, aircraft_info, itineraries, alt_flights,
                                              alt_airports, alt_aircraft, position, aircraft_type, airports, airports_capacity_total,
                                              flights, same_space_flights, flight_passengers, flight_ring, started_rotations_info,
                                              _path, _datapath, init_output_path, time_stamp, group_lookup_dic)



