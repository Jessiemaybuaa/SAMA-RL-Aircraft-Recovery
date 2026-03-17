import copy
import time
import random
import pickle
import numpy as np
from datetime import datetime, timedelta

def load_dict_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def get_left_num(element):
    return element['Left_num']

def query_group_by_aircraft(processed_flights_info, aircraft_id):
    return {group_id: info for group_id, info in processed_flights_info.items() if info['Aircraft'] == aircraft_id}

def query_flight_by_aircraft(flights_info, aircraft_id):
    return {flight_id: info for flight_id, info in flights_info.items() if info['Aircraft'] == aircraft_id}

def get_max_delay(aircraft_dic, flight_circles_dic, cancel_hour_dc, cancel_hour_i, cancel_decision):
    len_var = len(flight_circles_dic)
    max_delay = [len(aircraft_dic) - 1] * len_var + [0] * len_var
    tag = len_var
    for group_id, group_info in flight_circles_dic.items():
        delay_minutes = cancel_hour_i * 60
        for flight in group_info['Flights']:
            if flight['Type'] in ['D', 'C']:
                max_delay_minutes = cancel_hour_dc * 60
                delay_minutes = max_delay_minutes
            else:
                delay_minutes = cancel_hour_i * 60
        if cancel_decision == 1:
            max_delay[tag] = delay_minutes
        elif cancel_decision == 0:
            max_delay[tag] = delay_minutes + 1
        tag += 1

    return max_delay

def flight_circle_to_schedule(circles):
    schedule = {}
    for group_id, aircraft in circles.items():
        for flight in aircraft['Flights']:
            schedule[flight['Flight_ID']] = {'Orig': flight['Orig'], 'Dest': flight['Dest'],
                                             'DepTime': flight['DepTime'], 'ArrTime': flight['ArrTime'],
                                             'Overnight': flight['Overnight'], 'PrevFlight': flight['PrevFlight'],
                                             'DepDate': flight['DepDate'], 'Aircraft': aircraft}
    return schedule


def flight_circle_to_chorm(len_field1, len_field2, circles, aircraft_dic, cancel_hour_dc, cancel_hour_i, cancel_decision):
    schedule = np.zeros((1, len_field1 + len_field2))

    index = 0
    for group_index, info in circles.items():
        schedule[0, index] = int(aircraft_dic[info['Aircraft']]['Num_ID'])
        state = 0
        flight = info['Flights'][0]
        if flight['State'] == -1:
            state = -1
            # state = 0
        elif flight['Delay'] > 0:
            # if state == 0:
            state = flight['Delay']

        if cancel_decision == 0:
            if state != -1:
                schedule[0, len_field1 + index] = state
            else:
                if flight['Type'] in ['D', 'C']:
                    max_delay = cancel_hour_dc * 60
                    schedule[0, len_field1 + index] = max_delay + 1
                else:
                    max_delay = cancel_hour_i * 60
                    schedule[0, len_field1 + index] = max_delay + 1
        elif cancel_decision == 1:
            schedule[0, len_field1 + index] = state
        else:
            print('Change!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        index += 1

    return schedule


def get_group_id(flights, flight_circles_info):
    group_lookup_dic = {}
    for group_id, group_info in flight_circles_info.items():
        for flight in group_info['Flights']:
            if flight['New_Flight_ID'] not in group_lookup_dic:
                group_lookup_dic[flight['New_Flight_ID']] = {
                    'Group_ID': group_id,
                    'Flight_ID': flight['Flight_ID']
                }

    return group_lookup_dic

# 计算扰动后整个恢复期的机场时隙容量
def get_airports_capacity(config_info, airports_info, alt_airports_info, flights_info, slot_tag):
    config = copy.deepcopy(config_info)
    airports_capacity = copy.deepcopy(airports_info)
    flights = copy.deepcopy(flights_info)

    # 移除机场运行时隙约束
    if slot_tag == 0:
        for airport, capacity_list in airports_capacity.items():
            for capacity in capacity_list:
                capacity['Dep_Capacity'] = 1000
                capacity['Arr_Capacity'] = 1000

    config['Recovery_StartTime'] = datetime.strptime(config['Recovery_StartDate'] + ' ' + config['Recovery_StartTime'], '%d/%m/%y %H:%M')
    config['Recovery_EndTime'] = datetime.strptime(config['Recovery_EndDate'] + ' ' + config['Recovery_EndTime'], '%d/%m/%y %H:%M')
    airports_capacity_total = {
        'StartTime': config['Recovery_StartTime'],
        'EndTime': config['Recovery_EndTime'],
        'Capacity_Record': []
    }

    base_day = airports_capacity_total['StartTime'].day

    for airport, airport_cap in airports_capacity.items():
        for hour in airport_cap:
            hour['Dep_Flight'] = {}
            hour['Arr_Flight'] = {}

    # 创建初始机场时隙容量列表，一天一个
    start_day = config['Recovery_StartTime'].day
    end_day = config['Recovery_EndTime'].day

    airports_capacity_total['Capacity_Record'].append(copy.deepcopy(airports_capacity))
    for day in range(end_day - start_day + 1):
        airports_capacity_total['Capacity_Record'].append(copy.deepcopy(airports_capacity))
    airports_capacity_total['Capacity_Record'].append(copy.deepcopy(airports_capacity))
    airports_capacity_total['Capacity_Record'].append(copy.deepcopy(airports_capacity))
    airports_capacity_total['Capacity_Record'].append(copy.deepcopy(airports_capacity))

    # 对机场容量引入扰动
    for alt_airport_id, alt_airport in alt_airports_info.items():
        for alt_info_index in range(len(alt_airport)):
            alt_info = alt_airport[alt_info_index]
            day_change = 0
            alt_day = alt_info['StartTime'].day
            alt_index = alt_day - start_day
            start_time = alt_info['StartTime'].hour
            current_time = start_time
            # 该机场的容量扰动在一天内结束
            if alt_info['EndTime'].day == alt_day:
                end_time = alt_info['EndTime'].hour
            # 该机场的容量扰动未在一天内结束
            else:
                end_time = 24
                day_change = 1
            while current_time < end_time:
                next_time = current_time + 1
                airports_capacity_total['Capacity_Record'][alt_index][alt_airport_id][current_time]['Arr_Capacity'] = alt_info['ArrCap']
                airports_capacity_total['Capacity_Record'][alt_index][alt_airport_id][current_time]['Dep_Capacity'] = alt_info['DepCap']
                current_time = next_time

            # 继续处理跨天扰动的情况
            if day_change == 1:
                alt_index += 1
                start_time = 0
                end_time = alt_info['EndTime'].hour
                current_time = start_time
                while current_time < end_time:
                    next_time = current_time + 1
                    airports_capacity_total['Capacity_Record'][alt_index][alt_airport_id][current_time]['Arr_Capacity'] = alt_info['ArrCap']
                    airports_capacity_total['Capacity_Record'][alt_index][alt_airport_id][current_time]['Dep_Capacity'] = alt_info['DepCap']
                    current_time = next_time

    return airports_capacity_total


def init_chorm_recovery(chorm_num, config, init_flight_circles_info, position_info, aircraft_category,
                        canceled_flights_swap, max_delay, initialize_index, RVEA_init, recovery_start_time, recovery_end_time, flight_circles,
                        airports_info, airports_capacity_total, flights_info, aircraft_info, alt_airports_info,
                        alt_aircraft_info, group_lookup_dic, init_aircraft_assignment, alt_flights_info,
                        cancel_hour_dc, cancel_hour_i, cancel_decision, sort_str):
    len_field1 = len(flight_circles)
    len_field2 = len(flight_circles)
    group_situation = [0 for i in range(len(flight_circles))]


    num_groups = len(flight_circles)
    Vars = flight_circle_to_chorm(num_groups, num_groups, flight_circles, aircraft_info, cancel_hour_dc, cancel_hour_i, cancel_decision)

    solution = evaluation(Vars[0], config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                          alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                          init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total, flights_info,
                          group_lookup_dic, position_info, aircraft_category, cancel_decision, canceled_flights_swap,
                          max_delay, 2 * len(flight_circles), 'init_repair')

    print(solution['Violation'])

    cancel_flight_num = 0
    for group_id, group_info in solution['FlightCircles'].items():
        for flight in group_info['Flights']:
            if flight['State'] == -1:
                cancel_flight_num += 1

    init_schedule_vars = solution['Vars']

    solution = evaluation(Vars[0], config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info,
                          alt_flights_info, cancel_hour_dc, cancel_hour_i, sort_str, init_flight_circles_info,
                          init_aircraft_assignment, aircraft_info, airports_info, airports_capacity_total, flights_info,
                          group_lookup_dic, position_info, aircraft_category, cancel_decision, canceled_flights_swap,
                          max_delay, 2 * len(flight_circles), 'repair')

    print(solution['Violation'])


    cancel_flight_num = 0
    for group_id, group_info in solution['FlightCircles'].items():
        for flight in group_info['Flights']:
            if flight['State'] == -1:
                cancel_flight_num += 1

    init_schedule_vars_2 = solution['Vars']


    Chrom = np.zeros((chorm_num, len_field1 + len_field2))  # 初始化染色体

    if initialize_index == 1:
        for i in range(chorm_num):
            # 自定义初始化策略，例如：
            if i < 1:
                Chrom[i, :] = init_schedule_vars
            elif i == 1:
                Chrom[i, :] = init_schedule_vars_2
            init_num = 2
    elif initialize_index == 2:  # 在初始航班计划解附近生成一部分解作为初代种群个体
        init_num = int(0.05 * chorm_num)
        init_schedule_vars_array = np.zeros((init_num, len_field1 + len_field2))
        for j in range(init_num):
            # 保留原始计划
            if j == 0:
                init_schedule_vars_array[j, :] = init_schedule_vars
            # 生成变化个体
            else:
                new_schedule_vars = init_schedule_vars
                for k in range(len_field1):
                    if random.random() < (1 / len_field1):
                        new_schedule_vars[0][k] = random.randint(0, len(aircraft_info) - 1)
                check = 0
                for l in range(len(init_schedule_vars_array)):
                    if np.array_equal(new_schedule_vars[0, :], init_schedule_vars_array[l, :]):
                        check = 1
                        break
                if check == 0:
                    init_schedule_vars_array[j, :] = new_schedule_vars
                else:
                    change = random.randint(0, len_field1 - 1)
                    new_schedule_vars[0][change] = random.randint(0, len(aircraft_info) - 1)
                    init_schedule_vars_array[j, :] = new_schedule_vars

        for i in range(init_num):
            Chrom[i, :] = init_schedule_vars_array[i, :]

    elif initialize_index == 3:
        init_schedule_vars = RVEA_init
        for i in range(chorm_num):
            # 自定义初始化策略，例如：
            if i < 1:
                Chrom[i, :] = init_schedule_vars
            init_num = 1
    elif initialize_index == 0:
        init_num = 0

    return solution['FlightCircles'], Chrom, init_num, cancel_flight_num


# 用于调用geatpy的群体进化算法中种群的初始化
def init_Chrom(problem, population):
    initialize_index = problem.initialize
    chorm_num = problem.chorm_size
    RVEA_init = problem.RVEA_init
    recovery_start_time = problem.recovery_start_time
    recovery_end_time = problem.recovery_end_time
    flight_circles = copy.deepcopy(problem.flight_circles_info)
    airports_info = copy.deepcopy(problem.airports_info)
    airports_capacity_total = copy.deepcopy(problem.airports_capacity_total)
    flights_info = copy.deepcopy(problem.flights_info)
    aircraft_info = problem.aircraft_info
    alt_airports_info = problem.alt_airports_info
    alt_aircraft_info = problem.alt_aircraft_info
    alt_flights_info = problem.alt_flights_info
    cancel_hour_dc = problem.cancel_hour_dc
    cancel_hour_i = problem.cancel_hour_i
    cancel_decision = problem.cancel_decision
    group_lookup_dic = problem.group_lookup_dic
    config = problem.config_info
    init_flight_circles_info = copy.deepcopy(problem.flight_circles_info)
    position_info = copy.deepcopy(problem.position_info)
    aircraft_category = problem.aircraft_category
    canceled_flights_swap = problem.canceled_flights_count
    max_delay = problem.max_delay

    sort_str = problem.sort_str

    num_groups = len(flight_circles)

    init_schedule_vars = flight_circle_to_chorm(num_groups, num_groups, flight_circles, aircraft_info,
                                                cancel_hour_dc, cancel_hour_i, cancel_decision)
    init_aircraft_assignment = init_schedule_vars[0, :num_groups]

    flight_circles, Chrom, init_num, init_cancel_flight_num = init_chorm_recovery(chorm_num, config, init_flight_circles_info, position_info, aircraft_category,
                                                          canceled_flights_swap, max_delay, initialize_index, RVEA_init, recovery_start_time,
                                                          recovery_end_time, flight_circles, airports_info, airports_capacity_total, flights_info,
                                                          aircraft_info, alt_airports_info, alt_aircraft_info, group_lookup_dic, init_aircraft_assignment,
                                                          alt_flights_info, cancel_hour_dc, cancel_hour_i, cancel_decision, sort_str)

    population.ObjV = np.zeros((population.sizes, 2))
    population.CV = np.zeros((population.sizes, 5))

    if initialize_index == 1:
        population.Chroms[0][0] = Chrom[0][:num_groups]
        population.Chroms[1][0] = Chrom[0][num_groups:]
        population.Chroms[0][1] = Chrom[1][:num_groups]
        population.Chroms[1][1] = Chrom[1][num_groups:]
    elif initialize_index == 2:
        for i in range(init_num):
            population.Chroms[0][population.sizes - 1 - i] = Chrom[i][:num_groups]
            population.Chroms[1][population.sizes - 1 - i] = Chrom[i][num_groups:]
    elif initialize_index == 3:
        for i in range(init_num):
            population.Chroms[0][population.sizes - 1 - i] = Chrom[i][:num_groups]
            population.Chroms[1][population.sizes - 1 - i] = Chrom[i][num_groups:]
    elif initialize_index == 0:
        pass

    return population, init_cancel_flight_num, init_aircraft_assignment


def check_maintenance(aircraft, aircraft_dic, flights, maint_airport, maint_start, maint_end, maint_time_range):
    # 初始化
    successful_maintenance = False  # 是否能成功维修
    flights_during_maintenance = 0  # 维修期间需执行航班数
    pre_maint_duration = 0  # 维修前飞行小时数
    post_maint_duration = 0  # 维修后飞行小时数
    flag_in_maintenance_airport = False  # 维修时是否在计划维修机场
    maint_flight = 'None'

    aircraft_orig = aircraft_dic[aircraft]['Orig']
    flight_ids = list(flights.keys())
    if len(flight_ids) == 0:
        aircraft_position = aircraft_orig
    else:
        aircraft_position = flights[flight_ids[0]]['Orig']
    for flight_id, flight in flights.items():
        if flight['State'] != -1:
            dep_time = flight['New_DepTime']
            arr_time = flight['New_ArrTime']
            # 检查维修时间
            if dep_time <= maint_start <= arr_time:
                flights_during_maintenance += 1
            elif dep_time <= maint_end <= arr_time:
                flights_during_maintenance += 1
            elif maint_start <= dep_time <= maint_end:
                flights_during_maintenance += 1
            elif maint_start <= arr_time <= maint_end:
                flights_during_maintenance += 1
            else:
                if maint_start > arr_time:
                    pre_maint_duration += flight['Duration']
                    aircraft_position = flight['Dest']
                elif maint_end < dep_time:
                    post_maint_duration += flight['Duration']

    if aircraft_position == maint_airport:
        flag_in_maintenance_airport = True

    # 检查维修及飞行时间约束违反情况
    # 判断维修是否成功
    if flag_in_maintenance_airport and flights_during_maintenance == 0:
        successful_maintenance = True

    # 判断飞行时长是否超出限制
    over_duration_count = 0
    # if successful_maintenance:
    if pre_maint_duration > maint_time_range:
        over_duration_count = pre_maint_duration - maint_time_range
        over_tag = 'pre'
    elif post_maint_duration > maint_time_range:
        over_duration_count = post_maint_duration - maint_time_range
        over_tag = 'post'
    else:
        over_tag = 'None'

    return successful_maintenance, over_duration_count


def repair_maintenance(aircraft, aircraft_dic, flight_circles, flights, group_lookup_dic, maint_airport, maint_start, maint_end, maint_time_range):
    # 初始化
    successful_maintenance = False  # 是否能成功维修
    flights_during_maintenance = 0  # 维修期间需执行航班数
    pre_maint_duration = 0  # 维修前飞行小时数
    post_maint_duration = 0  # 维修后飞行小时数
    pre_maint_flights = {} # 维修前可调整航班
    post_maint_flights = {} # 维修后可调整航班
    flag_in_maintenance_airport = False  # 维修时是否在计划维修机场
    maint_flight = 'None'

    # if aircraft == 'A319#150':
    #     print(aircraft)

    aircraft_orig = aircraft_dic[aircraft]['Orig']
    flight_ids = list(flights.keys())
    if len(flight_ids) == 0:
        aircraft_position = aircraft_orig
    else:
        aircraft_position = flights[flight_ids[0]]['Orig']
    for flight_id, flight in flights.items():
        if flight['State'] != -1:
            dep_time = flight['New_DepTime']
            arr_time = flight['New_ArrTime']
            # 检查维修时间
            if dep_time <= maint_start <= arr_time:
                flights_during_maintenance += 1
            elif dep_time <= maint_end <= arr_time:
                flights_during_maintenance += 1
            elif maint_start <= dep_time <= maint_end:
                flights_during_maintenance += 1
            elif maint_start <= arr_time <= maint_end:
                flights_during_maintenance += 1
            else:
                if maint_start > arr_time:
                    # 记录可调整航班情况
                    if flight['New_Flight_ID'] in group_lookup_dic:
                        pre_maint_flights[flight['New_Flight_ID']] = flight
                    pre_maint_duration += flight['Duration']
                    aircraft_position = flight['Dest']
                elif maint_end < dep_time:
                    # 记录可调整航班情况
                    if flight['New_Flight_ID'] in group_lookup_dic:
                        post_maint_flights[flight['New_Flight_ID']] = flight
                    post_maint_duration += flight['Duration']

    if aircraft_position == maint_airport:
        flag_in_maintenance_airport = True

    # 检查维修及飞行时间约束违反情况
    # 判断维修是否成功
    if flag_in_maintenance_airport and flights_during_maintenance == 0:
        successful_maintenance = True

    # 判断飞行时长是否超出限制
    over_duration_count = 0
    cancel_flight = []
    while(pre_maint_duration > maint_time_range):
        # 尝试取消航班环，使得满足约束
        pre_maint_flights_list = list(pre_maint_flights.keys())
        flight_index = 0
        for flight_index in range(len(pre_maint_flights_list)):
            cancel_group_id = group_lookup_dic[pre_maint_flights_list[flight_index]]['Group_ID']
            if len(flight_circles[cancel_group_id]['Flights']) == 2:
                for flight in flight_circles[cancel_group_id]['Flights']:
                    cancel_flight.append(flight['New_Flight_ID'])
                    pre_maint_duration -= flight['Duration']
                if pre_maint_duration <= maint_time_range:
                    break

        if flight_index == len(pre_maint_flights_list)-1 or len(pre_maint_flights_list) == 0:
            break
        over_tag = 'pre'

    while(post_maint_duration > maint_time_range):
        # 尝试取消航班环，使得满足约束
        post_maint_flights_list = list(post_maint_flights.keys())
        flight_index = 0
        for flight_index in range(len(post_maint_flights_list)):
            index = len(post_maint_flights_list) - flight_index - 1
            cancel_group_id = group_lookup_dic[post_maint_flights_list[index]]['Group_ID']
            # if len(flight_circles[cancel_group_id]['Flights']) == 2:
            for flight in flight_circles[cancel_group_id]['Flights']:
                cancel_flight.append(flight['New_Flight_ID'])
                post_maint_duration -= flight['Duration']
            if post_maint_duration <= maint_time_range:
                break

        if flight_index == len(post_maint_flights_list) - 1 or len(post_maint_flights_list) == 0:
            break
        over_tag = 'post'
    else:
        over_tag = 'None'

    for flight_id, flight in flights.items():
        if flight['New_Flight_ID'] in cancel_flight:
            flights[flight['New_Flight_ID']]['State'] = -1
            flights[flight['New_Flight_ID']]['Delay'] = 0
            flights[flight['New_Flight_ID']]['New_DepTime'] = flights[flight['New_Flight_ID']]['DepTime'] + timedelta(minutes=flights[flight['New_Flight_ID']]['Delay'])
            flights[flight['New_Flight_ID']]['New_ArrTime'] = flights[flight['New_Flight_ID']]['ArrTime'] + timedelta(minutes=flights[flight['New_Flight_ID']]['Delay'])

    over_duration_count += max((pre_maint_duration - maint_time_range), 0)
    over_duration_count += max((post_maint_duration - maint_time_range), 0)
    # if over_duration_count > 0:
    #     print(aircraft)

    return successful_maintenance, over_duration_count, flights


def check_aircraft_position(aircraft_info, aircraft_position_dic, flights_info, recovery_end_time, sort_str):
    for aircraft, info in aircraft_info.items():
        # 汇合某架飞机的所有待执行航班环
        if aircraft[:9] == 'TranspCom':
            continue
        aircraft_id = aircraft
        flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

        sort_str_adapt = sort_str + 'Time'

        # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
        sorted_flights_for_aircraft = dict(sorted(
            flights_for_aircraft.items(),
            key=lambda item: (item[1][sort_str_adapt])
        ))

        # 记录飞机的最后位置
        flight_ids = list(sorted_flights_for_aircraft.keys())
        if len(flight_ids) == 0:
            last_dest = aircraft_info[aircraft]['Orig']
        else:
            last_dest = sorted_flights_for_aircraft[flight_ids[0]]['Orig']
        for flight_id, flight in sorted_flights_for_aircraft.items():
            if flight['State'] != -1:
                # 检查飞机位置
                if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                    last_dest = flight['Dest']
                else:
                    if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                        last_dest = 'Flying'

                    break

        # 记录每个机场停靠的飞机
        if last_dest != 'Flying':
            # 初始化字典条目
            if last_dest not in aircraft_position_dic:
                aircraft_position_dic[last_dest] = []

            aircraft_position_dic[last_dest].append(aircraft)

    return aircraft_position_dic


def check_capacity(airports_capacity_dic):
    violation = 0
    overload_info_dic = {}
    day_index = airports_capacity_dic['StartTime'].day
    for capacity_list in airports_capacity_dic['Capacity_Record']:
        for airport, airport_info in capacity_list.items():
            for time_tip in airport_info:
                if time_tip['Dep_Capacity'] < 0:
                    violation -= time_tip['Dep_Capacity']
                    if airport not in overload_info_dic:
                        overload_info_dic[airport] = {
                            'Total_Time': []
                        }
                    dep_hour = int(time_tip['StartTime'][:2])
                    overload_info_dic[airport]['Total_Time'].append(('Dep', day_index, dep_hour))
                    # overload_info_dic[airport]['Dep_Time'].append((day_index, dep_hour))
                if time_tip['Arr_Capacity'] < 0:
                    violation -= time_tip['Arr_Capacity']
                    if airport not in overload_info_dic:
                        overload_info_dic[airport] = {
                            'Total_Time': []
                        }
                    arr_hour = int(time_tip['StartTime'][:2])
                    overload_info_dic[airport]['Total_Time'].append(('Arr', day_index, arr_hour))
                    # overload_info_dic[airport]['Arr_Time'].append((day_index, arr_hour))
        day_index += 1

    return violation, overload_info_dic


def check_airport_capacity(airports_capacity_total, airports_info, flight_circles, flights_info, alt_airports_info):
    violation_3 = 0
    airports_capacity_dic = copy.deepcopy(airports_capacity_total)

    for group_id, group_info in flight_circles.items():
        aircraft = group_info['Aircraft']
        for flight in group_info['Flights']:
            new_flights_id = flight['New_Flight_ID']
            flights_info[new_flights_id]['Aircraft'] = aircraft
            flights_info[new_flights_id]['State'] = flight['State']
            flights_info[new_flights_id]['Delay'] = flight['Delay']
            flights_info[new_flights_id]['Recovery_Flight'] = 1
            flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
            flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
            flights_info[new_flights_id]['DepTime'] = flight['DepTime']
            flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']

    # 对航班按照第一个flight的DepTime排序
    sorted_flight = dict(sorted(
        flights_info.items(),
        key=lambda item: item[1]['DepTime']
    ))

    index = 0
    day_change = 0
    base_day = airports_capacity_dic['StartTime'].day

    # airports_capacity = copy.deepcopy(airports_info)
    for flight_id, flight in sorted_flight.items():
        if flight['State'] != -1:
            dep_day = flight['New_DepTime'].day
            dep_index = dep_day - base_day
            dep_time = flight['New_DepTime']
            dep_hour = dep_time.hour

            arr_day = flight['New_ArrTime'].day
            arr_index = arr_day - base_day
            arr_time = flight['New_ArrTime']
            arr_hour = arr_time.hour
            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] -= 1
            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] -= 1

    violation, overload_info_dic = check_capacity(airports_capacity_dic)
    violation_3 += violation

    return violation_3, flights_info, airports_capacity_dic


def repair_airports_capacity(recovery_start_time, airports_capacity_total, group_lookup_dic, init_aircraft_info, flight_circles, flights_info, sort_str, cancel_hour_dc, cancel_hour_i):
    violation_1 = 0
    violation_2 = 0
    overload_info_dic = {}
    airports_capacity_dic = copy.deepcopy(airports_capacity_total)
    aircraft_info = copy.deepcopy(init_aircraft_info)
    flight_circles = copy.deepcopy(flight_circles)
    # print('Repair Start!!')

    # 若航班在恢复期前开始，后续航班数设置为50
    for flight_id, flight in flights_info.items():
        if flight['New_Flight_ID'] not in group_lookup_dic:
            flight['Left_num'] = 50

    # 对航班按照第一个flight的DepTime排序
    sorted_flight = dict(sorted(
        flights_info.items(),
        key=lambda item: item[1]['DepTime']
    ))

    # index = 0
    # day_change = 0
    base_day = airports_capacity_dic['StartTime'].day

    # 记录航班及机场时隙情况
    violation_ini = 0
    for flight_id, flight in sorted_flight.items():
        # start_day = flight['New_DepTime'].day
        # 记录容量变化情况
        if flight['State'] != -1:
            dep_day = flight['New_DepTime'].day
            dep_index = dep_day - base_day
            dep_time = flight['New_DepTime']
            dep_hour = dep_time.hour

            arr_day = flight['New_ArrTime'].day
            arr_index = arr_day - base_day
            arr_time = flight['New_ArrTime']
            arr_hour = arr_time.hour

            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Flight'][flight['New_Flight_ID']] = {
                'New_Flight_ID': flight['New_Flight_ID'],
                'Delay': flight['Delay'],
                'Left_num': flight['Left_num']
            }

            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Flight'][flight['New_Flight_ID']] = {
                'New_Flight_ID': flight['New_Flight_ID'],
                'Delay': flight['Delay'],
                'Left_num': flight['Left_num']
            }

            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] -= 1
            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] -= 1

            if airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] < 0:
                violation_ini += 1
                if flight['Orig'] not in overload_info_dic:
                    overload_info_dic[flight['Orig']] = {
                        'Total_Time': []
                    }
                if ('Dep', dep_day, dep_hour) not in overload_info_dic[flight['Orig']]['Total_Time']:
                    overload_info_dic[flight['Orig']]['Total_Time'].append(('Dep', dep_day, dep_hour))
                # overload_info_dic[flight['Orig']]['Dep_Time'].append((dep_day, dep_hour))

            if airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] < 0:
                violation_ini += 1
                if flight['Dest'] not in overload_info_dic:
                    overload_info_dic[flight['Dest']] = {
                        'Total_Time': []
                    }
                if ('Arr', arr_day, arr_hour) not in overload_info_dic[flight['Dest']]['Total_Time']:
                    overload_info_dic[flight['Dest']]['Total_Time'].append(('Arr', arr_day, arr_hour))

    # 遍历各个机场的约束违反情况，执行修复操作
    repair_times = 0
    repair_max = 20
    repair_max = violation_ini
    repair_chance = True
    if violation_ini > repair_max:
        violation, overload_info_dic = check_capacity(airports_capacity_dic)

    else:
        violation = violation_ini
        # 记录起始，防止无法满足要求
        if violation > 0:
            overload_info_list = list(overload_info_dic.keys())
            loop_start = overload_info_dic[overload_info_list[0]]['Total_Time'][0]
            begin_tag = False
        while(violation > 0):
            if not repair_chance:
                break
            for airport, overload_info in overload_info_dic.items():
                # if airport == 'NCE':
                #     print('Debug')
                start_day = airports_capacity_dic['StartTime'].day
                if not repair_chance:
                    break
                # 先修复离场时隙约束
                for time_set in overload_info['Total_Time']:
                    if airport == overload_info_list[0] and time_set == loop_start:
                        if not begin_tag:
                            begin_tag = True
                        else:
                            repair_chance = False
                            break

                    repair_cap = time_set[0]
                    repair_day = time_set[1]
                    repair_hour = time_set[2]
                    if repair_cap == 'Dep':
                        check_tag = 'Dep_Capacity'
                    else:
                        check_tag = 'Arr_Capacity'
                    day_index = repair_day - start_day
                    # 获取需延误航班数量
                    overload_num = -1 * airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][check_tag]
                    flights_dic = airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][repair_cap + '_Flight']
                    flights_dic = dict(sorted(
                            flights_dic.items(),
                            key=lambda item: (item[1]['Left_num'])
                    ))
                    flights_list = list(flights_dic.keys())

                    flight_index = 0
                    while overload_num > 0:
                        # 已完整遍历可调整航班列表，无法满足修复需求
                        if flight_index >= len(flights_list):
                            repair_chance = False
                            break
                        # 获取需调整的航班信息
                        flight_id = flights_dic[flights_list[flight_index]]['New_Flight_ID']
                        curr_check_flight = flights_info[flight_id]
                        # 航班在恢复期前已出发，不能调整
                        if flight_id not in group_lookup_dic:
                            # 尝试调整其他航班
                            flight_index += 1
                            continue
                        # 航班在恢复期开始后出发，可以调整
                        else:
                            # 获取该航班对应的飞机周转、型号、维修信息
                            group_id = group_lookup_dic[flight_id]['Group_ID']
                            group_info = flight_circles[group_id]
                            aircraft_id = group_info['Aircraft']
                            # if aircraft_id == 'A319#20':
                            #     print('Debug')
                            groups_for_aircraft = query_group_by_aircraft(flight_circles, aircraft_id)
                            groups_for_aircraft = copy.deepcopy(groups_for_aircraft)
                            sort_str_adapt = sort_str + 'Time'

                            sorted_groups_for_aircraft = dict(sorted(
                                groups_for_aircraft.items(),
                                key=lambda item: (item[1]['Flights'][0][sort_str_adapt])
                            ))
                            # 获取该飞机的航班环
                            temp_sorted_groups_for_aircraft = copy.deepcopy(sorted_groups_for_aircraft)

                            # 获取型号相关信息
                            turnaround_time = aircraft_info[aircraft_id]['TurnRound']
                            transit_time = aircraft_info[aircraft_id]['Transit']
                            turnaround_time = max(turnaround_time, transit_time)

                            # 获取飞机维修信息
                            maint_check = False
                            post_maint_duration = 0
                            flag_in_maintenance_airport = False
                            successful_maintenance = False
                            if aircraft_info[aircraft_id]['Maint'] is None:
                                maint_check = True
                            else:
                                maint_info = aircraft_info[aircraft_id]['Maint']
                                maint_airport = maint_info['Maint_Airport']
                                maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
                                maint_time_range = int(maint_info['Maint_Time_Range'])
                                # 该航班在维修后起飞，维修约束已满足，飞行时长也必定满足
                                if maint_end < curr_check_flight['New_DepTime']:
                                    maint_check = True

                            # 获取飞机当前位置，以及飞机的最早起飞时间
                            last_dest = aircraft_info[aircraft_id]['Orig']
                            aircraft_available_time = aircraft_info[aircraft_id]['Available_time']
                            if aircraft_available_time is None:
                                aircraft_available_time = recovery_start_time
                            aircraft_earliest_dep_time = aircraft_available_time

                            find_flight = False
                            # # 不需要考虑维修情况，因为已经满足或没有维修计划
                            # if maint_check:
                            # 取消航班环标记
                            cancel_check = 0
                            # 遍历飞机的航班环序列
                            for check_group_id, check_group_info in temp_sorted_groups_for_aircraft.items():
                                for check_flight in check_group_info['Flights']:
                                    if check_flight['New_Flight_ID'] == flight_id:
                                        find_flight = True
                                        aircraft_earliest_dep_time = check_flight['New_ArrTime'] + timedelta(minutes=turnaround_time)
                                        # last_dest = check_flight['Orig']

                                    if check_flight['State'] != -1 and not maint_check:
                                        if check_flight['New_DepTime'] > maint_start:
                                            last_dest = check_flight['Orig']
                                        else:
                                            last_dest = check_flight['Dest']

                                    # 找到了需调整航班，调整其及后续航班，确保调整的航班未取消
                                    if find_flight and check_flight['State'] != -1:
                                        # 记录下当前航班的起降时刻，确认调整方案后，对时隙记录逐个调整
                                        orig_dep_time = copy.deepcopy(check_flight['New_DepTime'])
                                        orig_arr_time = copy.deepcopy(check_flight['New_ArrTime'])
                                        orig_dep_day_index = orig_dep_time.day - start_day
                                        orig_arr_day_index = orig_arr_time.day - start_day
                                        orig_dep_hour = orig_dep_time.hour
                                        orig_arr_hour = orig_arr_time.hour
                                        # 获取抵离港信息
                                        dep_day_index = check_flight['New_DepTime'].day - start_day
                                        arr_day_index = check_flight['New_ArrTime'].day - start_day
                                        dep_hour = check_flight['New_DepTime'].hour
                                        arr_hour = check_flight['New_ArrTime'].hour
                                        # 设置单次延误步长
                                        delay_step = 60
                                        delay_required = 0
                                        slot_check = False
                                        # 检查当前是否超出容量, 超出则延误航班至非拥挤时段，当前航班未取消，则可重复循环
                                        while cancel_check == 0:
                                            # 当前的起降时刻对应时隙容量非负
                                            if delay_required == 0:
                                                if airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Capacity'] >= 0 and airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Capacity'] >= 0:
                                                    slot_check = True
                                            # 当前的起降时刻对应时隙容量加入新航班后依旧非负
                                            else:
                                                if airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Capacity'] >= 1 and airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Capacity'] >= 1:
                                                    slot_check = True
                                            if not slot_check:
                                                delay_required = delay_step + check_flight['Delay']
                                                flight_delay = delay_required
                                                aircraft_delay = (aircraft_earliest_dep_time - check_flight['DepTime']).total_seconds() // 60
                                                delay_required = max(flight_delay, aircraft_delay)
                                                # 获取当前新的起降时刻
                                                check_flight['State'] = delay_required
                                                check_flight['Delay'] = delay_required
                                                check_flight['New_DepTime'] = check_flight['DepTime'] + timedelta(minutes=check_flight['Delay'])
                                                check_flight['New_ArrTime'] = check_flight['ArrTime'] + timedelta(minutes=check_flight['Delay'])
                                                # 检查维修情况
                                                if not maint_check:
                                                    dep_time = check_flight['New_DepTime']
                                                    arr_time = check_flight['New_ArrTime']
                                                    if dep_time <= maint_start <= arr_time:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    elif dep_time <= maint_end <= arr_time:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    elif maint_start <= dep_time <= maint_end:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    elif maint_start <= arr_time <= maint_end:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    else:
                                                        if maint_start > arr_time:
                                                            last_dest = check_flight['Dest']
                                                        elif maint_end < dep_time:
                                                            if post_maint_duration + check_flight['Duration'] > maint_time_range:
                                                                cancel_check = 1
                                                                break
                                                            else:
                                                                post_maint_duration += check_flight['Duration']

                                                if check_flight['Type'] in ['D', 'C']:
                                                    max_delay = cancel_hour_dc * 60
                                                else:
                                                    max_delay = cancel_hour_i * 60

                                                # 若延误超过最大值，取消航班，后续航班也全部取消
                                                if delay_required > max_delay:
                                                    cancel_check = 1

                                                # 延误没有超过最大值，采用当前的延误继续检查
                                                else:
                                                    check_flight['State'] = delay_required
                                                    check_flight['Delay'] = delay_required
                                                    check_flight['New_DepTime'] = check_flight['DepTime'] + timedelta(minutes=check_flight['Delay'])
                                                    check_flight['New_ArrTime'] = check_flight['ArrTime'] + timedelta(minutes=check_flight['Delay'])
                                                    # 更新抵离港信息，加入下次检验
                                                    dep_day_index = check_flight['New_DepTime'].day - start_day
                                                    arr_day_index = check_flight['New_ArrTime'].day - start_day
                                                    dep_hour = check_flight['New_DepTime'].hour
                                                    arr_hour = check_flight['New_ArrTime'].hour

                                            # 航班延误已能满足时隙容量要求，对时隙进行更改，然后跳出循环
                                            else:
                                                # 原始时隙容量增加，航班移出原始时隙航班记录
                                                airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Capacity'] += 1
                                                airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Capacity'] += 1
                                                flight_temp_dep = airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Flight'].pop(check_flight['New_Flight_ID'])
                                                flight_temp_arr = airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Flight'].pop(check_flight['New_Flight_ID'])
                                                # 更新时隙容量减少，航班加入更新时隙航班记录
                                                airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Capacity'] -= 1
                                                airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Capacity'] -= 1
                                                airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Flight'][check_flight['New_Flight_ID']] = {
                                                    **flight_temp_dep
                                                }
                                                airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Flight'][check_flight['New_Flight_ID']] = {
                                                    **flight_temp_arr
                                                }
                                                break

                                        # 若需要取消航班，取消当前飞机的后续所有航班
                                        if cancel_check == 1:
                                            check_flight['State'] = -1
                                            check_flight['Delay'] = 0
                                            check_flight['New_DepTime'] = check_flight['DepTime'] + timedelta(minutes=check_flight['Delay'])
                                            check_flight['New_ArrTime'] = check_flight['ArrTime'] + timedelta(minutes=check_flight['Delay'])
                                            # 原始时隙容量增加，航班移出原始时隙航班记录
                                            airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Capacity'] += 1
                                            airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Capacity'] += 1
                                            airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Flight'].pop(check_flight['New_Flight_ID'])
                                            airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Flight'].pop(check_flight['New_Flight_ID'])

                            if not maint_check:
                                if last_dest == maint_airport:
                                    flag_in_maintenance_airport = True
                                    successful_maintenance = True
                                else:
                                    violation_1 += 1
                                if post_maint_duration <= maint_time_range:
                                    if successful_maintenance:
                                        maint_check = True
                                    else:
                                        maint_check = False
                                else:
                                    violation_2 += 1
                                if violation_1 + violation_2 == 0:
                                    maint_check = True

                            # 将变化更新到航班环
                            for check_group_id, check_group_info in temp_sorted_groups_for_aircraft.items():
                                for check_flight in check_group_info['Flights']:
                                    new_flights_id = check_flight['New_Flight_ID']
                                    flights_info[new_flights_id]['Aircraft'] = aircraft_id
                                    flights_info[new_flights_id]['State'] = check_flight['State']
                                    flights_info[new_flights_id]['Delay'] = check_flight['Delay']
                                    flights_info[new_flights_id]['Recovery_Flight'] = 1
                                    flights_info[new_flights_id]['New_DepTime'] = check_flight['New_DepTime']
                                    flights_info[new_flights_id]['New_ArrTime'] = check_flight['New_ArrTime']
                                    flights_info[new_flights_id]['DepTime'] = check_flight['DepTime']
                                    flights_info[new_flights_id]['ArrTime'] = check_flight['ArrTime']
                                flight_circles[check_group_id] = check_group_info

                            if not maint_check:
                                # 未成功维修，终止修复
                                repair_chance = False
                                break

                        # 确认是否修复时隙
                        new_overload_num = -1 * airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][check_tag]
                        if new_overload_num < overload_num:
                            violation -= (overload_num - new_overload_num)
                            overload_num = new_overload_num

                        flight_index += 1

                    if not repair_chance:
                        break

                if not repair_chance:
                    break

            if not repair_chance:
                break


            new_violation, temp_overload_info_dic = check_capacity(airports_capacity_dic)
            if new_violation != violation:
                violation = new_violation

    return violation_1, violation_2, violation, airports_capacity_dic, flight_circles, flights_info, overload_info_dic


def init_re_repair_airports_capacity(recovery_start_time, airports_capacity_total, group_lookup_dic, init_aircraft_info, flight_circles, flights_info, sort_str, cancel_hour_dc, cancel_hour_i):
    violation_1 = 0
    violation_2 = 0
    overload_info_dic = {}
    airports_capacity_dic = copy.deepcopy(airports_capacity_total)
    aircraft_info = copy.deepcopy(init_aircraft_info)
    flight_circles = copy.deepcopy(flight_circles)

    # 若航班在恢复期前开始，后续航班数设置为50
    for flight_id, flight in flights_info.items():
        if flight['New_Flight_ID'] not in group_lookup_dic:
            flight['Left_num'] = 50

    # 对航班按照第一个flight的DepTime排序
    sorted_flight = dict(sorted(
        flights_info.items(),
        key=lambda item: item[1]['DepTime']
    ))

    base_day = airports_capacity_dic['StartTime'].day

    # 记录航班及机场时隙情况
    violation_ini = 0
    for flight_id, flight in sorted_flight.items():
        # 记录容量变化情况
        if flight['State'] != -1:
            dep_day = flight['New_DepTime'].day
            dep_index = dep_day - base_day
            dep_time = flight['New_DepTime']
            dep_hour = dep_time.hour

            arr_day = flight['New_ArrTime'].day
            arr_index = arr_day - base_day
            arr_time = flight['New_ArrTime']
            arr_hour = arr_time.hour

            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Flight'][flight['New_Flight_ID']] = {
                'New_Flight_ID': flight['New_Flight_ID'],
                'Delay': flight['Delay'],
                'Left_num': flight['Left_num']
            }

            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Flight'][flight['New_Flight_ID']] = {
                'New_Flight_ID': flight['New_Flight_ID'],
                'Delay': flight['Delay'],
                'Left_num': flight['Left_num']
            }

            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] -= 1
            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] -= 1

            if airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] < 0:
                violation_ini += 1
                if flight['Orig'] not in overload_info_dic:
                    overload_info_dic[flight['Orig']] = {
                        'Total_Time': []
                    }
                if ('Dep', dep_day, dep_hour) not in overload_info_dic[flight['Orig']]['Total_Time']:
                    overload_info_dic[flight['Orig']]['Total_Time'].append(('Dep', dep_day, dep_hour))

            if airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] < 0:
                violation_ini += 1
                if flight['Dest'] not in overload_info_dic:
                    overload_info_dic[flight['Dest']] = {
                        'Total_Time': []
                    }
                if ('Arr', arr_day, arr_hour) not in overload_info_dic[flight['Dest']]['Total_Time']:
                    overload_info_dic[flight['Dest']]['Total_Time'].append(('Arr', arr_day, arr_hour))

    # 遍历各个机场的约束违反情况，执行修复操作
    repair_times = 0
    repair_max = 20
    repair_max = violation_ini
    repair_chance = True
    if violation_ini > repair_max:
        violation, overload_info_dic = check_capacity(airports_capacity_dic)

    else:
        violation = violation_ini
        # 记录起始，防止无法满足要求
        if violation > 0:
            overload_info_list = list(overload_info_dic.keys())
            loop_start = overload_info_dic[overload_info_list[0]]['Total_Time'][0]
            begin_tag = False
        while(violation > 0):
            if not repair_chance:
                break
            for airport, overload_info in overload_info_dic.items():
                start_day = airports_capacity_dic['StartTime'].day
                if not repair_chance:
                    break
                # 先修复离场时隙约束
                for time_set in overload_info['Total_Time']:
                    if airport == overload_info_list[0] and time_set == loop_start:
                        if not begin_tag:
                            begin_tag = True
                        else:
                            repair_chance = False
                            break

                    repair_cap = time_set[0]
                    repair_day = time_set[1]
                    repair_hour = time_set[2]
                    if repair_cap == 'Dep':
                        check_tag = 'Dep_Capacity'
                    else:
                        check_tag = 'Arr_Capacity'
                    day_index = repair_day - start_day
                    # 获取需延误航班数量
                    overload_num = -1 * airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][check_tag]
                    flights_dic = airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][repair_cap + '_Flight']
                    flights_dic = dict(sorted(
                            flights_dic.items(),
                            key=lambda item: (item[1]['Left_num'])
                    ))
                    flights_list = list(flights_dic.keys())

                    flight_index = 0
                    while overload_num > 0:
                        # 已完整遍历可调整航班列表，无法满足修复需求
                        if flight_index >= len(flights_list):
                            repair_chance = False
                            break
                        # 获取需调整的航班信息
                        flight_id = flights_dic[flights_list[flight_index]]['New_Flight_ID']
                        curr_check_flight = flights_info[flight_id]
                        # 航班在恢复期前已出发，不能调整
                        if flight_id not in group_lookup_dic:
                            # 尝试调整其他航班
                            flight_index += 1
                            continue
                        # 航班在恢复期开始后出发，可以调整
                        else:
                            # 获取该航班对应的飞机周转、型号、维修信息
                            group_id = group_lookup_dic[flight_id]['Group_ID']
                            group_info = flight_circles[group_id]
                            aircraft_id = group_info['Aircraft']
                            groups_for_aircraft = query_group_by_aircraft(flight_circles, aircraft_id)
                            groups_for_aircraft = copy.deepcopy(groups_for_aircraft)
                            sort_str_adapt = sort_str + 'Time'

                            sorted_groups_for_aircraft = dict(sorted(
                                groups_for_aircraft.items(),
                                key=lambda item: (item[1]['Flights'][0][sort_str_adapt])
                            ))
                            # 获取该飞机的航班环
                            temp_sorted_groups_for_aircraft = copy.deepcopy(sorted_groups_for_aircraft)

                            # 获取型号相关信息
                            turnaround_time = aircraft_info[aircraft_id]['TurnRound']
                            transit_time = aircraft_info[aircraft_id]['Transit']
                            turnaround_time = max(turnaround_time, transit_time)

                            # 获取飞机维修信息
                            maint_check = False
                            post_maint_duration = 0
                            flag_in_maintenance_airport = False
                            successful_maintenance = False
                            if aircraft_info[aircraft_id]['Maint'] is None:
                                maint_check = True
                            else:
                                maint_info = aircraft_info[aircraft_id]['Maint']
                                maint_airport = maint_info['Maint_Airport']
                                maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
                                maint_time_range = int(maint_info['Maint_Time_Range'])
                                # 该航班在维修后起飞，维修约束已满足，飞行时长也必定满足
                                if maint_end < curr_check_flight['New_DepTime']:
                                    maint_check = True

                            # 获取飞机当前位置，以及飞机的最早起飞时间
                            last_dest = aircraft_info[aircraft_id]['Orig']
                            aircraft_available_time = aircraft_info[aircraft_id]['Available_time']
                            if aircraft_available_time is None:
                                aircraft_available_time = recovery_start_time
                            aircraft_earliest_dep_time = aircraft_available_time

                            find_flight = False
                            # # 不需要考虑维修情况，因为已经满足或没有维修计划
                            # if maint_check:
                            # 取消航班环标记
                            cancel_check = 0
                            # 遍历飞机的航班环序列
                            for check_group_id, check_group_info in temp_sorted_groups_for_aircraft.items():
                                for check_flight in check_group_info['Flights']:
                                    if check_flight['New_Flight_ID'] == flight_id:
                                        find_flight = True
                                        aircraft_earliest_dep_time = check_flight['New_ArrTime'] + timedelta(minutes=turnaround_time)
                                        # last_dest = check_flight['Orig']

                                    if check_flight['State'] != -1 and not maint_check:
                                        if check_flight['New_DepTime'] > maint_start:
                                            last_dest = check_flight['Orig']
                                        else:
                                            last_dest = check_flight['Dest']

                                    # 找到了需调整航班，调整其及后续航班，确保调整的航班未取消
                                    if find_flight and check_flight['State'] != -1:
                                        # 记录下当前航班的起降时刻，确认调整方案后，对时隙记录逐个调整
                                        orig_dep_time = copy.deepcopy(check_flight['New_DepTime'])
                                        orig_arr_time = copy.deepcopy(check_flight['New_ArrTime'])
                                        orig_dep_day_index = orig_dep_time.day - start_day
                                        orig_arr_day_index = orig_arr_time.day - start_day
                                        orig_dep_hour = orig_dep_time.hour
                                        orig_arr_hour = orig_arr_time.hour
                                        # 获取抵离港信息
                                        dep_day_index = check_flight['New_DepTime'].day - start_day
                                        arr_day_index = check_flight['New_ArrTime'].day - start_day
                                        dep_hour = check_flight['New_DepTime'].hour
                                        arr_hour = check_flight['New_ArrTime'].hour
                                        # 设置单次延误步长
                                        delay_step = 60
                                        delay_required = 0
                                        slot_check = False
                                        # 检查当前是否超出容量, 超出则延误航班至非拥挤时段，当前航班未取消，则可重复循环
                                        while cancel_check == 0:
                                            # 当前的起降时刻对应时隙容量非负
                                            if delay_required == 0:
                                                if airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Capacity'] >= 0 and airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Capacity'] >= 0:
                                                    slot_check = True
                                            # 当前的起降时刻对应时隙容量加入新航班后依旧非负
                                            else:
                                                if airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Capacity'] >= 1 and airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Capacity'] >= 1:
                                                    slot_check = True
                                            if not slot_check:
                                                delay_required = delay_step + check_flight['Delay']
                                                flight_delay = delay_required
                                                aircraft_delay = (aircraft_earliest_dep_time - check_flight['DepTime']).total_seconds() // 60
                                                delay_required = max(flight_delay, aircraft_delay)
                                                # 获取当前新的起降时刻
                                                check_flight['State'] = delay_required
                                                check_flight['Delay'] = delay_required
                                                check_flight['New_DepTime'] = check_flight['DepTime'] + timedelta(minutes=check_flight['Delay'])
                                                check_flight['New_ArrTime'] = check_flight['ArrTime'] + timedelta(minutes=check_flight['Delay'])
                                                # 检查维修情况
                                                if not maint_check:
                                                    dep_time = check_flight['New_DepTime']
                                                    arr_time = check_flight['New_ArrTime']
                                                    if dep_time <= maint_start <= arr_time:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    elif dep_time <= maint_end <= arr_time:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    elif maint_start <= dep_time <= maint_end:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    elif maint_start <= arr_time <= maint_end:
                                                        delay_required += (maint_end - dep_time).total_seconds() // 60
                                                    else:
                                                        if maint_start > arr_time:
                                                            last_dest = check_flight['Dest']
                                                        elif maint_end < dep_time:
                                                            if post_maint_duration + check_flight['Duration'] > maint_time_range:
                                                                cancel_check = 1
                                                                break
                                                            else:
                                                                post_maint_duration += check_flight['Duration']

                                                if check_flight['Type'] in ['D', 'C']:
                                                    max_delay = cancel_hour_dc * 60
                                                else:
                                                    max_delay = cancel_hour_i * 60

                                                # 若延误超过最大值，取消航班，后续航班也全部取消
                                                if delay_required > max_delay:
                                                    cancel_check = 1

                                                # 延误没有超过最大值，采用当前的延误继续检查
                                                else:
                                                    check_flight['State'] = delay_required
                                                    check_flight['Delay'] = delay_required
                                                    check_flight['New_DepTime'] = check_flight['DepTime'] + timedelta(minutes=check_flight['Delay'])
                                                    check_flight['New_ArrTime'] = check_flight['ArrTime'] + timedelta(minutes=check_flight['Delay'])
                                                    # 更新抵离港信息，加入下次检验
                                                    dep_day_index = check_flight['New_DepTime'].day - start_day
                                                    arr_day_index = check_flight['New_ArrTime'].day - start_day
                                                    dep_hour = check_flight['New_DepTime'].hour
                                                    arr_hour = check_flight['New_ArrTime'].hour

                                            # 航班延误已能满足时隙容量要求，对时隙进行更改，然后跳出循环
                                            else:
                                                # 原始时隙容量增加，航班移出原始时隙航班记录
                                                airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Capacity'] += 1
                                                airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Capacity'] += 1
                                                flight_temp_dep = airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Flight'].pop(check_flight['New_Flight_ID'])
                                                flight_temp_arr = airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Flight'].pop(check_flight['New_Flight_ID'])
                                                # 更新时隙容量减少，航班加入更新时隙航班记录
                                                airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Capacity'] -= 1
                                                airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Capacity'] -= 1
                                                airports_capacity_dic['Capacity_Record'][dep_day_index][check_flight['Orig']][dep_hour]['Dep_Flight'][check_flight['New_Flight_ID']] = {
                                                    **flight_temp_dep
                                                }
                                                airports_capacity_dic['Capacity_Record'][arr_day_index][check_flight['Dest']][arr_hour]['Arr_Flight'][check_flight['New_Flight_ID']] = {
                                                    **flight_temp_arr
                                                }
                                                break

                                        # 若需要取消航班，取消当前飞机的后续所有航班
                                        if cancel_check == 1:
                                            check_flight['State'] = -1
                                            check_flight['Delay'] = 0
                                            check_flight['New_DepTime'] = check_flight['DepTime'] + timedelta(minutes=check_flight['Delay'])
                                            check_flight['New_ArrTime'] = check_flight['ArrTime'] + timedelta(minutes=check_flight['Delay'])
                                            # 原始时隙容量增加，航班移出原始时隙航班记录
                                            airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Capacity'] += 1
                                            airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Capacity'] += 1
                                            airports_capacity_dic['Capacity_Record'][orig_dep_day_index][check_flight['Orig']][orig_dep_hour]['Dep_Flight'].pop(check_flight['New_Flight_ID'])
                                            airports_capacity_dic['Capacity_Record'][orig_arr_day_index][check_flight['Dest']][orig_arr_hour]['Arr_Flight'].pop(check_flight['New_Flight_ID'])


                            # # 需要考虑维修计划
                            # else:
                            #     continue

                            if not maint_check:
                                if last_dest == maint_airport:
                                    flag_in_maintenance_airport = True
                                    successful_maintenance = True
                                else:
                                    violation_1 += 1
                                if post_maint_duration <= maint_time_range:
                                    if successful_maintenance:
                                        maint_check = True
                                    else:
                                        maint_check = False
                                else:
                                    violation_2 += 1
                                if violation_1 + violation_2 == 0:
                                    maint_check = True

                            # 将变化更新到航班环
                            for check_group_id, check_group_info in temp_sorted_groups_for_aircraft.items():
                                for check_flight in check_group_info['Flights']:
                                    new_flights_id = check_flight['New_Flight_ID']
                                    flights_info[new_flights_id]['Aircraft'] = aircraft_id
                                    flights_info[new_flights_id]['State'] = check_flight['State']
                                    flights_info[new_flights_id]['Delay'] = check_flight['Delay']
                                    flights_info[new_flights_id]['Recovery_Flight'] = 1
                                    flights_info[new_flights_id]['New_DepTime'] = check_flight['New_DepTime']
                                    flights_info[new_flights_id]['New_ArrTime'] = check_flight['New_ArrTime']
                                    flights_info[new_flights_id]['DepTime'] = check_flight['DepTime']
                                    flights_info[new_flights_id]['ArrTime'] = check_flight['ArrTime']
                                #     flights_info[check_flight['New_Flight_ID']] =
                                flight_circles[check_group_id] = check_group_info

                            if not maint_check:
                                # 未成功维修，终止修复
                                repair_chance = False
                                break

                        # 确认是否修复时隙
                        new_overload_num = -1 * airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][check_tag]
                        if new_overload_num < overload_num:
                            violation -= (overload_num - new_overload_num)
                            overload_num = new_overload_num

                        flight_index += 1

                    if not repair_chance:
                        break

                if not repair_chance:
                    break

            if not repair_chance:
                break


            new_violation, temp_overload_info_dic = check_capacity(airports_capacity_dic)
            if new_violation != violation:
                violation = new_violation

    return violation_1, violation_2, violation, airports_capacity_dic, flight_circles, flights_info, overload_info_dic



def init_repair_airports_capacity(recovery_start_time, airports_capacity_total, group_lookup_dic, init_aircraft_info, flight_circles, flights_info, sort_str, cancel_hour_dc, cancel_hour_i):
    violation_1 = 0
    violation_2 = 0
    overload_info_dic = {}
    airports_capacity_dic = copy.deepcopy(airports_capacity_total)
    aircraft_info = copy.deepcopy(init_aircraft_info)
    flight_circles = copy.deepcopy(flight_circles)
    # print('Repair Start!!')

    # 若航班在恢复期前开始，后续航班数设置为50
    for flight_id, flight in flights_info.items():
        if flight['New_Flight_ID'] not in group_lookup_dic:
            flight['Left_num'] = 50

    # 对航班按照第一个flight的DepTime排序
    sorted_flight = dict(sorted(
        flights_info.items(),
        key=lambda item: item[1]['DepTime']
    ))

    # index = 0
    # day_change = 0
    base_day = airports_capacity_dic['StartTime'].day

    # 记录航班及机场时隙情况
    violation_ini = 0
    for flight_id, flight in sorted_flight.items():
        # start_day = flight['New_DepTime'].day
        # 记录容量变化情况
        if flight['State'] != -1:
            dep_day = flight['New_DepTime'].day
            dep_index = dep_day - base_day
            dep_time = flight['New_DepTime']
            dep_hour = dep_time.hour

            arr_day = flight['New_ArrTime'].day
            arr_index = arr_day - base_day
            arr_time = flight['New_ArrTime']
            arr_hour = arr_time.hour

            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Flight'][flight['New_Flight_ID']] = {
                'New_Flight_ID': flight['New_Flight_ID'],
                'Delay': flight['Delay'],
                'Left_num': flight['Left_num']
            }

            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Flight'][flight['New_Flight_ID']] = {
                'New_Flight_ID': flight['New_Flight_ID'],
                'Delay': flight['Delay'],
                'Left_num': flight['Left_num']
            }

            airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] -= 1
            airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] -= 1

            if airports_capacity_dic['Capacity_Record'][dep_index][flight['Orig']][dep_hour]['Dep_Capacity'] < 0:
                violation_ini += 1
                if flight['Orig'] not in overload_info_dic:
                    overload_info_dic[flight['Orig']] = {
                        'Total_Time': []
                    }
                if ('Dep', dep_day, dep_hour) not in overload_info_dic[flight['Orig']]['Total_Time']:
                    overload_info_dic[flight['Orig']]['Total_Time'].append(('Dep', dep_day, dep_hour))
                # overload_info_dic[flight['Orig']]['Dep_Time'].append((dep_day, dep_hour))

            if airports_capacity_dic['Capacity_Record'][arr_index][flight['Dest']][arr_hour]['Arr_Capacity'] < 0:
                violation_ini += 1
                if flight['Dest'] not in overload_info_dic:
                    overload_info_dic[flight['Dest']] = {
                        'Total_Time': []
                    }
                if ('Arr', arr_day, arr_hour) not in overload_info_dic[flight['Dest']]['Total_Time']:
                    overload_info_dic[flight['Dest']]['Total_Time'].append(('Arr', arr_day, arr_hour))

    # 遍历各个机场的约束违反情况，执行修复操作
    repair_times = 0
    repair_max = 20
    repair_max = violation_ini
    repair_chance = True
    if violation_ini > repair_max:
        violation, overload_info_dic = check_capacity(airports_capacity_dic)

    else:
        violation = violation_ini
        # 记录起始，防止无法满足要求
        if violation > 0:
            overload_info_list = list(overload_info_dic.keys())
            loop_start = overload_info_dic[overload_info_list[0]]['Total_Time'][0]
            begin_tag = False
        while(violation > 0):
            if not repair_chance:
                break
            for airport, overload_info in overload_info_dic.items():
                start_day = airports_capacity_dic['StartTime'].day
                if not repair_chance:
                    break
                # 先修复离场时隙约束
                for time_set in overload_info['Total_Time']:
                    if airport == overload_info_list[0] and time_set == loop_start:
                        if not begin_tag:
                            begin_tag = True
                        else:
                            repair_chance = False
                            break

                    repair_cap = time_set[0]
                    repair_day = time_set[1]
                    repair_hour = time_set[2]
                    if repair_cap == 'Dep':
                        check_tag = 'Dep_Capacity'
                    else:
                        check_tag = 'Arr_Capacity'
                    day_index = repair_day - start_day
                    # 获取需延误航班数量
                    overload_num = -1 * airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][check_tag]
                    flights_dic = airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][repair_cap + '_Flight']

                    flights_dic = dict(sorted(
                            flights_dic.items(),
                            key=lambda item: (item[1]['Left_num'])
                    ))
                    flights_list = list(flights_dic.keys())

                    flight_index = 0
                    while overload_num > 0:
                        # 已完整遍历可调整航班列表，无法满足修复需求
                        if flight_index >= len(flights_list):
                            repair_chance = False
                            break
                        # 获取需调整的航班信息
                        flight_id = flights_dic[flights_list[flight_index]]['New_Flight_ID']
                        curr_check_flight = flights_info[flight_id]
                        # 航班在恢复期前已出发，不能调整
                        if flight_id not in group_lookup_dic:
                            # 尝试调整其他航班
                            flight_index += 1
                            continue
                        # 航班在恢复期开始后出发，可以调整
                        else:
                            # 获取该航班对应的飞机周转、型号、维修信息
                            group_id = group_lookup_dic[flight_id]['Group_ID']
                            group_info = flight_circles[group_id]
                            aircraft_id = group_info['Aircraft']
                            groups_for_aircraft = query_group_by_aircraft(flight_circles, aircraft_id)
                            groups_for_aircraft = copy.deepcopy(groups_for_aircraft)
                            sort_str_adapt = sort_str + 'Time'

                            sorted_groups_for_aircraft = dict(sorted(
                                groups_for_aircraft.items(),
                                key=lambda item: (item[1]['Flights'][0][sort_str_adapt])
                            ))
                            # 获取该飞机的航班环
                            temp_sorted_groups_for_aircraft = copy.deepcopy(sorted_groups_for_aircraft)

                            # 获取型号相关信息
                            turnaround_time = aircraft_info[aircraft_id]['TurnRound']
                            transit_time = aircraft_info[aircraft_id]['Transit']
                            turnaround_time = max(turnaround_time, transit_time)

                            # 获取飞机维修信息
                            maint_check = False
                            post_maint_duration = 0
                            flag_in_maintenance_airport = False
                            successful_maintenance = False
                            if aircraft_info[aircraft_id]['Maint'] is None:
                                maint_check = True
                            else:
                                maint_info = aircraft_info[aircraft_id]['Maint']
                                maint_airport = maint_info['Maint_Airport']
                                maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
                                maint_time_range = int(maint_info['Maint_Time_Range'])
                                # 该航班在维修后起飞，维修约束已满足，飞行时长也必定满足
                                if maint_end < curr_check_flight['New_DepTime']:
                                    maint_check = True

                            # 获取飞机当前位置，以及飞机的最早起飞时间
                            last_dest = aircraft_info[aircraft_id]['Orig']
                            aircraft_available_time = aircraft_info[aircraft_id]['Available_time']
                            if aircraft_available_time is None:
                                aircraft_available_time = recovery_start_time
                            aircraft_earliest_dep_time = aircraft_available_time

                            find_flight = False
                            # # 不需要考虑维修情况，因为已经满足或没有维修计划
                            # if maint_check:
                            # 取消航班环标记
                            cancel_check = 0
                            # 遍历飞机的航班环序列
                            for check_group_id, check_group_info in temp_sorted_groups_for_aircraft.items():
                                for check_flight in check_group_info['Flights']:
                                    if check_flight['New_Flight_ID'] == flight_id:
                                        find_flight = True
                                        aircraft_earliest_dep_time = check_flight['New_ArrTime'] + timedelta(minutes=turnaround_time)
                                        # 飞机没有维修，取消航班环
                                        if aircraft_info[aircraft_id]['Maint'] is None:
                                            cancel_check = 1
                                        # 飞机有维修，判断航班环与维修的关系
                                        else:
                                            flight_circle_start = check_group_info['Flights'][0]['New_DepTime']
                                            flight_circle_end = check_group_info['Flights'][-1]['New_ArrTime']
                                            # 航班环在维修后开始，可以取消
                                            if flight_circle_start > maint_end:
                                                cancel_check = 1
                                            # 航班环在维修前结束
                                            elif flight_circle_end < maint_start:
                                                # 如果为航班环，可以取消
                                                if len(check_group_info['Flights']) == 2:
                                                    cancel_check = 1
                                                # 如果为单程航班，不可以取消
                                                else:
                                                    cancel_check = 0
                                            else:
                                                cancel_check = 0

                                        # 执行取消航班环操作
                                        if cancel_check == 1:
                                            for cancel_flight in check_group_info['Flights']:
                                                # 获取抵离港信息
                                                dep_day_index = cancel_flight['New_DepTime'].day - start_day
                                                arr_day_index = cancel_flight['New_ArrTime'].day - start_day
                                                dep_hour = cancel_flight['New_DepTime'].hour
                                                arr_hour = cancel_flight['New_ArrTime'].hour
                                                cancel_flight['State'] = -1
                                                cancel_flight['Delay'] = 0
                                                cancel_flight['New_DepTime'] = cancel_flight['DepTime'] + timedelta(minutes=cancel_flight['Delay'])
                                                cancel_flight['New_ArrTime'] = cancel_flight['ArrTime'] + timedelta(minutes=cancel_flight['Delay'])
                                                # 原始时隙容量增加，航班移出原始时隙航班记录
                                                airports_capacity_dic['Capacity_Record'][dep_day_index][cancel_flight['Orig']][dep_hour]['Dep_Capacity'] += 1
                                                airports_capacity_dic['Capacity_Record'][arr_day_index][cancel_flight['Dest']][arr_hour]['Arr_Capacity'] += 1
                                                airports_capacity_dic['Capacity_Record'][dep_day_index][cancel_flight['Orig']][dep_hour]['Dep_Flight'].pop(cancel_flight['New_Flight_ID'])
                                                airports_capacity_dic['Capacity_Record'][arr_day_index][cancel_flight['Dest']][arr_hour]['Arr_Flight'].pop(cancel_flight['New_Flight_ID'])

                            # 将变化更新到航班环
                            for check_group_id, check_group_info in temp_sorted_groups_for_aircraft.items():
                                for check_flight in check_group_info['Flights']:
                                    new_flights_id = check_flight['New_Flight_ID']
                                    flights_info[new_flights_id]['Aircraft'] = aircraft_id
                                    flights_info[new_flights_id]['State'] = check_flight['State']
                                    flights_info[new_flights_id]['Delay'] = check_flight['Delay']
                                    flights_info[new_flights_id]['Recovery_Flight'] = 1
                                    flights_info[new_flights_id]['New_DepTime'] = check_flight['New_DepTime']
                                    flights_info[new_flights_id]['New_ArrTime'] = check_flight['New_ArrTime']
                                    flights_info[new_flights_id]['DepTime'] = check_flight['DepTime']
                                    flights_info[new_flights_id]['ArrTime'] = check_flight['ArrTime']
                                flight_circles[check_group_id] = check_group_info

                        # 确认是否修复时隙
                        new_overload_num = -1 * airports_capacity_dic['Capacity_Record'][day_index][airport][repair_hour][check_tag]
                        if new_overload_num < overload_num:
                            violation -= (overload_num - new_overload_num)
                            overload_num = new_overload_num

                        flight_index += 1

                    if not repair_chance:
                        break

                if not repair_chance:
                    break

            if not repair_chance:
                break


            new_violation, temp_overload_info_dic = check_capacity(airports_capacity_dic)
            if new_violation != violation:
                print(violation, new_violation)
                violation = new_violation
            else:
                print(violation, new_violation)

    return violation_1, violation_2, violation, airports_capacity_dic, flight_circles, flights_info, overload_info_dic



def check_aircraft_requirements(config_info, aircraft_info, aircraft_position, position_info, aircraft_category):
    position_penalty = 0
    family_penalty = config_info['FamilyDismatchCost']
    model_penalty = config_info['ModelDismatchCost']
    config_penalty = config_info['ConfigDismatchCost']

    for airport, demand_list in position_info.items():
        # 创建字典来存储每个型号和配置的飞机数量,以及需求的数量
        available_aircraft_family = {}
        demand_aircraft_family = {}
        available_aircraft_model = {}
        demand_aircraft_model = {}
        available_aircraft_config = {}
        demand_aircraft_config = {}

        # 统计当前机场的飞机数量
        for aircraft_id in aircraft_position.get(airport, []):
            aircraft_data = aircraft_info.get(aircraft_id, {})
            model = aircraft_data.get("Model")
            for family_name in aircraft_category:
                if model in aircraft_category[family_name]:
                    family = family_name
            config = (aircraft_data['Cabin_Capacity'].get("F"),
                      aircraft_data['Cabin_Capacity'].get("B"),
                      aircraft_data['Cabin_Capacity'].get("E"))

            # 用 model 和 config 作为键
            # noinspection PyRedundantParentheses
            key_family = (family)
            key_model = (family, model)
            key_config = (family, model, config)

            if key_family in available_aircraft_family:
                available_aircraft_family[key_family] += 1
            else:
                available_aircraft_family[key_family] = 1

            if key_model in available_aircraft_model:
                available_aircraft_model[key_model] += 1
            else:
                available_aircraft_model[key_model] = 1

            if key_config in available_aircraft_config:
                available_aircraft_config[key_config] += 1
            else:
                available_aircraft_config[key_config] = 1

        # 遍历需求列表，统计不同需求
        for demand in demand_list:
            model = demand["Model"]
            for family_name in aircraft_category:
                if model in aircraft_category[family_name]:
                    family = family_name
            config = (demand["FirstCabin_Capacity"],
                      demand["BusinessCabin_Capacity"],
                      demand["EconomicCabin_Capacity"])
            # required_count = demand["Count"]
            # noinspection PyRedundantParentheses
            key_family = (family)
            key_model = (family, model)
            key_config = (family, model, config)

            if key_family in demand_aircraft_family:
                demand_aircraft_family[key_family] += 1
            else:
                demand_aircraft_family[key_family] = 1

            if key_model in demand_aircraft_model:
                demand_aircraft_model[key_model] += 1
            else:
                demand_aircraft_model[key_model] = 1

            if key_config in demand_aircraft_config:
                demand_aircraft_config[key_config] += 1
            else:
                demand_aircraft_config[key_config] = 1

        # 逐个检查family，model，config，若同时违反，惩罚叠加
        for key_family in demand_aircraft_family:
            family_demand_num = demand_aircraft_family.get(key_family, 0)
            family_available = available_aircraft_family.get(key_family, 0)
            family_displace = family_demand_num - family_available
            if family_displace > 0:
                position_penalty += family_displace * family_penalty
                # position_penalty += family_displace

        for key_model in demand_aircraft_model:
            model_demand_num = demand_aircraft_model.get(key_model, 0)
            model_available = available_aircraft_model.get(key_model, 0)
            model_displace = model_demand_num - model_available
            if model_displace > 0:
                position_penalty += model_displace * model_penalty
                # position_penalty += model_displace

        for key_config in demand_aircraft_config:
            config_demand_num = demand_aircraft_config.get(key_config, 0)
            config_available = available_aircraft_config.get(key_config, 0)
            config_displace = config_demand_num - config_available
            if config_displace > 0:
                position_penalty += config_displace * config_penalty
                # position_penalty += config_displace

    return position_penalty


# 评估该飞机执行航班序列，确保地点连续性/时间连续性
def evaluate_flight_schedule(recovery_start_time, recovery_end_time, aircraft, aircraft_dic,
                             sorted_processed_flights_info, cancel_hour_dc, cancel_hour_i, airport_capacity_total,
                             delay_situation, alt_flights_info, max_delay, num_groups, cancel_decision):
    # 初始化飞机状态记录
    successful_maintenance = False  # 是否能成功维修
    flag_in_maintenance_airport = False  # 维修时是否在计划维修机场
    flights_during_maintenance = 0  # 维修期间需执行航班数
    pre_maint_duration = 0  # 维修前飞行小时数
    post_maint_duration = 0  # 维修后飞行小时数
    aircraft_orig = aircraft_dic[aircraft]['Orig']
    turnround_time = aircraft_dic[aircraft]['TurnRound']
    transit_time = aircraft_dic[aircraft]['Transit']
    cost_per_hour = aircraft_dic[aircraft]['Cost_per_Hour']
    flight_range = aircraft_dic[aircraft]['Dist']

    # 查询飞机状态：可用时间，维修时间，维修地点
    aircraft_available_time = aircraft_dic[aircraft]['Available_time']
    if aircraft_available_time == None:
        aircraft_available_time = recovery_start_time

    maint_info = aircraft_dic[aircraft]['Maint']
    if maint_info is None:
        successful_maintenance = True
        flag_in_maintenance_airport = True
    else:
        maint_airport = maint_info['Maint_Airport']
        maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
        maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
        maint_time_range = int(maint_info['Maint_Time_Range'])

    # 初始化记录信息
    total_cost = 0
    violation_1 = 0  # 飞机维修时间/地点约束
    violation_2 = 0  # 飞行总时间约束
    violation_4 = 0  # 飞机航程约束
    violation = {
        'Flight_range': [],
        'Flying_time': [],
        'Maint': True
    } # 存储违反约束的信息

    # 遍历航班环，根据染色体、扰动取消、延误航班
    for group_id, group_info in sorted_processed_flights_info.items():
        index = group_info['Index']
        # 查看染色体中航班状态是否为取消
        chrom_situation = delay_situation[index]
        if cancel_decision == 0:
            if chrom_situation == max_delay[num_groups + index]:
                for flight in group_info['Flights']:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
            else:
                flight = group_info['Flights'][0]
                flight['State'] = chrom_situation
                flight['Delay'] = chrom_situation
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))

        elif cancel_decision == 1:
            if delay_situation[index] == -1:
                for flight in group_info['Flights']:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
            else:
                flight = group_info['Flights'][0]
                flight['State'] = chrom_situation
                flight['Delay'] = chrom_situation
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))

        # 引入航班受扰动情况
        alt_cancel = 0
        for flight in group_info['Flights']:
            # 确认是否位于扰动航班列表
            if flight['Flight_ID'] in alt_flights_info.keys() and flight['DepDate'] == alt_flights_info[flight['Flight_ID']]['DepDate']:  # 航班受到扰动
                if alt_flights_info[flight['Flight_ID']]['Delay'] == -1:
                    alt_cancel = 1
                else:
                    if flight['State'] != -1 and flight['Delay'] < alt_flights_info[flight['Flight_ID']]['Delay']:
                        flight['State'] = alt_flights_info[flight['Flight_ID']]['Delay']
                        flight['Delay'] = alt_flights_info[flight['Flight_ID']]['Delay']
                        flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                        flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
        if alt_cancel == 1:
            for flight in group_info['Flights']:
                flight['State'] = -1
                flight['Delay'] = 0
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))

    # 获取初始航班的最早起飞时刻
    aircraft_earliest_dep_time = aircraft_available_time
    aircraft_position = aircraft_orig
    last_dest = aircraft_orig

    # # 飞机没有维修
    # if maint_info is None:
    for group_id, group_info in sorted_processed_flights_info.items():
        for flight in group_info['Flights']:
            # 只看未取消航班
            if flight['State'] != -1:
                flight_orig = flight['Orig']
                # 飞机位于航班的起飞机场
                if flight_orig == aircraft_position:
                    flight_delay = (flight['New_DepTime'] - flight['DepTime']).total_seconds() // 60
                    aircraft_delay = (aircraft_earliest_dep_time - flight['DepTime']).total_seconds() // 60
                    delay_needed = max(flight_delay, aircraft_delay)
                    if flight['Type'] in ['D', 'C']:
                        max_delay = cancel_hour_dc * 60
                    else:
                        max_delay = cancel_hour_i * 60

                    if delay_needed > max_delay:
                        # 超出最大延误时间，取消航班
                        flight['State'] = -1
                        flight['Delay'] = 0
                        flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                        flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
                    else:
                        flight['State'] = delay_needed
                        flight['Delay'] = delay_needed
                        flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=delay_needed)
                        flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=delay_needed)
                        # 更新飞机的可用时间
                        aircraft_earliest_dep_time = flight['New_ArrTime'] + timedelta(minutes=max(turnround_time, transit_time))
                        # 更新飞机的位置
                        aircraft_position = flight['Dest']
                        if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                            last_dest = flight['Dest']
                        else:
                            if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                                last_dest = 'Flying'

                # 飞机未位于航班的起飞机场，此航班取消
                else:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))


    return total_cost, last_dest, sorted_processed_flights_info


# 修复该飞机执行航班序列，确保地点连续性/时间连续性/航程/尽量保证维修计划/飞行时长
def repair_flight_schedule(recovery_start_time, recovery_end_time, aircraft, aircraft_dic,
                             sorted_processed_flights_info, cancel_hour_dc, cancel_hour_i, airport_capacity_total,
                             delay_situation, alt_flights_info, max_delay, num_groups, cancel_decision):
    # 初始化飞机状态记录
    successful_maintenance = False  # 是否能成功维修
    flag_in_maintenance_airport = False  # 维修时是否在计划维修机场
    flights_during_maintenance = 0  # 维修期间需执行航班数
    pre_maint_duration = 0  # 维修前飞行小时数
    post_maint_duration = 0  # 维修后飞行小时数
    aircraft_orig = aircraft_dic[aircraft]['Orig']
    turnround_time = aircraft_dic[aircraft]['TurnRound']
    transit_time = aircraft_dic[aircraft]['Transit']
    cost_per_hour = aircraft_dic[aircraft]['Cost_per_Hour']
    flight_range = aircraft_dic[aircraft]['Dist']

    # 查询飞机状态：可用时间，维修时间，维修地点
    aircraft_available_time = aircraft_dic[aircraft]['Available_time']
    if aircraft_available_time == None:
        aircraft_available_time = recovery_start_time

    maint_info = aircraft_dic[aircraft]['Maint']
    if maint_info is None:
        successful_maintenance = True
        flag_in_maintenance_airport = True
    else:
        maint_airport = maint_info['Maint_Airport']
        maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
        maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
        maint_time_range = int(maint_info['Maint_Time_Range'])

    # 初始化记录信息
    total_cost = 0
    violation_1 = 0  # 飞机维修时间/地点约束
    violation_2 = 0  # 飞行总时间约束
    violation_4 = 0  # 飞机航程约束
    violation = {
        'Flight_range': [],
        'Flying_time': [],
        'Maint': True
    } # 存储违反约束的信息

    # 遍历航班环，根据染色体、扰动取消、延误航班
    for group_id, group_info in sorted_processed_flights_info.items():
        index = group_info['Index']
        # 查看染色体中航班状态是否为取消
        chrom_situation = delay_situation[index]
        if cancel_decision == 0:
            if chrom_situation == max_delay[num_groups + index]:
                for flight in group_info['Flights']:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
            else:
                flight = group_info['Flights'][0]
                flight['State'] = chrom_situation
                flight['Delay'] = chrom_situation
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))

        elif cancel_decision == 1:
            if delay_situation[index] == -1:
                for flight in group_info['Flights']:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
            else:
                flight = group_info['Flights'][0]
                flight['State'] = chrom_situation
                flight['Delay'] = chrom_situation
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))

        # 引入航班受扰动情况
        alt_cancel = 0
        for flight in group_info['Flights']:
            # 确认是否位于扰动航班列表
            if flight['Flight_ID'] in alt_flights_info.keys() and flight['DepDate'] == alt_flights_info[flight['Flight_ID']]['DepDate']:  # 航班受到扰动
                if alt_flights_info[flight['Flight_ID']]['Delay'] == -1:
                    alt_cancel = 1
                else:
                    if flight['State'] != -1 and flight['Delay'] < alt_flights_info[flight['Flight_ID']]['Delay']:
                        flight['State'] = alt_flights_info[flight['Flight_ID']]['Delay']
                        flight['Delay'] = alt_flights_info[flight['Flight_ID']]['Delay']
                        flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                        flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
        if alt_cancel == 1:
            for flight in group_info['Flights']:
                flight['State'] = -1
                flight['Delay'] = 0
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))

    # 获取初始航班的最早起飞时刻
    aircraft_earliest_dep_time = aircraft_available_time
    aircraft_position = aircraft_orig
    last_dest = aircraft_orig

    # 飞机没有维修
    if maint_info is None:
        for group_id, group_info in sorted_processed_flights_info.items():
            alt_cancel = 0
            for flight in group_info['Flights']:
                # 只看未取消航班
                if flight['State'] != -1:
                    # 检查飞机航程，若不满足航程，取消航班环
                    if flight['Duration'] > flight_range:
                        alt_cancel = 1
                        break
                    flight_orig = flight['Orig']
                    # 飞机位于航班的起飞机场
                    if flight_orig == aircraft_position:
                        flight_delay = (flight['New_DepTime'] - flight['DepTime']).total_seconds() // 60
                        aircraft_delay = (aircraft_earliest_dep_time - flight['DepTime']).total_seconds() // 60
                        delay_needed = max(flight_delay, aircraft_delay)
                        if flight['Type'] in ['D', 'C']:
                            max_delay = cancel_hour_dc * 60
                        else:
                            max_delay = cancel_hour_i * 60

                        if delay_needed > max_delay:
                            # 超出最大延误时间，取消航班
                            alt_cancel = 1
                            break
                        else:
                            flight['State'] = delay_needed
                            flight['Delay'] = delay_needed
                            flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=delay_needed)
                            flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=delay_needed)
                            # 更新飞机的可用时间
                            aircraft_earliest_dep_time = flight['New_ArrTime'] + timedelta(minutes=max(turnround_time, transit_time))
                            # 更新飞机的位置
                            aircraft_position = flight['Dest']
                            if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                                last_dest = flight['Dest']
                            else:
                                if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                                    last_dest = 'Flying'

                    # 飞机未位于航班的起飞机场，此航班取消
                    else:
                        alt_cancel = 1
                        break

            if alt_cancel == 1:
                for flight in group_info['Flights']:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))


    # 飞机有维修
    else:
        # 存储航班环列表
        group_id_list = list(sorted_processed_flights_info.keys())
        # 维修时间在恢复期前
        if maint_start < recovery_start_time:
            maint_flag = True
            find_maint_airport = True
        else:
            maint_flag = False
            if aircraft_orig == maint_airport:
                find_maint_airport = True
                maint_airport_tag = 'Orig'
            else:
                find_maint_airport = False

        for group_id, group_info in sorted_processed_flights_info.items():
            alt_cancel = 0
            for flight in group_info['Flights']:
                # 只看未取消航班
                if flight['State'] != -1:
                    # 检查飞机航程，若不满足航程，取消航班环
                    if flight['Duration'] > flight_range:
                        alt_cancel = 1
                        break
                    flight_orig = flight['Orig']
                    # 飞机位于航班的起飞机场
                    if flight_orig == aircraft_position:
                        flight_delay = (flight['New_DepTime'] - flight['DepTime']).total_seconds() // 60
                        aircraft_delay = (aircraft_earliest_dep_time - flight['DepTime']).total_seconds() // 60
                        delay_needed = max(flight_delay, aircraft_delay)
                        if flight['Type'] in ['D', 'C']:
                            max_delay = cancel_hour_dc * 60
                        else:
                            max_delay = cancel_hour_i * 60

                        if delay_needed > max_delay:
                            # 超出最大延误时间，取消航班
                            alt_cancel = 1
                            break
                        else:
                            flight['State'] = delay_needed
                            flight['Delay'] = delay_needed
                            flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=delay_needed)
                            flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=delay_needed)
                            # 检查维修情况
                            dep_time = flight['New_DepTime']
                            arr_time = flight['New_ArrTime']
                            if dep_time <= maint_start <= arr_time:
                                alt_cancel = 1
                                break
                            elif dep_time <= maint_end <= arr_time:
                                alt_cancel = 1
                                break
                            elif maint_start <= dep_time <= maint_end:
                                alt_cancel = 1
                                break
                            elif maint_start <= arr_time <= maint_end:
                                alt_cancel = 1
                                break
                            else:
                                # 记录并检查维修前的飞行时长
                                if maint_start > arr_time:
                                    if (pre_maint_duration + flight['Duration']) > maint_time_range:
                                        alt_cancel = 1
                                        break
                                    else:
                                        pre_maint_duration += flight['Duration']

                                # 应该已完成维修
                                elif maint_end < dep_time:
                                    # 维修尚未完成
                                    if not maint_flag:
                                        if aircraft_position == maint_airport:
                                            flag_in_maintenance_airport = True
                                            maint_flag = True
                                            if (post_maint_duration + flight['Duration']) > maint_time_range:
                                                if flight['New_Flight_ID'] == group_info['Flights'][0]['New_Flight_ID']:
                                                    alt_cancel = 1
                                                    break
                                                else:
                                                    flight['State'] = -1
                                                    flight['Delay'] = 0
                                                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                                                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))
                                                    # alt_cancel = 1
                                                    break
                                            else:
                                                post_maint_duration += flight['Duration']
                                        else:
                                            if find_maint_airport:
                                                maint_flag = True
                                                # 飞机只有起始时位于维修机场，取消本航班及之前的所有航班
                                                if maint_airport_tag == 'Orig':
                                                    # 取消本航班环
                                                    alt_cancel = 1
                                                    # 取消本航班环之前的航班环
                                                    for cancel_group_id in group_id_list:
                                                        if cancel_group_id != group_id:
                                                            for cancel_flight in sorted_processed_flights_info[cancel_group_id]['Flights']:
                                                                cancel_flight['State'] = -1
                                                                cancel_flight['Delay'] = 0
                                                                cancel_flight['New_DepTime'] = cancel_flight['DepTime'] + timedelta(minutes=int(cancel_flight['Delay']))
                                                                cancel_flight['New_ArrTime'] = cancel_flight['ArrTime'] + timedelta(minutes=int(cancel_flight['Delay']))
                                                        else:
                                                            break
                                                    last_dest = aircraft_orig
                                                    aircraft_position = aircraft_orig
                                                    break
                                                # 找到使飞机位于维修机场的航班环，取消本航班与其之间的所有航班
                                                else:
                                                    # 取消本航班环
                                                    alt_cancel = 1
                                                    # 取消本航班环与其之间的所有航班
                                                    airport_find_tag = False
                                                    last_dest = aircraft_orig
                                                    for cancel_group_id in group_id_list:
                                                        if cancel_group_id == maint_airport_tag:
                                                            for cancel_flight in sorted_processed_flights_info[cancel_group_id]['Flights']:
                                                                # 找到使飞机位于维修机场的航班环
                                                                if cancel_flight['Dest'] == maint_airport:
                                                                    airport_find_tag = True
                                                                    # 找到使飞机位于维修机场的航班环，判断取消系列航班后飞机的位置
                                                                    if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                                                                        last_dest = flight['Dest']
                                                                    else:
                                                                        if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                                                                            last_dest = 'Flying'

                                                                    continue
                                                                # 找到使飞机位于维修机场的航班环之后，取消后续的航班环
                                                                if airport_find_tag:
                                                                    cancel_flight['State'] = -1
                                                                    cancel_flight['Delay'] = 0
                                                                    cancel_flight['New_DepTime'] = cancel_flight['DepTime'] + timedelta(minutes=int(cancel_flight['Delay']))
                                                                    cancel_flight['New_ArrTime'] = cancel_flight['ArrTime'] + timedelta(minutes=int(cancel_flight['Delay']))
                                                                # 若未找到，更新飞机位置
                                                                else:
                                                                    if cancel_flight['State'] != -1:
                                                                        if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                                                                            last_dest = flight['Dest']
                                                                        else:
                                                                            if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                                                                                last_dest = 'Flying'

                                                        if cancel_group_id != group_id:
                                                            if airport_find_tag:
                                                                for cancel_flight in sorted_processed_flights_info[cancel_group_id]['Flights']:
                                                                    cancel_flight['State'] = -1
                                                                    cancel_flight['Delay'] = 0
                                                                    cancel_flight['New_DepTime'] = cancel_flight['DepTime'] + timedelta(minutes=int(cancel_flight['Delay']))
                                                                    cancel_flight['New_ArrTime'] = cancel_flight['ArrTime'] + timedelta(minutes=int(cancel_flight['Delay']))
                                                            else:
                                                                for cancel_flight in sorted_processed_flights_info[cancel_group_id]['Flights']:
                                                                    if cancel_flight['State'] != -1:
                                                                        if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                                                                            last_dest = flight['Dest']
                                                                        else:
                                                                            if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                                                                                last_dest = 'Flying'
                                                        else:
                                                            break
                                                    aircraft_position = maint_airport
                                                    break

                                            # 无法完成维修，后续检验可发现
                                            else:
                                                maint_flag = True
                                                if (post_maint_duration + flight['Duration']) > maint_time_range:
                                                    alt_cancel = 1
                                                    break
                                                else:
                                                    post_maint_duration += flight['Duration']

                                    # 维修为恢复期前开始，无法控制维修场地，需后续检验
                                    else:
                                        if (post_maint_duration + flight['Duration']) > maint_time_range:
                                            alt_cancel = 1
                                            break
                                        else:
                                            post_maint_duration += flight['Duration']

                            # 更新飞机的可用时间
                            aircraft_earliest_dep_time = flight['New_ArrTime'] + timedelta(minutes=max(turnround_time, transit_time))
                            # 更新飞机的位置
                            aircraft_position = flight['Dest']
                            if aircraft_position == maint_airport:
                                find_maint_airport = True
                                maint_airport_tag = group_id
                            if (recovery_end_time - flight['New_ArrTime']).total_seconds() > 0:
                                last_dest = flight['Dest']
                            else:
                                if (recovery_end_time - flight['New_DepTime']).total_seconds() > 0:
                                    last_dest = 'Flying'


                    # 飞机未位于航班的起飞机场，此航班取消
                    else:
                        alt_cancel = 1
                        break

            if alt_cancel == 1:
                for flight in group_info['Flights']:
                    flight['State'] = -1
                    flight['Delay'] = 0
                    flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=int(flight['Delay']))
                    flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=int(flight['Delay']))


    return total_cost, last_dest, sorted_processed_flights_info


def evaluation(Vars, config, recovery_start_time, recovery_end_time, num_groups, alt_airports_info, alt_flights_info,
               cancel_hour_dc, cancel_hour_i, sort_str, flight_circles_info, init_aircraft_assignment, aircraft_info,
               airports_info, airports_capacity_total, flights_info, group_lookup_dic, position_info, aircraft_category,
               cancel_decision, canceled_flights_swap, max_delay, init_cancel_flight_num, evaluate_tag):

    solution = {}
    fitness_1 = 0
    fitness_2 = 0
    cost1_1 = 0
    constraint_violation_1 = 0
    constraint_violation_2 = 0
    constraint_violation_3 = 0
    constraint_violation_4 = 0
    constraint_violation_5 = 0

    aircraft_assignment = Vars[:num_groups]
    delay_situation = Vars[num_groups:]
    flight_circles = copy.deepcopy(flight_circles_info)

    # 按飞机顺序搜索航班环
    aircraft_position = {}
    violation_dic = {
        'Flight_range': [],
        'Flying_time': [],
        'Maint': []
    }  # 存储违反约束的信息

    # 对照染色体改变飞机分配
    for group_index, circle in flight_circles.items():
        index = circle['Index']
        if aircraft_assignment[index] != init_aircraft_assignment[index]:
            target_id = aircraft_assignment[index]
            for aircraft, info in aircraft_info.items():
                if info['Num_ID'] == target_id:
                    matching_aircraft = aircraft
                    circle['Aircraft'] = matching_aircraft
                    break

    # 获取飞机对应航班序列并进行调整评估
    for aircraft, info in aircraft_info.items():
        # 汇合某架飞机的所有待执行航班环
        if aircraft[:9] == 'TranspCom':
            continue
        aircraft_id = aircraft
        groups_for_aircraft = query_group_by_aircraft(flight_circles, aircraft_id)

        sort_str_adapt = sort_str + 'Time'

        # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
        sorted_groups_for_aircraft = dict(sorted(
            groups_for_aircraft.items(),
            key=lambda item: (item[1]['Flights'][0][sort_str_adapt])
        ))

        if evaluate_tag == 'evaluate':
            # 评估该飞机执行航班序列，确保地点连续性/时间连续性
            cost, last_dest, aircraft_groups = repair_flight_schedule(recovery_start_time, recovery_end_time,
                                                                        aircraft, aircraft_info,
                                                                        sorted_groups_for_aircraft, cancel_hour_dc,
                                                                        cancel_hour_i, airports_capacity_total,
                                                                        delay_situation, alt_flights_info, max_delay,
                                                                        num_groups, cancel_decision)


        elif evaluate_tag == 'repair':
            # if aircraft == 'A330#609':
            #     print('Debug')
            # 修复该飞机执行航班序列，确保地点连续性/时间连续性/航程/尽量保证维修计划/飞行时长
            cost, last_dest, aircraft_groups = repair_flight_schedule(recovery_start_time, recovery_end_time,
                                                                        aircraft, aircraft_info,
                                                                        sorted_groups_for_aircraft, cancel_hour_dc,
                                                                        cancel_hour_i, airports_capacity_total,
                                                                        delay_situation, alt_flights_info, max_delay,
                                                                        num_groups, cancel_decision)

        elif evaluate_tag == 'init_re_repair':
            # if aircraft == 'A330#609':
            #     print('Debug')
            # 修复该飞机执行航班序列，确保地点连续性/时间连续性/航程/尽量保证维修计划/飞行时长
            cost, last_dest, aircraft_groups = repair_flight_schedule(recovery_start_time, recovery_end_time,
                                                                        aircraft, aircraft_info,
                                                                        sorted_groups_for_aircraft, cancel_hour_dc,
                                                                        cancel_hour_i, airports_capacity_total,
                                                                        delay_situation, alt_flights_info, max_delay,
                                                                        num_groups, cancel_decision)

        elif evaluate_tag == 'init_repair':
            # 评估该飞机执行航班序列，确保地点连续性/时间连续性
            cost, last_dest, aircraft_groups = repair_flight_schedule(recovery_start_time, recovery_end_time,
                                                                        aircraft, aircraft_info,
                                                                        sorted_groups_for_aircraft, cancel_hour_dc,
                                                                        cancel_hour_i, airports_capacity_total,
                                                                        delay_situation, alt_flights_info, max_delay,
                                                                        num_groups, cancel_decision)

        # 将恢复过程中的变化更新到数据中
        for group_id, group_info in aircraft_groups.items():
            flight_circles[group_id] = group_info


    # 计算/修复时隙约束
    if evaluate_tag == 'evaluate':
        # 整合所有飞机的航班信息，同时计算航程约束-constraint_violation_4
        for group_id, group_info in flight_circles.items():
            aircraft = group_info['Aircraft']
            # total_flights = aircraft_info[aircraft]['Flights_num']
            for flight in group_info['Flights']:
                new_flights_id = flight['New_Flight_ID']
                flights_info[new_flights_id]['Aircraft'] = aircraft
                flights_info[new_flights_id]['State'] = flight['State']
                flights_info[new_flights_id]['Delay'] = flight['Delay']
                flights_info[new_flights_id]['Recovery_Flight'] = 1
                flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                if flight['State'] != -1:
                    # 计算航程约束
                    if flight['Duration'] > aircraft_info[aircraft]['Dist']:
                        constraint_violation_4 += 1
                        violation_dic['Flight_range'].append(new_flights_id)

        # 计算维修时间/地点、飞行时长约束
        for aircraft, info in aircraft_info.items():
            maint_info = info['Maint']
            if maint_info is None:
                successful_maintenance = True
                flag_in_maintenance_airport = True
            else:
                maint_airport = maint_info['Maint_Airport']
                maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
                maint_time_range = int(maint_info['Maint_Time_Range'])
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                # 检查维修时间/地点、飞行时长约束
                successful_maintenance, over_duration_count = check_maintenance(aircraft, aircraft_info, sorted_flights_for_aircraft, maint_airport, maint_start, maint_end, maint_time_range)
                if not successful_maintenance:
                    constraint_violation_1 += 1  # 维修时间/地点约束
                constraint_violation_2 += over_duration_count  # 飞行时长约束

        for group_id, group_info in flight_circles.items():
            for flight in group_info['Flights']:
                flight['State'] = flights_info[flight['New_Flight_ID']]['State']
                flight['Delay'] = flights_info[flight['New_Flight_ID']]['Delay']
                flight['New_DepTime'] = flights_info[flight['New_Flight_ID']]['New_DepTime']
                flight['New_ArrTime'] = flights_info[flight['New_Flight_ID']]['New_ArrTime']
                flight['DepTime'] = flights_info[flight['New_Flight_ID']]['DepTime']
                flight['ArrTime'] = flights_info[flight['New_Flight_ID']]['ArrTime']

        violation_3, flights_info, airports_capacity_dic = check_airport_capacity(airports_capacity_total, airports_info, flight_circles, flights_info, alt_airports_info)
        constraint_violation_3 += violation_3

        # 整合航班具体信息
        for group_id, group_info in flight_circles.items():
            aircraft = group_info['Aircraft']
            # total_flights = aircraft_info[aircraft]['Flights_num']
            for flight in group_info['Flights']:
                new_flights_id = flight['New_Flight_ID']
                flights_info[new_flights_id]['Aircraft'] = aircraft
                flights_info[new_flights_id]['State'] = flight['State']
                flights_info[new_flights_id]['Delay'] = flight['Delay']
                flights_info[new_flights_id]['Recovery_Flight'] = 1
                flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']

    elif evaluate_tag == 'repair':
        # 整合航班具体信息, 并统计航程违反情况
        for group_id, group_info in flight_circles.items():
            aircraft = group_info['Aircraft']
            # total_flights = aircraft_info[aircraft]['Flights_num']
            for flight in group_info['Flights']:
                new_flights_id = flight['New_Flight_ID']
                flights_info[new_flights_id]['Aircraft'] = aircraft
                flights_info[new_flights_id]['State'] = flight['State']
                flights_info[new_flights_id]['Delay'] = flight['Delay']
                flights_info[new_flights_id]['Recovery_Flight'] = 1
                flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                # flights_info[new_flights_id]['Left_num'] = total_flights - flights_info[new_flights_id]['Sequence']
                if flight['State'] != -1:
                    # 计算航程约束
                    if flight['Duration'] > aircraft_info[aircraft]['Dist']:
                        constraint_violation_4 += 1
                        violation_dic['Flight_range'].append(new_flights_id)

        # 修复维修时间/地点、飞行时长约束
        for aircraft, info in aircraft_info.items():
            # 记录维修情况
            maint_info = info['Maint']
            # 记录每个航班的后续航班情况
            flights_num_total = 0
            canceled_flights_num = 0
            flights_num = 0
            total_flights_num = 0
            flights_state = []

            if maint_info is None:
                successful_maintenance = True
                flag_in_maintenance_airport = True
                # 记录后续航班情况
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    if flight['State'] == -1:
                        canceled_flights_num += 1
                        flights_state.append(0)
                    else:
                        flights_num += 1
                        total_flights_num += 1
                        flights_state.append(1)
                    flights_num_total += 1
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Sequence'] = flights_num
                info['Flights_num'] = total_flights_num

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Left_num'] = total_flights_num - flights_info[new_flights_id]['Sequence']

            else:
                total_flights_num = 50
                maint_airport = maint_info['Maint_Airport']
                maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
                maint_time_range = int(maint_info['Maint_Time_Range'])
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                # 检查维修时间/地点、飞行时长约束
                successful_maintenance, over_duration_count, sorted_flights_for_aircraft = repair_maintenance(aircraft, aircraft_info, flight_circles, sorted_flights_for_aircraft, group_lookup_dic, maint_airport, maint_start, maint_end, maint_time_range)
                if not successful_maintenance:
                    constraint_violation_1 += 1  # 维修时间/地点约束
                constraint_violation_2 += over_duration_count  # 飞行时长约束

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    if flight['State'] == -1:
                        canceled_flights_num += 1
                        flights_state.append(0)
                    else:
                        flights_num += 1
                        total_flights_num += 1
                        flights_state.append(1)
                    flights_num_total += 1
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Sequence'] = flights_num

                info['Flights_num'] = total_flights_num

                # 信息更新到flights_info中
                for flight_id, flight in sorted_flights_for_aircraft.items():
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Aircraft'] = aircraft
                    flights_info[new_flights_id]['State'] = flight['State']
                    flights_info[new_flights_id]['Delay'] = flight['Delay']
                    flights_info[new_flights_id]['Recovery_Flight'] = 1
                    flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                    flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                    flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                    flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                    flights_info[new_flights_id]['Left_num'] = total_flights_num - flights_info[new_flights_id]['Sequence']

        # print(constraint_violation_1, constraint_violation_2)
        for group_id, group_info in flight_circles.items():
            for flight in group_info['Flights']:
                flight['State'] = flights_info[flight['New_Flight_ID']]['State']
                flight['Delay'] = flights_info[flight['New_Flight_ID']]['Delay']
                flight['New_DepTime'] = flights_info[flight['New_Flight_ID']]['New_DepTime']
                flight['New_ArrTime'] = flights_info[flight['New_Flight_ID']]['New_ArrTime']
                flight['DepTime'] = flights_info[flight['New_Flight_ID']]['DepTime']
                flight['ArrTime'] = flights_info[flight['New_Flight_ID']]['ArrTime']

        if constraint_violation_1 == 0 and constraint_violation_2 == 0:

            # 修复时隙容量约束、飞机维修约束及飞行时长约束
            violation_1, violation_2, violation_3, airports_capacity_dic, flight_circles, flights_info, overload_info_dic = repair_airports_capacity(recovery_start_time, airports_capacity_total, group_lookup_dic, aircraft_info, flight_circles, flights_info, sort_str, cancel_hour_dc, cancel_hour_i)
            constraint_violation_1 += violation_1
            constraint_violation_2 += violation_2
            constraint_violation_3 += violation_3

        else:
            violation_3, flights_info, airports_capacity_dic = check_airport_capacity(airports_capacity_total, airports_info, flight_circles, flights_info, alt_airports_info)
            constraint_violation_3 += violation_3


    elif evaluate_tag == 'init_re_repair':
        # 整合航班具体信息, 并统计航程违反情况
        for group_id, group_info in flight_circles.items():
            aircraft = group_info['Aircraft']
            # total_flights = aircraft_info[aircraft]['Flights_num']
            for flight in group_info['Flights']:
                new_flights_id = flight['New_Flight_ID']
                flights_info[new_flights_id]['Aircraft'] = aircraft
                flights_info[new_flights_id]['State'] = flight['State']
                flights_info[new_flights_id]['Delay'] = flight['Delay']
                flights_info[new_flights_id]['Recovery_Flight'] = 1
                flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                # flights_info[new_flights_id]['Left_num'] = total_flights - flights_info[new_flights_id]['Sequence']
                if flight['State'] != -1:
                    # 计算航程约束
                    if flight['Duration'] > aircraft_info[aircraft]['Dist']:
                        constraint_violation_4 += 1
                        violation_dic['Flight_range'].append(new_flights_id)

        # 修复维修时间/地点、飞行时长约束
        for aircraft, info in aircraft_info.items():
            # 记录维修情况
            maint_info = info['Maint']
            # 记录每个航班的后续航班情况
            flights_num_total = 0
            canceled_flights_num = 0
            flights_num = 0
            total_flights_num = 0
            flights_state = []

            if maint_info is None:
                successful_maintenance = True
                flag_in_maintenance_airport = True
                # 记录后续航班情况
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    if flight['State'] == -1:
                        canceled_flights_num += 1
                        flights_state.append(0)
                    else:
                        flights_num += 1
                        total_flights_num += 1
                        flights_state.append(1)
                    flights_num_total += 1
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Sequence'] = flights_num
                info['Flights_num'] = total_flights_num

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Left_num'] = total_flights_num - flights_info[new_flights_id]['Sequence']

            else:
                total_flights_num = 50
                maint_airport = maint_info['Maint_Airport']
                maint_start = datetime.strptime(maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'], '%d/%m/%y %H:%M')
                maint_time_range = int(maint_info['Maint_Time_Range'])
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                # 检查维修时间/地点、飞行时长约束
                successful_maintenance, over_duration_count, sorted_flights_for_aircraft = repair_maintenance(aircraft, aircraft_info, flight_circles, sorted_flights_for_aircraft, group_lookup_dic, maint_airport, maint_start, maint_end, maint_time_range)
                if not successful_maintenance:
                    constraint_violation_1 += 1  # 维修时间/地点约束
                constraint_violation_2 += over_duration_count  # 飞行时长约束

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    if flight['State'] == -1:
                        canceled_flights_num += 1
                        flights_state.append(0)
                    else:
                        flights_num += 1
                        total_flights_num += 1
                        flights_state.append(1)
                    flights_num_total += 1
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Sequence'] = flights_num

                info['Flights_num'] = total_flights_num

                # 信息更新到flights_info中
                for flight_id, flight in sorted_flights_for_aircraft.items():
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Aircraft'] = aircraft
                    flights_info[new_flights_id]['State'] = flight['State']
                    flights_info[new_flights_id]['Delay'] = flight['Delay']
                    flights_info[new_flights_id]['Recovery_Flight'] = 1
                    flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                    flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                    flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                    flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                    flights_info[new_flights_id]['Left_num'] = total_flights_num - flights_info[new_flights_id]['Sequence']

        # print(constraint_violation_1, constraint_violation_2)
        for group_id, group_info in flight_circles.items():
            for flight in group_info['Flights']:
                flight['State'] = flights_info[flight['New_Flight_ID']]['State']
                flight['Delay'] = flights_info[flight['New_Flight_ID']]['Delay']
                flight['New_DepTime'] = flights_info[flight['New_Flight_ID']]['New_DepTime']
                flight['New_ArrTime'] = flights_info[flight['New_Flight_ID']]['New_ArrTime']
                flight['DepTime'] = flights_info[flight['New_Flight_ID']]['DepTime']
                flight['ArrTime'] = flights_info[flight['New_Flight_ID']]['ArrTime']

        if constraint_violation_1 == 0 and constraint_violation_2 == 0:

            # 修复时隙容量约束、飞机维修约束及飞行时长约束
            violation_1, violation_2, violation_3, airports_capacity_dic, flight_circles, flights_info, overload_info_dic = init_re_repair_airports_capacity(recovery_start_time, airports_capacity_total, group_lookup_dic, aircraft_info, flight_circles, flights_info, sort_str, cancel_hour_dc, cancel_hour_i)
            constraint_violation_1 += violation_1
            constraint_violation_2 += violation_2
            constraint_violation_3 += violation_3

        else:
            violation_3, flights_info, airports_capacity_dic = check_airport_capacity(airports_capacity_total, airports_info, flight_circles, flights_info, alt_airports_info)
            constraint_violation_3 += violation_3

    elif evaluate_tag == 'init_repair':
        # 整合所有飞机的航班信息，同时计算航程约束-constraint_violation_4
        for group_id, group_info in flight_circles.items():
            aircraft = group_info['Aircraft']
            # total_flights = aircraft_info[aircraft]['Flights_num']
            for flight in group_info['Flights']:
                new_flights_id = flight['New_Flight_ID']
                flights_info[new_flights_id]['Aircraft'] = aircraft
                flights_info[new_flights_id]['State'] = flight['State']
                flights_info[new_flights_id]['Delay'] = flight['Delay']
                flights_info[new_flights_id]['Recovery_Flight'] = 1
                flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                if flight['State'] != -1:
                    # 计算航程约束
                    if flight['Duration'] > aircraft_info[aircraft]['Dist']:
                        constraint_violation_4 += 1
                        violation_dic['Flight_range'].append(new_flights_id)


        # 修复维修时间/地点、飞行时长约束
        for aircraft, info in aircraft_info.items():
            # 记录维修情况
            maint_info = info['Maint']
            # 记录每个航班的后续航班情况
            flights_num_total = 0
            canceled_flights_num = 0
            flights_num = 0
            total_flights_num = 0
            flights_state = []

            if maint_info is None:
                successful_maintenance = True
                flag_in_maintenance_airport = True
                # 记录后续航班情况
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    if flight['State'] == -1:
                        canceled_flights_num += 1
                        flights_state.append(0)
                    else:
                        flights_num += 1
                        total_flights_num += 1
                        flights_state.append(1)
                    flights_num_total += 1
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Sequence'] = flights_num
                info['Flights_num'] = total_flights_num

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Left_num'] = total_flights_num - flights_info[new_flights_id][
                        'Sequence']

            else:
                total_flights_num = 50
                maint_airport = maint_info['Maint_Airport']
                maint_start = datetime.strptime(
                    maint_info['Maint_StartDate'] + ' ' + maint_info['Maint_StartTime'], '%d/%m/%y %H:%M')
                maint_end = datetime.strptime(maint_info['Maint_EndDate'] + ' ' + maint_info['Maint_EndTime'],
                                              '%d/%m/%y %H:%M')
                maint_time_range = int(maint_info['Maint_Time_Range'])
                # 汇合某架飞机的所有待执行航班环
                if aircraft[:9] == 'TranspCom':
                    continue
                aircraft_id = aircraft
                flights_for_aircraft = query_flight_by_aircraft(flights_info, aircraft_id)

                sort_str_adapt = sort_str + 'Time'

                # 对该飞机的待执行航班环按照第一个flight的DepDate和DepTime排序
                sorted_flights_for_aircraft = dict(sorted(
                    flights_for_aircraft.items(),
                    key=lambda item: (item[1][sort_str_adapt])
                ))

                # 检查维修时间/地点、飞行时长约束
                successful_maintenance, over_duration_count, sorted_flights_for_aircraft = repair_maintenance(aircraft,
                                                                                                              aircraft_info,
                                                                                                              flight_circles,
                                                                                                              sorted_flights_for_aircraft,
                                                                                                              group_lookup_dic,
                                                                                                              maint_airport,
                                                                                                              maint_start,
                                                                                                              maint_end,
                                                                                                              maint_time_range)
                if not successful_maintenance:
                    constraint_violation_1 += 1  # 维修时间/地点约束
                constraint_violation_2 += over_duration_count  # 飞行时长约束

                for flight_id, flight in sorted_flights_for_aircraft.items():
                    if flight['State'] == -1:
                        canceled_flights_num += 1
                        flights_state.append(0)
                    else:
                        flights_num += 1
                        total_flights_num += 1
                        flights_state.append(1)
                    flights_num_total += 1
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Sequence'] = flights_num

                info['Flights_num'] = total_flights_num

                # 信息更新到flights_info中
                for flight_id, flight in sorted_flights_for_aircraft.items():
                    new_flights_id = flight['New_Flight_ID']
                    flights_info[new_flights_id]['Aircraft'] = aircraft
                    flights_info[new_flights_id]['State'] = flight['State']
                    flights_info[new_flights_id]['Delay'] = flight['Delay']
                    flights_info[new_flights_id]['Recovery_Flight'] = 1
                    flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                    flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                    flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                    flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']
                    flights_info[new_flights_id]['Left_num'] = total_flights_num - flights_info[new_flights_id]['Sequence']

        for group_id, group_info in flight_circles.items():
            for flight in group_info['Flights']:
                flight['State'] = flights_info[flight['New_Flight_ID']]['State']
                flight['Delay'] = flights_info[flight['New_Flight_ID']]['Delay']
                flight['New_DepTime'] = flights_info[flight['New_Flight_ID']]['New_DepTime']
                flight['New_ArrTime'] = flights_info[flight['New_Flight_ID']]['New_ArrTime']
                flight['DepTime'] = flights_info[flight['New_Flight_ID']]['DepTime']
                flight['ArrTime'] = flights_info[flight['New_Flight_ID']]['ArrTime']

        if constraint_violation_1 == 0 and constraint_violation_2 == 0:
            # 修复时隙容量约束、飞机维修约束及飞行时长约束
            violation_1, violation_2, violation_3, airports_capacity_dic, flight_circles, flights_info, overload_info_dic = init_repair_airports_capacity(recovery_start_time, airports_capacity_total, group_lookup_dic, aircraft_info, flight_circles, flights_info, sort_str, cancel_hour_dc, cancel_hour_i)
            constraint_violation_1 += violation_1
            constraint_violation_2 += violation_2
            constraint_violation_3 += violation_3
        else:
            violation_3, flights_info, airports_capacity_dic = check_airport_capacity(airports_capacity_total, airports_info, flight_circles, flights_info, alt_airports_info)
            constraint_violation_3 += violation_3

        # 整合航班具体信息
        for group_id, group_info in flight_circles.items():
            aircraft = group_info['Aircraft']
            # total_flights = aircraft_info[aircraft]['Flights_num']
            for flight in group_info['Flights']:
                new_flights_id = flight['New_Flight_ID']
                flights_info[new_flights_id]['Aircraft'] = aircraft
                flights_info[new_flights_id]['State'] = flight['State']
                flights_info[new_flights_id]['Delay'] = flight['Delay']
                flights_info[new_flights_id]['Recovery_Flight'] = 1
                flights_info[new_flights_id]['New_DepTime'] = flight['New_DepTime']
                flights_info[new_flights_id]['New_ArrTime'] = flight['New_ArrTime']
                flights_info[new_flights_id]['DepTime'] = flight['DepTime']
                flights_info[new_flights_id]['ArrTime'] = flight['ArrTime']

    # 计算恢复期结束时飞机的位置
    aircraft_position = check_aircraft_position(aircraft_info, aircraft_position, flights_info, recovery_end_time, sort_str)

    # 检验恢复期结束时机场停靠的飞机，计算目标函数1-3
    cost1_3 = check_aircraft_requirements(config, aircraft_info, aircraft_position, position_info, aircraft_category)
    fitness_1 += cost1_3

    vars = flight_circle_to_chorm(num_groups, num_groups, flight_circles, aircraft_info, cancel_hour_dc, cancel_hour_i, cancel_decision)
    Vars = vars[0, :]
    Vars.astype(int)


    aircraft_assignment = Vars[:num_groups]
    group_situation = Vars[num_groups:]

    cost1_1_total = 0
    for flight_id, flight in flights_info.items():
        cost_per_hour = aircraft_info[flight['Aircraft']]['Cost_per_Hour']
        if flight['State'] != -1:
            duration = flight['Duration']
            if flight['New_Flight_ID'] in group_lookup_dic:
                cost1_1 += (duration / 60) * cost_per_hour
            cost1_1_total += (duration / 60) * cost_per_hour

    time_check = time.time()
    # 计算航班取消延误成本
    cost1_2 = 0
    Cabin = ['F', 'B', 'E']
    total_flight_num = 0
    adaptable_flight_num = 0
    cancel_flight_num = 0
    delay_flight_num = 0
    cancel_passenger_num = 0
    delay_passenger_num = 0
    delay_flight_time = 0
    cancel_delay_flight_time = 0
    changed_flight_num = 0
    cancel_delay_passenger_time = 0
    delay_passenger_time = 0
    cancel_cost_sum = 0
    delay_cost_sum = 0
    for flight_id, flight in flights_info.items():
        total_flight_num += 1
        if flight['DepTime'] > recovery_start_time:
            adaptable_flight_num += 1
        flight_type = flight['Type']
        if flight['State'] == -1:
            if flight['Type'] in ['D', 'C']:
                cancel_delay_flight_time += cancel_hour_dc * 60
                changed_flight_num += 1
            else:
                cancel_delay_flight_time += cancel_hour_i * 60
                changed_flight_num += 1
            cancel_flight_num += 1
            for cabin_type in Cabin:
                cancel_cost_tag = f'CancelInCost_' + cabin_type + '_' + flight_type
                cancel_cost = config[cancel_cost_tag]
                # delay_cost_tag = f'DelayCost_'+ cabin_type + '_' + flight_type
                # cancel_cost = config[delay_cost_tag] * 60 * cancel_delayhour
                cost1_2 += cancel_cost * flight[cabin_type]
                cancel_cost_sum += cancel_cost * flight[cabin_type]
                cancel_passenger_num += flight[cabin_type]
        elif flight['Delay'] != 0:
            if flight['Delay'] > 15:
                changed_flight_num += 1
            aircraft_capacity_num = aircraft_info[flight['Aircraft']]['Cabin_Capacity']
            delay_flight_num += 1
            delay_flight_time += flight['Delay']
            for cabin_type in Cabin:
                cancel_num = flight[cabin_type] - aircraft_capacity_num[cabin_type]
                if cancel_num > 0:
                    cancel_cost_tag = f'CancelInCost_' + cabin_type + '_' + flight_type
                    cancel_cost = config[cancel_cost_tag]
                    cost1_2 += cancel_cost * cancel_num
                    cancel_cost_sum += cancel_cost * cancel_num
                    cancel_passenger_num += cancel_num
                    delay_cost_tag = f'DelayCost_' + cabin_type + '_' + flight_type
                    delay_cost = config[delay_cost_tag]
                    cost1_2 += delay_cost * aircraft_capacity_num[cabin_type] * flight['Delay']
                    delay_cost_sum += delay_cost * aircraft_capacity_num[cabin_type] * flight['Delay']
                    delay_passenger_num += aircraft_capacity_num[cabin_type]
                    delay_passenger_time += aircraft_capacity_num[cabin_type] * flight['Delay']
                else:
                    delay_cost_tag = f'DelayCost_' + cabin_type + '_' + flight_type
                    delay_cost = config[delay_cost_tag]
                    cost1_2 += delay_cost * flight[cabin_type] * flight['Delay']
                    delay_cost_sum += delay_cost * flight[cabin_type] * flight['Delay']
                    delay_passenger_num += flight[cabin_type]
                    delay_passenger_time += flight[cabin_type] * flight['Delay']
        else:
            aircraft_capacity_num = aircraft_info[flight['Aircraft']]['Cabin_Capacity']
            for cabin_type in Cabin:
                cancel_num = flight[cabin_type] - aircraft_capacity_num[cabin_type]
                if cancel_num > 0:
                    cancel_cost_tag = f'CancelInCost_' + cabin_type + '_' + flight_type
                    cancel_cost = config[cancel_cost_tag]
                    cost1_2 += cancel_cost * cancel_num
                    cancel_passenger_num += cancel_num
                    cancel_cost_sum += cancel_cost * cancel_num

    time_pass = time.time() - time_check
    # print(time_pass)

    fitness_1 += cost1_1_total # 总运行油耗
    fitness_1 += cost1_2
    cancel_delay_flight_time += delay_flight_time

    # 最后检测飞机交换情况
    index = 0
    for group_index, circle in flight_circles.items():
        if aircraft_assignment[index] != init_aircraft_assignment[index]:  # 检测航班环的飞机指派是否改变
            for flight in circle['Flights']:
                if canceled_flights_swap == 0:  # 取消航班不加入计算
                    if flight['State'] != -1:
                        fitness_2 += 1
                else:
                    fitness_2 += 1
        index += 1

    aircraft_exchanges = fitness_2

    cancel_standard = max(round(init_cancel_flight_num * 1.0), round(total_flight_num * 0.05))

    if cancel_flight_num >= cancel_standard:
        constraint_violation_5 = cancel_flight_num - cancel_standard

    constraint_violation_5 = 0

    constraints = np.zeros((1, 5))
    constraints[0][0] = constraint_violation_1  # 飞机维修时间/地点约束
    constraints[0][1] = constraint_violation_2  # 飞行总时间约束
    constraints[0][2] = constraint_violation_3  # 机场时隙约束
    constraints[0][3] = constraint_violation_4  # 飞机航程约束
    constraints[0][4] = constraint_violation_5  # 航班取消比例约束

    evaluation_record = {
        'total_flight_num': total_flight_num,
        'cancel_flight_num': cancel_flight_num,
        'delay_flight_num': delay_flight_num,
        'cancel_passenger_num': cancel_passenger_num,
        'delay_passenger_num': delay_passenger_num,
        'delay_flight_time': delay_flight_time,
        'delay_passenger_time': delay_passenger_time,
        'cancel_cost_sum': cancel_cost_sum,
        'delay_cost_sum': delay_cost_sum,
        'aircraft_exchanges': aircraft_exchanges,
        'fitness_1': fitness_1,
        'fitness_2': delay_flight_time,
        'cost1_1_total': cost1_1_total,
        'cost1_1': cost1_1,
        'cost1_2': cost1_2,
        'cost1_3': cost1_3
    }

    solution = {
        'FlightCircles': flight_circles,
        'Vars': Vars,
        'Cost': evaluation_record['fitness_1'],
        'Exchanges': evaluation_record['fitness_2'],
        'Violation': constraints,
        'EvaluationRecord': evaluation_record,

    }

    return solution
