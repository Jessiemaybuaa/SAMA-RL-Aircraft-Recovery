import copy
import random
from datetime import datetime, timedelta
import numpy as np



def try_insert_flight(index, aircraft_id, aircraft_dic, group_id, cancelled_group_info, flight_sequence):
    """
    尝试将航班插入到指定位置，调整延误并验证约束
    """
    # 插入取消航班至指定位置
    former_flight_sequence = copy.deepcopy(flight_sequence)
    cancelled_group = {}
    cancelled_group = {
        'Group_ID': group_id,
        **cancelled_group_info
    }

    for flight in cancelled_group['Flights']:
        flight['State'] = 0

    flight_sequence.insert(index, cancelled_group)

    # 调整该位置及之后的航班延误时间
    if index == 0:
        index = 1
    for i in range(index, len(flight_sequence)):
        previous_flight = flight_sequence[i - 1]
        current_flight = flight_sequence[i]

        # 计算周转时间并检查是否需要延误
        check_time = calculate_turnaround_time(previous_flight, current_flight)
        turnround_time = aircraft_dic[aircraft_id]['TurnRound']
        transit_time = aircraft_dic[aircraft_id]['Transit']
        turnround_time = max(turnround_time, transit_time)

        if check_time < turnround_time:
            delay_needed = turnround_time - check_time
            for flight in current_flight['Flights']:
                flight['Delay'] += delay_needed
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=flight['Delay'])
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=flight['Delay'])
                flight['State'] = flight['Delay']

    return flight_sequence


def try_cross_flight(index_pair, aircraft_id, aircraft_dic, group_id, cancelled_group_info, flight_sequence):
    """
    尝试将航班与指定位置航班交换，调整延误并验证约束
    """
    # 交换取消航班至指定位置，并存储交换的航班
    new_cancelled_flights_set = []
    former_flight_sequence = copy.deepcopy(flight_sequence)
    cross_group = {}
    cross_group = {
        'Group_ID': group_id,
        **cancelled_group_info
    }

    for flight in cross_group['Flights']:
        flight['State'] = 0

    for group_index in range(index_pair[0], index_pair[1] + 1):
        new_cancelled_flights = {}
        group = flight_sequence[group_index]
        new_cancelled_flights[group['Group_ID']] = {
            **group
        }

        new_cancelled_flights_set.append(new_cancelled_flights)

    del flight_sequence[index_pair[0]: index_pair[1] + 1]
    flight_sequence.insert(index_pair[0], cross_group)

    # 调整该位置及之后的航班延误时间
    if index_pair[0] == 0:
        index = index_pair[1] + 1
    else:
        index = index_pair[0]
    for i in range(index, len(flight_sequence)):
        previous_flight = flight_sequence[i - 1]
        current_flight = flight_sequence[i]

        # 计算周转时间并检查是否需要延误
        check_time = calculate_turnaround_time(previous_flight, current_flight)
        turnround_time = aircraft_dic[aircraft_id]['TurnRound']
        transit_time = aircraft_dic[aircraft_id]['Transit']
        turnround_time = max(turnround_time, transit_time)

        if check_time < turnround_time:
            delay_needed = turnround_time - check_time
            for flight in current_flight['Flights']:
                flight['Delay'] += delay_needed
                flight['New_DepTime'] = flight['DepTime'] + timedelta(minutes=flight['Delay'])
                flight['New_ArrTime'] = flight['ArrTime'] + timedelta(minutes=flight['Delay'])
                flight['State'] = flight['Delay']

        flight_sequence[i] = current_flight

    return flight_sequence, new_cancelled_flights_set


def calculate_turnaround_time(previous_flight, current_flight):
    """
    计算航班之间的周转时间
    """
    # 假设起降时间都已转换为datetime格式
    previous_arrival = previous_flight['Flights'][-1]['New_ArrTime']
    current_departure = current_flight['Flights'][0]['New_DepTime']

    turnaround_time = (current_departure - previous_arrival).total_seconds() // 60  # 转换为分钟
    return turnaround_time


def Here_Insert(problem, cancelled_set, active_flights):
    """
    将取消航班插入飞机的航班序列
    """
    # problem = self.problem
    aircraft_info = problem.aircraft_info
    alt_flights_info = problem.alt_flights_info
    alt_airports_info = problem.alt_airports_info
    alt_aircraft_info = problem.alt_aircraft_info

    # 每个取消航班环的选择概率
    prob_num_1 = min(1, len(cancelled_set))
    # 插入的飞机序列的选择概率
    prob_num_2 = min(5, len(active_flights))

    new_cancelled_set = {}

    select_list = []
    for group_id, group_info in cancelled_set.items():
        if len(cancelled_set[group_id]['Flights']) == 1:  # 原文中不成环航班无法插入
            continue
        else:
            select_list.append(group_id)

    # 获取所有航班编号和对应的CancelCost
    group_ids = select_list
    cancel_costs = [cancelled_set[fid]['CancelCost'] for fid in group_ids]

    # 以CancelCost为权重进行加权随机抽取
    if len(group_ids) > 0:
        selected_group_id = random.choices(group_ids, weights=cancel_costs, k=1)[0]
    else:
        selected_group_id = random.choice(list(cancelled_set.keys()))

    for group_id, group_info in cancelled_set.items():
        aircraft_id = group_info['Aircraft']
        cancelled_flights = group_info['Flights']

        # 判断是否插入此取消航班
        if group_id == selected_group_id:
        # if random.random() < 1 / prob_num_1:
            if len(cancelled_flights) == 1:  # 原文中不成环航班无法插入
                cancelled_flight = cancelled_flights[0]
                orig = cancelled_flight['Orig']
                dest = cancelled_flight['Dest']
                new_cancelled_set[group_id] = {
                    **group_info
                    # 'CancelCost': cancel_cost
                }
                continue
            else:
                orig = cancelled_flights[0]['Orig']
                dest = cancelled_flights[-1]['Dest']
        else:
            new_cancelled_set[group_id] = {
                **group_info
                # 'CancelCost': cancel_cost
            }
            continue

        # 遍历每个飞机序列，随机决定是否要插入此飞机序列
        insert_flag = 0
        insert_places = []
        for aircraft_id, groups in active_flights.items():  # 插入飞机已有的航班序列中
            feasible_flight_sequence = active_flights[aircraft_id]['FlightGroups']
            if random.random() < 0.1:
                if orig == dest:  # 取消航班成环，枢纽机场相同即可插入
                    # 查看是否将取消航班插入作为第一个航班（第一个航班环之前）
                    if aircraft_info[aircraft_id]['Orig'] == orig:
                        insert_places.append(0)

                    else:  # 遍历飞机的每个航班环，看是否可以插入此航班环之后
                        for group_index in range(0, len(feasible_flight_sequence)):
                            if feasible_flight_sequence[group_index]['Flights'][-1]['Dest'] == dest:
                                insert_places.append(group_index + 1)
                            else:
                                continue
                else:  # 取消航班不成环，只可在尾部插入
                    if feasible_flight_sequence[-1]['Flights'][-1]['Dest'] == orig:
                        insert_places.append(len(feasible_flight_sequence))

                if len(insert_places) != 0:
                    insert_place = random.choice(insert_places)
                    new_flight_sequence = try_insert_flight(insert_place,
                                                                 aircraft_id, aircraft_info,
                                                                 group_id, group_info,
                                                                 feasible_flight_sequence)
                    insert_flag = 1
                else:
                    continue

                if insert_flag == 1:
                    active_flights[aircraft_id]['FlightGroups'] = new_flight_sequence
                    break

        if insert_flag == 0:
            new_cancelled_set[group_id] = {
                **group_info
            }

    return new_cancelled_set, active_flights


def Here_Cross(problem, cancelled_set, active_flights):
    """
    将取消航班和飞机的航班序列中航班交换
    """
    # problem = self.problem
    aircraft_info = problem.aircraft_info
    alt_flights_info = problem.alt_flights_info
    alt_airports_info = problem.alt_airports_info
    alt_aircraft_info = problem.alt_aircraft_info

    # 每个交换航班环的选择概率
    prob_num_1 = min(1, len(cancelled_set))
    # 交换的飞机序列的选择概率
    prob_num_2 = min(100, len(active_flights))

    new_cancelled_set = {}

    select_list = []
    for group_id, group_info in cancelled_set.items():
        if len(cancelled_set[group_id]['Flights']) == 1:  # 原文中不成环航班无法插入
            continue
        else:
            select_list.append(group_id)

    # 获取所有航班编号和对应的CancelCost
    group_ids = select_list
    cancel_costs = [cancelled_set[fid]['CancelCost'] for fid in group_ids]

    # 以CancelCost为权重进行加权随机抽取
    # while (1):
    if len(group_ids) > 0:
        selected_group_id = random.choices(group_ids, weights=cancel_costs, k=1)[0]
    else:
        selected_group_id = random.choice(list(cancelled_set.keys()))


    for group_id, group_info in cancelled_set.items():
        aircraft_id = group_info['Aircraft']
        cancelled_flights = group_info['Flights']

        # 判断是否交换此航班
        if group_id == selected_group_id:
            if len(cancelled_flights) == 1:
                cancelled_flight = cancelled_flights[0]
                orig = cancelled_flight['Orig']
                dest = cancelled_flight['Dest']
            else:
                orig = cancelled_flights[0]['Orig']
                dest = cancelled_flights[-1]['Dest']
        else:
            new_cancelled_set[group_id] = {
                **group_info
                # 'CancelCost': cancel_cost
            }
            continue

        # 遍历每个飞机序列，随机决定是否要插入此飞机序列
        cross_flag = 0
        cross_places = []
        for aircraft_id, groups in active_flights.items():  # 插入飞机已有的航班序列中
            feasible_flight_sequence = active_flights[aircraft_id]['FlightGroups']
            if random.random() < 1 / prob_num_2:
                for group_index_i in range(0, len(feasible_flight_sequence)):
                    for group_index_j in range(group_index_i, len(feasible_flight_sequence)):
                        # 遍历飞机的航班环，查看是否可以对航班环进行交换
                        if feasible_flight_sequence[group_index_i]['Flights'][0]['Orig'] == orig \
                                and feasible_flight_sequence[group_index_j]['Flights'][-1]['Dest'] == dest:
                            cross_place = [group_index_i, group_index_j]
                            cross_places.append(cross_place)

                if len(cross_places) != 0:
                    cross_place_select = random.choice(cross_places)
                    new_flight_sequence, cross_sequence = try_cross_flight(cross_place_select,
                                                                                aircraft_id, aircraft_info,
                                                                                group_id, group_info,
                                                                                feasible_flight_sequence)
                    cross_flag = 1
                else:
                    continue

                if cross_flag == 1:
                    active_flights[aircraft_id]['FlightGroups'] = new_flight_sequence
                    # 取消航班集合加入交叉部分航班
                    for i in range(0, len(cross_sequence)):
                        group = cross_sequence[i]
                        for group_id_new, group_info_new in group.items():
                            for flight in group_info_new['Flights']:
                                flight['State'] = -1
                                flight['Delay'] = 0
                            new_cancelled_set[group_id_new] = {
                                **group_info_new
                            }
                    break

        if cross_flag == 0:
            new_cancelled_set[group_id] = {
                **group_info
            }

    return new_cancelled_set, active_flights


def Homo_Insert(problem, cancelled_set, active_flights):
    """
    将一个航班飞机序列中的航班插入另一个飞机航班序列中
    """

    aircraft_info = problem.aircraft_info
    alt_flights_info = problem.alt_flights_info
    alt_airports_info = problem.alt_airports_info
    alt_aircraft_info = problem.alt_aircraft_info


    while (1):
        # 选择两个飞机序列进行插入操作
        if len(active_flights.keys()) >= 2:
            flight_sequences_id = random.sample(active_flights.keys(), 2)
        else:
            return cancelled_set, active_flights
        flight_sequence_1 = active_flights[flight_sequences_id[0]]['FlightGroups']
        flight_sequence_2 = active_flights[flight_sequences_id[1]]['FlightGroups']
        aircraft_id_1 = flight_sequences_id[0]
        aircraft_id_2 = flight_sequences_id[1]
        if len(flight_sequence_1) > 0:
            break

    insert_flag = False
    insert_count = 0
    stop_tag = False
    while (stop_tag != True):
        # 随机选择飞机序列1中的一个航班环
        insert_circle_index = random.randint(0, len(flight_sequence_1) - 1)
        insert_flights = flight_sequence_1[insert_circle_index]['Flights']

        group_id = flight_sequence_1[insert_circle_index]['Group_ID']
        group_info = flight_sequence_1[insert_circle_index]

        if len(insert_flights) == 1:
            insert_flight = insert_flights[0]
            orig = insert_flight['Orig']
            dest = insert_flight['Dest']
        else:
            orig = insert_flights[0]['Orig']
            dest = insert_flights[-1]['Dest']

        # 查询飞机序列2中可以插入的地方
        insert_places = []
        if orig == dest:  # 取消航班成环，枢纽机场相同即可插入
            # 查看是否将取消航班插入作为第一个航班（第一个航班环之前）
            if aircraft_info[aircraft_id_2]['Orig'] == orig:
                insert_places.append(0)

            else:  # 遍历飞机的每个航班环，看是否可以插入此航班环之后
                for group_index in range(0, len(flight_sequence_2)):
                    if flight_sequence_2[group_index]['Flights'][-1]['Dest'] == dest:
                        insert_places.append(group_index + 1)
                    else:
                        continue
        else:  # 取消航班不成环，只可在尾部插入
            if len(flight_sequence_2) == 0:
                if aircraft_info[aircraft_id_2]['Orig'] == orig:
                    insert_places.append(0)
            else:
                if flight_sequence_2[-1]['Flights'][-1]['Dest'] == orig:
                    insert_places.append(len(flight_sequence_2))

        if len(insert_places) != 0:
            insert_place = random.choice(insert_places)
            # print(insert_place)
            new_flight_sequence_2 = try_insert_flight(insert_place,
                                                           aircraft_id_2, aircraft_info,
                                                           group_id, group_info,
                                                           flight_sequence_2)
            insert_flag = True
        else:
            insert_count += 1
            # insert_flag = True

        if insert_flag == True:
            active_flights[aircraft_id_2]['FlightGroups'] = new_flight_sequence_2
            new_flight_sequence_1 = []
            for circle_index in range(0, len(flight_sequence_1)):
                if circle_index != insert_circle_index:
                    new_flight_sequence_1.append(flight_sequence_1[circle_index])
            active_flights[aircraft_id_1]['FlightGroups'] = new_flight_sequence_1

        if insert_count == 3 or insert_flag == True:
            stop_tag = True

    return cancelled_set, active_flights


def Homo_Cross(problem, cancelled_set, active_flights):
    """
    将一个航班飞机序列中的航班与另一个飞机航班序列中的航班交换
    """
    # problem = self.problem
    aircraft_info = problem.aircraft_info
    alt_flights_info = problem.alt_flights_info
    alt_airports_info = problem.alt_airports_info
    alt_aircraft_info = problem.alt_aircraft_info

    while (1):
        # 选择两个飞机序列进行插入操作
        if len(active_flights.keys()) >= 2:
            flight_sequences_id = random.sample(active_flights.keys(), 2)
        else:
            return cancelled_set, active_flights
        flight_sequence_1 = active_flights[flight_sequences_id[0]]['FlightGroups']
        flight_sequence_2 = active_flights[flight_sequences_id[1]]['FlightGroups']
        aircraft_id_1 = flight_sequences_id[0]
        aircraft_id_2 = flight_sequences_id[1]
        if len(flight_sequence_1) > 0 and len(flight_sequence_2) > 0:
            break

    cross_flag = False
    cross_count = 0
    stop_tag = False
    while (stop_tag != True):
        # 随机选择飞机序列1中的一个航班环
        cross_circle_index = random.randint(0, len(flight_sequence_1) - 1)
        cross_flights = flight_sequence_1[cross_circle_index]['Flights']

        group_id = flight_sequence_1[cross_circle_index]['Group_ID']
        group_info = flight_sequence_1[cross_circle_index]

        if len(cross_flights) == 1:
            cross_flight = cross_flights[0]
            orig = cross_flight['Orig']
            dest = cross_flight['Dest']
        else:
            orig = cross_flights[0]['Orig']
            dest = cross_flights[-1]['Dest']

        # 查询飞机序列2中可以交换的地方
        cross_places = []
        if cross_circle_index != len(flight_sequence_1) - 1:  # 交换的航班非飞机的尾部航班，起止点相同即可交换
            for group_index_i in range(0, len(flight_sequence_2)):
                for group_index_j in range(group_index_i, len(flight_sequence_2)):
                    # 遍历飞机的航班环，查看是否可以对航班环进行交换
                    if flight_sequence_2[group_index_i]['Flights'][0]['Orig'] == orig \
                            and flight_sequence_2[group_index_j]['Flights'][-1]['Dest'] == dest:
                        cross_place = [group_index_i, group_index_j]
                        cross_places.append(cross_place)
        elif cross_circle_index == len(flight_sequence_1) - 1:  # 交换的航班是飞机的尾部航班，起点相同即可交换
            for group_index_i in range(0, len(flight_sequence_2)):
                group_index_j = len(flight_sequence_2) - 1
                if flight_sequence_2[group_index_i]['Flights'][0]['Orig'] == orig:
                    cross_place = [group_index_i, group_index_j]
                    cross_places.append(cross_place)

        if len(cross_places) != 0:
            cross_place = random.choice(cross_places)
            # print(insert_place)
            new_flight_sequence_2, cross_sequence = try_cross_flight(cross_place,
                                                                          aircraft_id_2, aircraft_info,
                                                                          group_id, group_info,
                                                                          flight_sequence_2)
            cross_flag = True
        else:
            cross_count += 1

        if cross_flag == True:
            active_flights[aircraft_id_2]['FlightGroups'] = new_flight_sequence_2
            new_flight_sequence_1 = []
            for circle_index in range(0, len(flight_sequence_1)):
                if circle_index != cross_circle_index:  # 不是交换的航班则原样复制
                    new_flight_sequence_1.append(flight_sequence_1[circle_index])
                elif circle_index == cross_circle_index:  # 是交换的航班则执行替换
                    # 取消航班集合加入交叉部分航班
                    for i in range(0, len(cross_sequence)):
                        group = cross_sequence[i]
                        for group_id_new, group_info_new in group.items():
                            new_flight_sequence_1.append(group_info_new)

            active_flights[aircraft_id_1]['FlightGroups'] = new_flight_sequence_1

        if cross_count == 3 or cross_flag == True:
            stop_tag = True

    return cancelled_set, active_flights


def Cancel_Flight(problem, cancelled_set, active_flights, obj):
    """
    将一个航班飞机序列中延误较多的航班随机取消
    """
    # problem = self.problem
    aircraft_info = problem.aircraft_info
    alt_flights_info = problem.alt_flights_info
    alt_airports_info = problem.alt_airports_info
    alt_aircraft_info = problem.alt_aircraft_info
    group_lookup_dic = problem.group_lookup_dic
    alt_flights_info = problem.alt_flights_info

    # 对集合里每个飞机序列按照调换延误时间从长到短排序
    active_flights = dict(sorted(
        active_flights.items(),
        key=lambda item: (item[1]['Delay']), reverse=True
    ))

    active_flights_list = list(active_flights.keys())
    active_flights_weight = [active_flights.get(active_flights_list[i], {}).get('Delay', 0) for i in range(len(active_flights_list))]
    if len(active_flights_weight) == 0:
        max_delay = 0
    else:
        max_delay = np.max(active_flights_weight)
    flights_num = len(active_flights_weight)
    if flights_num > 0:
        mean_delay = int(max_delay / flights_num)
        active_flights_weight = [a + mean_delay for a in active_flights_weight]

    while(len(active_flights_list) > 0):
            aircraft_id = random.choices(active_flights_list, weights=active_flights_weight, k=1)[0]
            groups = active_flights[aircraft_id]
            if len(groups['FlightGroups']) >= 1:
                choose_list = [index for index in range(len(groups['FlightGroups']))]
                np.random.shuffle(choose_list)
                for index in choose_list:
                    cancel_false = 0
                    for flight in groups['FlightGroups'][index]['Flights']:
                        if flight['New_Flight_ID'] not in group_lookup_dic:
                            cancel_false = 1
                            break
                    if cancel_false == 1:
                        continue
                    else:
                        select = index
                        break
            elif len(groups['FlightGroups']) == 0:
                break
            if cancel_false == 1:
                break
            cancel_group = groups['FlightGroups'][select]
            for flight in cancel_group['Flights']:
                flight['State'] = -1
            cancelled_set[cancel_group['Group_ID']] = cancel_group
            del groups['FlightGroups'][select]
            break


    # 计算当前每个飞机的飞行时间
    remove_aircraft_list = []
    for aircraft_id, groups in active_flights.items():
        flying_time = 0
        exchange = 0
        delay_time = 0
        if len(groups['FlightGroups']) == 0:
            remove_aircraft_list.append(aircraft_id)
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
        # 对取消航班按取消成本排序
        cancelled_set = dict(sorted(
            cancelled_set.items(),
            key=lambda item: (item[1]['CancelCost']),
            reverse=True
        ))
        # 对飞机序列按照航班插入难易度排序（当前飞机序列飞行时长）
        active_flights = dict(sorted(
            active_flights.items(),
            key=lambda item: (item[1]['FlyingTime'])
        ))
    elif obj == 'Exchanges':
        # 对取消航班按照如造成延误，延误损失量排序：与取消成本有一定关系
        cancelled_set = dict(sorted(
            cancelled_set.items(),
            key=lambda item: (item[1]['CancelCost']),
            reverse=True
        ))
        # 对集合里每个飞机序列按照调换飞机次数排序
        active_flights = dict(sorted(
            active_flights.items(),
            key=lambda item: (item[1]['Delay'])
        ))


    return cancelled_set, active_flights



