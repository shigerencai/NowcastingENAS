import copy
import numpy as np
from loggers.logger import LOG
from default_configs.default_config import DefaultConfig

LOG = LOG.get_instance().get_logger()
dominate_objectives = DefaultConfig.get_config('compare_objectives')

def fast_non_dominated_sort(population, population_all_models_fitness):
    population = copy.deepcopy(population)
    population_all_models_fitness = copy.deepcopy(population_all_models_fitness)
    LOG.info(f'Start the selection process using the NSGA-II algorithm')
    s = {}
    for model_name in population:
        s[model_name] = []
    n = {}
    for model_name in population:
        n[model_name] = 0
    rank = {}
    for model_name in population:
        rank[model_name] = 0
    front = [[]]
    for model_name_1 in population:
        for model_name_2 in population:
            if is_dominate(population_all_models_fitness[model_name_1], population_all_models_fitness[model_name_2]):
                s[model_name_1].append(model_name_2)
            elif is_dominate(population_all_models_fitness[model_name_2], population_all_models_fitness[model_name_1]):
                n[model_name_1] += 1
        if n[model_name_1] == 0:
            front[0].append(model_name_1)

    i = 0
    while (front[i] != []):
        LOG.info(f'Individual in the {i}th layer of the Pareto frontier: {front[i]}')
        Q = []
        for model_name in front[i]:
            for dominate_model_name in s[model_name]:
                n[dominate_model_name] -= 1
                if (n[dominate_model_name] == 0):
                    rank[dominate_model_name] = i + 1
                    if dominate_model_name not in Q:
                        Q.append(dominate_model_name)
        i = i + 1
        front.append(Q)
    del front[len(front) - 1]
    return copy.deepcopy(front)


def is_dominate(model_all_values_1, model_all_values_2):
    model_all_values_1 = copy.deepcopy(model_all_values_1)
    model_all_values_2 = copy.deepcopy(model_all_values_2)
    model_values_dict_1 = model_all_values_1.get_values_dict()
    model_values_dict_2 = model_all_values_2.get_values_dict()
    equal_num = 0
    for objective in dominate_objectives:
        if objective == 'csi' or objective == 'hss':
            if model_values_dict_1[objective] < model_values_dict_2[objective]:
                return False
            if model_values_dict_1[objective] == model_values_dict_2[objective]:
                equal_num += 1
        else:
            if model_values_dict_1[objective] > model_values_dict_2[objective]:
                return False
            if model_values_dict_1[objective] == model_values_dict_2[objective]:
                equal_num += 1
    if equal_num == len(dominate_objectives):
        return False

    return True

def calculate_crowding_distance(pareto_front, population_all_models_fitness):
    pareto_front = copy.deepcopy(pareto_front)
    population_all_models_fitness = copy.deepcopy(population_all_models_fitness)
    crowding_distance = []
    t_obj1 = t_obj2 = 0
    x1 = x2 = 0
    for i in range(len(pareto_front)-1):
        lenth = len(pareto_front[i])
        distance = {}
        for model_name in pareto_front[i]:
            distance[model_name] = 0
        values_dict = {}
        for objective in dominate_objectives:
            values = {}
            if objective == 'mae_loss' or objective == 'mse_loss':
                for model_name in pareto_front[i]:
                    values[model_name] = population_all_models_fitness[model_name].get_values_dict()[objective]
                    if values[model_name] != np.inf:
                        t_obj1 += values[model_name]
                        x1 += 1
            elif objective == 'csi' or objective == 'hss':
                for model_name in pareto_front[i]:
                    values[model_name] = population_all_models_fitness[model_name].get_values_dict()[objective]
                    if values[model_name] != 0:
                        t_obj1 += values[model_name]
                        x1 += 1
            else:
                for model_name in pareto_front[i]:
                    values[model_name] = population_all_models_fitness[model_name].get_values_dict()[objective]
                    if values[model_name] != np.inf:
                        t_obj2 += values[model_name]
                        x2 += 1
            values_dict[objective] = values

        for objective in values_dict.keys():
            max_objective = -np.inf
            max_name = None
            min_objective = np.inf
            min_name = None
            for model_name in values_dict[objective].keys():
                if values_dict[objective][model_name] > max_objective:
                    max_objective = values_dict[objective][model_name]
                    max_name = model_name
                if values_dict[objective][model_name] < min_objective:
                    min_objective = values_dict[objective][model_name]
                    min_name = model_name
            distance[max_name] = np.inf
            distance[min_name] = np.inf
            if lenth > 2:
                for model_name in pareto_front[i]:
                    if distance[model_name] != np.inf:
                        left_neighbor_value, right_neighbor_value = search_neighbor_values(values_dict[objective][model_name], values_dict[objective])
                        distance[model_name] = distance[model_name] + abs((right_neighbor_value - left_neighbor_value)/(max_objective - min_objective))
        crowding_distance.append(distance)
        LOG.info(f'The individual crowding distance on the {i}th layer of the Pareto frontier: {distance}')
    avg_obj1 = t_obj1 / x1
    avg_obj2 = t_obj2 / x2
    return copy.deepcopy(crowding_distance), avg_obj1, avg_obj2, dominate_objectives

def search_neighbor_values(value, values_dict):
    value = copy.deepcopy(value)
    values_dict = copy.deepcopy(values_dict)
    values_list = []
    for model_name in values_dict.keys():
        values_list.append(values_dict[model_name])
    values_list = sorted(values_list)
    posite = values_list.index(value)
    left_neighbor_value = values_list[posite-1]
    right_neighbor_value = values_list[posite + 1]

    return copy.deepcopy(left_neighbor_value), copy.deepcopy(right_neighbor_value)
