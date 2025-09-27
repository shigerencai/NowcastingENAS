import os
import csv
import copy
import glob
import json
import random
import time
import numpy as np
import torch
import jsonpickle
from loggers.logger import LOG
import matplotlib.pyplot as plt
from default_configs.default_config import DefaultConfig
from random_generator import RandomGenerator
from core.designer.state_block_designer import StateBlockDesigner
from model_values.model_allvalues import Modelallvalues
from model_values.evaluate_values import ObjectiveEvaluation
from core.transformation.block_transformation import BlockTransformation
from core.NSGA_II import selection, crossover, mutation


LOG = LOG.get_instance().get_logger()


class SearchDelegator:

    def __init__(self, data_name):

        self.generation = 0
        self.population = set()
        self.architectures = {}
        self.arch_history = {}
        self.population_all_models_fitness = {}
        self.history_all_models_fitness = {}
        self.pareto_front = []
        self.use_alternate_nsga_crowding = DefaultConfig.get_config('alternative_nsga_crowding')
        self.data_name = data_name

    def search(self):
        population_size = DefaultConfig.get_config('population_size')
        population_to_select = DefaultConfig.get_config('population_to_select')
        is_restore_success = self.restore_snapshot()
        if not is_restore_success:
            transformer = BlockTransformation(generation=self.generation, num_layers=DefaultConfig.get_config('num_layers'))
            self.initialise_population(population_size, transformer)
            self.save_architectures()
            self.evaluate_objective_functions(list(self.population))
        else:
            if self.evaluating_architecture is None:
                evaluating_architecture_position = None
                for p, model_name in enumerate(list(self.population)):
                    if model_name not in self.population_all_models_fitness.keys():
                        evaluating_architecture_position = p
                        break
                if evaluating_architecture_position is not None:
                    self.evaluate_objective_functions(list(self.population)[evaluating_architecture_position:])
            else:
                evaluating_architecture_position = None
                for p, model_name in enumerate(list(self.population)):
                    if model_name == self.evaluating_architecture:
                        evaluating_architecture_position = p
                self.evaluate_objective_functions(list(self.population)[evaluating_architecture_position:])
        avg_objective1 = []
        avg_objective2 = []
        gen = []
        for num in range(DefaultConfig.get_config('number_of_generations')):
            pareto_front = selection.fast_non_dominated_sort(self.population, self.population_all_models_fitness)
            crowding_distance, avg_obj1, avg_obj2, obj_list = selection.calculate_crowding_distance(pareto_front, self.population_all_models_fitness)
            avg_objective1.append(avg_obj1)
            avg_objective2.append(avg_obj2)
            gen.append(num)
            with open(DefaultConfig.get_config('experimental_pic')+f'{self.data_name} Average target values of each generation of the population.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Generation', obj_list[0], obj_list[1]])
                for g, obj1, obj2 in zip(gen, avg_objective1, avg_objective2):
                    writer.writerow([g, obj1, obj2])
            self.pareto_front = copy.deepcopy(pareto_front)
            new_population = []
            new_architectures = {}
            new_population_all_models_fitness = {}
            self.generation += 1

            for i in range(len(pareto_front)-1):
                crowding_distance_list = []
                model_dict = {}
                for model_name in crowding_distance[i].keys():
                    if crowding_distance[i][model_name] == np.inf:
                        new_population.append(model_name)
                        new_architectures[model_name] = copy.deepcopy(self.architectures[model_name])
                        new_population_all_models_fitness[model_name] = copy.deepcopy(self.population_all_models_fitness[model_name])
                        if len(new_population) == population_to_select:
                            break
                    else:
                        crowding_distance_list.append(crowding_distance[i][model_name])
                        model_dict[crowding_distance[i][model_name]] = model_name
                if len(new_population) == population_to_select:
                    break
                crowding_distance_list = sorted(crowding_distance_list)
                crowding_distance_list.reverse()
                for distance in crowding_distance_list:
                    new_population.append(model_dict[distance])
                    new_architectures[model_dict[distance]] = copy.deepcopy(self.architectures[model_dict[distance]])
                    new_population_all_models_fitness[model_dict[distance]] = copy.deepcopy(self.population_all_models_fitness[model_dict[distance]])
                    if len(new_population) == population_to_select:
                        break
                if len(new_population) == population_to_select:
                    break
            LOG.info(f'From the old population, the best {population_to_select} individuals were selected and {new_population} of them were added to the new population.')
            self.delete_architectures(new_population)
            self.delete_architectures_parameter(new_population)
            LOG.info('>'*50+''f'The beginning of genetics'+'<'*50)
            for i in range(int((population_size - population_to_select)/2)):
                individual_id1 = random.randint(0, population_to_select - 1)
                individual_id2 = random.randint(0, population_to_select - 1)
                while individual_id1 == individual_id2:
                    individual_id2 = random.randint(0, population_to_select - 1)
                individual_name1 = new_population[individual_id1]
                individual_name2 = new_population[individual_id2]
                LOG.info(f'In the {self.generation} generation, the {2*i+1}th individual and the {2*i+2}th individual were derived from the parent population through the reproduction of {individual_name1} and {individual_name2}.')
                new_architecture1, new_architecture2 = crossover.crossover(self.architectures[individual_name1], self.architectures[individual_name2])
                new_architecture1.identifier = f'Evo{self.generation}_{2*i}'
                new_architecture2.identifier = f'Evo{self.generation}_{2*i+1}'
                new_architecture1.generation = self.generation
                new_architecture2.generation = self.generation

                transformer = BlockTransformation(generation=self.generation, num_layers=DefaultConfig.get_config('num_layers'))
                if random.uniform(0, 1) > 0.8:
                    LOG.info('>'*25+f'Carry out variation'+'<'*25)
                    new_architecture1 = mutation.mutation(new_architecture1, transformer)
                else:
                    LOG.info('>'*25+'No mutation occurs'+'<'*25)
                if random.uniform(0, 1) > 0.8:
                    LOG.info('>'*25+f'Carry out variation'+'<'*25)
                    new_architecture2 = mutation.mutation(new_architecture2, transformer)
                else:
                    LOG.info('>'*25+'No mutation occurs'+'<'*25)
                LOG.info('-' * 100)
                new_population.append(new_architecture1.identifier)
                new_population.append(new_architecture2.identifier)
                new_architectures[new_architecture1.identifier] = new_architecture1
                new_architectures[new_architecture2.identifier] = new_architecture2
            self.population = copy.deepcopy(new_population)
            self.architectures = copy.deepcopy(new_architectures)
            self.population_all_models_fitness = copy.deepcopy(new_population_all_models_fitness)
            self.save_arch_in_history()
            self.evaluate_objective_functions(list(self.population))
        return


    def restore_snapshot(self):
        if not os.path.exists(os.path.join(DefaultConfig.get_config('snapshots_save_dir'), self.data_name)) or not DefaultConfig.get_config('restore_if_possible'):
            LOG.info('No recovery of files')
            return False
        files = glob.glob(os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name, 'snapshot_*_*.pt'), recursive=False)
        if len(files) == 0:
            LOG.info('Attempted to restore, but failed to find the snapshot file')
            return False

        versions = self.get_restore_versions(files)
        snapshot = torch.load(os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name, f'snapshot_{versions[0][0]}_{versions[0][1]}.pt'))
        self.generation = snapshot['generation']
        self.population_all_models_fitness = snapshot['population_all_models_fitness']
        self.history_all_models_fitness = snapshot['history_all_models_fitness']
        self.evaluating_architecture = snapshot['evaluating_architecture']
        self.pareto_front = snapshot['pareto_front']
        self.arch_history = snapshot['arch_history']
        self.population = snapshot['population']
        for key in snapshot['architecture_keys']:
            f = open(os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), self.data_name, f'{key}.json'))
            json_str = f.read()
            architecture = jsonpickle.decode(json_str)
            f.close()
            self.architectures[key] = architecture
        LOG.info(f'Successfully restored from snapshot_{versions[0][0]}_{versions[0][1]}')
        return True

    def get_restore_versions(self, files=None):
        if files is None:
            files = glob.glob(os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name, 'snapshot_*_*.pt'), recursive=False)
            if len(files) == 0:
                return []

        versions = []
        for file in files:
            first_pos = file.find('_')
            second_pos = file.find('_', first_pos + 1)
            second_end = file.find('.', second_pos)
            first_num = file[first_pos + 1:second_pos]
            second_num = file[second_pos + 1:second_end]
            versions.append((int(first_num), int(second_num)))
        versions = sorted(versions, key=lambda x: (x[0], x[1]), reverse=True)
        return versions


    def initialise_population(self, population_size, transformer):
        designer = StateBlockDesigner('BASIC')


        if 'convlstm_0' not in self.architectures.keys() or 'convlstm_0' not in self.population_all_models_fitness.keys() or 'convlstm_0' not in self.history_all_models_fitness.keys():
            self.architectures['convlstm_0'] = designer.get_convlstm_architecture()
            self.architectures['convlstm_0'].identifier = 'convlstm_0'
            self.population.add('convlstm_0')

        if 'convgru_0' not in self.architectures.keys() or 'convgru_0' not in self.population_all_models_fitness.keys() or 'convgru_0' not in self.history_all_models_fitness.keys():
            self.architectures['convgru_0'] = designer.get_convgru_architecture()
            self.architectures['convgru_0'].identifier = 'convgru_0'
            self.population.add('convgru_0')

        self.save_arch_in_history()


        if (population_size-len(self.population)) > 0:
            LOG.info(f'Total population size: {population_size}, Current initial population size: {len(self.population)}, Continue to generate {population_size - len(self.population)} model individuals')
            for i in range(population_size-len(self.population)):
                self.generate_random_architecture(0, transformer)

    def save_arch_in_history(self):
        for architectures_name in self.architectures.keys():
            if architectures_name not in self.arch_history.keys():
                _hash = json.dumps(self.architectures[architectures_name].get_block_strings())
                self.arch_history[architectures_name] = str(hash(_hash))


    def generate_random_architecture(self, count, transformer):
        new_architecture = None
        transformation_count = None
        transformations_dict = None
        while new_architecture is None or self.test_new_architecture_similarity(new_architecture):
            initial_model = RandomGenerator.get_initial_model_function()
            if initial_model == 'convlstm':
                new_architecture = StateBlockDesigner.get_convlstm_architecture()
            elif initial_model == 'convgru':
                new_architecture = StateBlockDesigner.get_convgru_architecture()


            new_architecture.identifier = f'Evo0_{count}'
            new_count = copy.copy(count)
            while new_architecture.identifier in self.architectures.keys():
                new_count += 1
                new_architecture.identifier = f'Evo0_{new_count}'
            new_architecture = transformer.transform_architecture(new_architecture, transformation_count=transformation_count)
            if transformation_count is None:
                transformation_count = 5
            else:
                transformation_count += 1

        self.architectures[new_architecture.identifier] = new_architecture
        self.population.add(new_architecture.identifier)
        return copy.deepcopy(new_architecture.identifier)

    def test_new_architecture_similarity(self, new_architecture):
        for architectures_name in self.architectures.keys():
            if new_architecture.similar_with_other(self.architectures[architectures_name]):
                LOG.debug(f'The new model is exactly the same as model {architectures_name}, and we will start generating it again.')
                return True
            if self.check_if_hash_exist(new_architecture):
                LOG.debug(f'The new model and the model {architectures_name} have the same hash value. Starting to regenerate.')
                return True
        return False


    def check_if_hash_exist(self, new_architecture):
        if len(self.arch_history.keys()) == 0:
            self.save_arch_in_history()

        lst_hashes = list(self.arch_history.values())
        _compare_json = json.dumps(new_architecture.get_block_strings())
        _compare_hash = str(hash(_compare_json))
        if _compare_hash in lst_hashes:
            return True
        return False

    def save_architectures(self):
        if not os.path.exists(os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), self.data_name)):
            os.makedirs(os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), self.data_name))
        for model_name in self.architectures.keys():
            architecture_path = os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), self.data_name, f'{model_name}.json')
            if not os.path.exists(architecture_path):
                f = open(architecture_path, 'w')
                json_object = jsonpickle.encode(self.architectures[model_name])
                f.write(json_object)
                f.close()


    def delete_architectures(self, new_population):
        for model_name in self.population:
            if model_name not in new_population:
                if os.path.exists(os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), self.data_name, f'{model_name}.json')):
                    os.remove(os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), self.data_name, f'{model_name}.json'))

    def delete_architectures_parameter(self, new_population):
        for model_name in self.population:
            if model_name not in new_population:
                if os.path.exists(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), self.data_name, f'{model_name}.ckpt')):
                    os.remove(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), self.data_name, f'{model_name}.ckpt'))


    def evaluate_objective_functions(self, population):
        population = copy.deepcopy(population)
        LOG.info(f'{">" * 50} Start the assessment{population} {"<" * 50}')
        for i, model_name in enumerate(population):
            if model_name in self.population_all_models_fitness.keys() and model_name in self.history_all_models_fitness.keys():
                print(f'The target value of the model {model_name} already exists.')
                continue

            self.evaluating_architecture = model_name
            previous_snapshot = self.save_snapshot()
            architecture = self.architectures[model_name]
            if model_name not in self.population_all_models_fitness.keys():
                model_all_values = Modelallvalues(model_name)
                model_all_values = ObjectiveEvaluation.evaluate_cheap_objectives(architecture, model_all_values)
                model_all_values = ObjectiveEvaluation.evaluate_expensive_objectives(architecture, model_all_values, self.data_name, self.generation)
                self.population_all_models_fitness[model_name] = copy.deepcopy(model_all_values)
                self.history_all_models_fitness[model_name] = copy.deepcopy(model_all_values)
            else:
                model_all_values = ObjectiveEvaluation.evaluate_expensive_objectives(architecture, self.history_all_models_fitness[model_name], self.data_name, self.generation)
                self.population_all_models_fitness[model_name] = copy.deepcopy(model_all_values)
                self.history_all_models_fitness[model_name] = copy.deepcopy(model_all_values)
            print(f'Evaluate the progress: {i+1}/{len(population)}')
            self.evaluating_architecture = None
            self.save_snapshot(add_count=1)
            if os.path.exists(previous_snapshot):
                os.remove(previous_snapshot)
        LOG.info(f'{">" * 50} Evaluation is complete {"<" * 50}')


    def save_snapshot(self, add_count=0):
        if not os.path.exists(os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name)):
            os.makedirs(os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name))
        count = len(glob.glob(os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name, f'snapshot_{self.generation}_*.pt'), recursive=True)) + add_count
        snapshot_path = os.path.join(DefaultConfig.get_config("snapshots_save_dir"), self.data_name, f'snapshot_{self.generation}_{count}.pt')
        torch.save({
            'generation': self.generation,
            'population_all_models_fitness': self.population_all_models_fitness,
            'architecture_keys': list(self.architectures.keys()),
            'history_all_models_fitness': self.history_all_models_fitness,
            'evaluating_architecture': self.evaluating_architecture,
            'pareto_front': self.pareto_front,
            'arch_history': self.arch_history,
            'population': self.population
        }, snapshot_path)
        LOG.debug('Snapshot successfully saved.')
        self.save_architectures()

        return snapshot_path


