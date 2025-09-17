import os
import pickle
import numpy as np
from collections import Counter
from default_configs.default_config import DefaultConfig


class RandomGenerator:
    __instance = None

    def __init__(self):
        if RandomGenerator.__instance is None:
            self.rng = None
            self.path = './restore/randomgenerator.pkl'
            RandomGenerator.__instance = self

    @staticmethod
    def setup(seed):
        RandomGenerator.get_instance()._restore_rng_state(seed)


    def _restore_rng_state(self, seed):
        if os.path.exists(self.path):
            file = open(self.path, 'rb')
            pickle_state = pickle.load(file)
            self.rng = np.random.RandomState()
            self.rng.set_state(pickle_state)
            file.close()
            print('Random state has been loaded.')
        else:
            self._set_seed_value(seed)
            self._save_rng_state()
            print('A new random state has been created.')

    def _set_seed_value(self, seed_value):
        self.rng = np.random.RandomState(seed_value)

    def _save_rng_state(self):
        state = self.rng.get_state()
        p_file = open(self.path, 'wb')
        pickle.dump(state, p_file)
        p_file.close()

    @staticmethod
    def get_conv_kernel_function():
        options = DefaultConfig.get_config('conv_kernel')
        distribution = RandomGenerator.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_activation_function(include_none=False):
        options = DefaultConfig.get_config('activation_functions')
        if include_none:
            options.append('None')
        elif 'None' in options:
            options.remove('None')
        distribution = RandomGenerator.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_combination_function():
        options = DefaultConfig.get_config('combination_methods')
        distribution = RandomGenerator.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_initial_model_function():
        options = DefaultConfig.get_config("initial_models")
        distribution = RandomGenerator.get_distribution(options)
        return list(distribution.keys())[0]


    @staticmethod
    def get_transformation_function(generation):
        options = DefaultConfig.get_config('network_transformations')
        if generation == 0:
            for _opt in ["remove_unit", "remove_connection"]:
                if _opt in options:
                    options.remove(_opt)
        distribution = RandomGenerator.get_distribution(options)
        return list(distribution.keys())[0]

    @staticmethod
    def get_distribution(options_list, samples=1):
        options = {}
        for key in options_list:
            if DefaultConfig.get_config(key):
                options[key] = 1
        count = len(options.keys())
        for key in options.keys():
            options[key] = 1.0 / count
        samples = RandomGenerator.choice(list(options.keys()), size=samples, p=list(options.values()))
        return Counter(samples)


    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        return RandomGenerator.get_instance()._choice(a, size, replace, p)


    def _choice(self, a, size=None, replace=True, p=None):
        result = self.rng.choice(a, size, replace, p)
        self._save_rng_state()
        return result

    @staticmethod
    def randint(low, high=None, size=None):
        return RandomGenerator.get_instance()._get_random_int(low, high, size)

    def _get_random_int(self, low, high=None, size=None):   # low = 1
        result = self.rng.randint(low, high, size)
        self._save_rng_state()
        return result

    @staticmethod
    def get_instance():
        if RandomGenerator.__instance is None:
            RandomGenerator()
        return RandomGenerator.__instance







