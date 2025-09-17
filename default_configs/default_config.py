import os
import copy
import json



class DefaultConfig:
    __instance = None

    def __init__(self):
        self.default_configs_path = './default_configs'
        if not os.path.exists(self.default_configs_path):
            os.mkdir(self.default_configs_path)
        if not os.path.exists(self.default_configs_path+'/default_configs_dict.json'):
            self.initialise_default_config()
        self.load_config()

        DefaultConfig.__instance = self

    def initialise_default_config(self):
        default = {
            "env_name": "DEFAULT",
            "persist": True,
            "slack_webhook": -1,
            "post_update_freq": 1,
            "time_limit": -1,
            "restore_if_possible": True,
            "logging_debug": False,
            "models_parameter_save_dir": "restore/architectures/models_parameter",
            "models_architecture_save_dir": "restore/architectures/models_architecture",
            "snapshots_save_dir": "restore/snapshots",
            "valid_pred_imgs": "output/valid_pred_imgs/",
            "test_pred_imgs": "output/test_pred_imgs/",
            "experimental_pic": "output/",

            "population_size": 100,
            "population_to_select": 50,
            "number_of_generations": 20,
            "override_transformation_count": 3,
            "alternative_nsga_crowding": False,
            "initial_population_transformations_min": 1,
            "initial_population_transformations": 5,

            "dataset": "CIKM2017",
            "input_length": 10,
            "total_length": 20,
            "CIKM_img_width": 128,
            "CIKM_img_height": 128,
            "CIKM_img_channel": 1,
            "Shanghai_img_width": 512,
            "Shanghai_img_height": 512,
            "Shanghai_img_channel": 1,

            "reverse_input": True,
            "max_iterations": 10000,
            "lr": 0.0001,
            "batch_size": 4,
            "patch_size": 4,
            "num_layers": 4,
            "num_hidden": 128,
            "scheduled_sampling": 1,
            "sampling_stop_iter": 5000,
            "sampling_start_value": 1.0,
            "sampling_changing_rate": 0.00002,
            "compare_objectives": ["mae_loss", "number_of_all_blocks"],
            "cheap_objectives": ["number_of_parameters", "number_of_blocks"],
            "default_persistence_columns": ["time", "model_id", "model_hash", "number_of_blocks", "number_of_parameters", "training_time"],
            "expensive_objectives": ["mae_loss", "mse_loss", "csi", "hss"],
            "conv_kernel": ["conv_1x1", "conv_3x3", "conv_5x5"],
            "initial_models": ["convlstm", "convgru"],
            "activation_functions": ["tanh", "sigmoid", "relu", "leaky_relu", "softplus"],
            "combination_methods": ["add", "elem_mul", "sub"],
            "network_transformations": ["add_unit", "remove_unit", "add_connection", "remove_connection",
                                        "random_change_conv_kernel", "random_change_activation", "random_change_combination"],

            "conv_1x1": 1,
            "conv_3x3": 1,
            "conv_5x5": 1,

            "convlstm": 1,
            "convgru": 1,
            "predrnn": 1,

            "identity": 1,
            "tanh": 1,
            "sigmoid": 1,
            "relu": 1,
            "leaky_relu": 1,
            "None": 1,

            "add": 1,
            "elem_mul": 1,
            "sub": 1,

            "add_unit": 1,
            "remove_unit": 1,
            "add_connection": 1,
            "remove_connection": 1,
            "random_change_conv_kernel": 1,
            "random_change_activation": 1,
            "random_change_combination": 1
        }
        f = open(self.default_configs_path+'/default_configs_dict.json', 'w')
        json_object = json.dumps(default)
        f.write(json_object)
        f.close()

    def load_config(self):
        with open(self.default_configs_path+'/default_configs_dict.json') as file:
            self.config = json.load(file)

    def get_config_loaded(self, key):
        if key in self.config.keys():
            return copy.deepcopy(self.config[key])
        return None

    @staticmethod
    def get_instance():
        if DefaultConfig.__instance is None:
            DefaultConfig()
            return DefaultConfig.__instance
        else:
            return DefaultConfig.__instance

    @staticmethod
    def get_config(key):
        return DefaultConfig.get_instance().get_config_loaded(key)



