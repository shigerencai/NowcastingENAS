import time
from loggers.logger import LOG
from core.trainer.models_trainer import CIKM_ModelsTrainer,Shanghai_ModelsTrainer
from core.builder.build_convrnn import ConvRNN
from default_configs.default_config import DefaultConfig



class ObjectiveEvaluation:
    __instance = None

    def __init__(self):
        if ObjectiveEvaluation.__instance is not None:
            raise Exception('The example already exists.')
        ObjectiveEvaluation.__instance = self

    @staticmethod
    def get_instance():
        if ObjectiveEvaluation.__instance is None:
            ObjectiveEvaluation()
        return ObjectiveEvaluation.__instance

    @staticmethod
    def evaluate_cheap_objectives(architecture, model_all_values):
        number_of_all_blocks = 0
        number_of_conv_blocks = 0
        number_of_activation_blocks = 0
        number_of_combination_blocks = 0
        number_of_add_blocks = 0
        number_of_sub_blocks = 0
        number_of_mul_blocks = 0
        for layer_id in range(architecture.num_layers):
            number_of_all_blocks += len(architecture.layers[layer_id].keys())
            for blocks_id in architecture.layers[layer_id].keys():
                block = architecture.layers[layer_id][blocks_id]
                if len(block.inputs) == 2:
                    number_of_combination_blocks += 1
                    if block.combination == 'sub':
                        number_of_sub_blocks += 1
                    elif block.combination == 'elem_mul':
                        number_of_mul_blocks += 1
                    else:
                        number_of_add_blocks += 1
                elif len(block.inputs) == 1:
                    if block.conv_kernel is not None:
                        number_of_conv_blocks += 1
                    elif block.activation is not None:
                        number_of_activation_blocks += 1

        model_all_values.number_of_all_blocks = number_of_all_blocks
        model_all_values.number_of_conv_blocks = number_of_conv_blocks
        model_all_values.number_of_activation_blocks = number_of_activation_blocks
        model_all_values.number_of_combination_blocks = number_of_combination_blocks
        model_all_values.number_of_add_blocks = number_of_add_blocks
        model_all_values.number_of_sub_blocks = number_of_sub_blocks
        model_all_values.number_of_mul_blocks = number_of_mul_blocks

        return model_all_values

    @staticmethod
    def evaluate_expensive_objectives(architecture, model_all_values, data_name, generation=0):
        if data_name == 'CIKM':
            modelstrainer = CIKM_ModelsTrainer(architecture, generation)
            model_all_values = modelstrainer.train(model_all_values)
        elif data_name == 'Shanghai':
            modelstrainer = Shanghai_ModelsTrainer(architecture, generation)
            model_all_values = modelstrainer.train(model_all_values)
        is_new_best_list = model_all_values.is_objectives_new_best(DefaultConfig.get_config('compare_objectives'))
        if len(is_new_best_list) != 0:
           LOG.info(f'Model {architecture.identifier} has achieved the new optimal goal {is_new_best_list}')
        return model_all_values
