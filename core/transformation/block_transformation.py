import copy
import os.path
import string
import jsonpickle
from sympy.solvers.diophantine.diophantine import transformation_to_normal

from loggers.logger import LOG
from default_configs.default_config import DefaultConfig

from model.block import Block
from random_generator import RandomGenerator


LOG = LOG.get_instance().get_logger()


class BlockTransformation:
    __instance = None

    def __init__(self, generation=0, num_layers=4):
        self.generation = generation
        self.transformations = {}
        self.num_layers = num_layers
        BlockTransformation.__instance = self

    def transform_architecture(self, architecture, transformation_count=None, initial_generation=False):
        architecture = copy.deepcopy(architecture)
        transformations_dict = {}
        if transformation_count is None:
            if self.generation == 0:
                number_of_transformations = RandomGenerator.randint(DefaultConfig.get_config('initial_population_transformations_min'), DefaultConfig.get_config('initial_population_transformations') + 1)
            else:
                number_of_transformations = RandomGenerator.randint(1, DefaultConfig.get_config('override_transformation_count') + 1)
        else:
            number_of_transformations = RandomGenerator.randint(1, transformation_count + 1)

        LOG.info('>'*10+ f'Model{architecture.identifier}' + '<'*10)
        for i in range(number_of_transformations):
            if self.generation == 0:
                layer_id = 0
            else:
                layer_id = RandomGenerator.randint(0, architecture.num_layers)
            transformation = RandomGenerator.get_transformation_function(self.generation)
            LOG.info(f'{i + 1}/{number_of_transformations} times the {transformation} operation was selected')
            if transformation == 'add_unit':
                new_transformation = self.generate_add_single_unit(architecture, layer_id)
                architecture = self.add_single_unit(architecture, new_transformation)
            elif transformation == 'remove_unit':
                new_transformation = self.generate_remove_singe_unit(architecture, layer_id)
                architecture = self.remove_single_unit(architecture, new_transformation)
            elif transformation == 'add_connection':
                new_transformation = self.generate_add_connection(architecture, layer_id)
                architecture = self.add_connection(architecture, new_transformation)
            elif transformation == 'remove_connection':
                new_transformation = self.generate_remove_connection(architecture, layer_id)
                architecture = self.remove_connection(architecture, new_transformation)
            elif transformation == 'random_change_conv_kernel':
                new_transformation = self.generate_random_change_conv_kernel(architecture, layer_id)
                architecture = self.random_change_conv_kernel(architecture, new_transformation)
            elif transformation == 'random_change_activation':
                new_transformation = self.generate_random_change_activation(architecture, layer_id)
                architecture = self.random_change_activation(architecture, new_transformation)
            elif transformation == 'random_change_combination':
                new_transformation = self.generate_random_change_combination(architecture, layer_id)
                architecture = self.random_change_combination(architecture, new_transformation)
            else:
                raise Exception(f'无法识别{transformation}变换')
            new_transformation.architecture_identifier = copy.deepcopy(architecture.identifier)
            architecture.transformation_history.append(new_transformation.hash)
            transformations_dict[new_transformation.hash] = copy.deepcopy(new_transformation)

        self.transformations[architecture.identifier] = transformations_dict
        return copy.deepcopy(architecture)

    def generate_add_single_unit(self, architecture, layer_id):
        architecture =copy.deepcopy(architecture)
        blocks_to_select_list = copy.deepcopy(architecture.get_all_block_ids(layer_id))
        blocks_to_select_list.remove('0')
        blocks_to_select_list.remove('1')
        blocks_to_select_list.remove('2')
        blocks_to_select_list.remove('3')
        for block_id in architecture.layers[layer_id].keys():
            if len(architecture.layers[layer_id][block_id].inputs) == 1 and architecture.layers[layer_id][block_id].conv_kernel is not None:
                if architecture.layers[layer_id][block_id].inputs[0] == '0':
                    blocks_to_select_list.remove(block_id)
        behind_block_id = str(RandomGenerator.choice(blocks_to_select_list))
        behind_block_inputs = []
        for input_id in architecture.layers[layer_id][behind_block_id].inputs:
            if type(input_id) is not int:
                behind_block_inputs.append(input_id)

        if len(behind_block_inputs) == 2:
            front_block_id = str(RandomGenerator.choice(behind_block_inputs))
        else:
            front_block_id = behind_block_inputs[0]

        options = RandomGenerator.choice(['activation','conv_kernel'])
        current_block_id = self.get_new_block_id(architecture, layer_id)
        return AddUnitTransformation(front_block_id, current_block_id, behind_block_id, options, layer_id)

    def add_single_unit(self, architecture, _transformation):
        architecture =copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        front_block_id = transformation.front_block_id
        current_block_id = transformation.current_block_id
        behind_block_id = transformation.behind_block_id
        options = transformation.options
        layer_id = transformation.layer_id
        if options == 'conv_kernel':
            option = RandomGenerator.get_conv_kernel_function()
            current_block_name = f'{front_block_id}_{option}'
            new_unit = Block([front_block_id], current_block_name, current_block_id, conv_kernel=option)
            architecture.add_block(new_unit, layer_id)
        else:
            option = RandomGenerator.get_activation_function()
            current_block_name = f'{front_block_id}_{option}'
            new_unit = Block([front_block_id], current_block_name, current_block_id, activation=option)
            architecture.add_block(new_unit, layer_id)
        behind_block = architecture.layers[layer_id][behind_block_id]
        for i, input_id in enumerate(behind_block.inputs):
            if input_id == front_block_id:
                behind_block.inputs[i] = current_block_id

        if current_block_id in architecture.layers[layer_id][behind_block_id].inputs:
            LOG.info(f'In the {layer_id}th layer of the model, a {option} block {current_block_id} was added between the blocks {front_block_id} and {behind_block_id}.')
            architecture.verify_blocks()
            architecture.layers[layer_id][current_block_id].transformations.append(transformation)
            architecture.layers[layer_id][behind_block_id].transformations.append(transformation)
        else:
            raise Exception(f'It is impossible to add an {option} activation function block {current_block_id} between block {front_block_id} and block {behind_block_id} in layer {layer_id} of the model.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def generate_remove_singe_unit(self, architecture, layer_id):
        architecture = copy.deepcopy(architecture)
        probably_select_block_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if len(architecture.layers[layer_id][block_id].inputs) == 1 and architecture.layers[layer_id][block_id].name not in ['h_next', 'c_next', 'm_next']:
                if architecture.layers[layer_id][block_id].inputs[0] != '0' or architecture.layers[layer_id][block_id].conv_kernel ==  None:
                    probably_select_block_id.append(block_id)

        if len(probably_select_block_id) == 0:
            LOG.info(f'In the {layer_id} layer of the model, there are 0 single-input blocks that can be deleted. Skip the "remove_single_unit" block.')
            return RemoveUnitTransformation(None, None, None)
        else:
            current_block_id = str(RandomGenerator.choice(probably_select_block_id))
            front_block_id = architecture.layers[layer_id][current_block_id].inputs[0]
            return RemoveUnitTransformation(front_block_id, current_block_id, layer_id)

    def remove_single_unit(self, architecture, _transformation):
        if _transformation.current_block_id is None:
            return
        architecture = copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        front_block_id = transformation.front_block_id
        current_block_id = transformation.current_block_id
        layer_id = transformation.layer_id
        if architecture.layers[layer_id][current_block_id].activation is not None:
            removed_operation = copy.deepcopy(architecture.layers[layer_id][current_block_id].activation)
        else:
            removed_operation = copy.deepcopy(architecture.layers[layer_id][current_block_id].conv_kernel)

        uses_this_block_list = architecture.get_blocks_that_use(current_block_id, layer_id)
        for block_id in uses_this_block_list:
            for i, input_id in enumerate(architecture.layers[layer_id][block_id].inputs):
                if input_id == current_block_id:
                    architecture.layers[layer_id][block_id].inputs[i] = architecture.layers[layer_id][current_block_id].inputs[0]
                    if architecture.layers[layer_id][block_id].activation is not None:
                        architecture.layers[layer_id][block_id].name = f'{architecture.layers[layer_id][block_id].inputs[0]}_{architecture.layers[layer_id][block_id].activation}'
                    elif architecture.layers[layer_id][block_id].conv_kernel is not None:
                        architecture.layers[layer_id][block_id].name = f'{architecture.layers[layer_id][block_id].inputs[0]}_{architecture.layers[layer_id][block_id].conv_kernel}'
                    elif architecture.layers[layer_id][block_id].combination is not None:
                        architecture.layers[layer_id][block_id].name = f'{architecture.layers[layer_id][block_id].inputs[0]}_{architecture.layers[layer_id][block_id].inputs[1]}_{architecture.layers[layer_id][block_id].combination}'

        architecture.layers[layer_id].pop(current_block_id)

        if current_block_id not in architecture.layers[layer_id].keys():
            architecture.verify_blocks()
            architecture.removed_blocks.append(current_block_id)
            LOG.info(f'Removed the {removed_operation} block {current_block_id} from the {layer_id} layer of the model.')
        else:
            raise Exception(f'It is impossible to remove the {removed_operation} block {current_block_id} from the {layer_id} layer of the model.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def generate_add_connection(self, architecture, layer_id):
        architecture = copy.deepcopy(architecture)
        possible_front_blocks_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if architecture.layers[layer_id][block_id].name not in ['h_next', 'c_next', 'm_next', 'x']:
                possible_front_blocks_id.append(block_id)

        possible_behind_blocks_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if architecture.layers[layer_id][block_id].name not in ['x', 'h', 'c', 'm', 'h_next', 'c_next', 'm_next']:
                possible_behind_blocks_id.append(block_id)

        front_block_id = str(RandomGenerator.choice(possible_front_blocks_id))
        front_block = architecture.layers[layer_id][front_block_id]
        behind_block_id = str(RandomGenerator.choice(possible_behind_blocks_id))

        count = 0
        while behind_block_id in front_block.build_block_chain(architecture, layer_id):
            front_block_id = str(RandomGenerator.choice(possible_front_blocks_id))
            front_block = architecture.layers[layer_id][front_block_id]
            behind_block_id = str(RandomGenerator.choice(possible_behind_blocks_id))
            if count > 10:
                return AddConnectionTransformation(None, None, None, None, None)
            count += 1
        new_current_id = self.get_new_block_id(architecture, layer_id)
        combination = RandomGenerator.get_combination_function()
        return AddConnectionTransformation(front_block_id, new_current_id, behind_block_id, combination, layer_id)

    def add_connection(self, architecture, _transformation):
        if _transformation.behind_block_id is None:
            return
        architecture = copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        front_block_id = transformation.front_block_id
        new_current_id = transformation.new_current_id
        behind_block_id = transformation.behind_block_id
        combination = transformation.combination
        layer_id = transformation.layer_id
        architecture.layers[layer_id][behind_block_id].id = new_current_id
        architecture.layers[layer_id][new_current_id] = architecture.layers[layer_id].pop(behind_block_id)
        connection_block = Block([front_block_id, new_current_id], f'{front_block_id}_{new_current_id}_{combination}', behind_block_id, combination=combination)
        architecture.add_block(connection_block, layer_id)

        if new_current_id in architecture.layers[layer_id].keys() and behind_block_id in architecture.layers[layer_id].keys() and new_current_id in architecture.layers[layer_id][behind_block_id].inputs and front_block_id in architecture.layers[layer_id][behind_block_id].inputs:
            architecture.verify_blocks()
            LOG.info(f'In the {layer_id}th layer of the model, a {combination} block {behind_block_id} was added after blocks {front_block_id} and {new_current_id}.')
            architecture.layers[layer_id][front_block_id].transformations.append(transformation)
            architecture.layers[layer_id][new_current_id].transformations.append(transformation)
            architecture.layers[layer_id][behind_block_id].transformations.append(transformation)
        else:
            raise Exception(f'It is impossible to add a {combination} block {behind_block_id} after block {front_block_id} and block {new_current_id} in layer {layer_id} of the model.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def generate_remove_connection(self, architecture, layer_id):
        architecture = copy.deepcopy(architecture)
        possible_blocks_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if len(architecture.layers[layer_id][block_id].inputs) == 2:
                possible_blocks_id.append(block_id)

        if len(possible_blocks_id) == 0:
            LOG.info(f'There are no connection blocks in the {layer_id}th layer of the model, so the remove_connection operation is skipped.')
            return RemoveConnectionTransformation(None, None, None, None)

        delete_block_id = str(RandomGenerator.choice(possible_blocks_id))
        flag = True
        while flag:
            for block_id in architecture.layers[layer_id].keys():
                if delete_block_id in architecture.layers[layer_id][block_id].inputs:
                    flag = False
            if flag:
                possible_blocks_id.remove(delete_block_id)
                if len(possible_blocks_id) == 0:
                    LOG.info(f'There are no valid connection blocks in the {layer_id} layer of the model. Skip the remove_connection operation.')
                    return RemoveConnectionTransformation(None, None, None, None)
                delete_block_id = str(RandomGenerator.choice(possible_blocks_id))
        if type(architecture.layers[layer_id][delete_block_id].inputs[0]) is str and type(architecture.layers[layer_id][delete_block_id].inputs[1]) is str:
            front_block_id = str(RandomGenerator.choice(architecture.layers[layer_id][delete_block_id].inputs))
        elif type(architecture.layers[layer_id][delete_block_id].inputs[0]) is int:
            front_block_id = architecture.layers[layer_id][delete_block_id].inputs[1]
        else:
            front_block_id = architecture.layers[layer_id][delete_block_id].inputs[0]
        combination = architecture.layers[layer_id][delete_block_id].combination

        return RemoveConnectionTransformation(front_block_id, delete_block_id, combination, layer_id)

    def remove_connection(self, architecture, _transformation):
        if _transformation.delete_block_id is None:
            return
        architecture = copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        front_block_id = transformation.front_block_id
        delete_block_id = transformation.delete_block_id
        combination = transformation.combination
        layer_id = transformation.layer_id

        uses_this_block_list = architecture.get_blocks_that_use(delete_block_id, layer_id)
        for block_id in uses_this_block_list:
            for i, input_id in enumerate(architecture.layers[layer_id][block_id].inputs):
                if input_id == delete_block_id:
                    architecture.layers[layer_id][block_id].inputs[i] = front_block_id
                    if architecture.layers[layer_id][block_id].activation is not None:
                        architecture.layers[layer_id][block_id].name = f'{architecture.layers[layer_id][block_id].inputs[0]}_{architecture.layers[layer_id][block_id].activation}'
                    elif architecture.layers[layer_id][block_id].conv_kernel is not None:
                        architecture.layers[layer_id][block_id].name = f'{architecture.layers[layer_id][block_id].inputs[0]}_{architecture.layers[layer_id][block_id].conv_kernel}'
                    elif architecture.layers[layer_id][block_id].combination is not None:
                        architecture.layers[layer_id][block_id].name = f'{architecture.layers[layer_id][block_id].inputs[0]}_{architecture.layers[layer_id][block_id].inputs[1]}_{architecture.layers[layer_id][block_id].combination}'
        architecture.layers[layer_id].pop(delete_block_id)

        if delete_block_id not in architecture.layers[layer_id].keys():
            architecture.verify_blocks()
            LOG.info(f'Deleted the {combination} connection block {delete_block_id} after the {front_block_id} block in the {layer_id} layer of the model')
            architecture.layers[layer_id][front_block_id].transformations.append(transformation)
            architecture.removed_blocks.append(delete_block_id)
        else:
            raise Exception(f'It is impossible to delete the {combination} connection block {delete_block_id} that follows the block {front_block_id} in the {layer_id}th layer of the model.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def generate_random_change_conv_kernel(self, architecture, layer_id):
        architecture = copy.deepcopy(architecture)
        probably_select_block_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if architecture.layers[layer_id][block_id].conv_kernel is not None:
                probably_select_block_id.append(block_id)

        if len(probably_select_block_id) == 0:
            LOG.info(f'There are no convolution blocks in the {layer_id} layer of the model. Skip the random_change_conv_kernel operation.')
            return RandomChangeConvKernelTransformation(None, None, None, None)
        current_block_id = str(RandomGenerator.choice(probably_select_block_id))
        original_conv_kernel = copy.deepcopy(architecture.layers[layer_id][current_block_id].conv_kernel)
        options = DefaultConfig.get_config('conv_kernel')
        if original_conv_kernel in options:
            options.remove(original_conv_kernel)
        conv_kernel = str(RandomGenerator.choice(options))

        return RandomChangeConvKernelTransformation(current_block_id, conv_kernel, original_conv_kernel, layer_id)

    def random_change_conv_kernel(self, architecture, _transformation):
        if _transformation.current_block_id is None:
            return
        architecture = copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        current_block_id = transformation.current_block_id
        conv_kernel= transformation.conv_kernel
        original_conv_kernel = transformation.original_conv_kernel
        layer_id = transformation.layer_id
        architecture.layers[layer_id][current_block_id].conv_kernel = conv_kernel
        architecture.layers[layer_id][current_block_id].name = f'{architecture.layers[layer_id][current_block_id].inputs[0]}_{conv_kernel}'
        if architecture.layers[layer_id][current_block_id].conv_kernel == conv_kernel:
            architecture.verify_blocks()
            LOG.info(f'The convolution kernel {original_conv_kernel} in the block {current_block_id} of the {layer_id}th layer of the model has been changed to the convolution kernel {conv_kernel}.')
            architecture.layers[layer_id][current_block_id].transformations.append(transformation)
        else:
            raise Exception(f'It is impossible to convert the {original_conv_kernel} convolution kernel in block {current_block_id} of layer {layer_id} of the model into the {conv_kernel} convolution kernel.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def generate_random_change_activation(self, architecture, layer_id):
        architecture = copy.deepcopy(architecture)
        probably_select_block_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if architecture.layers[layer_id][block_id].activation is not None:
                probably_select_block_id.append(block_id)

        if len(probably_select_block_id) == 0:
            LOG.info(f'There is no activation function block in the {layer_id}th layer of the model. Skip the random_change_activation operation.')
            return RandomChangeActivationTransformation(None, None, None, None)
        current_block_id = str(RandomGenerator.choice(probably_select_block_id))
        original_activation = copy.deepcopy(architecture.layers[layer_id][current_block_id].activation)
        options = DefaultConfig.get_config('activation_functions')
        if original_activation in options:
            options.remove(original_activation)
        activation = str(RandomGenerator.choice(options))

        return RandomChangeActivationTransformation(current_block_id, activation, original_activation, layer_id)

    def random_change_activation(self, architecture, _transformation):
        if _transformation.current_block_id is None:
            return
        architecture = copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        current_block_id = transformation.current_block_id
        activation = transformation.activation
        original_activation = transformation.original_activation
        layer_id = transformation.layer_id
        architecture.layers[layer_id][current_block_id].activation = activation
        architecture.layers[layer_id][current_block_id].name = f'{architecture.layers[layer_id][current_block_id].inputs[0]}_{activation}'
        if architecture.layers[layer_id][current_block_id].activation == activation:
            architecture.verify_blocks()
            LOG.info(f'The activation function {original_activation} in the block {current_block_id} of the model has been changed to {activation}.')
            architecture.layers[layer_id][current_block_id].transformations.append(transformation)
        else:
            raise Exception(f'It is impossible to transform the activation function {original_activation} in the block {current_block_id} of the model into the activation function {activation}.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def generate_random_change_combination(self, architecture, layer_id):
        architecture = copy.deepcopy(architecture)
        possible_blocks_id = []
        for block_id in architecture.get_all_block_ids(layer_id):
            if len(architecture.layers[layer_id][block_id].inputs) == 2:
                possible_blocks_id.append(block_id)

        if len(possible_blocks_id) == 0:
            LOG.info(f'There are no connection blocks in the {layer_id} layer of the model. Skip the random_change_combination operation.')
            return RandomChangeCombinationTransformation(None, None, None, None)

        current_block_id = str(RandomGenerator.choice(possible_blocks_id))
        original_combination = architecture.layers[layer_id][current_block_id].combination
        options = DefaultConfig.get_config('combination_methods')
        if original_combination in options:
            options.remove(original_combination)
        combination = str(RandomGenerator.choice(options))

        return RandomChangeCombinationTransformation(current_block_id, combination, original_combination, layer_id)

    def random_change_combination(self, architecture, _transformation):
        if _transformation.current_block_id is None:
            return
        architecture = copy.deepcopy(architecture)
        transformation = copy.deepcopy(_transformation)
        current_block_id = transformation.current_block_id
        combination = transformation.combination
        orginal_combination = transformation.orginal_combination
        layer_id = transformation.layer_id
        architecture.layers[layer_id][current_block_id].combination = combination
        architecture.layers[layer_id][current_block_id].name = f'{architecture.layers[layer_id][current_block_id].inputs[0]}_{architecture.layers[layer_id][current_block_id].inputs[1]}_{combination}'
        if architecture.layers[layer_id][current_block_id].combination == combination:
            architecture.verify_blocks()
            LOG.info(f'The combination method of {orginal_combination} in the {current_block_id} block of the {layer_id} layer of the model has been changed to {combination}.')
            architecture.layers[layer_id][current_block_id].transformations.append(transformation)
        else:
            raise Exception(f'It is impossible to transform the combination method of {orginal_combination} in the {current_block_id} block of the {layer_id} layer of the model into the combination method of {combination}.')
        if self.generation == 0:
            for layer_id in range(1, self.num_layers):
                architecture.layers[layer_id] = architecture.layers[0]
        return architecture

    def get_new_block_id(self, architecture, layer_id):
        max_id = -1
        for block_id in architecture.layers[layer_id].keys():
            if int(block_id) > max_id:
                max_id = int(block_id)
        return str(max_id + 1)


class AddUnitTransformation(object):
    def __init__(self, front_block_id, current_block_id, behind_block_id, options, layer_id):
        self.front_block_id = front_block_id
        self.current_block_id = current_block_id
        self.behind_block_id = behind_block_id
        self.options= options
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'In the {self.architecture_identifier} model, at the {self.layer_id} layer, add a {self.options} block {self.current_block_id} between the block {self.front_block_id} and the block {self.behind_block_id}.'


class RemoveUnitTransformation(object):
    def __init__(self, front_block_id, current_block_id, layer_id):
        self.front_block_id = front_block_id
        self.current_block_id = current_block_id
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Remove the block {self.current_block_id} from the layer {self.layer_id} of the model {self.architecture_identifier}.'

class AddConnectionTransformation(object):
    def __init__(self, front_block_id, new_current_id, behind_block_id, combination, layer_id):
        self.front_block_id = front_block_id
        self.new_current_id = new_current_id
        self.behind_block_id = behind_block_id
        self.combination = combination
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'In the model {self.architecture_identifier} at the {self.layer_id} layer, after the blocks {self.front_block_id} and {self.new_current_id}, the {self.combination} block {self.behind_block_id} was added.'


class RemoveConnectionTransformation(object):
    def __init__(self, front_block_id, delete_block_id, combination, layer_id):
        self.front_block_id = front_block_id
        self.delete_block_id = delete_block_id
        self.combination = combination
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Removed the {self.combination} block {self.delete_block_id} from the {self.layer_id} layer of the model {self.architecture_identifier}.'


class RandomChangeConvKernelTransformation(object):
    def __init__(self, current_block_id, conv_kernel, original_conv_kernel, layer_id):
        self.current_block_id = current_block_id
        self.conv_kernel = conv_kernel
        self.original_conv_kernel = original_conv_kernel
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Change the convolution kernel {self.original_conv_kernel} in the block {self.current_block_id} of the {self.layer_id}-th layer of the model {self.architecture_identifier} to the convolution kernel {self.conv_kernel}.'

class RandomChangeActivationTransformation(object):
    def __init__(self, current_block_id, activation, original_activation, layer_id):
        self.current_block_id = current_block_id
        self.activation = activation
        self.original_activation = original_activation
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Change the activation function in the block {self.current_block_id} of the {self.layer_id}th layer of the model {self.architecture_identifier} to the activation function {self.activation}.'

class RandomChangeCombinationTransformation(object):
    def __init__(self, current_block_id, combination, orginal_combination, layer_id):
        self.current_block_id = current_block_id
        self.combination = combination
        self.orginal_combination = orginal_combination
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'The combination method of {self.orginal_combination} in the {self.current_block_id} block of the {self.layer_id} layer of the model {self.architecture_identifier} has been changed to {self.combination}.'

class DifferentiableCombineActivateGateTransformation(object):
    def __init__(self, current_block_id, input_id_1, input_id_2, conv_kernel, combination, layer_id):
        self.current_block_id = current_block_id
        self.input_id_1 = input_id_1
        self.input_id_2 = input_id_2
        self.conv_kernel = conv_kernel
        self.combination = combination
        self.layer_id = layer_id
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'sigmoid((block {self.input_id_1} processed by {self.conv_kernel}) + (block {self.input_id_2} processed by {self.conv_kernel})) {self.combination} {self.current_block_id}.}'



