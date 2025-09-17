import copy
import random
from loggers.logger import LOG
from model.architecture import Architecture
from default_configs.default_config import DefaultConfig


LOG = LOG.get_instance().get_logger()

def crossover(architecture1, architecture2):
    architecture1 = copy.deepcopy(architecture1)
    architecture2 = copy.deepcopy(architecture2)
    layer_1= random.randint(0, DefaultConfig.get_config('num_layers') - 1)
    layer_2= random.randint(0, DefaultConfig.get_config('num_layers') - 1)
    while layer_1 == layer_2:
        layer_2 = random.randint(0, DefaultConfig.get_config('num_layers') - 1)
    new_architecture1 = Architecture()
    new_architecture1.layers = copy.deepcopy(architecture1.layers)
    new_architecture1.layers[layer_1] = copy.deepcopy(architecture2.layers[layer_1])
    new_architecture1.layers[layer_2] = copy.deepcopy(architecture2.layers[layer_2])
    new_architecture1.ancestors.append(architecture1.identifier)
    new_architecture1.ancestors.append(architecture2.identifier)
    new_architecture1.parents[architecture1.identifier] = [i for i in range(DefaultConfig.get_config('num_layers')) if i != layer_1 and i != layer_2]
    new_architecture1.parents[architecture2.identifier] = [layer_1, layer_2]
    new_architecture2 = Architecture()
    new_architecture2.layers = copy.deepcopy(architecture1.layers)
    new_architecture2.layers[layer_1] = copy.deepcopy(architecture2.layers[layer_1])
    new_architecture2.layers[layer_2] = copy.deepcopy(architecture2.layers[layer_2])
    new_architecture2.ancestors.append(architecture1.identifier)
    new_architecture2.ancestors.append(architecture2.identifier)
    new_architecture2.parents[architecture1.identifier] = [i for i in range(DefaultConfig.get_config('num_layers')) if i != layer_1 and i != layer_2]
    new_architecture2.parents[architecture2.identifier] = [layer_1, layer_2]
    LOG.info(f'Perform crossover operations on the {layer_1} and {layer_2} layers of {architecture1.identifier} and {architecture2.identifier}.')
    return new_architecture1, new_architecture2