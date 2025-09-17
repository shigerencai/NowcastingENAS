import copy
from loggers.logger import LOG

LOG = LOG.get_instance().get_logger()


def mutation(architecture, transformer):
    architecture = copy.deepcopy(architecture)
    transformation_count = 3
    new_architecture = transformer.transform_architecture(architecture, transformation_count=transformation_count)
    return copy.deepcopy(new_architecture)