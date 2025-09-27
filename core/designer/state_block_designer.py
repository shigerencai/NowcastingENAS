from loggers.logger import LOG
from model.block import Block
from model.architecture import Architecture
from default_configs.default_config import DefaultConfig

LOG = LOG.get_instance()

class StateBlockDesigner:
    def __init__(self, identifier):
        self.model_identifier = identifier

    @staticmethod
    def get_convlstm_architecture():
        architecture = Architecture()
        for i in range(DefaultConfig.get_config('num_layers')):
            architecture.add_block(Block([], 'x', '0'), i)
            architecture.add_block(Block([], 'h', '1'), i)
            architecture.add_block(Block([], 'c', '2'), i)
            architecture.add_block(Block([], 'm', '3'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '4', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['1'], '1_conv_3x3', '5', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['4', '5'], '4_5_add', '6', combination='add'), i)
            architecture.add_block(Block(['6'], '6_sigmoid', '7', activation='sigmoid'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '8', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['1'], '1_conv_3x3', '9', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['8', '9'], '8_9_add', '10', combination='add'), i)
            architecture.add_block(Block(['10'], '10_sigmoid', '11', activation='sigmoid'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '12', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['1'], '1_conv_3x3', '13', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['12', '13'], '12_13_add', '14', combination='add'), i)
            architecture.add_block(Block(['14'], '14_sigmoid', '15', activation='sigmoid'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '16', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['1'], '1_conv_3x3', '17', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['16', '17'], '16_17_add', '18', combination='add'), i)
            architecture.add_block(Block(['18'], '18_tanh', '19', activation='tanh'), i)
            architecture.add_block(Block(['2', '11'], '2_11_elem_mul', '20', combination='elem_mul'), i)
            architecture.add_block(Block(['7', '19'], '7_19_elem_mul', '21', combination='elem_mul'), i)
            architecture.add_block(Block(['20', '21'], '20_21_add', '22', combination='add'), i)
            architecture.add_block(Block(['22'], '22_tanh', '23', activation='tanh'), i)
            architecture.add_block(Block(['15', '23'], '15_23_elem_mul', '24', combination='elem_mul'), i)
            architecture.add_block(Block(['24'], 'h_next', '25'), i)
            architecture.add_block(Block(['22'], 'c_next', '26'), i)
            architecture.add_block(Block(['3'], 'm_next', '27'), i)
        return architecture

    @staticmethod
    def get_convgru_architecture():
        architecture = Architecture()
        for i in range(DefaultConfig.get_config('num_layers')):
            architecture.add_block(Block([], 'x', '0'), i)
            architecture.add_block(Block([], 'h', '1'), i)
            architecture.add_block(Block([], 'c', '2'), i)
            architecture.add_block(Block([], 'm', '3'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '4', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['1'], '1_conv_3x3', '5', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['4', '5'], '4_5_add', '6', combination='add'), i)
            architecture.add_block(Block(['6'], '6_sigmoid', '7', activation='sigmoid'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '8', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['1'], '1_conv_3x3', '9', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['8', '9'], '8_9_add', '10', combination='add'), i)
            architecture.add_block(Block(['10'], '10_sigmoid', '11', activation='sigmoid'), i)
            architecture.add_block(Block(['0'], '0_conv_3x3', '12', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['7', '1'], '7_1_elem_mul', '13', combination='elem_mul'), i)
            architecture.add_block(Block(['13'], '13_conv_3x3', '14', conv_kernel='conv_3x3'), i)
            architecture.add_block(Block(['12', '13'], '12_13_add', '15', combination='add'), i)
            architecture.add_block(Block(['15'], '15_tanh', '16', activation='tanh'), i)
            architecture.add_block(Block([1, '11'], 'one_11_sub', '17', combination='sub'), i)
            architecture.add_block(Block(['17', '1'], '17_1_elem_mul',  '18', combination='elem_mul'), i)
            architecture.add_block(Block(['11', '16'], '11_16_elem_mul', '19', combination='elem_mul'), i)
            architecture.add_block(Block(['18', '19'], '18_19_add', '20', combination='add'), i)
            architecture.add_block(Block(['20'], 'h_next', '21'), i)
            architecture.add_block(Block(['2'], 'c_next', '22'), i)
            architecture.add_block(Block(['3'], 'm_next', '23'), i)
        return architecture






