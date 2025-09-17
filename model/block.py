

class Block(object):
    def __init__(self, inputs, name, id, input_dimension=None, output_dimension=None, conv_kernel=None, activation=None, combination=None):
        self.inputs = inputs
        self.name = name
        self.id = id
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.conv_kernel = conv_kernel
        self.activation = activation
        self.combination = combination
        self.chains = []
        self.output = None
        self.transformations = []

    def get_str(self):
        if len(self.inputs) == 1:
            if self.activation is not None:
                return f'{self.activation}(block{self.inputs[0]})'
            elif self.conv_kernel is not None:
                return f'{self.conv_kernel}(block{self.inputs[0]})'
            else:
                return f'{self.inputs[0]}->{self.name}'
        elif len(self.inputs) == 2:
            if type(self.inputs[0]) is str and type(self.inputs[1]) is str:
                return f'block{self.inputs[0]}_{self.combination}_block{self.inputs[1]}'
            elif type(self.inputs[0]) is str:
                return f'block{self.inputs[0]}_{self.combination}_{self.inputs[1]}'
            else:
                return f'{self.inputs[0]}_{self.combination}_block{self.inputs[1]}'
        else:
            return f'{self.name}'

    def validate(self, architecture, encountered, layer_id):
        if self.name in ['x', 'h', 'c', 'm']:
            return
        if self.name not in encountered.keys():
            encountered[self.name] = 0
        encountered[self.name] = encountered[self.name] + 1
        if encountered[self.name] > (len(architecture.layers[layer_id].keys()) * 3):
            raise f'A circular reference of the {self.name} block was detected in the {layer_id} layer of the {architecture.identifier} model.'
        for input_id in self.inputs:
            if type(input_id) is not int:
                architecture.layers[layer_id][input_id].validate(architecture, encountered, layer_id)

    def build_block_chain(self, architecture, layer_id):
        chain = [self.id]
        for input_id in self.inputs:
            if type(input_id) is int:
                chain.append(1)
            else:
                chain = chain + architecture.layers[layer_id][input_id].build_block_chain(architecture, layer_id)
        self.chains = chain
        return self.chains

