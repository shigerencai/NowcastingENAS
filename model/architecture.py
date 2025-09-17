from default_configs.default_config import DefaultConfig


class Architecture(object):
    def __init__(self, identifier=None):
        self.identifier = identifier
        self.num_layers = DefaultConfig.get_config('num_layers')
        self.layers = [{} for i in range(self.num_layers)]
        self.generation = 0
        self.transformation_history = []
        self.removed_blocks = []
        self.ancestors = []
        self.parents = {}
        self.output_block_name = ['h_next', 'c_next', 'm_next']

    def add_block(self, block, layer_id):
        if block.id in self.layers[layer_id].keys():
            print(f'The {layer_id} layer already contains the block {block.id}')
        self.layers[layer_id][block.id] = block

    def get_block_strings(self):
        all_block_strings = {}
        for layer_id in range(self.num_layers):
            block_strings = {}
            for block_id in self.layers[layer_id].keys():
                block = self.layers[layer_id][block_id]
                for input_id in block.inputs:
                    if type(input_id) is not int:
                        if input_id not in block_strings.keys():
                            block_strings[input_id] = self.layers[layer_id][input_id].get_str()
                if block_id not in block_strings.keys():
                    block_strings[block_id] = block.get_str()
            all_block_strings[str(layer_id)] = block_strings
        return all_block_strings

    def similar_with_other(self, other):
        similarity_count = 0
        for layer_id in range(self.num_layers):
            block_strings = []
            for block_id in self.layers[layer_id].keys():
                block_strings.append(self.layers[layer_id][block_id].get_str())
            for block_id in other.layers[layer_id].keys():
                other_block = other.layers[layer_id][block_id].get_str()
                if other_block in block_strings:
                    similarity_count += 1
            if similarity_count != len(self.layers[layer_id].keys()) or len(self.layers[layer_id].keys()) != len(other.layers[layer_id].keys()):
                return False
        return True


    def get_all_block_ids(self, layer_id):
        return list(self.layers[layer_id].keys())


    def verify_blocks(self):
        for layer_id in range(self.num_layers):
            output_block_id_list = []
            for block_id in self.layers[layer_id].keys():
                if self.layers[layer_id][block_id].name in self.output_block_name:
                    output_block_id_list.append(block_id)
            for output_block_id in output_block_id_list:
                inputs = self.layers[layer_id][output_block_id].inputs
                for input in inputs:
                    encountered = {}
                    if type(input) is not int:
                        self.layers[layer_id][input].validate(self, encountered, layer_id)

    def get_blocks_that_use(self, input_id, layer_id):
        blocks_that_use = set()
        for block_id in self.layers[layer_id].keys():
            if block_id != input_id:
                if input_id in self.layers[layer_id][block_id].inputs:
                    blocks_that_use.add(block_id)
        return list(blocks_that_use)

    def update_generation(self):
        self.generation += 1

    def get_parent(self):
        if len(self.ancestors) < 1:
            return None
        return self.ancestors[-1]

