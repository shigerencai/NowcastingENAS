import torch
import torch.nn as nn


class ConvRNNCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, height, blocks):
        super(ConvRNNCell, self).__init__()
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.width = width
        self.height = height
        self.blocks = blocks
        self.conv_list = {}
        for block_id in blocks.keys():
            if blocks[block_id].conv_kernel == 'conv_1x1':
                if (blocks[block_id].inputs[0] == '0'):
                    self.conv_list[block_id] = nn.Conv2d(in_channel, num_hidden, kernel_size=1, stride=1, padding=0).cuda()
                    nn.init.normal_(self.conv_list[block_id].weight, std=0.01)
                else:
                    self.conv_list[block_id] = nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0).cuda()
                    nn.init.normal_(self.conv_list[block_id].weight, std=0.01)
            elif blocks[block_id].conv_kernel == 'conv_3x3':
                if blocks[block_id].inputs[0] == '0':
                    self.conv_list[block_id] = nn.Conv2d(in_channel, num_hidden, kernel_size=3, stride=1,padding=1).cuda()
                    nn.init.normal_(self.conv_list[block_id].weight, std=0.01)
                else:
                    self.conv_list[block_id] = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1,padding=1).cuda()
                    nn.init.normal_(self.conv_list[block_id].weight, std=0.01)
            elif blocks[block_id].conv_kernel == 'conv_5x5':
                if blocks[block_id].inputs[0] == '0':
                    self.conv_list[block_id] = nn.Conv2d(in_channel, num_hidden, kernel_size=5, stride=1,padding=2).cuda()
                    nn.init.normal_(self.conv_list[block_id].weight, std=0.01)
                else:
                    self.conv_list[block_id] = nn.Conv2d(num_hidden, num_hidden, kernel_size=5, stride=1,padding=2).cuda()
                    nn.init.normal_(self.conv_list[block_id].weight, std=0.01)

        self.conv_list = nn.ModuleDict(self.conv_list)

        self.activation_list = {"tanh": nn.Tanh().cuda(), "sigmoid": nn.Sigmoid().cuda(), "relu": nn.ReLU().cuda(),
                                "leaky_relu": nn.LeakyReLU().cuda(), "softplus": nn.Softplus().cuda()}
        self.all_blocks_output = {}

    def calculate_block_result(self, block, x_t, h_t, c_t, m_t):
        if len(block.inputs) == 1:
            if block.inputs[0] in self.all_blocks_output.keys():
                input_result = self.all_blocks_output[block.inputs[0]]
            else:
                input_block = self.blocks[block.inputs[0]]
                input_result = self.calculate_block_result(input_block, x_t, h_t, c_t, m_t)

            if block.conv_kernel is not None:
                output_result = self.conv_list[block.id](input_result)
            elif block.activation is not None:
                output_result = self.activation_list[block.activation](input_result)
            elif block.name in ['h_next', 'c_next', 'm_next']:
                output_result = input_result
            else:
                raise 'An error occurred in calculate_block_result in ConvRNNCell.'
        elif len(block.inputs) == 2:
            # 获取该块的第一个输入值
            if block.inputs[0] in self.all_blocks_output.keys():
                input_result_1 = self.all_blocks_output[block.inputs[0]]
            elif block.inputs[0] == 1:
                input_result_1 = 1
            else:
                input_block = self.blocks[block.inputs[0]]
                input_result_1 = self.calculate_block_result(input_block, x_t, h_t, c_t, m_t)
            # 获取该块的第二个输入值
            if block.inputs[1] in self.all_blocks_output.keys():
                input_result_2 = self.all_blocks_output[block.inputs[1]]
            elif block.inputs[1] == 1:
                input_result_2 = 1
            else:
                input_block = self.blocks[block.inputs[1]]
                input_result_2 = self.calculate_block_result(input_block, x_t, h_t, c_t, m_t)
            # 计算该块输出值
            if block.combination == "add":
                output_result = input_result_1 + input_result_2
            elif block.combination == "elem_mul":
                output_result = input_result_1 * input_result_2
            elif block.combination == "sub":
                output_result = input_result_1 - input_result_2
            else:
                raise 'An error occurred in calculate_block_result in ConvRNNCell.'
        else:
            raise 'An error occurred in calculate_block_result in ConvRNNCell.'
        self.all_blocks_output[block.id] = output_result
        return output_result

    def forward(self, x_t, h_t, c_t, m_t):
        self.all_blocks_output = dict()
        self.all_blocks_output['0'] = x_t
        self.all_blocks_output['1'] = h_t
        self.all_blocks_output['2'] = c_t
        self.all_blocks_output['3'] = m_t
        h_next = c_next = m_next = None
        for block_id in self.blocks.keys():
            if self.blocks[block_id].name == 'h_next':
                h_next = self.calculate_block_result(self.blocks[block_id], x_t, h_t, c_t, m_t)
            elif self.blocks[block_id].name == 'c_next':
                c_next = self.calculate_block_result(self.blocks[block_id], x_t, h_t, c_t, m_t)
            elif self.blocks[block_id].name == 'm_next':
                m_next = self.calculate_block_result(self.blocks[block_id], x_t, h_t, c_t, m_t)

        return h_next, c_next, m_next

