import torch
import torch.nn as nn
from core.builder.build_conv_rnn_cell import ConvRNNCell
from default_configs.default_config import DefaultConfig


class ConvRNN(nn.Module):
    def __init__(self, architecture, data_name):
        super(ConvRNN, self).__init__()
        self.data_name = data_name
        self.model_identifier = architecture.identifier
        self.num_layers = architecture.num_layers
        self.input_length = DefaultConfig.get_config('input_length')
        self.total_length = DefaultConfig.get_config('total_length')
        self.frame_channel = DefaultConfig.get_config(f'{data_name}_img_channel') * DefaultConfig.get_config('patch_size')**2
        self.num_hidden = DefaultConfig.get_config('num_hidden')
        self.patch_size = DefaultConfig.get_config('patch_size')
        self.img_width = DefaultConfig.get_config(f'{data_name}_img_width')
        self.img_height = DefaultConfig.get_config(f'{data_name}_img_height')

        cell_list = []
        width = self.img_width // self.patch_size
        height = self.img_height // self.patch_size
        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden
            cell_list.append(
                             ConvRNNCell(in_channel, self.num_hidden, width, height, architecture.layers[i])
                            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden, self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true):
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden, height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
        m_t= torch.zeros([batch, self.num_hidden, height, width]).cuda()
        x_gen = 0
        for t in range(self.total_length-1):
            if t < self.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.input_length] * frames[:, t] + (1 - mask_true[:, t - self.input_length]) * x_gen

            h_t[0], c_t[0], m_t = self.cell_list[0](net, h_t[0], c_t[0], m_t)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], m_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames