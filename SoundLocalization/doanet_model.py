import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.relu_(self.bn(self.conv(x)))

class CRNN(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.use_hnet = params['use_hnet']
        self.use_activity_out = not params['use_dmot_only']
        self.conv_block_list = torch.nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = int(params['nb_cnn2d_filt'] * (in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        self.fnn_act_list = torch.nn.ModuleList()
        if self.use_hnet and self.use_activity_out:
            for fc_cnt in range(params['nb_fnn_act_layers']):
                self.fnn_act_list.append(
                    torch.nn.Linear(params['fnn_act_size'] if fc_cnt else params['rnn_size'] , params['fnn_act_size'], bias=True)
                )
            self.fnn_act_list.append(
                torch.nn.Linear(params['fnn_act_size'] if params['nb_fnn_act_layers'] else params['rnn_size'], params['unique_classes'], bias=True)
            )

    def forward(self, x):
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            x = torch.tanh(x)
        x_rnn = x
        for fnn_cnt in range(len(self.fnn_list)-1):
            x = torch.relu_(self.fnn_list[fnn_cnt](x))
        doa = torch.tanh(self.fnn_list[-1](x))
        if self.use_hnet and self.use_activity_out:
            for fnn_cnt in range(len(self.fnn_act_list)-1):
                x_rnn = torch.relu_(self.fnn_act_list[fnn_cnt](x_rnn))
            activity = torch.tanh(self.fnn_act_list[-1](x_rnn))
            return doa, activity
        else:
            return doa