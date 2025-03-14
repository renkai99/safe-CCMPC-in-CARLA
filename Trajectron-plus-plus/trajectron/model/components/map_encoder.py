import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMapEncoder(nn.Module):
    def __init__(self, map_channels, hidden_channels, output_size, masks, strides, patch_size):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        patch_size_x = patch_size[0] + patch_size[2]
        patch_size_y = patch_size[1] + patch_size[3]
        input_size = (map_channels, patch_size_x, patch_size_y)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float('nan'))

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hidden_channels[i-1],
                                        hidden_channels[i], masks[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, x, training):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

def pairwise(l):
    """Make a list of consecutive pairs given a list. 
    Example: [1,2,3] -> [(1,2),(2,3)]"""
    a, b = itertools.tee(l)
    next(b, None)
    return list(zip(a, b))

class CNNMapEncoderV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config['patch_size']
        self.patch_size = (
                patch_size[0] + patch_size[2],
                patch_size[1] + patch_size[3],)
        self.image_channels  = config['map_channels']
        self.hidden_channels = config['hidden_channels']
        self.output_size     = config['output_size']
        self.filters         = config['masks']
        try:
            self.strides  = config['strides']
        except:
            self.strides  = [1 for _ in self.filters]
        try:
            self.paddings = config['paddings']
        except:
            self.paddings = [0 for _ in self.filters]
        try:
            self.poolings = config['poolings']
        except:
            self.poolings = [None for _ in self.filters]
        if 'fc_features' in config:
            self.fc_features     = config.fc_features
        else:
            self.fc_features = []
        dummy = torch.ones((self.image_channels, *self.patch_size,)).unsqueeze(0) \
                * torch.tensor(float('nan'))
        conv_layers = []
        _channels = pairwise((self.image_channels, *self.hidden_channels,))
        for idx, ((_in_channel, _out_channel), _filter, _stride, _padding, _pool) in enumerate(
                zip(_channels, self.filters, self.strides, self.paddings, self.poolings)):
            conv = nn.Conv2d(_in_channel, _out_channel, _filter, stride=_stride, padding=_padding)
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU())
            if _pool == 'max':
                pool = nn.MaxPool2d(2, 2)
                conv_layers.append(pool)
            elif _pool is None:
                pass
        self.convs = nn.Sequential(*conv_layers)
        dummy = self.convs(dummy)
        self.fc_inputs = dummy.numel()
        _features = pairwise((self.fc_inputs, *self.fc_features, self.output_size))
        fc_layers = []
        for idx, (_in_features, _out_features) in enumerate(_features):
            fc = nn.Linear(_in_features, _out_features)
            fc_layers.append(fc)
            fc_layers.append(nn.ReLU())
        self.fcs = nn.Sequential(*fc_layers)

    def forward(self, x, training):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

class CNNMapEncoderV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config['patch_size']
        self.patch_size = (
                patch_size[0] + patch_size[2],
                patch_size[1] + patch_size[3],)
        self.image_channels  = config['map_channels']
        self.hidden_channels = config['hidden_channels']
        self.filters         = config['masks']
        self.output_size     = config['output_size']
        try:
            self.strides  = config['strides']
        except:
            self.strides  = [1 for _ in self.filters]
        try:
            self.paddings = config['paddings']
        except:
            self.paddings = [0 for _ in self.filters]
        try:
            self.poolings = config['poolings']
        except:
            self.poolings = [None for _ in self.filters]
        self.apply_last_cnn_activation = ('apply_last_cnn_activation' not in config) \
                or config['apply_last_cnn_activation']
        conv_layers = []
        _channels = pairwise((self.image_channels, *self.hidden_channels,))
        for idx, ((_in_channel, _out_channel), _filter, _stride, _padding, _pool) in enumerate(
                zip(_channels, self.filters, self.strides, self.paddings, self.poolings)):
            conv = nn.Conv2d(_in_channel, _out_channel, _filter, stride=_stride, padding=_padding)
            conv_layers.append(conv)
            if idx != len(_channels) - 1 or self.apply_last_cnn_activation:
                conv_layers.append(nn.ReLU())
            if _pool == 'max':
                pool = nn.MaxPool2d(2, 2)
                conv_layers.append(pool)
            elif _pool is None:
                pass
        conv_layers.append(nn.Flatten())
        self.convs = nn.Sequential(*conv_layers)
        dummy = torch.ones((self.image_channels, *self.patch_size,)).unsqueeze(0) \
                * torch.tensor(float('nan'))
        dummy = self.convs(dummy)
        self.has_fc = ('has_fc' not in config) or config['has_fc']
        if self.has_fc:
            # Add the FC layers after the CNN layers
            if 'fc_features' in config:
                self.fc_features     = config.fc_features
            else:
                self.fc_features = []
            self.apply_last_fc_activation = ('apply_last_fc_activation' not in config) \
                    or config['apply_last_fc_activation']
            self.fc_inputs = dummy.numel()
            _features = pairwise((self.fc_inputs, *self.fc_features, self.output_size))
            fc_layers = []
            for idx, (_in_features, _out_features) in enumerate(_features):
                fc = nn.Linear(_in_features, _out_features)
                fc_layers.append(fc)
                if idx != len(_features) - 1 or self.apply_last_fc_activation:
                    fc_layers.append(nn.ReLU())
            self.fcs = nn.Sequential(*fc_layers)
        else:
            assert self.output_size == dummy.numel()

    def forward(self, x, training):
        x = self.convs(x)
        if self.has_fc:
            x = self.fcs(x)
        return x
