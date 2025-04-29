import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

from models.helpers.utils import make_divisible


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class CondBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_devices, eps=1e-5, momentum=0.1):
        super(CondBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.num_devices = num_devices
        self.eps = eps
        self.momentum = momentum

        # Initialize weights and biases for each device
        self.weight = nn.Parameter(torch.ones(num_devices, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_devices, num_features, 1, 1))

    def forward(self, x, device_ids):
        # device_ids should be a tensor of shape (batch_size,)
        enocoded_ids = {'a': 0, 'b': 1, 'c': 2, 's1':3, 's2': 4, 's3': 5}
        device_ids = torch.tensor([enocoded_ids[id] for id in device_ids], dtype=torch.long, device=x.device)
        batch_size = x.size(0)
        assert device_ids.size(0) == batch_size, "device_ids must match the batch size"

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Loop through each sample in the batch and apply conditional normalization
        for i in range(batch_size):
            device_id = device_ids[i].item()  # Get device ID for the current sample
            weight = self.weight[device_id]  # Get the weight for the current device
            bias = self.bias[device_id]      # Get the bias for the current device

            # Calculate the mean and variance for batch normalization
            mean = x[i].mean(dim=[1, 2], keepdim=True)
            var = x[i].var(dim=[1, 2], keepdim=True)

            # Apply normalization for the current sample
            x_normalized = (x[i] - mean) / torch.sqrt(var + self.eps)
            output[i] = weight * x_normalized + bias  # Apply scaling and shifting

        return output

class Conv2dNormActivationCond(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, groups=1, norm_layer=None, activation_layer=nn.ReLU, inplace=True, num_devices=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(out_channels, num_devices)
        else:
            self.norm = None
        if activation_layer is not None:
            self.act = activation_layer(inplace=inplace)
        else:
            self.act = None

    def forward(self, x, device_id):

        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, device_id)
        if self.act is not None:
            x = self.act(x)
        return x

class SequentialWithDeviceID(nn.Sequential):
    def forward(self, x, device_id):
        for module in self:
            x = module(x, device_id)
        return x

class IdentityWithDeviceID(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, device_id):
        return self.module(x)


class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride, 
            num_devices
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # create the three factorized convs that make up the inverted bottleneck block
        self.exp_conv = Conv2dNormActivationCond(in_channels,
                                        exp_channels,
                                        kernel_size=1,
                                        stride=1,
                                        norm_layer=CondBatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False,
                                        num_devices=num_devices
                                        )

        # depthwise convolution with possible stride
        self.depth_conv =Conv2dNormActivationCond(exp_channels,
                                          exp_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=exp_channels,
                                          norm_layer=CondBatchNorm2d,
                                          activation_layer=nn.ReLU,
                                          inplace=False,
                                          num_devices=num_devices
                                          )

        self.proj_conv = Conv2dNormActivationCond(exp_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         norm_layer=CondBatchNorm2d,
                                         activation_layer=None,
                                         inplace=False,
                                         num_devices=num_devices
                                         )
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels and (stride == 1 or stride == (1, 1)):
            self.use_shortcut = True
            self.shortcut = nn.Identity()
        elif in_channels == out_channels:
            self.use_shortcut = True
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.use_shortcut = True
            self.shortcut = SequentialWithDeviceID(
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity(),
                Conv2dNormActivationCond(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                        norm_layer=CondBatchNorm2d, activation_layer=None, num_devices=num_devices)
            )


    def forward(self, x, device_id):
        out = self.exp_conv(x, device_id)
        out = self.depth_conv(out, device_id)
        out = self.proj_conv(out, device_id)

        if self.use_shortcut:
            if isinstance(self.shortcut, nn.Sequential):
                shortcut = self.shortcut[0](x)  # AvgPool2d or Identity
                shortcut = self.shortcut[1](shortcut, device_id)  # CondConv needs device ID
            else:
                shortcut = self.shortcut(x)
            out = out + shortcut

        out = self.after_block_activation(out)
        return out



class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)
        num_devices = config['num_devices']

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0

        self.in_c = SequentialWithDeviceID(
            Conv2dNormActivationCond(in_channels,
                                 channels_per_stage[0] // 4,
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
            Conv2dNormActivationCond(channels_per_stage[0] // 4,
                                 channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
        )

        self.stages = SequentialWithDeviceID()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate,
                                     num_devices=num_devices
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        ff_list = []
        ff_list += [Conv2dNormActivationCond(
            channels_per_stage[-1],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            norm_layer=CondBatchNorm2d,
            activation_layer=None,
            num_devices=num_devices
        )]

        

        ff_list.append(IdentityWithDeviceID(nn.AdaptiveAvgPool2d((1, 1))))


        self.feed_forward = SequentialWithDeviceID(
            *ff_list
        )

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate,
                    num_devices):
        stage = SequentialWithDeviceID()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate,
                num_devices=num_devices
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate,
                    num_devices
                    ):

        block = Block(in_channels,
                      out_channels,
                      expansion_rate,
                      stride,
                      num_devices
                      )
        return block

    def _forward_conv(self, x, device_id):
        x = self.in_c(x, device_id)
        x = self.stages(x, device_id)
        return x

    def forward(self, x, device_id):
        x = self._forward_conv(x, device_id)
        x = self.feed_forward(x, device_id)
        logits = x.squeeze(2).squeeze(2)
        return logits


def get_model(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=2.3, expansion_rate=3.0,
              n_blocks=(3, 2, 1), strides=None, num_devices= 9):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @param num_devices: number of devices
    @return: full neural network model based on the specified configs
    """

    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides,
        "num_devices" : num_devices
    }

    m = Network(model_config)
    return m
