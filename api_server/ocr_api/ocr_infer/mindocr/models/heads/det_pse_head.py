from mindspore import nn
from mindspore.common.initializer import TruncatedNormal


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same', has_bias=False):
    init_value = TruncatedNormal(0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=init_value, has_bias=has_bias)


def _bn(channels, momentum=0.1):
    return nn.BatchNorm2d(channels, momentum=momentum)


class PSEHead(nn.Cell):
    """
    PSE Head module for text detection.

    This module takes a single input feature map and applies convolutional operations
    to generate the output feature map.

    Args:
       in_channels (int): The feature dimension of a single feature map generated by the neck (FPN).
       hidden_size (int): The hidden size for intermediate convolutions.
       out_channels (int): The output channel size.

    Returns:
       Tensor: The output feature map of shape [batch_size, out_channels, H, W].

    """
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.conv1 = _conv(4 * in_channels,
                           hidden_size,
                           kernel_size=3,
                           stride=1,
                           has_bias=True)
        self.bn1 = _bn(hidden_size)
        self.relu1 = nn.ReLU()
        self.conv2 = _conv(hidden_size,
                           out_channels,
                           kernel_size=1,
                           has_bias=True)

    def construct(self, feature):
        output = self.conv1(feature)
        output = self.relu1(self.bn1(output))
        output = self.conv2(output)

        return output
