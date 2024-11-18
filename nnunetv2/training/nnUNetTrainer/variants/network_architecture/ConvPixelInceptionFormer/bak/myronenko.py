from torch import nn as nn
from .resnet import conv3x3x3, conv1x1x1
from functools import partial


class MyronenkoConvolutionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(MyronenkoConvolutionBlock, self).__init__()
        self.norm_groups = norm_groups
        if norm_layer is None:
            self.norm_layer = nn.GroupNorm
        else:
            self.norm_layer = norm_layer
        self.norm1 = self.create_norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3x3(in_planes, planes, stride, kernel_size=kernel_size)

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

    def create_norm_layer(self, planes, error_on_non_divisible_norm_groups=False):
        if planes < self.norm_groups:
            return self.norm_layer(planes, planes)
        elif not error_on_non_divisible_norm_groups and (planes % self.norm_groups) > 0:
            # This will just make a the number of norm groups equal to the number of planes
            print("Setting number of norm groups to {} for this convolution block.".format(planes))
            return self.norm_layer(planes, planes)
        else:
            return self.norm_layer(self.norm_groups, planes)


class MyronenkoResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(MyronenkoResidualBlock, self).__init__()
        self.conv1 = MyronenkoConvolutionBlock(in_planes=in_planes, planes=planes, stride=stride,
                                               norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        self.conv2 = MyronenkoConvolutionBlock(in_planes=planes, planes=planes, stride=stride, norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        if in_planes != planes:
            self.sample = conv1x1x1(in_planes, planes)
        else:
            self.sample = None

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.sample is not None:
            identity = self.sample(identity)

        x += identity

        return x


class MyronenkoLayer(nn.Module):
    def __init__(self, n_blocks, block, in_planes, planes, *args, dropout=None, kernel_size=3, **kwargs):
        super(MyronenkoLayer, self).__init__()
        self.block = block
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(block(in_planes, planes, *args, kernel_size=kernel_size, **kwargs))
            in_planes = planes
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.dropout is not None:
                x = self.dropout(x)
        return x


class MyronenkoEncoder(nn.Module):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        in_width = n_features
        for i, n_blocks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_width = layer_widths[i]
            else:
                out_width = base_width * (feature_dilation ** i)
            if dropout and i == 0:
                layer_dropout = dropout
            else:
                layer_dropout = None
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     dropout=layer_dropout, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x):
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            x = downsampling(x)
        x = self.layers[-1](x)
        return x
    




class BasicDecoder(nn.Module):
    def __init__(self, in_planes, layers, block=resnet.BasicBlock, plane_dilation=2, upsampling_mode="trilinear",
                 upsampling_scale=2):
        super(BasicDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.upsampling_mode = upsampling_mode
        self.upsampling_scale = upsampling_scale
        layer_planes = in_planes
        for n_blocks in layers:
            self.conv1s.append(resnet.conv1x1x1(in_planes=layer_planes,
                                                out_planes=int(layer_planes/plane_dilation)))
            layer = nn.ModuleList()
            layer_planes = int(layer_planes/plane_dilation)
            for i_block in range(n_blocks):
                layer.append(block(layer_planes, layer_planes))
            self.layers.append(layer)

    def forward(self, x):
        for conv1, layer in zip(self.conv1s, self.layers):
            x = conv1(x)
            x = nn.functional.interpolate(x, scale_factor=self.upsampling_scale, mode=self.upsampling_mode)
            for block in layer:
                x = block(x)
        return x


class MyronenkoDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, kernal_size=3):
        super(MyronenkoDecoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 1, 1]
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = list()
        for i, n_blocks in enumerate(layer_blocks):
            depth = len(layer_blocks) - (i + 1)
            if layer_widths is not None:
                out_width = layer_widths[depth]
                in_width = layer_widths[depth + 1]
            else:
                out_width = base_width * (feature_reduction_scale ** depth)
                in_width = out_width * feature_reduction_scale
            if use_transposed_convolutions:
                self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                      mode=upsampling_mode, align_corners=align_corners))
            else:
                self.pre_upsampling_blocks.append(nn.Sequential())
                self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernal_size,
                                                                 stride=upsampling_scale, padding=1))
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=out_width, planes=out_width,
                                     kernal_size=kernal_size))

    def forward(self, x):
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers):
            x = pre(x)
            x = up(x)
            x = lay(x)
        return x


class MirroredDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, kernel_size=3):
        super(MirroredDecoder, self).__init__()
        self.use_transposed_convolutions = use_transposed_convolutions
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)

            if depth != 0:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=in_width,
                                         kernel_size=kernel_size))
                if self.use_transposed_convolutions:
                    self.pre_upsampling_blocks.append(nn.Sequential())
                    self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernel_size,
                                                                     stride=upsampling_scale, padding=1))
                else:
                    self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                    self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                          mode=upsampling_mode, align_corners=align_corners))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernel_size))

    def calculate_layer_widths(self, depth):
        if self.layer_widths is not None:
            out_width = self.layer_widths[depth]
            in_width = self.layer_widths[depth + 1]
        else:
            if depth > 0:
                out_width = int(self.base_width * (self.feature_reduction_scale ** (depth - 1)))
                in_width = out_width * self.feature_reduction_scale
            else:
                out_width = self.base_width
                in_width = self.base_width
        return in_width, out_width

    def forward(self, x):
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1]):
            x = lay(x)
            x = pre(x)
            x = up(x)
        x = self.layers[-1](x)
        return x


class Decoder1D(nn.Module):
    def __init__(self, input_features, output_features, layer_blocks, layer_channels, block=resnet.BasicBlock1D,
                 kernel_size=3, upsample_factor=2, interpolation_mode="linear", interpolation_align_corners=True):
        super(Decoder1D, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.output_features = output_features
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.upsample_factor = upsample_factor
        in_channels = input_features
        for n_blocks, out_channels in zip(layer_blocks, layer_channels):
            layer = nn.ModuleList()
            self.conv1s.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                         stride=1, bias=False))
            for i_block in range(n_blocks):
                layer.append(block(in_channels=out_channels, channels=out_channels, kernel_size=kernel_size, stride=1))
            in_channels = out_channels
            self.layers.append(layer)

    def forward(self, x):
        for (layer, conv1) in zip(self.layers, self.conv1s):
            x = nn.functional.interpolate(x,
                                          size=(x.shape[-1] * self.upsample_factor),
                                          mode=self.interpolation_mode,
                                          align_corners=self.interpolation_align_corners)
            x = conv1(x)
            for block in layer:
                x = block(x)
        return x