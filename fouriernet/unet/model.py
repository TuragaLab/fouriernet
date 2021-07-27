"""
Taken from https://github.com/wolny/pytorch-3dunet

Original License:

MIT License

Copyright (c) 2018 Adrian Wolny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch.nn as nn
import torch.nn.functional as F

from .buildingblocks import (
    Decoder,
    DoubleConv,
    Encoder,
    ExtResNetBlock,
)


class FUNet3D(nn.Module):
    """
    Fourier UNet for 3D volume reconstruction.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_shape,
        num_planes,
        scale_factors,
        final_sigmoid=False,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="crb",
        num_groups=8,
        is_segmentation=False,
        testing=False,
        conv_kernel_size=(5, 5, 5),
        conv_padding=(2, 2, 2),
        **kwargs
    ):
        super().__init__()

        num_levels = len(scale_factors)
        self.testing = testing
        self.num_planes = num_planes

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        fourier_f_maps = [f * self.num_planes for f in f_maps]

        self.encoder = d.MultiscaleFourierConv2D(
            in_channels,
            fourier_f_maps,
            in_shape,
            stride=2,
            scale_factors=scale_factors,
        )
        self.do_encoder_relu = False
        self.do_encoder_batchnorm = False
        if "do_encoder_relu" in kwargs:
            self.do_encoder_relu = True
        if "do_encoder_batchnorm" in kwargs:
            self.encoder_batchnorms = nn.ModuleList(
                [
                    nn.BatchNorm2d(fourier_f_maps[l], momentum=0.1)
                    for l in range(num_levels)
                ]
            )
            self.do_encoder_batchnorm = True

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (1, 2, 2)
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                scale_factor=(1, 2, 2),
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = list(reversed(self.encoder(x)))
        all_encoders_features = encoders_features

        # apply relu if specified
        if self.do_encoder_relu:
            encoders_features = [F.leaky_relu(ef) for ef in encoders_features]

        # apply batchnorm if specified
        if self.do_encoder_batchnorm:
            encoders_features = [
                bn(ef) for bn, ef in zip(self.encoder_batchnorms, encoders_features)
            ]

        # reshape features to 3d
        for scale_idx in range(len(encoders_features)):
            ef = encoders_features[scale_idx]
            shape = ef.shape
            encoders_features[scale_idx] = ef.view(
                -1,
                int(shape[1] / self.num_planes),
                self.num_planes,
                shape[2],
                shape[3],
            )

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # WARN(dip): we have to assign the first feature in the list to the
        # variable x explicitly because we removed the loop through encoders in
        # the original unet
        x = encoders_features[0]
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class FUNet2D(nn.Module):
    """
    Fourier UNet for 2D image reconstruction (e.g. RGB diffused/lensless camera
    images).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_shape,
        scale_factors,
        final_sigmoid=False,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="crb",
        num_groups=8,
        is_segmentation=False,
        testing=False,
        conv_kernel_size=(1, 3, 3),
        conv_padding=(0, 1, 1),
        **kwargs
    ):
        super().__init__()

        num_levels = len(scale_factors)
        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        fourier_f_maps = list(f_maps)

        self.encoder = d.MultiscaleFourierConv2D(
            in_channels,
            fourier_f_maps,
            in_shape,
            stride=2,
            scale_factors=scale_factors,
        )
        self.do_encoder_relu = False
        self.do_encoder_batchnorm = False
        if "do_encoder_relu" in kwargs:
            self.do_encoder_relu = True
        if "do_encoder_batchnorm" in kwargs:
            self.encoder_batchnorms = nn.ModuleList(
                [
                    nn.BatchNorm2d(fourier_f_maps[l], momentum=0.1)
                    for l in range(num_levels)
                ]
            )
            self.do_encoder_batchnorm = True

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (1, 2, 2)
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                scale_factor=(1, 2, 2),
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = list(reversed(self.encoder(x)))
        all_encoders_features = encoders_features

        # apply relu if specified
        if self.do_encoder_relu:
            encoders_features = [F.leaky_relu(ef) for ef in encoders_features]

        # apply batchnorm if specified
        if self.do_encoder_batchnorm:
            encoders_features = [
                bn(ef) for bn, ef in zip(self.encoder_batchnorms, encoders_features)
            ]

        # add empty depth dimension to allow 3d convs
        for scale_idx in range(len(encoders_features)):
            ef = encoders_features[scale_idx]
            shape = ef.shape
            encoders_features[scale_idx] = ef.view(
                -1,
                shape[1],
                1,
                shape[2],
                shape[3],
            )

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # WARN(dip): we have to assign the first feature in the list to the
        # variable x explicitly because we removed the loop through encoders in
        # the original unet
        x = encoders_features[0]
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        # remove empty depth dimension
        x = x.squeeze(2)

        return x


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]
