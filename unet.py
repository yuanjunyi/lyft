import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet import init
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn
from bilinear_kernel import bilinear_kernel


ctx = mx.gpu()

class DecoderBlock(nn.HybridBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        with self.name_scope():
            self.block = nn.HybridSequential()
            with self.block.name_scope():
                self.block.add(
                    nn.Conv2DTranspose(in_channels=in_channels, channels=out_channels, kernel_size=4, strides=2, padding=1),
                    nn.Activation('relu'),
                    nn.BatchNorm(),
                )
            self.block.initialize(ctx=ctx)
            self.block[0].weight.set_data(bilinear_kernel(in_channels, out_channels, kernel_size=4))

    def hybrid_forward(self, F, x):
        return self.block(x)

class UnetResnet34(nn.HybridBlock):
    def __init__(self, encoder, nclass, nfilter=32):
        super().__init__()
        with self.name_scope():

            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                for layer in encoder[:5]:
                    self.conv1.add(layer)

            self.conv2 = encoder[5]
            self.conv3 = encoder[6]
            self.conv4 = encoder[7]
            self.conv5 = encoder[8]
            self.pool = nn.MaxPool2D(pool_size=2, strides=2)

            self.center = DecoderBlock(512, nfilter * 8)
            
            self.dec5 = DecoderBlock(512 + nfilter * 8, nfilter * 8)
            self.dec4 = DecoderBlock(256 + nfilter * 8, nfilter * 8)
            self.dec3 = DecoderBlock(128 + nfilter * 8, nfilter * 2)
            self.dec2 = DecoderBlock(64 + nfilter * 2, nfilter * 2 * 2)    
            self.dec1 = nn.Conv2DTranspose(in_channels=nfilter * 2 * 2, channels=nclass, kernel_size=4, strides=2, padding=1)
            self.dec1.initialize(ctx=ctx)
            self.dec1.weight.set_data(bilinear_kernel(nfilter * 2 * 2, nclass, kernel_size=4))

    def hybrid_forward(self, F, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(F.concat(center, conv5, dim=1))
        dec4 = self.dec4(F.concat(dec5, conv4, dim=1))
        dec3 = self.dec3(F.concat(dec4, conv3, dim=1))
        dec2 = self.dec2(F.concat(dec3, conv2, dim=1))
        dec1 = self.dec1(dec2)

        return dec1
