from mxnet.gluon.model_zoo import vision as models
from mxnet import init
from bilinear_kernel import bilinear_kernel

def get_model(pretrained, nclass, input_shape, batch_size, ctx):
    backbone = models.resnet18_v2(pretrained=not pretrained, ctx=ctx)
    net = nn.HybridSequential()

    with net.name_scope():
        for layer in backbone.features[:-2]:
            net.add(layer)

        net.add(
            nn.Conv2D(nclass, kernel_size=1, weight_initializer=init.Xavier()),
            nn.Conv2DTranspose(nclass, kernel_size=64, padding=16, strides=32)
        )

    if pretrained:
        net.load_params('/tmp/lyft/net_fcn.params', ctx=ctx)
    else:
        net.initialize(ctx=ctx)
        x = nd.zeros((batch_size, 3, *input_shape), ctx=ctx)
        net(x)
        shape = net[-1].weight.data().shape
        net[-1].weight.set_data(bilinear_kernel(*shape[0:3]))

    return net