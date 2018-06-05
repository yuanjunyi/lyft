from mxnet.gluon import nn
from mxnet import gluon
from mxnet import image
import utils
import lyft
import data_aug
import predict

class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

def resnet18(num_classes):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.BatchNorm(),
            nn.Conv2D(64, kernel_size=3, strides=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            Residual(64),
            Residual(64),
            Residual(128, same_shape=False),
            Residual(128),
            Residual(256, same_shape=False),
            Residual(256),
            nn.Conv2D(num_classes, kernel_size=1),
            nn.Conv2DTranspose(num_classes, kernel_size=16, padding=4, strides=8)
        )
    return net

if __name__ == '__main__':
    # train_data, train_label, test_data, test_label = lyft.read_images()
    # calar_trainset = lyft.CALARSegDataset(train_data, train_label, need_to_aug=False, transform=data_aug.transform)
    # calar_testset = lyft.CALARSegDataset(test_data, test_label, need_to_aug=False, transform=None)

    # input_shape = (64, 800)
    # net = resnet18(num_classes=2)
    # net.initialize()
    # loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

    # trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lyft.baselr, 'wd': 1e-3})
    # trainset = gluon.data.DataLoader(calar_trainset, lyft.batch_size, shuffle=True, last_batch='discard')
    # testset = gluon.data.DataLoader(calar_testset, lyft.batch_size, last_batch='discard')

    # utils.train(trainset, testset, net, loss, trainer, lyft.ctx, num_epochs=lyft.nepochs,
    #     save_epochs=True, train_car=lyft.train_car, small_car=lyft.small_car)

    net_car = lyft.get_model(False, 2, input_shape=(268,800), batch_size=lyft.batch_size, small_car=False)
    net_small_car = lyft.get_model(False, 2, input_shape=(64,800), batch_size=lyft.batch_size, small_car=True)
    # net_small_car.load_params('net_small_car.params')

    

    rgb = []
    cars = []
    small_cars = []
    idx = [578, 23, 789, 56, 923, 234, 800, 698, 102]
    for i in idx:
        im = image.imread('Train/CameraRGB/%d.png' % i)
        rgb.append(im)
        cars.append(predict.predict(im, net_car, small_car=False))
        small_cars.append(predict.predict(im, net_small_car, small_car=True))

    utils.show_images(rgb, cars, small_cars)

