from mxnet import image
from mxnet import nd
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss, SoftmaxCrossEntropyLoss
from mxnet.gluon.model_zoo import vision as models
from time import time
from trainer import Trainer
from unet import UnetResnet34
from loss import MyLoss
from fcn import get_model
import numpy as np
import mxnet as mx
import random
import utils
import data_aug

use_gluoncv_FCN = False
is_unet = True
batch_size = 16
nepochs = 30
baselr = 0.001
ctx = mx.gpu()


def crop_image(data):
    data = image.fixed_crop(data, x0=0, y0=203, w=800, h=320)
    if is_unet:
        h = data.shape[0]
        # padding = nd.zeros((h, 16, 3), dtype=np.uint8, ctx=ctx)
        padding = nd.zeros((h, 16, 3), dtype=np.uint8)
        return nd.concat(padding, data, padding, dim=1)
    else:
        return data


def preprocess_data(data):
    return crop_image(data)


def preprocess_label(label):
    # Only red channel is useful.
    label = label[:, :, 0]
    
    # Crop sky and most part of hood
    label = label[203:523, :]
    
    nplabel = label.asnumpy()
    
    # Set remaining hood to None
    nplabel[-26:, :][nplabel[-26:, :] == 10] = 0
    
    # Set Road Line to Road
    nplabel[nplabel == 6] = 7
    
    # Set non-road and non-vehicle to None
    nplabel[((nplabel != 7) & (nplabel != 10))] = 0
    
    # Set road to 1
    nplabel[nplabel == 7] = 1
    
    # Set vechicle to 2
    nplabel[nplabel == 10] = 2
    label = nd.array(nplabel, dtype='uint8')

    if is_unet:
        h = label.shape[0]
        padding = nd.zeros((h, 16), dtype=np.uint8)
        return nd.concat(padding, label, padding, dim=1)
    else:
        return label


def read_images():
    train_data, train_label, test_data, test_label = [], [], [], []

    for s in ['town1-1000', 'town2-1000']:
        for i in range(15):
            for j in range(25, 1025):
                if j % 10 == 0:
                    test_data.append(preprocess_data(image.imread('%s/episode%d/CameraRGB/%d.png' % (s, i, j))))
                    test_label.append(preprocess_label(image.imread('%s/episode%d/CameraSeg/%d.png' % (s, i, j))))
                else:
                    train_data.append(preprocess_data(image.imread('%s/episode%d/CameraRGB/%d.png' % (s, i, j))))
                    train_label.append(preprocess_label(image.imread('%s/episode%d/CameraSeg/%d.png' % (s, i, j))))
    print('read town1-1000 and town2-1000')

    for s in ['town1-600', 'town2-600']:
        for i in range(7, 15):
            for j in range(26, 625):
                if j % 10 == 0:
                    test_data.append(preprocess_data(image.imread('%s/episode%d/CameraRGB/%d.png' % (s, i, j))))
                    test_label.append(preprocess_label(image.imread('%s/episode%d/CameraSeg/%d.png' % (s, i, j))))
                else:
                    train_data.append(preprocess_data(image.imread('%s/episode%d/CameraRGB/%d.png' % (s, i, j))))
                    train_label.append(preprocess_label(image.imread('%s/episode%d/CameraSeg/%d.png' % (s, i, j))))
    print('read episode 7-14 of town1-600 and town2-600')

    for i in range(7):
        for j in range(300, 900):
            if j % 10 == 0:
                test_data.append(preprocess_data(image.imread('town1-600/episode%d/CameraRGB/%d.png' % (i, j))))
                test_label.append(preprocess_label(image.imread('town1-600/episode%d/CameraSeg/%d.png' % (i, j))))
            else:
                train_data.append(preprocess_data(image.imread('town1-600/episode%d/CameraRGB/%d.png' % (i, j))))
                train_label.append(preprocess_label(image.imread('town1-600/episode%d/CameraSeg/%d.png' % (i, j))))
    print('read episode 0-6 of town1-600')

    for i in range(7):
        for j in range(26, 625):
            if j % 10 == 0:
                test_data.append(preprocess_data(image.imread('town2-600/episode%d/CameraRGB/%d.png' % (i, j))))
                test_label.append(preprocess_label(image.imread('town2-600/episode%d/CameraSeg/%d.png' % (i, j))))
            else:
                train_data.append(preprocess_data(image.imread('town2-600/episode%d/CameraRGB/%d.png' % (i, j))))
                train_label.append(preprocess_label(image.imread('town2-600/episode%d/CameraSeg/%d.png' % (i, j))))
    print('read episode 0-6 of town2-600')

    for s in ['town1', 'town2-fps2']:
        for i in range(15):
            for j in range(25, 300):
                if j % 10 == 0:
                    test_data.append(preprocess_data(image.imread('%s/episode%d/CameraRGB/%d.png' % (s, i, j))))
                    test_label.append(preprocess_label(image.imread('%s/episode%d/CameraSeg/%d.png' % (s, i, j))))
                else:
                    train_data.append(preprocess_data(image.imread('%s/episode%d/CameraRGB/%d.png' % (s, i, j))))
                    train_label.append(preprocess_label(image.imread('%s/episode%d/CameraSeg/%d.png' % (s, i, j))))
    print('read town1 and town2-fps2')

    for i in range(1000):
        if i % 10 == 0:
            test_data.append(preprocess_data(image.imread('Train/CameraRGB/%d.png' % i)))
            test_label.append(preprocess_label(image.imread('Train/CameraSeg/%d.png' % i)))
        else:
            train_data.append(preprocess_data(image.imread('Train/CameraRGB/%d.png' % i)))
            train_label.append(preprocess_label(image.imread('Train/CameraSeg/%d.png' % i)))
    print('read Train')

    return train_data, train_label, test_data, test_label


def normalize_image(data):
    return (data.astype('float32') - 128) / 128


class CALARSegDataset(gluon.data.Dataset):

    def __init__(self, data, label, need_to_aug, transform):
        self.data = data
        self.label = label
        self.transform = transform
        print('Read %d images' % len(self.data))

        if need_to_aug:
            N = len(self.data)
            for i in range(N):
                aug_data, aug_label = self.transform(
                    self.data[i], self.label[i])
                self.data.append(aug_data)
                self.label.append(aug_label)
            print('Augmented to %d images' % len(self.data))

    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        data = data.transpose((2, 0, 1))
        return normalize_image(data).as_in_context(ctx), label.astype('float32').as_in_context(ctx)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_images()
    calar_trainset = CALARSegDataset(
        train_data, train_label, need_to_aug=False, transform=data_aug.transform)
    calar_testset = CALARSegDataset(
        test_data, test_label, need_to_aug=False, transform=None)

    if not use_gluoncv_FCN:

        if is_unet:
            encoder = models.resnet34_v2(pretrained=True, ctx=ctx).features
            net = UnetResnet34(encoder, nclass=3, nfilter=16)
            # encoder = models.resnet34_v2(pretrained=False, ctx=ctx).features
            # net = UnetResnet34(encoder, nclass=3, nfilter=16)
            # net.load_params('net_25_0.961_0.720.params', ctx=ctx)
        else:
            input_shape = (320, 832)    
            net = get_model(False, nclass, input_shape, batch_size)
            # net = get_model(True, nclass, input_shape, batch_size)

        loss = MyLoss(axis=1, jaccard_weight=1, nclass=3)
        net.collect_params().reset_ctx(ctx)

        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': baselr})
        trainset = gluon.data.DataLoader(
            calar_trainset, batch_size, shuffle=True, last_batch='discard')
        testset = gluon.data.DataLoader(
            calar_testset, batch_size, last_batch='discard')
        utils.train(trainset, testset, net, loss, trainer, ctx,
                    nepochs, None, True)
    else:
        trainer = Trainer(calar_trainset, calar_testset, nclass, ctx,
                          batch_size, nepochs, baselr)
        for epoch in range(nepochs):
            start = time()
            training_loss = trainer.training(epoch)
            pixAcc, mIoU = trainer.validation(epoch)

            print('Epoch %d, training loss %.3f, validation pixAcc %.3f, mIoU %.3f, time %.1f sec'
                  % (epoch, training_loss, pixAcc, mIoU, time() - start))

            trainer.net.module.save_params(
                'net_%d_%.3f_%.3f_%.3f.params' % (epoch, training_loss, pixAcc, mIoU))
    print('training finished')
