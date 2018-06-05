import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from gluoncv.utils import PolyLRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo.fcn import *
from gluoncv.utils.parallel import *
from tqdm import tqdm

class Trainer(object):
    def __init__(self, calar_trainset, calar_testset, nclass, ctx, batch_size, nepochs, baselr=0.001, wd=0.0001, momentum=0.9):
        self.batch_size = batch_size
        self.train_data = gluon.data.DataLoader(calar_trainset, batch_size, shuffle=True, last_batch='discard')
        self.eval_data = gluon.data.DataLoader(calar_testset, batch_size, last_batch='discard')
        
        model = FCN(nclass=nclass, backbone='resnet50', aux=True, ctx=ctx)

        ctx_list = [ctx]
        self.net = DataParallelModel(model, ctx_list)
        self.evaluator = DataParallelModel(SegEvalModel(model), ctx_list)

        criterion = SoftmaxCrossEntropyLossWithAux(aux=True)
        self.criterion = DataParallelCriterion(criterion, ctx_list)

        self.lr_scheduler = PolyLRScheduler(baselr=baselr, niters=len(self.train_data), nepochs=nepochs)
        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       {'lr_scheduler': self.lr_scheduler,
                                        'wd': wd,
                                        'momentum': momentum,
                                        'multi_precision': True})

    def training(self, epoch):
        train_loss = 0.0
        for i, (data, target) in enumerate(self.train_data):
            self.lr_scheduler.update(i, epoch)
            with autograd.record(True):
                outputs = self.net(data)
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.batch_size)
            for loss in losses:
                train_loss += loss.asnumpy()[0] / len(losses)
            mx.nd.waitall()
        return train_loss / len(self.train_data)

    def validation(self, epoch):
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        for i, (data, target) in enumerate(self.eval_data):
            outputs = self.evaluator(data, target)
            for (correct, labeled, inter, union) in outputs:
                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
            mx.nd.waitall()
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU
