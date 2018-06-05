from mxnet.gluon.loss import Loss, SoftmaxCrossEntropyLoss

class MyLoss(Loss):
    def __init__(self, axis, jaccard_weight, nclass):
        super().__init__(weight=1.0, batch_axis=0)
        self.loss = SoftmaxCrossEntropyLoss(axis=axis)
        self.axis = axis
        self.jaccard_weight = jaccard_weight
        self.nclass = nclass

    def hybrid_forward(self, F, pred, label):
        loss = self.loss(pred, label)
        if self.jaccard_weight:
            pred = F.softmax(pred, axis=self.axis)
            cls_weight = self.jaccard_weight / self.nclass
            eps = 1e-15
            for c in range(self.nclass):
                jaccard_label = label == c
                jaccard_pred = pred[:, c, :, :]
                intersection = (jaccard_label * jaccard_pred).sum()
                union = jaccard_label.sum() + jaccard_label.sum() + eps
                loss = loss + (1 - intersection / (union - intersection)) * cls_weight
            loss = loss / (1 + self.jaccard_weight)
        return loss