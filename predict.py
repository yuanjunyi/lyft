from mxnet import nd
from lyft import use_gluoncv_FCN, preprocess_data, normalize_image


def preprocess_image(img):
    return normalize_image(preprocess_data(img)).transpose((2, 0, 1))


def predict_prob(img, net):
    data = preprocess_image(img).expand_dims(axis=0)
    yhat = net(data)
    if use_gluoncv_FCN:
        yhat = yhat[0]
    pred_prob = nd.softmax(yhat, axis=1)
    return pred_prob[0]


def predict_batch_prob(img_batch, net):
    img_batch = [preprocess_image(img) for img in img_batch]
    X = nd.stack(*img_batch)
    yhat = net(X)
    if use_gluoncv_FCN:
        yhat = yhat[0]
    pred_prob = nd.softmax(yhat, axis=1)
    return pred_prob


def predict(img, net):
    data = preprocess_image(img).expand_dims(axis=0)
    yhat = net(data)
    if use_gluoncv_FCN:
        yhat = yhat[0]
    pred = nd.argmax(yhat, axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def predict_batch(img_batch, net):
    img_batch = [preprocess_image(img) for img in img_batch]
    X = nd.stack(*img_batch)
    yhat = net(X)
    if use_gluoncv_FCN:
        yhat = yhat[0]
    pred = nd.argmax(yhat, axis=1)
    return pred
