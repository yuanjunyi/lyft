import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import lyft
import predict
import mxnet as mx
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models
from unet import UnetResnet34
from fcn import get_model

if lyft.is_unet:
    encoder = models.resnet34_v2(pretrained=False, ctx=lyft.ctx).features
    net = UnetResnet34(encoder, nclass=3, nfilter=16)
    net.load_params('/tmp/lyft/net_m_9_0.995_0.995.params', ctx=lyft.ctx)
else:
    net = get_model(pretrained=False, nclass=13, input_shape=(320,800), batch_size=lyft.batch_size)

top = np.zeros((203, 800), dtype='uint8')
bottom = np.zeros((77, 800), dtype='uint8')

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def encode_fast(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")

def postprocess(pred):
    pred = pred.asnumpy()
    if lyft.is_unet:
        pred = pred[:, 16:816]

    binary_car_result = np.where(pred==2, 1, 0).astype('uint8')
    car_result = np.vstack([top, binary_car_result, bottom])
    binary_road_result = np.where(pred==1, 1, 0).astype('uint8')
    road_result = np.vstack([top, binary_road_result, bottom])

    return car_result, road_result


def postprocess_prob(pred_prob):
    pred_prob = pred_prob.asnumpy()
    if lyft.is_unet:
        pred_prob = pred_prob[:, :, 16:816]

    binary_car_result = np.where(pred_prob[2,:,:] > 0.95, 1, 0).reshape((320, 800)).astype('uint8')
    car_result = np.vstack([top, binary_car_result, bottom])
    binary_road_result = np.where(pred_prob[1,:,:] > 0.2, 1, 0).reshape((320, 800)).astype('uint8')
    road_result = np.vstack([top, binary_road_result, bottom])

    return car_result, road_result


file = sys.argv[-1]
video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

image_batch = []
for rgb_frame in video:
    im = nd.array(rgb_frame, dtype=np.uint8).as_in_context(mx.gpu())
    image_batch.append(im)

    if len(image_batch) == lyft.batch_size:
        prediction = predict.predict_batch(image_batch, net)
        for pred in prediction:
            car_result, road_result = postprocess(pred)
            answer_key[frame] = [encode(car_result), encode(road_result)]
            # Increment frame
            frame += 1
        image_batch = []

if len(image_batch) != 0:
    for im in image_batch:
        pred = predict.predict(im, net)
        car_result, road_result = postprocess(pred)
        answer_key[frame] = [encode(car_result), encode(road_result)]
        # Increment frame
        frame += 1

# Print output in proper json format
print (json.dumps(answer_key))
