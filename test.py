from mxnet import image
from utils import show_rows
import matplotlib.pyplot as plt
import cv2


def annotate(img, car, road):
    car[:, :, 0] = 0
    car[:, :, 1] = 0
    car = car * 200

    road[:, :, 0] = 0
    road[:, :, 2] = 0
    road = road * 100

    return (img.astype('float32') +
            road.astype('float32') +
            car.astype('float32')).clip(0, 255).astype('uint8')


if __name__ == '__main__':
    for i in range(26, 30):
        img = image.imread('../CameraRGB/%d.png' % i)
        car = image.imread('../prediction/car%d.png' % i)
        road = image.imread('../prediction/road%d.png' % i)
        annotated = annotate(img, car, road)
        cv2.imwrite('annotated%d.png' % i,
                    cv2.cvtColor(annotated.asnumpy(), cv2.COLOR_RGB2BGR))
