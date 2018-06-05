from mxnet import image
from utils import show_rows
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # encoder = models.resnet34_v2(pretrained=False, ctx=mx.gpu()).features
    # net = UnetResnet34(encoder, nclass=3, nfilter=16)
    # net.load_params('net_4_0.992_0.993.params', ctx=mx.gpu())
    
    # j = 23
    # for i in range(61, 71):
    #     img = image.imread('Carla/CameraRGB/F%d-%d.png' % (i, j)).as_in_context(mx.gpu())
    #     pred = predict(img, net)
    #     car_result, road_result = postprocess(pred)
    #     cv2.imwrite('car%d.png' % i, car_result)
    #     cv2.imwrite('road%d.png' % i, road_result)

    imgs = []
    labels = []
    cars = []
    roads = []

    j = 23
    for i in range(61, 71):
        img = image.imread('Carla/CameraRGB/F%d-%d.png' % (i, j))
        label = image.imread('Carla/CameraSeg/F%d-%d.png' % (i, j))[:,:,0]
        
        car = image.imread('car%d.png' % i)[:,:,0]
        road = image.imread('road%d.png' % i)[:,:,0]

        imgs.append(img)
        labels.append(label)
        cars.append(car)
        roads.append(road)
        
    show_rows([imgs, labels, cars, roads])