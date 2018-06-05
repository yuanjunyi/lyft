from mxnet.image import RandomCropAug, ResizeAug, HorizontalFlipAug
from mxnet.image import ColorJitterAug
import mxnet as mx

def positional_augmentation(joint):
    # Random crop
    #crop_height, crop_width = 450, 600
    #aug = RandomCropAug(size=(crop_width, crop_height))
    #aug_joint = aug(joint)

    # Deterministic resize
    #short_edge = 600
    #aug = ResizeAug(short_edge)
    #aug_joint = aug(aug_joint)

    # Horizontal flip
    aug = HorizontalFlipAug(p=1)
    aug_joint = aug(joint) # Change to aug_joint if random crop is on

    return aug_joint

def color_augmentation(base):
    # Only applied to the base image, and not the mask layers.
    aug = mx.image.ColorJitterAug(brightness=1, contrast=1, saturation=1)
    aug_base = aug(base)
    return aug_base

def joint_transform(base, mask):
    ### Join
    # Concatinate on channels dim, to obtain an 6 channel image
    # (3 channels for the base image, plus 3 channels for the mask)
    base_channels = base.shape[2] # so we know where to split later on
    joint = mx.nd.concat(base, mask, dim=2)

    ### Augmentation Part 1: positional
    aug_joint = positional_augmentation(joint)
    
    ### Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    
    ### Augmentation Part 2: color
    aug_base = aug_base.astype('float32') / 255
    aug_base = color_augmentation(aug_base).clip(0, 1)
    aug_base = (aug_base * 255).astype('uint8')

    return aug_base, aug_mask

def transform(base, mask):
    aug = HorizontalFlipAug(p=1)
    aug_base = aug(base)
    aug_mask = aug(mask)

    aug = mx.image.ColorJitterAug(brightness=1, contrast=1, saturation=1)
    aug_base = aug_base.astype('float32') / 255
    aug_base = aug(aug_base).clip(0, 1)
    aug_base = (aug_base * 255).astype('uint8')

    return aug_base, aug_mask

if __name__ == '__main__':
    import random
    import utils
    images = []
    for i in range(200):
        images.append(mx.image.imread('Train/CameraRGB/%d.png' % i))
    
    samples = random.sample(images, 5)
    augmented = []
    for img in samples:
        img = img.astype('float32') / 255
        img = color_augmentation(img).clip(0, 1)
        img = (img * 255).astype('uint8')
        augmented.append(img)
    utils.show_images(samples, augmented)
