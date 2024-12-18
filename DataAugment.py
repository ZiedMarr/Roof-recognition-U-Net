import torchvision.transforms as T
import random

#Define individual transformations
horizontal_flip = T.RandomHorizontalFlip(p=1.0)
vertical_flip = T.RandomVerticalFlip(p=1.0)
rotate_and_shift = T.RandomAffine(degrees=15, translate=(0.10, 0.10))  # random rotation in range -15,15. random shift in range 10% for width qnd length
brightness_contrast = T.ColorJitter(brightness=0.3, contrast=0.3)  #random brightness and contrasts adjustments in specified ranges


#custom function for random augmentations
def apply_augmentations(image, label):
    #horizontal flip
    if random.random() > 0.5:
        image = horizontal_flip(image)
        label = horizontal_flip(label)

    #vertical flip
    if random.random() > 0.5:
        image = vertical_flip(image)
        label = vertical_flip(label)

    #rotation and shift
    if random.random() > 0.5:
        image = rotate_and_shift(image)
        label = rotate_and_shift(label)

    #brightness and contrast adjustment
    #we only apply it on the image
    if random.random() > 0.5:
        image = brightness_contrast(image)

    return image, label


