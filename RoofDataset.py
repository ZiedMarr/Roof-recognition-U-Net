#In this file we define the Dataset class

import os
import numpy as np
from cProfile import label

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RoofDataset(Dataset):                                             #we use the Dataset superclass
    def __init__(self, image_dir, label_dir , transform=None, augmentation=None , labeled = True ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labeled = labeled
        self.augmentation = augmentation
        # get the image names and sort them, inorder to form a label list later that is corresponding
        self.image_filenames = sorted(os.listdir(image_dir))

        #This is relevant to discriminate between the training datapoints and the 5 remaining images
        if labeled :
            #change the image_filenames list to only include the ones that have a corresponding label
            self.image_filenames = [img for img in self.image_filenames if img in os.listdir(label_dir)]
            #fill the label_filenames list which is basically the same, since they have the same names
            self.label_filenames = self.image_filenames
        else :
            # we only get the images that do not have a corresponding label
            self.image_filenames = [img for img in self.image_filenames if img not in os.listdir(label_dir)]

    #this function give out the length of the data set, meaning the number of images
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        #This function loads (image,label), or just image if in the case of non labeled

        #get the path of the image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        #load the image
        image = Image.open(image_path).convert("RGB")

        #get label
        if self.labeled :
            #in the case, we want to get a labeled dataset

            #get the path to the label
            label_path = os.path.join(self.label_dir, self.label_filenames[idx])
            #load the label
            lbl = Image.open(label_path).convert("L")  #Images are black and white so we can use grayscale


        #apply augmentation
        if self.augmentation :
            image , lbl = self.augmentation(image, lbl)

        # apply transform
        if self.transform :
            image = self.transform(image)

        if self.labeled :
            if self.transform:
                lbl = self.transform(lbl)

                #binarize the gray scale, to black and white
                lbl = (lbl > 0.5).float()
            return image, lbl
        else:
            return image



