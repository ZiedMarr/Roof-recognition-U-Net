import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import DataAugment
import tests
from RoofDataset import RoofDataset
from PIL import Image
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# Directories for images, labels and outputs
image_dir = './dida_test_task/images'
label_dir = './dida_test_task/labels'
output_dir = './outputs'

#define a transform :
transform = transforms.ToTensor()

augmentation = DataAugment.apply_augmentations
# Create datasets
train_dataset = RoofDataset(image_dir=image_dir, label_dir=label_dir, transform=transform,augmentation=augmentation, labeled=True)
predict_dataset = RoofDataset(image_dir=image_dir, label_dir=label_dir, transform=transform, labeled=False)



# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)


#########################################################################################################

#################### Defining the Neural Network ########################################################

#########################################################################################################

# We are using a U-Net since it is proven to have good performance in semantic segmentation
# To define the U-Net, we first define the 'building blocks' of the U-Net : Convolutional block, Encoder block and Decoder block


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# the Encoder block outputs 2 Tensors: p is going to be passed to the next encoder block or bottleneck.
# And x is going to be passed to the 'parallel' Decoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv_block(x)
        return x

# Now we define the U-Net using the 'building blocks'
# This is an implementation of the U-Net defined in the paper : "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Olaf Ronneberger, Philipp Fischer, Thomas Brox)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        #encoder
        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        #bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        #decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        #output layer
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        #Encoder path
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        #Bottleneck
        b1 = self.bottleneck(p4)

        #Decoder path
        d1 = self.decoder1(b1, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        # output layer
        outputs = self.outputs(d4)

        return outputs

#########################################################################################################

#################### Defining the Loss Function, optimizer ########################################################

#########################################################################################################

#define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#initiate model
model = UNet().to(device)

#define the Loss function
#we use BCEWithLogitsLoss as it works well for semantic segmentation

#Since we have only one class (roof) we use a size 1 Tensor for the wight
#The pos_weight is used to adjust the loss function based on the imbalance between positive and negative samples
#In the provided Dataset the negative area : "no-roof" is bigger than the positive are : "roof" in most samples
pos_weight = torch.tensor([2.645]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# define the optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#########################################################################################################

#################### Defining the Training loop ########################################################

#########################################################################################################


#set the number of epochs
num_epochs = 30

#nve model and criterion to the appropriate device
model = model.to(device)
criterion = criterion.to(device)


#Training Loop
for epoch in range(num_epochs):
    model.train()
    #initialize accumulated loss to evaluate the model while training
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add current loss to the accumulated loss
        running_loss += loss.item()

    #calculating the average loss
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")




#########################################################################################################

#################### Defining the Evaluation using the non-labeled images ########################################################

#########################################################################################################

#set the model to evaluation mode
model.eval()
# generate predictions on test images
with torch.no_grad():
    for idx, image in enumerate(predict_loader):
        image = image.to(device)
        output = model(image)

        # Apply sigmoid and threshold to get a binary mask
        output = torch.sigmoid(output)  # Converts logits to probabilities

        # apply threshold to get binary mask (roof or no roof)
        # since our model outputs a gray-scale image, because having the outputs
        # in a continuous space makes the training more efficient
        # we set the threshhold to defining wether a pixel is white(1) or black(0) to be 0.5
        pred_mask = (output > 0.5).float()




        # Convert the tensor to a PIL image
        pred_mask_np = pred_mask.squeeze().cpu().numpy()  # Convert to numpy for saving
        pred_mask_img = Image.fromarray((pred_mask_np * 255).astype(np.uint8))  # Scale to 0-255

        # Save the image to the output directory
        output_path = os.path.join(output_dir, f"predicted_label_{idx}.png")
        pred_mask_img.save(output_path)

        print(f"Saved predicted label for image {idx} at {output_path}")

        #Display the original image and the predicted mask
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask.cpu().squeeze(), cmap="gray")
        plt.title("Predicted Roof Mask")

        plt.show()

