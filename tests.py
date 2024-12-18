#this code is used to test some essential functionnalities of the program


import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms
import DataAugment
from RoofDataset import RoofDataset
from torch.utils.data import DataLoader
from PIL import Image

#Testing the Dataset and Dataloader
def show_images_with_labels(data_loader, num_images=3):
    images_shown = 0
    plt.figure(figsize=(10, num_images * 5))

    # Iterate through the DataLoader
    for images, labels in data_loader:
        if images_shown >= num_images:
            break  # Show only the specified number of images

        # Denormalize and convert tensor to PIL images for display
        image = F.to_pil_image(images[0])  # Convert the first image in the batch
        label = F.to_pil_image(labels[0])  # Convert the corresponding label

        # Plot the image and label side by side
        plt.subplot(num_images, 2, images_shown * 2 + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(num_images, 2, images_shown * 2 + 2)
        plt.imshow(label, cmap='gray')
        plt.title("Label")
        plt.axis("off")

        images_shown += 1
        plt.show()


######################################


# Function to visualize augmentations
def visualize_augmentation(image_path, mask_path, num_augmentations=3):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    plt.figure(figsize=(10, 5))

    # Display the original image and mask
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, num_augmentations + 1, num_augmentations + 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Original Mask")
    plt.axis("off")

    # Display augmented images and masks
    for i in range(num_augmentations):
        aug_image, aug_mask = DataAugment.apply_augmentations(image.copy(), mask.copy())

        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(aug_image)
        plt.title(f"Augmented Image {i + 1}")
        plt.axis("off")

        plt.subplot(2, num_augmentations + 1, num_augmentations + i + 3)
        plt.imshow(aug_mask, cmap="gray")
        plt.title(f"Augmented Mask {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage of show_images_with_labels :
# Directories for images and labels
#image_dir = './dida_test_task/images'
#label_dir = './dida_test_task/labels'

#define a transform :
#transform = transforms.ToTensor()
# Create dataset
#example_dataset = RoofDataset(image_dir=image_dir, label_dir=label_dir, transform=transform, labeled=True)
#create Dataloader
#example_loader = DataLoader(example_dataset, batch_size=4, shuffle=True)
#show_images_with_labels(example_loader)


# Example usage of visualize_augmentation
#image_path = './dida_test_task/images/121.png'
#mask_path = './dida_test_task/labels/121.png'
#visualize_augmentation(image_path, mask_path)

