from albumentations.pytorch import ToTensorV2
import albumentations as A
import mahotas
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


def resize_and_replace_images(directory_path, target_size=(224, 224)):
    """
    Resize all images in the specified directory to the target size and convert them to 1 channel (grayscale).
    The original images will be replaced.

    Parameters:
    - directory_path (str): Path to the directory containing the images.
    - target_size (tuple): The target size (width, height) to which the images will be resized.

    Returns:
    None
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is an image (you can add more image extensions if needed)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            try:
                # Open the image
                img = Image.open(file_path)

                # Resize the image to the target size
                img = img.resize(target_size, Image.ANTIALIAS)

                # Convert the image to grayscale (1 channel)
                img = img.convert('L')

                # Replace the original image with the resized and converted one
                img.save(file_path)
                print(f"Resized and replaced {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


def remove_background(image_path):
    # Load the image
    img = mahotas.imread(image_path)

    # Perform Otsu thresholding to separate foreground and background
    T_otsu = mahotas.otsu(img)
    thresholded_img = img > T_otsu

    # Create a mask to keep the foreground in grayscale and set the background to white
    grayscale_img = img.copy()
    grayscale_img[thresholded_img] = 255  # Set foreground to white

    return grayscale_img.astype(np.uint8)


class SignatureDataset(Dataset):
    def __init__(self, root_org, root_forg, transform=None):
        self.root_org = root_org
        self.root_forg = root_forg
        self.transform = transform

        self.org_img = os.listdir(root_org)
        self.forg_img = os.listdir(root_forg)
        self.length_dataset = max(len(self.org_img), len(self.forg_img))
        self.org_len = len(self.org_img)
        self.forg_len = len(self.forg_img)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        org_img = self.org_img[index % self.org_len]
        forg_img = self.forg_img[index % self.forg_len]

        org_path = os.path.join(self.root_org, org_img)
        forg_path = os.path.join(self.root_forg, forg_img)

        org_img = remove_background(org_path)
        forg_img = remove_background(forg_path)

        if self.transform:
            augmentations = self.transform(image=org_img, image0=forg_img)
            org_img = augmentations["image"]
            forg_img = augmentations["image0"]

        return org_img, forg_img, org_path, forg_path
