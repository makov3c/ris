import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt


input_folder = r'patient_0'

pet_path = os.path.join(input_folder, 'PET.nii.gz')
ct_path = os.path.join(input_folder, 'CT.nii.gz')
mask_path = os.path.join(input_folder, 'MASK.nii.gz')

pet_img = nib.load(pet_path).get_fdata()
ct_img = nib.load(ct_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

print(pet_img.shape)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    
    return input_image, input_mask


input_image, input_mask = normalize(pet_img, mask_img)

print(np.min(input_image), np.max(input_image)) # vrednosti so med 0 in 1
print(np.min(ct_img), np.max(ct_img))
print(np.min(mask_img), np.max(mask_img)) # vrednosti so med 0 in 1, gl # Risanje maske



"""
# Risanje slik maske

def plot_mask_slice(mask_img):
    count = 0
    for i in range(mask_img.shape[2]):
        slice = mask_img[:, :, i]
        if (np.min(slice) == 0) and (np.max(slice) == 1) and count == 20:
            plt.imshow(slice, cmap='gray')
            plt.show()
            break
        elif (np.min(slice) == 0) and (np.max(slice) == 1):
            count += 1
            continue

plot_mask_slice(mask_img)
"""

def load_image(datapoint): #datapoint je dictionary 
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']

    # Initialize lists to hold resized slices
    resized_image_slices = []
    resized_mask_slices = []

    # Iterate over slices along the third dimension
    for i in range(input_image.shape[2]):
        # Resize each slice and append to list
        resized_image_slices.append(tf.image.resize(input_image[:, :, i:i+1], (400, 400)))
        resized_mask_slices.append(tf.image.resize(input_mask[:, :, i:i+1], (400, 400), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

    # Stack resized slices along third dimension
    input_image = tf.stack(resized_image_slices, axis=2)
    input_mask = tf.stack(resized_mask_slices, axis=2)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# Create dictionary to hold image and mask
datapoint = {'image': pet_img, 'segmentation_mask': mask_img}

load_image(datapoint)