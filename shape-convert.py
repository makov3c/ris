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
    input_mask -= 1
    return input_image, input_mask


input_image, input_mask = normalize(pet_img, mask_img)

print(np.min(input_image), np.max(input_image)) # vrednosti so med 0 in 1
print(np.min(ct_img), np.max(ct_img))
print(np.min(mask_img), np.max(mask_img)) # vrednosti so med -1 in 0, gl # Risanje maske



"""
# Risanje slik maske

def plot_mask_slice(mask_img):
    count = 0
    for i in range(mask_img.shape[2]):
        slice = mask_img[:, :, i]
        if (np.min(slice) == -1) and (np.max(slice) == 0) and count == 20:
            plt.imshow(slice, cmap='gray')
            plt.show()
            break
        elif (np.min(slice) == -1) and (np.max(slice) == 0):
            count += 1
            continue

plot_mask_slice(mask_img)
"""