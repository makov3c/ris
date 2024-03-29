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



# Create dictionary to hold image and mask
# Ko imamo enkrat več slik, jih bomo dali v dictionary
datapoint = {'image': pet_img, 'segmentation_mask': mask_img}

def load_image(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']

    # Initialize lists to hold resized slices
    resized_image_slices = []
    resized_mask_slices = []

    # Iterate over slices along the third dimension
    for i in range(input_image.shape[2]):
        # Resize each slice and append to list
        resized_image_slices.append(tf.image.resize(input_image[:, :, i:i+1], (128, 128)))
        resized_mask_slices.append(tf.image.resize(input_mask[:, :, i:i+1], (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

    # Stack resized slices along third dimension
    input_image = tf.stack(resized_image_slices, axis=2)
    input_mask = tf.stack(resized_mask_slices, axis=2)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

input_image, input_mask = load_image(datapoint)

print(np.min(input_image), np.max(input_image)) # vrednosti so med 0 in 1
print(np.min(ct_img), np.max(ct_img))
print(np.min(mask_img), np.max(mask_img)) # vrednosti so med 0 in 1, gl # Risanje maske

print("Input Image Shape:", input_image.shape)
print("Input Mask Shape:", input_mask.shape)

# TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
# bSTEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# Treba je prebrat tole dokumentacijo
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# da lahko konstruiramo datasete pravilne oblike

train_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3]) # kako passati najine podatke sem notri v pravem formatu?
test_dataset = tf.data.Dataset.from_tensor_slices([4, 5, 6])

dataset = {'train': train_dataset, 'test': test_dataset}

# dataset is dictionary with keys 'train' and 'test', where values are tf.data.Dataset objects
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


