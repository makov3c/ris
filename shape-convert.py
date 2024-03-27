import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib


input_folder = r'images/patient_0'

pet_path = os.path.join(input_folder, 'PET.nii.gz')
ct_path = os.path.join(input_folder, 'CT.nii.gz')
mask_path = os.path.join(input_folder, 'MASK.nii.gz')

pet_img = nib.load(pet_path).get_fdata()
ct_img = nib.load(ct_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

print(pet_img.shape)