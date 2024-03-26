import torch
import numpy as np
import os
import nibabel as nib

img = nib.load(os.path.join("/media/sf_p2/RIS/data/input/images/ct/patient_0-ct.nii.gz"))

print (img.header.get_data_shape())

data = img.get_data_dtype()
print(data.type.shape)

t = torch.from_numpy(img)
print(t[0])



#output 3d tensor