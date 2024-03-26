import os
import nibabel as nib
import matplotlib.pyplot as plt

def save_images_as_png(input_folder):
    pet_path = os.path.join(input_folder, 'PET.nii.gz')
    ct_path = os.path.join(input_folder, 'CT.nii.gz')
    mask_path = os.path.join(input_folder, 'MASK.nii.gz')

    pet_img = nib.load(pet_path).get_fdata()
    ct_img = nib.load(ct_path).get_fdata()
    mask_img = nib.load(mask_path).get_fdata()

    # middle slice 
    pet_slice = pet_img[pet_img.shape[0] // 2]
    ct_slice = ct_img[ct_img.shape[0] // 2]
    mask_slice = mask_img[mask_img.shape[0] // 2]

    plt.imsave(os.path.join(input_folder, 'pet.png'), pet_slice, cmap='gray')
    plt.imsave(os.path.join(input_folder, 'ct.png'), ct_slice, cmap='gray')
    plt.imsave(os.path.join(input_folder, 'mask.png'), mask_slice, cmap='gray')


#save_images_as_png(r'\Users\Fedja\p2\RIS\data\input\images\patient_10')

#---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

def run_image(input_folder):
    pet_path = os.path.join(input_folder, 'PET.nii.gz')
    ct_path = os.path.join(input_folder, 'CT.nii.gz')
    mask_path = os.path.join(input_folder, 'MASK.nii.gz')

    pet_img = nib.load(pet_path).get_fdata()
    ct_img = nib.load(ct_path).get_fdata()
    mask_img = nib.load(mask_path).get_fdata()

    # middle slice 
    pet_slice = pet_img[pet_img.shape[0] // 2]
    ct_slice = ct_img[ct_img.shape[0] // 2]
    mask_slice = mask_img[mask_img.shape[0] // 2]

    mask_img[mask_img == 0.] = None
    # The parametrized function to be plotted
    def f(t, list):
        return list[int(t)]

    # Create the figure and the line that we will manipulate
    fig, axs = plt.subplots(1,4, sharey=True)
    line1 = axs[0].imshow(f(200, pet_img), cmap='gray')
    line1_1 = axs[0].imshow(f(200, mask_img), cmap='YlOrRd_r')
    line3 = axs[1].imshow(f(200, pet_img), cmap='gray')

    line2 = axs[2].imshow(f(200, ct_img), cmap='gray')
    line2_1 = axs[2].imshow(f(200, mask_img), cmap='YlOrRd_r')
    line4 = axs[3].imshow(f(200, ct_img), cmap='gray')
    

    axs[0].set_xlabel('PET + mask')
    axs[2].set_xlabel('CT + mask')
    axs[1].set_xlabel('PET')
    axs[3].set_xlabel('CT')

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label='time of scan',
        valmin=0,
        valmax=pet_img.shape[0]-1,
        valinit=0,
        valstep=1,
    )



    # The function to be called anytime a slider's value changes
    def update(val):
        line1.set_data(f(val, pet_img))
        line1_1.set_data(f(val, mask_img))
        line2.set_data(f(val, ct_img))
        line2_1.set_data(f(val, mask_img))
        line3.set_data(f(val, pet_img))
        line4.set_data(f(val, ct_img))

        fig.canvas.draw_idle()


    # register the update function with each slider
    freq_slider.on_changed(update)
    plt.show()

#wind
#run_image(r'\Users\Fedja\p2\RIS\data\patient\patient_10')
    
#ubuntu
run_image(r'/media/sf_p2/RIS/data/patient/patient_0')  
