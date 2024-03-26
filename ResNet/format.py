import os
import shutil

#main_folder = r'C:\Users\Fedja\p2\RIS\data\patient'
main_folder = r'/media/sf_p2/RIS/data/patient'

# Output folders for CT, MASK, and PET files
input_folder = r'/media/sf_p2/RIS/data/input/images'
#input_folder = r'C:\Users\Fedja\p2\RIS\data\input\images'

ct_folder = os.path.join(input_folder, 'ct')
mask_folder = os.path.join(input_folder, 'mask')
pet_folder = os.path.join(input_folder, 'pet')

os.makedirs(ct_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(pet_folder, exist_ok=True)

# Iterate through each patient folder
for patient_folder in os.listdir(main_folder):
    patient_folder_path = os.path.join(main_folder, patient_folder)
    if os.path.isdir(patient_folder_path):
        # Iterate through each file in the patient folder
        for file_name in os.listdir(patient_folder_path):
            file_path = os.path.join(patient_folder_path, file_name)
            if file_name.endswith('.nii.gz'):
                # Determine the file type (CT, MASK, or PET)
                file_type = file_name.split('.')[0].lower()
                # Create the new file name
                new_file_name = f'{patient_folder}-{file_type}.nii.gz'
                # Move the file to the appropriate output folder
                if file_type == 'ct':
                    shutil.move(file_path, os.path.join(ct_folder, new_file_name))
                    print(f"Moved CT file '{file_name}' to '{ct_folder}'.")
                elif file_type == 'mask':
                    shutil.move(file_path, os.path.join(mask_folder, new_file_name))
                    print(f"Moved MASK file '{file_name}' to '{mask_folder}'.")
                elif file_type == 'pet':
                    shutil.move(file_path, os.path.join(pet_folder, new_file_name))
                    print(f"Moved PET file '{file_name}' to '{pet_folder}'.")