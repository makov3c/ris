import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import nibabel as nib

# Define custom dataset class
class CancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        #patient_id = sample['patient_id']
        #pet_filename = f"patient_{patient_id}-pet.nii.gz"
        #ct_filename = f"patient_{patient_id}-ct.nii.gz"
        #mask_filename = f"patient_{patient_id}-mask.nii.gz"

        pet_filename = sample['pet_filename']
        ct_filename = sample['ct_filename']
        mask_filename = sample['mask_filename']


        self._load_nifti(os.path.join(self.root_dir, pet_filename))
        ct_image = self._load_nifti(os.path.join(self.root_dir, ct_filename))
        mask = self._load_nifti(os.path.join(self.root_dir, mask_filename))
        label = sample['label']
        
        if self.transform:
            pet_image = self.transform(pet_image)
            ct_image = self.transform(ct_image)
            mask = self.transform(mask)
        
        return {'pet_image': pet_image, 'ct_image': ct_image, 'mask': mask, 'label': label}
    
    #---------------------------------------------------------------------------------------------

    def _load_samples(self):
        samples = []

        for patient_dir in os.listdir(self.root_dir):
            if not os.path.isdir(os.path.join(self.root_dir, patient_dir)):
                continue

            pet_filename = os.path.join(patient_dir, f"{patient_dir}-pet.nii.gz")
            ct_filename = os.path.join(patient_dir, f"{patient_dir}-ct.nii.gz")
            mask_filename = os.path.join(patient_dir, f"{patient_dir}-mask.nii.gz")

        # ID če mi bo nucal
            patient_id = int(patient_dir.split('_')[1])

        # NI DEFINIRANA!!!!!!!!!!!!!!!!!!!!!!!!!
            label = get_label(patient_id)  

        # vsi podatki
            sample_info = {
                'patient_id': patient_id,
                'pet_filename': pet_filename,
                'ct_filename': ct_filename,
                'mask_filename': mask_filename,
                'label': label
            }
            samples.append(sample_info)

        return samples

    def _load_nifti(self, path):
        # Load NIfTI file using nibabel
        image = nib.load(path).get_fdata()
        # Normalize or preprocess the image as needed
        return image
    
    #---------------------------------------------------------------------------------

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

# arhitektura
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_layer(64, 64, 3)
        self.block2 = self._make_layer(64, 128, 4, stride=2)
        self.block3 = self._make_layer(128, 256, 6, stride=2)
        self.block4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)  # 2 output classes (healthy, cancer)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define main function for training and evaluation
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataset and dataloader
    dataset = CancerDataset(root_dir=r'/media/sf_p2/RIS/data/patient', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define model
    model = ResNet34().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            pet_images = batch['pet_image'].to(device)
            ct_images = batch['ct_image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(pet_images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'cancer_detection_model.pth')

if __name__ == '__main__':
    main()