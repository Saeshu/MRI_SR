import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the NIfTI file
nifti_file_path = '/content/drive/MyDrive/IXI-T2/IXI013-HH-1212-T2.nii.gz'  # Replace with your file path
nifti_img = nib.load(nifti_file_path)

# Get the data from the NIfTI file as a NumPy array
nifti_data = nifti_img.get_fdata()

# Show some details about the NIfTI file
print(f"Data shape: {nifti_data.shape}")

# Function to display a slice
def display_slice(data, slice_index, axis=2):
    if axis == 0:
        slice_data = data[slice_index, :, :]
    elif axis == 1:
        slice_data = data[:, slice_index, :]
    elif axis == 2:
        slice_data = data[:, :, slice_index]

    plt.imshow(slice_data.T, cmap="gray", origin="lower")
    plt.show()

# Display a middle slice along the chosen axis (0, 1, or 2)
slice_index = nifti_data.shape[2] // 2  # Adjust axis and index as needed
display_slice(nifti_data, slice_index, axis=2)
