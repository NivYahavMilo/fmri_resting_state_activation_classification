import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from nilearn import datasets, plotting

# Load MNI152 template
template = datasets.load_mni152_template()
template_data = template.get_fdata()

# Define the coordinates (R, A, S)
R, A, S = -34, -41, -21

# Convert R, A, S to voxel indices
x_idx = int(round((R - template.affine[0, 3]) / np.abs(template.affine[0, 0])))
y_idx = int(round((A - template.affine[1, 3]) / np.abs(template.affine[1, 1])))
z_idx = int(round((S - template.affine[2, 3]) / np.abs(template.affine[2, 2])))

# Get the value at the voxel indices
roi_value = 0.7  # Placeholder for the ROI value (normalized value 0.7)

# Create a 3D array with the ROI value at the specific voxel
roi_data = np.zeros(template_data.shape)
roi_data[x_idx, y_idx, z_idx] = roi_value

# Plotting
plotting.plot_stat_map(nib.Nifti1Image(roi_data, template.affine), bg_img=template,
                        display_mode='ortho', cut_coords=[R, A, S], colorbar=True, title="ROI Visualization")

plt.show()