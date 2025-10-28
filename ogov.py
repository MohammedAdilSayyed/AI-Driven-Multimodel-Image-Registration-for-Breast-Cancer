import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# ==== File paths ====
ct_path = r"D:\projects\breast_reg\data\patients\patient04\ct.nii.gz"
pet_path = r"D:\projects\breast_reg\data\patients\patient04\pet.nii.gz"

# ==== Read images ====
ct_img = sitk.ReadImage(ct_path)
pet_img = sitk.ReadImage(pet_path)

# ==== Resample PET to CT geometry ====
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ct_img)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetTransform(sitk.Transform())
pet_resampled = resampler.Execute(pet_img)

# ==== Convert both to NumPy arrays ====
ct_arr = sitk.GetArrayFromImage(ct_img)
pet_arr = sitk.GetArrayFromImage(pet_resampled)

# ==== Choose a middle slice ====
slice_idx = ct_arr.shape[0] // 2
ct_slice = ct_arr[slice_idx]
pet_slice = pet_arr[slice_idx]

# ==== Normalize for display ====
def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

ct_norm = normalize(ct_slice)
pet_norm = normalize(pet_slice)

# ==== Overlay visualization ====
plt.figure(figsize=(8, 8))
plt.imshow(ct_norm, cmap='gray')
plt.imshow(pet_norm, cmap='hot', alpha=0.4)  # adjust alpha as needed
plt.title(f"CT–PET Overlay (Resampled PET) — slice {slice_idx}")
plt.axis('off')
plt.show()
