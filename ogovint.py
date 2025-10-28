import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# ==== Convert to NumPy arrays ====
ct_arr = sitk.GetArrayFromImage(ct_img)
pet_arr = sitk.GetArrayFromImage(pet_resampled)

# ==== Normalize function ====
def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

ct_arr = np.array([normalize(s) for s in ct_arr])
pet_arr = np.array([normalize(s) for s in pet_arr])

# ==== Setup plot ====
slice_idx = ct_arr.shape[0] // 2

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

# Initial slice
ct_display = ax.imshow(ct_arr[slice_idx], cmap='gray')
pet_display = ax.imshow(pet_arr[slice_idx], cmap='hot', alpha=0.4)
ax.set_title(f"CT–PET Overlay — Slice {slice_idx}")
ax.axis('off')

# ==== Slider ====
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, ct_arr.shape[0]-1, valinit=slice_idx, valstep=1)

# ==== Update function ====
def update(val):
    idx = int(slider.val)
    ct_display.set_data(ct_arr[idx])
    pet_display.set_data(pet_arr[idx])
    ax.set_title(f"CT–PET Overlay — Slice {idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
