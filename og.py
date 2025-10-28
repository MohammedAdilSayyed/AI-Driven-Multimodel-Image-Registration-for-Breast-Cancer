import nibabel as nib
import matplotlib.pyplot as plt

# Paths to original unregistered CT and PET
ct_path = r"D:\projects\breast_reg\data\patients\patient04\ct.nii.gz"
pet_path = r"D:\projects\breast_reg\data\patients\patient04\pet.nii.gz"

# Load volumes using nibabel
ct_img = nib.load(ct_path)
pet_img = nib.load(pet_path)

ct_arr = ct_img.get_fdata()
pet_arr = pet_img.get_fdata()

print("CT shape:", ct_arr.shape)
print("PET shape:", pet_arr.shape)

# Visualize a slice
slice_idx = ct_arr.shape[2] // 2  # middle slice
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(ct_arr[:, :, slice_idx], cmap='gray')
plt.title("Original CT")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(pet_arr[:, :, slice_idx], cmap='hot')
plt.title("Original PET")
plt.axis('off')

plt.show()
