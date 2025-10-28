import numpy as np
import matplotlib.pyplot as plt

# Load your saved aligned CT and PET arrays
aligned_ct = np.load("aligned_ct.npy")
pet = np.load("pet.npy")

# If 3D, pick a slice index to visualize (e.g., middle slice)
slice_idx = aligned_ct.shape[0] // 2

# Normalize images for visualization
ct_slice = aligned_ct[slice_idx]
pet_slice = pet[slice_idx]

ct_norm = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice) + 1e-8)
pet_norm = (pet_slice - np.min(pet_slice)) / (np.max(pet_slice) - np.min(pet_slice) + 1e-8)

# Overlay visualization
plt.figure(figsize=(8, 8))
plt.imshow(ct_norm, cmap='gray')
plt.imshow(pet_norm, cmap='hot', alpha=0.4)  # adjust alpha (0.3–0.6) for clarity
plt.title(f"CT–PET Overlay (slice {slice_idx})")
plt.axis('off')
plt.show()
