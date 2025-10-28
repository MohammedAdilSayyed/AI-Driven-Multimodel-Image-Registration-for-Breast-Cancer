import numpy as np
import matplotlib.pyplot as plt

# Load already-registered data
aligned_ct = np.load("aligned_ct.npy")
pet_arr = np.load("pet.npy")

# Show a slice
slice_idx = 40
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(aligned_ct[slice_idx], cmap='gray')
plt.title("Aligned CT")
plt.subplot(1,2,2)
plt.imshow(pet_arr[slice_idx], cmap='hot')
plt.title("PET")
plt.show()
