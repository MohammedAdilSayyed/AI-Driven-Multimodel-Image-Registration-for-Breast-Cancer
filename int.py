import numpy as np
import matplotlib.pyplot as plt

aligned_ct = np.load("aligned_ct.npy")
pet = np.load("pet.npy")

fig, ax = plt.subplots(figsize=(8, 8))
slice_idx = aligned_ct.shape[0] // 2

def show_slice(idx):
    ax.clear()
    ct_slice = aligned_ct[idx]
    pet_slice = pet[idx]

    ct_norm = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice) + 1e-8)
    pet_norm = (pet_slice - np.min(pet_slice)) / (np.max(pet_slice) - np.min(pet_slice) + 1e-8)

    ax.imshow(ct_norm, cmap='gray')
    ax.imshow(pet_norm, cmap='hot', alpha=0.4)
    ax.set_title(f"CT–PET Overlay (slice {idx})")
    ax.axis('off')
    fig.canvas.draw_idle()

def on_scroll(event):
    global slice_idx
    if event.button == 'up':
        slice_idx = (slice_idx + 1) % aligned_ct.shape[0]
    elif event.button == 'down':
        slice_idx = (slice_idx - 1) % aligned_ct.shape[0]
    show_slice(slice_idx)

fig.canvas.mpl_connect('scroll_event', on_scroll)
show_slice(slice_idx)
plt.show()
