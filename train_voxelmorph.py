import os
import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import voxelmorph as vxm

# ==========================================================
# CONFIGURATION
# ==========================================================
data_dir = r"D:\projects\breast_reg\data\patients"
new_shape = (128, 128, 128)
batch_size = 1
epochs = 20
learning_rate = 1e-4
save_dir = "./trained_model"
os.makedirs(save_dir, exist_ok=True)

# ==========================================================
# RESAMPLING FUNCTION
# ==========================================================
def resample_image(img, new_shape=(128, 128, 128)):
    """Resample SimpleITK image to a fixed shape."""
    original_size = np.array(img.GetSize())
    original_spacing = np.array(img.GetSpacing())
    new_size = np.array(new_shape)
    new_spacing = original_spacing * (original_size / new_size)

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(tuple(new_spacing))
    resample.SetSize([int(s) for s in new_size])
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetDefaultPixelValue(0)

    return resample.Execute(img)

# ==========================================================
# LOAD AND PREPROCESS DATA
# ==========================================================
def load_patient_data(data_dir, new_shape):
    patients = sorted(glob.glob(os.path.join(data_dir, "patient*")))
    ct_vols, pet_vols = [], []

    for p in patients:
        ct_path = os.path.join(p, "ct.nii.gz")
        pet_path = os.path.join(p, "pet.nii.gz")

        if os.path.exists(ct_path) and os.path.exists(pet_path):
            ct_img = sitk.ReadImage(ct_path)
            pet_img = sitk.ReadImage(pet_path)

            # Resample both to the same fixed size
            ct_resampled = resample_image(ct_img, new_shape)
            pet_resampled = resample_image(pet_img, new_shape)

            ct_np = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
            pet_np = sitk.GetArrayFromImage(pet_resampled).astype(np.float32)

            # Normalize (0–1)
            ct_np = (ct_np - np.min(ct_np)) / (np.max(ct_np) - np.min(ct_np) + 1e-8)
            pet_np = (pet_np - np.min(pet_np)) / (np.max(pet_np) - np.min(pet_np) + 1e-8)

            ct_vols.append(ct_np)
            pet_vols.append(pet_np)
        else:
            print(f"Missing CT or PET file for {p}")

    return np.array(ct_vols), np.array(pet_vols)

print("Loading and resampling patient data...")
ct_vols, pet_vols = load_patient_data(data_dir, new_shape)
print(f"Loaded {len(ct_vols)} patient volumes. Shape: {ct_vols[0].shape}")

# ==========================================================
# PREPARE DATA FOR TRAINING
# ==========================================================
# Add channel dimension: (B, H, W, D, 1)
ct_vols = np.expand_dims(ct_vols, -1)
pet_vols = np.expand_dims(pet_vols, -1)
inshape = new_shape

train_dataset = tf.data.Dataset.from_tensor_slices((pet_vols, ct_vols))
train_dataset = train_dataset.shuffle(buffer_size=len(ct_vols)).batch(batch_size)

# ==========================================================
# BUILD VOXELMORPH MODEL
# ==========================================================
nb_unet_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=nb_unet_features,
    int_steps=0
)

# Compile with mean squared error + gradient regularization
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
weights = [1.0, 0.01]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=losses, loss_weights=weights)

# ==========================================================
# TRAINING
# ==========================================================
print("Starting training...")
for epoch in range(epochs):
    for step, (moving, fixed) in enumerate(train_dataset):
        loss = model.train_on_batch([moving, fixed], [fixed, np.zeros_like(fixed)])
    print(f"Epoch {epoch+1}/{epochs} completed.")

# ==========================================================
# SAVE MODEL
# ==========================================================
model.save(os.path.join(save_dir, "vxm_breast_model.h5"))
print("✅ Training complete! Model saved at:", os.path.join(save_dir, "vxm_breast_model.h5"))
