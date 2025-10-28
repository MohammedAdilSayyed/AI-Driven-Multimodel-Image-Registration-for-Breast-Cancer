import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# ==== File paths ====
ct_path = r"D:\projects\breast_reg\data\patients\patient04\ct.nii.gz"
pet_path = r"D:\projects\breast_reg\data\patients\patient04\pet.nii.gz"

# ==== Read images ====
ct_img = sitk.ReadImage(ct_path, sitk.sitkFloat32)
pet_img = sitk.ReadImage(pet_path, sitk.sitkFloat32)

# ==== Initialize registration ====
registration_method = sitk.ImageRegistrationMethod()

# Metric: Mutual Information (good for multi-modal)
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.2)

# Optimizer: Regular Step Gradient Descent
registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=2.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5
)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

# Initial transform (rigid)
initial_transform = sitk.CenteredTransformInitializer(
    ct_img, pet_img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
)
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# ==== Run registration ====
final_transform = registration_method.Execute(ct_img, pet_img)
print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())

# ==== Apply transform ====
pet_registered = sitk.Resample(pet_img, ct_img, final_transform, sitk.sitkLinear, 0.0, pet_img.GetPixelID())

# ==== Convert to numpy ====
ct_arr = sitk.GetArrayFromImage(ct_img)
pet_arr = sitk.GetArrayFromImage(pet_registered)

# ==== Visualization ====
slice_idx = ct_arr.shape[0] // 2
def normalize(img): return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

ct_norm = normalize(ct_arr[slice_idx])
pet_norm = normalize(pet_arr[slice_idx])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ct_norm, cmap='gray')
plt.imshow(pet_norm, cmap='hot', alpha=0.4)
plt.title("Registered CT–PET Overlay")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ct_norm, cmap='gray')
plt.title("CT Only")
plt.axis('off')
plt.show()
