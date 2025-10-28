import SimpleITK as sitk
import numpy as np
import os

def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # Shape: [slices, H, W]
    return img, arr

def normalize_ct(ct_arr):
    # Clip Hounsfield units and scale to 0-1
    ct_arr = np.clip(ct_arr, -1000, 400)
    ct_arr = (ct_arr + 1000) / 1400
    return ct_arr

def normalize_pet(pet_arr):
    pet_arr = np.clip(pet_arr, 0, 20)  # SUV normalization
    pet_arr = pet_arr / 20
    return pet_arr

def preprocess(ct_path, pet_path):

    ct_img, ct_arr = load_nifti(ct_path)
    pet_img, pet_arr = load_nifti(pet_path)
    
    # Resample CT to PET spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    ct_img_resampled = resampler.Execute(ct_img)
    ct_arr_resampled = sitk.GetArrayFromImage(ct_img_resampled)
    
    # Normalize
    ct_arr_resampled = normalize_ct(ct_arr_resampled)
    pet_arr = normalize_pet(pet_arr)
    
    return ct_arr_resampled, pet_arr
