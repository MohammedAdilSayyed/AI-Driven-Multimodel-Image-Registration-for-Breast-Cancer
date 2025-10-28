import os

ct_path = r"D:\projects\breast_reg\data\patients\patient01\ct.nii.gz"
pet_path = r"D:\projects\breast_reg\data\patients\patient01\pet.nii.gz"

print("CT exists?", os.path.exists(ct_path))
print("PET exists?", os.path.exists(pet_path))
