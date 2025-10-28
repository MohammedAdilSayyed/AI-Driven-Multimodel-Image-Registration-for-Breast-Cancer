import SimpleITK as sitk

ct = sitk.ReadImage(r"D:\projects\breast_reg\data\patients\patient01\ct.nii.gz")
pet = sitk.ReadImage(r"D:\projects\breast_reg\data\patients\patient01\pet.nii.gz")

print("CT spacing:", ct.GetSpacing(), "direction:", ct.GetDirection())
print("PET spacing:", pet.GetSpacing(), "direction:", pet.GetDirection())
