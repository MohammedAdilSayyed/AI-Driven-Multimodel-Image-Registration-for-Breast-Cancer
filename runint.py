import torch
import nibabel as nib
import voxelmorph as vxm
import numpy as np

model = vxm.networks.VxmDense.load('vxm_breast_reg.pth')
model.eval()

ct = nib.load(r"D:\projects\breast_reg\data\patients\patient04\ct.nii.gz").get_fdata()
pet = nib.load(r"D:\projects\breast_reg\data\patients\patient04\pet.nii.gz").get_fdata()

ct = (ct - ct.min()) / (ct.max() - ct.min())
pet = (pet - pet.min()) / (pet.max() - pet.min())

ct_t = torch.from_numpy(ct[None, None]).float()
pet_t = torch.from_numpy(pet[None, None]).float()

pred, flow = model(pet_t, ct_t, registration=True)
registered_pet = pred.detach().numpy().squeeze()

nib.save(nib.Nifti1Image(registered_pet, np.eye(4)), "registered_pet_ai.nii.gz")
print("✅ AI-based registration complete!")
