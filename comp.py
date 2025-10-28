from pre import preprocess
from affi import affine_register

def run_registration(ct_path, pet_path):
    ct_arr, pet_arr = preprocess(ct_path, pet_path)
    
    # Classical affine registration
    aligned_ct, transform = affine_register(ct_arr, pet_arr)
    
    # Optional: DL fine-tuning
    # model = LightweightUNet()
    # train model here with slices (requires more code)
    
    return aligned_ct, pet_arr

if __name__ == "__main__":
    ct_path = r"D:\projects\breast_reg\data\patients\patient01\ct.nii.gz"
    pet_path = r"D:\projects\breast_reg\data\patients\patient01\pet.nii.gz"
    
    aligned_ct, pet_arr = run_registration(ct_path, pet_path)
    
    import numpy as np
    np.save("aligned_ct.npy", aligned_ct)
    np.save("pet.npy", pet_arr)
    print("Saved aligned_ct.npy and pet.npy")
    print("Registration complete. Shapes:", aligned_ct.shape, pet_arr.shape)
