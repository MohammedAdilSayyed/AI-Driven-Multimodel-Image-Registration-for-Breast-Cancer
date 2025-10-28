import SimpleITK as sitk

def match_voxel_geometry(fixed_img, moving_img):
    """Resample moving_img to match fixed_img spacing, size, and orientation."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    resampled = resampler.Execute(moving_img)
    return resampled

def affine_register(ct_arr, pet_arr):
    ct_img = sitk.GetImageFromArray(ct_arr)
    pet_img = sitk.GetImageFromArray(pet_arr)
    
    # Match PET to CT geometry
    pet_resampled = match_voxel_geometry(ct_img, pet_img)
    
    # Centered initialization
    initial_transform = sitk.CenteredTransformInitializer(
        ct_img, pet_resampled,
        sitk.AffineTransform(ct_img.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # Setup registration
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    def iteration_callback():
        print(f"Iteration: {registration_method.GetOptimizerIteration()}, "
              f"Metric: {registration_method.GetMetricValue():.4f}")
    
    registration_method.AddCommand(sitk.sitkIterationEvent, iteration_callback)
    
    transform = registration_method.Execute(ct_img, pet_resampled)
    print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())
    
    # Deformable registration for fine alignment
    registration_method.SetInitialTransform(transform, inPlace=False)
    registration_method.SetMetricAsMattesMutualInformation(50)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel([4,2,1])
    registration_method.SetSmoothingSigmasPerLevel([2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # BSplineTransform for local deformation
    transform_domain_mesh_size = [8, 8, 8]
    bspline_transform = sitk.BSplineTransformInitializer(ct_img, transform_domain_mesh_size)
    
    registration_method.SetMovingInitialTransform(transform)
    registration_method.SetInitialTransform(bspline_transform, inPlace=False)
    
    final_bspline = registration_method.Execute(ct_img, pet_resampled)
    
    # Apply final transform
    aligned_ct = sitk.Resample(ct_img, pet_resampled, final_bspline, sitk.sitkLinear, 0.0)
    aligned_ct_arr = sitk.GetArrayFromImage(aligned_ct)
    
    return aligned_ct_arr, final_bspline
