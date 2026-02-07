"""
Rigid and Affine Registration Module for CT-PET Image Alignment

This module provides robust image registration using SimpleITK,
supporting rigid (6 DOF) and affine (12 DOF) transformations.
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class ImageRegistration:
    """
    CT-PET Image Registration using SimpleITK.
    
    Supports:
    - Rigid registration (translation + rotation)
    - Affine registration (+ scaling + shearing)
    - Multiple similarity metrics (MI, NCC, MSE)
    """
    
    def __init__(self, 
                 metric: str = "mutual_information",
                 learning_rate: float = 1.0,
                 num_iterations: int = 200,
                 shrink_factors: list = [4, 2, 1],
                 smoothing_sigmas: list = [2, 1, 0]):
        """
        Initialize the registration pipeline.
        
        Args:
            metric: Similarity metric ('mutual_information', 'correlation', 'mse')
            learning_rate: Optimizer learning rate
            num_iterations: Maximum iterations per resolution level
            shrink_factors: Multi-resolution pyramid shrink factors
            smoothing_sigmas: Gaussian smoothing at each level
        """
        self.metric = metric
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.shrink_factors = shrink_factors
        self.smoothing_sigmas = smoothing_sigmas
        self.registration_method = None
        self.final_transform = None
        self.metric_values = []
        
    def _setup_registration(self, fixed_image: sitk.Image, moving_image: sitk.Image,
                           transform_type: str = "rigid") -> sitk.ImageRegistrationMethod:
        """
        Configure the registration method.
        
        Args:
            fixed_image: Reference image (typically CT)
            moving_image: Image to be registered (typically PET)
            transform_type: 'rigid' or 'affine'
            
        Returns:
            Configured ImageRegistrationMethod
        """
        registration = sitk.ImageRegistrationMethod()
        
        # Set up the similarity metric
        if self.metric == "mutual_information":
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif self.metric == "correlation":
            registration.SetMetricAsCorrelation()
        elif self.metric == "mse":
            registration.SetMetricAsMeanSquares()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Metric sampling strategy
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.1)
        
        # Interpolator for moving image
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer: Gradient Descent with line search
        registration.SetOptimizerAsGradientDescent(
            learningRate=self.learning_rate,
            numberOfIterations=self.num_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Multi-resolution framework
        registration.SetShrinkFactorsPerLevel(shrinkFactors=self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Initial transform
        if transform_type == "rigid":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif transform_type == "affine":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image,
                sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
            
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        # Store metric values during optimization
        self.metric_values = []
        registration.AddCommand(sitk.sitkIterationEvent, 
                               lambda: self.metric_values.append(registration.GetMetricValue()))
        
        return registration
    
    def register(self, fixed_image: sitk.Image, moving_image: sitk.Image,
                transform_type: str = "rigid") -> sitk.Transform:
        """
        Perform registration with improved initialization.
        
        Args:
            fixed_image: Reference image (CT)
            moving_image: Image to register (PET)
            transform_type: 'rigid', 'affine', or 'rigid+affine'
            
        Returns:
            Final optimized transform
        """
        print(f"Starting registration (Type: {transform_type})...")
        print(f"Fixed: {fixed_image.GetSize()}, Moving: {moving_image.GetSize()}")
        
        # 1. Initialization: Align Centers of Mass (Moments)
        # This is more robust than Geometry center if the patient is shifted
        print("Initializing with Center of Mass alignment...")
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        
        # 2. Rigid Registration Stage
        print("Running Rigid Registration stage...")
        registration = self._setup_registration(fixed_image, moving_image, "rigid")
        self.registration_method = registration  # Track for stats
        registration.SetInitialTransform(initial_transform, inPlace=False)
        rigid_transform = registration.Execute(fixed_image, moving_image)
        
        print(f"Rigid Stage Metric: {registration.GetMetricValue():.6f}")
        
        if transform_type == "rigid":
            self.final_transform = rigid_transform
            return self.final_transform
            
        # 3. Affine Registration Stage (Optional)
        # Uses the result of Rigid as the starting point
        if "affine" in transform_type:
            print("Running Affine Registration stage...")
            affine_registration = self._setup_registration(fixed_image, moving_image, "affine")
            self.registration_method = affine_registration # Track for stats
            
            # Convert Rigid (Euler3D) to Affine for initialization
            composite_transform = sitk.Transform(rigid_transform)
            affine_registration.SetInitialTransform(composite_transform, inPlace=False)
            
            self.final_transform = affine_registration.Execute(fixed_image, moving_image)
            print(f"Affine Stage Metric: {affine_registration.GetMetricValue():.6f}")
            
        return self.final_transform
    
    def apply_transform(self, fixed_image: sitk.Image, moving_image: sitk.Image,
                       transform: Optional[sitk.Transform] = None,
                       interpolator: int = sitk.sitkLinear) -> sitk.Image:
        """
        Apply transform to resample moving image to fixed image space.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to transform
            transform: Transform to apply (uses self.final_transform if None)
            interpolator: Interpolation method
            
        Returns:
            Resampled image
        """
        if transform is None:
            transform = self.final_transform
            
        if transform is None:
            raise ValueError("No transform available. Run register() first.")
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        
        return resampler.Execute(moving_image)
    
    def get_registration_stats(self) -> Dict[str, Any]:
        """Get registration statistics."""
        if self.registration_method is None:
            return {}
            
        return {
            "final_metric": self.registration_method.GetMetricValue(),
            "iterations": len(self.metric_values),
            "stop_condition": self.registration_method.GetOptimizerStopConditionDescription(),
            "metric_history": self.metric_values
        }


def load_image(path: str) -> sitk.Image:
    """Load a medical image file."""
    return sitk.ReadImage(str(path))


def save_image(image: sitk.Image, path: str):
    """Save a medical image file."""
    sitk.WriteImage(image, str(path))


def quick_register(ct_path: str, pet_path: str, 
                   output_path: Optional[str] = None,
                   transform_type: str = "rigid") -> Tuple[sitk.Image, sitk.Transform]:
    """
    Quick registration function for simple use cases.
    
    Args:
        ct_path: Path to CT image
        pet_path: Path to PET image
        output_path: Optional path to save registered PET
        transform_type: 'rigid' or 'affine'
        
    Returns:
        Tuple of (registered PET image, transform)
    """
    ct_img = load_image(ct_path)
    pet_img = load_image(pet_path)
    
    # Cast to float for registration
    ct_float = sitk.Cast(ct_img, sitk.sitkFloat32)
    pet_float = sitk.Cast(pet_img, sitk.sitkFloat32)
    
    registrar = ImageRegistration()
    transform = registrar.register(ct_float, pet_float, transform_type)
    registered_pet = registrar.apply_transform(ct_float, pet_float, transform)
    
    if output_path:
        save_image(registered_pet, output_path)
        print(f"Saved registered PET to: {output_path}")
    
    return registered_pet, transform


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python registration.py <ct_path> <pet_path> [output_path]")
        sys.exit(1)
    
    ct_path = sys.argv[1]
    pet_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    registered_pet, transform = quick_register(ct_path, pet_path, output_path)
    print("Registration complete!")
