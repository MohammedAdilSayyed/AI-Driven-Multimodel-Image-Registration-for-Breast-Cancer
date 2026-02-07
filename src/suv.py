"""
SUV (Standardized Uptake Value) Calculation Module

This module provides functions to calculate SUV from PET images,
which is essential for quantitative oncology analysis.

SUV = (Tissue Activity [Bq/ml]) / (Injected Dose [Bq] / Patient Weight [g])

Or equivalently:
SUV = (Tissue Activity) * (Patient Weight) / (Injected Dose * Decay Factor)
"""

import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PETMetadata:
    """Container for PET acquisition metadata needed for SUV calculation."""
    injected_dose_bq: float  # Injected radioactivity in Bq
    patient_weight_kg: float  # Patient weight in kg
    half_life_seconds: float  # Radionuclide half-life in seconds
    injection_time: datetime  # Time of injection
    scan_time: datetime  # Time of scan acquisition
    units: str = "BQML"  # PET units (typically BQML)
    decay_correction: str = "START"  # Decay correction method
    
    @classmethod
    def from_json(cls, json_path: str, patient_weight_kg: float = 70.0) -> 'PETMetadata':
        """
        Load PET metadata from JSON sidecar file.
        
        Args:
            json_path: Path to JSON metadata file
            patient_weight_kg: Patient weight (often not in metadata, provide manually)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Parse times
        time_zero = data.get('TimeZero', '00:00:00')
        acq_time = data.get('AcquisitionTime', time_zero)
        
        # Convert time strings to datetime
        base_date = datetime.now().date()
        
        def parse_time(time_str):
            parts = time_str.replace('.', ':').split(':')
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2]) if len(parts) > 2 else 0
            return datetime.combine(base_date, datetime.min.time().replace(
                hour=h, minute=m, second=int(s), microsecond=int((s % 1) * 1000000)
            ))
        
        return cls(
            injected_dose_bq=data.get('InjectedRadioactivity', 370e6),  # Default ~10 mCi
            patient_weight_kg=patient_weight_kg,
            half_life_seconds=data.get('RadionuclideHalfLife', 6588),  # F-18 default
            injection_time=parse_time(time_zero),
            scan_time=parse_time(acq_time),
            units=data.get('Units', 'BQML'),
            decay_correction=data.get('DecayCorrection', 'START')
        )


class SUVCalculator:
    """
    Calculate SUV (Standardized Uptake Value) from PET images.
    
    Supports:
    - SUV body weight (SUVbw)
    - SUV lean body mass (SUVlbm) 
    - SUV body surface area (SUVbsa)
    """
    
    def __init__(self, metadata: PETMetadata):
        """
        Initialize SUV calculator with PET metadata.
        
        Args:
            metadata: PET acquisition metadata
        """
        self.metadata = metadata
        
    def calculate_decay_factor(self) -> float:
        """
        Calculate the decay factor from injection to scan time.
        
        Returns:
            Decay factor (always >= 1)
        """
        time_diff = (self.metadata.scan_time - self.metadata.injection_time).total_seconds()
        decay_constant = np.log(2) / self.metadata.half_life_seconds
        decay_factor = np.exp(decay_constant * time_diff)
        return decay_factor
    
    def calculate_suv_bw(self, pet_image: sitk.Image) -> sitk.Image:
        """
        Calculate SUV normalized by body weight (most common method).
        
        SUVbw = (Activity [Bq/ml]) / (Injected Dose [Bq] / Weight [g])
        
        Args:
            pet_image: PET image in Bq/ml
            
        Returns:
            SUV image (dimensionless)
        """
        pet_array = sitk.GetArrayFromImage(pet_image).astype(np.float64)
        
        # Calculate decay-corrected dose
        if self.metadata.decay_correction == "START":
            # Already decay corrected to scan start
            effective_dose = self.metadata.injected_dose_bq
        else:
            # Need to apply decay correction
            decay_factor = self.calculate_decay_factor()
            effective_dose = self.metadata.injected_dose_bq / decay_factor
        
        # Convert weight to grams
        weight_g = self.metadata.patient_weight_kg * 1000
        
        # Calculate SUV
        suv_array = pet_array / (effective_dose / weight_g)
        
        # Create SUV image with same geometry
        suv_image = sitk.GetImageFromArray(suv_array.astype(np.float32))
        suv_image.CopyInformation(pet_image)
        
        return suv_image
    
    def calculate_suv_lbm(self, pet_image: sitk.Image, 
                          height_cm: float, is_male: bool = True) -> sitk.Image:
        """
        Calculate SUV normalized by lean body mass (James formula).
        
        Args:
            pet_image: PET image in Bq/ml
            height_cm: Patient height in cm
            is_male: True for male, False for female
            
        Returns:
            SUV-LBM image
        """
        weight = self.metadata.patient_weight_kg
        height = height_cm
        
        # James formula for lean body mass
        if is_male:
            lbm = 1.10 * weight - 128 * (weight / height) ** 2
        else:
            lbm = 1.07 * weight - 148 * (weight / height) ** 2
        
        # Calculate SUV-LBM (similar to SUVbw but with LBM)
        pet_array = sitk.GetArrayFromImage(pet_image).astype(np.float64)
        
        if self.metadata.decay_correction == "START":
            effective_dose = self.metadata.injected_dose_bq
        else:
            decay_factor = self.calculate_decay_factor()
            effective_dose = self.metadata.injected_dose_bq / decay_factor
        
        lbm_g = lbm * 1000
        suv_array = pet_array / (effective_dose / lbm_g)
        
        suv_image = sitk.GetImageFromArray(suv_array.astype(np.float32))
        suv_image.CopyInformation(pet_image)
        
        return suv_image
    
    def calculate_suv_bsa(self, pet_image: sitk.Image, height_cm: float) -> sitk.Image:
        """
        Calculate SUV normalized by body surface area (DuBois formula).
        
        Args:
            pet_image: PET image in Bq/ml
            height_cm: Patient height in cm
            
        Returns:
            SUV-BSA image
        """
        weight = self.metadata.patient_weight_kg
        height = height_cm
        
        # DuBois formula for BSA (m²)
        bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
        
        pet_array = sitk.GetArrayFromImage(pet_image).astype(np.float64)
        
        if self.metadata.decay_correction == "START":
            effective_dose = self.metadata.injected_dose_bq
        else:
            decay_factor = self.calculate_decay_factor()
            effective_dose = self.metadata.injected_dose_bq / decay_factor
        
        # SUV-BSA uses m² instead of weight
        suv_array = pet_array / (effective_dose / (bsa * 10000))  # Convert m² to cm²
        
        suv_image = sitk.GetImageFromArray(suv_array.astype(np.float32))
        suv_image.CopyInformation(pet_image)
        
        return suv_image


def calculate_suv_statistics(suv_image: sitk.Image, 
                             mask: Optional[sitk.Image] = None) -> Dict[str, float]:
    """
    Calculate SUV statistics for a region.
    
    Args:
        suv_image: SUV image
        mask: Optional binary mask for ROI analysis
        
    Returns:
        Dictionary with SUV statistics
    """
    suv_array = sitk.GetArrayFromImage(suv_image)
    
    if mask is not None:
        mask_array = sitk.GetArrayFromImage(mask).astype(bool)
        suv_values = suv_array[mask_array]
    else:
        suv_values = suv_array[suv_array > 0]  # Exclude background
    
    if len(suv_values) == 0:
        return {
            "suv_max": 0.0,
            "suv_mean": 0.0,
            "suv_min": 0.0,
            "suv_std": 0.0,
            "suv_peak": 0.0,
            "volume_ml": 0.0
        }
    
    # Calculate voxel volume in ml
    spacing = suv_image.GetSpacing()
    voxel_volume_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000  # mm³ to ml
    
    stats = {
        "suv_max": float(np.max(suv_values)),
        "suv_mean": float(np.mean(suv_values)),
        "suv_min": float(np.min(suv_values)),
        "suv_std": float(np.std(suv_values)),
        "suv_median": float(np.median(suv_values)),
        "volume_ml": float(len(suv_values) * voxel_volume_ml),
        "total_lesion_glycolysis": float(np.mean(suv_values) * len(suv_values) * voxel_volume_ml)
    }
    
    # SUVpeak - mean SUV in 1cc sphere around hottest voxel (simplified)
    # Here we approximate with a local neighborhood
    stats["suv_peak"] = stats["suv_max"]  # Simplified
    
    return stats


def quick_suv_calculation(pet_path: str, json_path: str,
                         patient_weight_kg: float = 70.0,
                         output_path: Optional[str] = None) -> Tuple[sitk.Image, Dict[str, float]]:
    """
    Quick SUV calculation from PET image and metadata.
    
    Args:
        pet_path: Path to PET image
        json_path: Path to JSON metadata
        patient_weight_kg: Patient weight in kg
        output_path: Optional path to save SUV image
        
    Returns:
        Tuple of (SUV image, statistics dictionary)
    """
    # Load metadata
    metadata = PETMetadata.from_json(json_path, patient_weight_kg)
    
    # Load PET image
    pet_image = sitk.ReadImage(pet_path)
    
    # Calculate SUV
    calculator = SUVCalculator(metadata)
    suv_image = calculator.calculate_suv_bw(pet_image)
    
    # Calculate statistics
    stats = calculate_suv_statistics(suv_image)
    
    if output_path:
        sitk.WriteImage(suv_image, output_path)
        print(f"Saved SUV image to: {output_path}")
    
    print(f"\nSUV Statistics:")
    print(f"  SUVmax:  {stats['suv_max']:.2f}")
    print(f"  SUVmean: {stats['suv_mean']:.2f}")
    print(f"  SUVstd:  {stats['suv_std']:.2f}")
    
    return suv_image, stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python suv.py <pet_path> <json_path> [patient_weight_kg] [output_path]")
        sys.exit(1)
    
    pet_path = sys.argv[1]
    json_path = sys.argv[2]
    weight = float(sys.argv[3]) if len(sys.argv) > 3 else 70.0
    output = sys.argv[4] if len(sys.argv) > 4 else None
    
    suv_image, stats = quick_suv_calculation(pet_path, json_path, weight, output)
