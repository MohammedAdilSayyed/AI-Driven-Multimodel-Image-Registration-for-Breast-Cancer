"""
Tumor Segmentation Module for PET/CT Images

This module provides multiple segmentation methods:
- Fixed threshold segmentation
- Adaptive threshold (percentage of SUVmax)
- Region growing
- Otsu's method
- Connected component analysis
"""

import SimpleITK as sitk
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SegmentationMethod(Enum):
    """Available segmentation methods."""
    FIXED_THRESHOLD = "fixed_threshold"
    PERCENT_MAX = "percent_max"
    OTSU = "otsu"
    REGION_GROWING = "region_growing"
    ADAPTIVE = "adaptive"


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    mask: sitk.Image
    num_lesions: int
    total_volume_ml: float
    lesion_stats: List[Dict[str, Any]]
    method: str
    parameters: Dict[str, Any]


class TumorSegmentation:
    """
    Tumor segmentation for PET/SUV images.
    
    Provides multiple methods for identifying metabolically active regions
    that may indicate tumors or other areas of high FDG uptake.
    """
    
    def __init__(self, min_lesion_volume_ml: float = 0.5,
                 max_lesion_volume_ml: float = 1000.0):
        """
        Initialize segmentation.
        
        Args:
            min_lesion_volume_ml: Minimum lesion volume to consider (filter noise)
            max_lesion_volume_ml: Maximum lesion volume (filter large organs)
        """
        self.min_lesion_volume_ml = min_lesion_volume_ml
        self.max_lesion_volume_ml = max_lesion_volume_ml
        
    def _get_voxel_volume_ml(self, image: sitk.Image) -> float:
        """Calculate voxel volume in milliliters."""
        spacing = image.GetSpacing()
        return (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    
    def fixed_threshold(self, image: sitk.Image, threshold: float = 2.5) -> sitk.Image:
        """
        Simple fixed threshold segmentation.
        
        For SUV images, threshold of 2.5 is commonly used as a cutoff
        for differentiating benign from malignant lesions.
        
        Args:
            image: SUV or PET image
            threshold: Fixed SUV threshold value
            
        Returns:
            Binary mask image
        """
        binary = sitk.BinaryThreshold(image, 
                                       lowerThreshold=threshold,
                                       upperThreshold=float('inf'),
                                       insideValue=1,
                                       outsideValue=0)
        return sitk.Cast(binary, sitk.sitkUInt8)
    
    def percent_max_threshold(self, image: sitk.Image, 
                              percentage: float = 0.4,
                              min_threshold: float = 2.5) -> sitk.Image:
        """
        Threshold at a percentage of the maximum value.
        
        Common percentages:
        - 40% (0.4): Standard for tumor delineation
        - 50% (0.5): More conservative
        - 70% (0.7): For identifying hottest regions
        
        Args:
            image: SUV or PET image
            percentage: Percentage of max (0-1)
            min_threshold: Minimum threshold to apply
            
        Returns:
            Binary mask image
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        max_val = stats.GetMaximum()
        
        threshold = max(max_val * percentage, min_threshold)
        print(f"Using {percentage*100:.0f}% of max ({max_val:.2f}) = {threshold:.2f}")
        
        return self.fixed_threshold(image, threshold)
    
    def otsu_threshold(self, image: sitk.Image, 
                       num_thresholds: int = 1) -> Tuple[sitk.Image, float]:
        """
        Automatic threshold selection using Otsu's method.
        
        Args:
            image: Input image
            num_thresholds: Number of threshold levels
            
        Returns:
            Tuple of (binary mask, computed threshold)
        """
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(0)
        otsu.SetOutsideValue(1)
        binary = otsu.Execute(image)
        threshold = otsu.GetThreshold()
        
        print(f"Otsu threshold: {threshold:.2f}")
        return sitk.Cast(binary, sitk.sitkUInt8), threshold
    
    def region_growing(self, image: sitk.Image, 
                       seed_points: List[Tuple[int, int, int]],
                       lower_threshold: float = 2.5,
                       upper_threshold: Optional[float] = None) -> sitk.Image:
        """
        Region growing from seed points.
        
        Args:
            image: SUV or PET image
            seed_points: List of (x, y, z) seed coordinates
            lower_threshold: Lower bound for inclusion
            upper_threshold: Upper bound for inclusion (None = no limit)
            
        Returns:
            Binary mask image
        """
        if upper_threshold is None:
            stats = sitk.StatisticsImageFilter()
            stats.Execute(image)
            upper_threshold = stats.GetMaximum()
        
        # Convert seed points to SimpleITK format
        seeds = [image.TransformPhysicalPointToIndex(
            image.TransformIndexToPhysicalPoint(seed)) for seed in seed_points]
        
        region_growing = sitk.ConnectedThreshold(
            image,
            seedList=seeds,
            lower=lower_threshold,
            upper=upper_threshold
        )
        
        return sitk.Cast(region_growing, sitk.sitkUInt8)
    
    def adaptive_threshold(self, image: sitk.Image,
                          background_suv: float = 2.0,
                          source_to_background_ratio: float = 0.5) -> sitk.Image:
        """
        Adaptive threshold based on background-corrected uptake.
        
        Uses the formula: T = background + (fraction * (SUVmax - background))
        
        Args:
            image: SUV image
            background_suv: Estimated background SUV (e.g., liver or blood pool)
            source_to_background_ratio: Fraction of uptake above background
            
        Returns:
            Binary mask image
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        max_val = stats.GetMaximum()
        
        threshold = background_suv + source_to_background_ratio * (max_val - background_suv)
        print(f"Adaptive threshold: {threshold:.2f} (background={background_suv}, max={max_val:.2f})")
        
        return self.fixed_threshold(image, threshold)
    
    def filter_by_size(self, mask: sitk.Image) -> Tuple[sitk.Image, List[Dict[str, Any]]]:
        """
        Filter connected components by size and compute statistics.
        
        Args:
            mask: Binary mask image
            
        Returns:
            Tuple of (filtered mask, list of lesion statistics)
        """
        voxel_vol = self._get_voxel_volume_ml(mask)
        min_voxels = int(self.min_lesion_volume_ml / voxel_vol)
        max_voxels = int(self.max_lesion_volume_ml / voxel_vol)
        
        # Label connected components
        cc = sitk.ConnectedComponent(mask)
        
        # Get component statistics
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        
        lesion_stats = []
        labels_to_keep = []
        
        for label in stats.GetLabels():
            num_voxels = stats.GetNumberOfPixels(label)
            volume_ml = num_voxels * voxel_vol
            
            if min_voxels <= num_voxels <= max_voxels:
                labels_to_keep.append(label)
                centroid = stats.GetCentroid(label)
                bbox = stats.GetBoundingBox(label)
                
                lesion_stats.append({
                    "label": label,
                    "volume_ml": volume_ml,
                    "centroid": centroid,
                    "bounding_box": bbox,
                    "num_voxels": num_voxels,
                    "elongation": stats.GetElongation(label),
                    "roundness": stats.GetRoundness(label)
                })
        
        # Create filtered mask
        if labels_to_keep:
            filtered = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
            filtered.CopyInformation(mask)
            
            for label in labels_to_keep:
                label_mask = sitk.BinaryThreshold(cc, label, label, 1, 0)
                filtered = sitk.Or(filtered, sitk.Cast(label_mask, sitk.sitkUInt8))
        else:
            filtered = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
            filtered.CopyInformation(mask)
        
        return filtered, lesion_stats
    
    def segment(self, image: sitk.Image,
                method: SegmentationMethod = SegmentationMethod.PERCENT_MAX,
                **kwargs) -> SegmentationResult:
        """
        Main segmentation interface.
        
        Args:
            image: SUV or PET image
            method: Segmentation method to use
            **kwargs: Method-specific parameters
            
        Returns:
            SegmentationResult with mask and statistics
        """
        print(f"Segmenting using {method.value} method...")
        
        # Apply segmentation method
        if method == SegmentationMethod.FIXED_THRESHOLD:
            threshold = kwargs.get('threshold', 2.5)
            raw_mask = self.fixed_threshold(image, threshold)
            params = {'threshold': threshold}
            
        elif method == SegmentationMethod.PERCENT_MAX:
            percentage = kwargs.get('percentage', 0.4)
            min_threshold = kwargs.get('min_threshold', 2.5)
            raw_mask = self.percent_max_threshold(image, percentage, min_threshold)
            params = {'percentage': percentage, 'min_threshold': min_threshold}
            
        elif method == SegmentationMethod.OTSU:
            raw_mask, threshold = self.otsu_threshold(image)
            params = {'computed_threshold': threshold}
            
        elif method == SegmentationMethod.REGION_GROWING:
            seed_points = kwargs.get('seed_points', [])
            if not seed_points:
                raise ValueError("seed_points required for region growing")
            lower = kwargs.get('lower_threshold', 2.5)
            upper = kwargs.get('upper_threshold', None)
            raw_mask = self.region_growing(image, seed_points, lower, upper)
            params = {'seed_points': seed_points, 'lower': lower, 'upper': upper}
            
        elif method == SegmentationMethod.ADAPTIVE:
            background = kwargs.get('background_suv', 2.0)
            ratio = kwargs.get('source_to_background_ratio', 0.5)
            raw_mask = self.adaptive_threshold(image, background, ratio)
            params = {'background_suv': background, 'ratio': ratio}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply morphological cleanup
        raw_mask = sitk.BinaryMorphologicalOpening(raw_mask, [1, 1, 1])
        raw_mask = sitk.BinaryMorphologicalClosing(raw_mask, [2, 2, 2])
        
        # Filter by size
        filtered_mask, lesion_stats = self.filter_by_size(raw_mask)
        
        # Calculate total volume
        voxel_vol = self._get_voxel_volume_ml(filtered_mask)
        mask_array = sitk.GetArrayFromImage(filtered_mask)
        total_volume = np.sum(mask_array > 0) * voxel_vol
        
        result = SegmentationResult(
            mask=filtered_mask,
            num_lesions=len(lesion_stats),
            total_volume_ml=total_volume,
            lesion_stats=lesion_stats,
            method=method.value,
            parameters=params
        )
        
        print(f"Found {result.num_lesions} lesions, total volume: {result.total_volume_ml:.2f} ml")
        
        return result


def quick_segment(image_path: str, 
                  method: str = "percent_max",
                  output_path: Optional[str] = None,
                  **kwargs) -> SegmentationResult:
    """
    Quick segmentation function.
    
    Args:
        image_path: Path to SUV/PET image
        method: Segmentation method name
        output_path: Optional path to save mask
        **kwargs: Method parameters
        
    Returns:
        SegmentationResult
    """
    image = sitk.ReadImage(image_path)
    
    segmentor = TumorSegmentation()
    seg_method = SegmentationMethod(method)
    result = segmentor.segment(image, seg_method, **kwargs)
    
    if output_path:
        sitk.WriteImage(result.mask, output_path)
        print(f"Saved mask to: {output_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <suv_image_path> [method] [threshold]")
        print("Methods: fixed_threshold, percent_max, otsu, adaptive")
        sys.exit(1)
    
    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "percent_max"
    
    kwargs = {}
    if len(sys.argv) > 3:
        if method == "fixed_threshold":
            kwargs['threshold'] = float(sys.argv[3])
        elif method == "percent_max":
            kwargs['percentage'] = float(sys.argv[3])
    
    result = quick_segment(image_path, method, **kwargs)
    
    print(f"\nSegmentation Results:")
    print(f"  Method: {result.method}")
    print(f"  Lesions found: {result.num_lesions}")
    print(f"  Total volume: {result.total_volume_ml:.2f} ml")
    for i, lesion in enumerate(result.lesion_stats):
        print(f"  Lesion {i+1}: {lesion['volume_ml']:.2f} ml")
