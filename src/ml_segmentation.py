"""
Machine Learning Segmentation Module

This module uses Unsupervised Machine Learning (K-Means Clustering) 
to segment tumors by combining information from both CT (anatomical) 
and PET (metabolic) images.

Unlike simple thresholding, this method learns the statistical 
distributions of intensity values to separate tissues.
"""

import SimpleITK as sitk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MLSegmentationResult:
    mask: sitk.Image
    n_clusters: int
    tumor_cluster_id: int
    cluster_centers: np.ndarray
    method: str = "k-means_clustering"

class MLTumorSegmentor:
    """
    Unsupervised ML Segmentor using K-Means Clustering on Multi-modal data.
    """

    def __init__(self, n_clusters: int = 3):
        """
        Initialize ML Segmentor.
        
        Args:
            n_clusters: Number of tissue classes to find (e.g., Background, Soft Tissue, Tumor)
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def segment(self, ct_image: sitk.Image, suv_image: sitk.Image) -> MLSegmentationResult:
        """
        Perform ML-based segmentation using both CT and SUV data.
        
        Args:
            ct_image: CT image (anatomical structure)
            suv_image: SUV image (metabolic activity)
            
        Returns:
            MLSegmentationResult object
        """
        print(f"Running ML Segmentation with {self.n_clusters} clusters...")

        # 1. Prepare Features
        # Get arrays
        ct_arr = sitk.GetArrayFromImage(ct_image).flatten()
        suv_arr = sitk.GetArrayFromImage(suv_image).flatten()
        
        # Stack features: We use SUV and CT intensity as our 2 features for every voxel
        features = np.column_stack((suv_arr, ct_arr))
        
        # 2. Preprocessing
        # Only consider inside body (approx) to avoid background dominance
        # Simple threshold: CT > -900 (air is ~ -1000)
        mask_indices = np.where(ct_arr > -900) 
        valid_features = features[mask_indices]
        
        if len(valid_features) == 0:
            print("Warning: No valid tissue found for ML segmentation.")
            return self._create_empty_result(suv_image)

        # Scale features so SUV (0-20) and CT (-1000 to 1000) have equal weight
        scaled_features = self.scaler.fit_transform(valid_features)

        # 3. Clustering (The AI Step)
        labels = self.model.fit_predict(scaled_features)

        # 4. Identify Tumor Cluster
        # The tumor cluster typically has the highest SUV center.
        cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        
        # Column 0 is SUV, Column 1 is CT
        suv_centers = cluster_centers[:, 0]
        tumor_cluster_id = np.argmax(suv_centers)
        
        print(f"  Cluster centers (SUV, CT):")
        for i, center in enumerate(cluster_centers):
            is_tumor = " [TUMOR CLASS]" if i == tumor_cluster_id else ""
            print(f"    Cluster {i}: SUV={center[0]:.2f}, CT={center[1]:.0f}{is_tumor}")

        # 5. Reconstruct 3D Mask
        # Create empty array matching original shape
        full_labels = np.zeros(ct_arr.shape, dtype=np.uint8)
        
        # Map predicted labels back to original indices
        # We assign the tumor cluster ID to 1, everything else to 0 for binary mask
        binary_labels = np.zeros_like(labels, dtype=np.uint8)
        binary_labels[labels == tumor_cluster_id] = 1
        
        full_labels[mask_indices] = binary_labels
        
        # Reshape back to 3D image
        mask_arr = full_labels.reshape(sitk.GetArrayFromImage(ct_image).shape)
        
        # Convert to SimpleITK image
        mask_image = sitk.GetImageFromArray(mask_arr)
        mask_image.CopyInformation(ct_image)

        # 6. Post-processing (Clean up noise)
        # Morphological opening to remove small disconnected speckles
        mask_image = sitk.BinaryMorphologicalOpening(mask_image, [1, 1, 1])
        
        return MLSegmentationResult(
            mask=mask_image,
            n_clusters=self.n_clusters,
            tumor_cluster_id=int(tumor_cluster_id),
            cluster_centers=cluster_centers
        )

    def _create_empty_result(self, ref_image: sitk.Image) -> MLSegmentationResult:
        empty_mask = sitk.GetImageFromArray(np.zeros(sitk.GetArrayFromImage(ref_image).shape, dtype=np.uint8))
        empty_mask.CopyInformation(ref_image)
        return MLSegmentationResult(empty_mask, 0, 0, np.array([]))
