"""
Complete Analysis Pipeline for Breast CT-PET Registration

This script runs the full analysis pipeline:
1. Load CT and PET images
2. Perform rigid registration
3. Calculate SUV
4. Segment tumors
5. Generate report

Usage:
    python pipeline.py <patient_path> [--output <output_dir>] [--weight <kg>]
"""

import argparse
import json
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from src.registration import ImageRegistration, load_image, save_image
from src.suv import SUVCalculator, PETMetadata, calculate_suv_statistics
from src.segmentation import TumorSegmentation, SegmentationMethod
from src.ml_segmentation import MLTumorSegmentor
from src.visualization import normalize, plot_slice, create_3d_montage


def run_pipeline(patient_dir: Path, output_dir: Path, patient_weight: float = 70.0):
    """
    Run the complete analysis pipeline for a patient.
    
    Args:
        patient_dir: Path to patient data directory
        output_dir: Path to save outputs
        patient_weight: Patient weight in kg
    """
    print("=" * 60)
    print("BREAST CT-PET ANALYSIS PIPELINE")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load images
    print("\n[1/5] Loading images...")
    ct_path = patient_dir / "ct.nii.gz"
    pet_path = patient_dir / "pet.nii.gz"
    pet_json = patient_dir / "pet.json"
    
    ct_img = load_image(str(ct_path))
    pet_img = load_image(str(pet_path))
    
    print(f"  CT size: {ct_img.GetSize()}")
    print(f"  PET size: {pet_img.GetSize()}")
    
    # 2. Registration
    print("\n[2/5] Performing rigid registration...")
    ct_float = sitk.Cast(ct_img, sitk.sitkFloat32)
    pet_float = sitk.Cast(pet_img, sitk.sitkFloat32)
    
    registrar = ImageRegistration(
        metric="mutual_information",
        num_iterations=200,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0]
    )
    
    transform = registrar.register(ct_float, pet_float, "rigid")
    pet_registered = registrar.apply_transform(ct_float, pet_float, transform)
    
    # Save registered PET
    reg_output = output_dir / "pet_registered.nii.gz"
    save_image(pet_registered, str(reg_output))
    print(f"  Saved registered PET to: {reg_output}")
    
    # Save registration stats
    stats = registrar.get_registration_stats()
    with open(output_dir / "registration_stats.json", 'w') as f:
        json.dump({k: v for k, v in stats.items() if k != 'metric_history'}, f, indent=2)
    
    # 3. SUV Calculation
    print("\n[3/5] Calculating SUV...")
    metadata = PETMetadata.from_json(str(pet_json), patient_weight)
    calculator = SUVCalculator(metadata)
    suv_img = calculator.calculate_suv_bw(pet_registered)
    
    # Save SUV image
    suv_output = output_dir / "suv.nii.gz"
    save_image(suv_img, str(suv_output))
    print(f"  Saved SUV image to: {suv_output}")
    
    # Calculate whole-image statistics
    suv_stats = calculate_suv_statistics(suv_img)
    print(f"  SUVmax: {suv_stats['suv_max']:.2f}")
    print(f"  SUVmean: {suv_stats['suv_mean']:.2f}")
    
    # 4. Tumor Segmentation
    print("\n[4/5] Segmenting tumors...")
    segmentor = TumorSegmentation(min_lesion_volume_ml=0.5, max_lesion_volume_ml=500)
    
    # Try multiple methods
    methods = [
        (SegmentationMethod.PERCENT_MAX, {'percentage': 0.4}),
        (SegmentationMethod.FIXED_THRESHOLD, {'threshold': 2.5}),
    ]
    
    best_result = None
    for method, kwargs in methods:
        result = segmentor.segment(suv_img, method, **kwargs)
        if best_result is None or result.num_lesions > 0:
            best_result = result
            if result.num_lesions > 0:
                break
    
    # Save segmentation mask
    mask_output = output_dir / "tumor_mask.nii.gz"
    save_image(best_result.mask, str(mask_output))
    print(f"  Saved tumor mask to: {mask_output}")
    print(f"  Lesions found: {best_result.num_lesions}")
    print(f"  Total volume: {best_result.total_volume_ml:.2f} ml")
    
    # Calculate SUV statistics within mask
    if best_result.num_lesions > 0:
        tumor_suv_stats = calculate_suv_statistics(suv_img, best_result.mask)
        print(f"  Tumor SUVmax: {tumor_suv_stats['suv_max']:.2f}")
        print(f"  Tumor SUVmean: {tumor_suv_stats['suv_mean']:.2f}")
        print(f"  TLG: {tumor_suv_stats['total_lesion_glycolysis']:.2f}")
    
    # --- 4b. AI Segmentation (K-Means) ---
    print("\n[4b/5] Running AI Segmentation (K-Means Clustering)...")
    ml_segmentor = MLTumorSegmentor(n_clusters=3) # Background, Tissue, Tumor
    ai_result = ml_segmentor.segment(ct_img, suv_img)
    
    ai_mask_output = output_dir / "tumor_mask_ai.nii.gz"
    save_image(ai_result.mask, str(ai_mask_output))
    print(f"  Saved AI tumor mask to: {ai_mask_output}")
    
    # 5. Generate visualizations
    print("\n[5/5] Generating visualizations...")
    
    ct_arr = sitk.GetArrayFromImage(ct_img)
    pet_reg_arr = sitk.GetArrayFromImage(pet_registered)
    suv_arr = sitk.GetArrayFromImage(suv_img)
    mask_arr = sitk.GetArrayFromImage(best_result.mask)
    ai_mask_arr = sitk.GetArrayFromImage(ai_result.mask)
    
    # Find slice with maximum tumor (or middle slice)
    if np.any(mask_arr > 0):
        slice_sums = np.sum(mask_arr, axis=(1, 2))
        best_slice = np.argmax(slice_sums)
    elif np.any(ai_mask_arr > 0):
        slice_sums = np.sum(ai_mask_arr, axis=(1, 2))
        best_slice = np.argmax(slice_sums)
    else:
        best_slice = ct_arr.shape[0] // 2
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: CT, PET, Fusion
    axes[0, 0].imshow(normalize(ct_arr[best_slice]), cmap='gray')
    axes[0, 0].set_title('CT')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize(pet_reg_arr[best_slice]), cmap='hot')
    axes[0, 1].set_title('Registered PET')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(normalize(ct_arr[best_slice]), cmap='gray')
    pet_norm = normalize(pet_reg_arr[best_slice])
    pet_masked = np.ma.masked_where(pet_norm < 0.1, pet_norm)
    axes[0, 2].imshow(pet_masked, cmap='hot', alpha=0.5)
    axes[0, 2].set_title('CT-PET Fusion')
    axes[0, 2].axis('off')
    
    # Row 2: SUV, Segmentation Comparison, Stats
    im = axes[1, 0].imshow(suv_arr[best_slice], cmap='hot')
    axes[1, 0].set_title('SUV')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Segmentation Comparison
    axes[1, 1].imshow(normalize(ct_arr[best_slice]), cmap='gray')
    
    # Plot Manual (Percent Max) in Green
    if np.any(mask_arr[best_slice] > 0):
        axes[1, 1].contour(mask_arr[best_slice], levels=[0.5], 
                          colors=['lime'], linewidths=2, label='Manual')
    
    # Plot AI (K-Means) in Cyan
    if np.any(ai_mask_arr[best_slice] > 0):
        axes[1, 1].contour(ai_mask_arr[best_slice], levels=[0.5], 
                          colors=['cyan'], linewidths=2, linestyles='dashed', label='AI')
                          
    axes[1, 1].set_title('Segmentation: Green=Manual, Cyan=AI')
    axes[1, 1].axis('off')
    
    # Stats panel
    axes[1, 2].axis('off')
    stats_text = f"""Analysis Results
================

Registration:
  Final metric: {stats.get('final_metric', 'N/A'):.4f}

SUV Statistics:
  SUVmax: {suv_stats['suv_max']:.2f}
  SUVmean: {suv_stats['suv_mean']:.2f}

Manual Seg ({best_result.method}):
  Lesions: {best_result.num_lesions}
  Volume: {best_result.total_volume_ml:.2f} ml

AI Seg (K-Means):
  Clusters: {ai_result.n_clusters}
  Volume: {np.sum(ai_mask_arr > 0) * (suv_img.GetSpacing()[0]**3) / 1000:.2f} ml
"""
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(output_dir / "analysis_report.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved report to: {output_dir / 'analysis_report.png'}")
    
    # Save montage
    fig = create_3d_montage(ct_arr, 16, pet_reg_arr)
    fig.savefig(output_dir / "montage.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved montage to: {output_dir / 'montage.png'}")
    
    # Save JSON report
    report = {
        "patient": patient_dir.name,
        "registration": {
            "final_metric": stats.get('final_metric'),
            "iterations": stats.get('iterations'),
        },
        "suv": suv_stats,
        "segmentation": {
            "method": best_result.method,
            "parameters": best_result.parameters,
            "num_lesions": best_result.num_lesions,
            "total_volume_ml": best_result.total_volume_ml,
            "lesions": [
                {
                    "volume_ml": l["volume_ml"],
                    "centroid": list(l["centroid"]) if "centroid" in l else None
                }
                for l in best_result.lesion_stats
            ]
        }
    }
    
    with open(output_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved JSON report to: {output_dir / 'report.json'}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Breast CT-PET Analysis Pipeline")
    parser.add_argument("patient_path", type=str, help="Path to patient data directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: patient_path/output)")
    parser.add_argument("--weight", "-w", type=float, default=70.0,
                       help="Patient weight in kg (default: 70)")
    
    args = parser.parse_args()
    
    patient_dir = Path(args.patient_path)
    if not patient_dir.exists():
        print(f"Error: Patient directory not found: {patient_dir}")
        return
    
    output_dir = Path(args.output) if args.output else patient_dir / "output"
    
    run_pipeline(patient_dir, output_dir, args.weight)


if __name__ == "__main__":
    main()
