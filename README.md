# Multi-Modal Medical Image Registration for Breast CT and PET Images

## Project Overview

This project implements a complete pipeline for multi-modal image registration of breast CT and PET images using SimpleITK. The work focuses on aligning CT (Computed Tomography) anatomical images with PET (Positron Emission Tomography) functional images to enable accurate combined visualization for diagnostic applications.

## Technology Stack

- **SimpleITK**: Core registration engine
- **NumPy**: Numerical array operations
- **Matplotlib**: Visualization and plotting
- **NiBabel**: Medical image I/O
- **Python 3.x**: Programming language

## Dataset

The project uses a breast imaging dataset containing paired CT and PET volumes from multiple patients:
- **Modality**: CT (anatomical) and PET (functional) 
- **Manufacturer**: GE Discovery STE scanner
- **Imaging Position**: Prone breast position (HFP - Head First Prone)
- **Format**: NIfTI (.nii.gz) converted from DICOM using dcm2niix
- **Patients**: 4 patients (patient01-patient04)
- **Protocol**: 7.3 PRONE BREAST

### Dataset Structure
```
data/
└── patients/
    ├── patient01/
    │   ├── ct.nii.gz
    │   ├── ct.json (metadata)
    │   ├── pet.nii.gz
    │   └── pet.json (metadata)
    ├── patient02/
    └── ...
```

## Core Files Description

### 1. `og.py` - Original Images Visualization
**Purpose**: Loads and displays unregistered CT and PET images side-by-side for baseline assessment.

**Key Features**:
- Uses NiBabel for loading NIfTI volumes
- Extracts middle slice for 2D visualization
- Displays CT in grayscale and PET in hot colormap
- Prints volume dimensions for spatial awareness

**Usage**:
```python
python og.py
```

**Output**: Side-by-side comparison of original CT and PET slices showing misalignment before registration.

---

### 2. `ogov.py` - Resampled Overlay Visualization
**Purpose**: Creates overlay visualization of CT-PET by resampling PET to CT spatial geometry.

**Key Features**:
- SimpleITK-based resampling using `ResampleImageFilter`
- Transforms PET to match CT pixel grid (spacing, origin, direction)
- Linear interpolation for smooth resampling
- Overlays PET (hot colormap) with CT (grayscale) using alpha blending
- Normalizes intensity values to [0,1] range

**Technical Details**:
- **Resampling**: Reference image (CT) defines the output geometry
- **Interpolation**: Linear (preserves edge quality)
- **Alpha channel**: 0.4 opacity for PET overlay
- **No geometric transformation**: Only isotropic resampling

**Usage**:
```python
python ogov.py
```

---

### 3. `ogovint.py` - Interactive Slice Viewer
**Purpose**: Interactive slider-based viewer for browsing through all slices of the CT-PET overlay.

**Key Features**:
- Dynamic slice navigation with matplotlib slider widget
- Real-time slice update as slider moves
- Pre-normalizes all slices for consistent display
- Frame-by-frame analysis capability
- Value range: 0 to (total_slices - 1)

**UI Components**:
- Main axis: CT-PET overlay display
- Bottom slider: Slice selection
- Automatic normalization per slice
- Dynamic title updates

**Usage**:
```python
python ogovint.py
```

**Interactivity**: Move slider to browse through all axial slices of the registered volume.

---

### 4. `reg.py` - Image Registration Pipeline
**Purpose**: Performs rigid multi-modal image registration using mutual information optimization.

**Algorithm**: Mattes Mutual Information (MMI) with Regular Step Gradient Descent

#### Registration Components

**1. Similarity Metric**
```python
MattesMutualInformation(numberOfHistogramBins=50)
```
- **Rationale**: Robust for multi-modal (CT-PET) registration
- **Histogram bins**: 50 for MI calculation
- **Sampling strategy**: Random (20% of pixels)
- **Advantage**: Intensity-independent matching

**2. Optimizer**
```python
RegularStepGradientDescent(
    learningRate=2.0,
    minStep=1e-4,
    numberOfIterations=200,
    relaxationFactor=0.5
)
```
- **Type**: Gradient descent with adaptive step size
- **Learning rate**: 2.0
- **Convergence**: Minimum step 1e-4
- **Iterations**: Maximum 200
- **Relaxation**: 0.5 per iteration

**3. Transform**
```python
Euler3DTransform()  # 6 degrees of freedom
```
- **Type**: Rigid (rotation + translation)
- **DOF**: 6 (3 rotations, 3 translations)
- **Initialization**: Centered transform initializer
- **Method**: Geometry-based initialization

**4. Interpolator**
```python
sitkLinear
```
- Smooth interpolation during resampling
- Balances speed and accuracy

#### Registration Workflow

1. **Image Loading**: Read CT and PET as Float32
2. **Initialize Transform**: Compute initial rigid transformation from image geometry centers
3. **Optimize**: Minimize MI metric to find optimal alignment
4. **Apply Transform**: Resample PET using final transform
5. **Visualize**: Display registered overlay

**Usage**:
```python
python reg.py
```

**Output**: Registered PET overlaid on CT showing improved alignment.

---

## Technical Implementation Details

### Coordinate System Handling
- **Image orientation**: Both CT and PET have `[-1, 0, 0; 0, -1, 0]` direction matrix
- **Patient position**: HFP (Head First Prone)
- **Spatial reference**: SimpleITK handles voxel-to-physical mapping

### Intensity Normalization
```python
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
```
- Maps intensities to [0,1] range
- Prevents division by zero with epsilon
- Enables consistent visualization across modalities

### Resampling Pipeline
```python
ResampleImageFilter()
    → SetReferenceImage(ct_img)
    → SetInterpolator(sitkLinear)
    → SetTransform(transform)
    → Execute(pet_img)
```
- Output matches reference image (CT) exactly
- Linear interpolation for sub-voxel accuracy
- Transform applied: identity (ogov.py) or optimized (reg.py)

---

## Installation

### Prerequisites
```bash
pip install SimpleITK numpy matplotlib nibabel
```

### System Requirements
- Python 3.7+
- 8GB+ RAM for volumetric processing
- Windows/Linux/macOS

---

## Usage Workflow

### Step 1: View Original Unregistered Images
```bash
python og.py
```
Analyzes baseline misalignment between CT and PET.

### Step 2: Resampled Overlay
```bash
python ogov.py
```
Creates overlay by geometric resampling without optimization.

### Step 3: Interactive Inspection
```bash
python ogovint.py
```
Browse through all slices with slider.

### Step 4: Rigid Registration
```bash
python reg.py
```
Performs full rigid registration with mutual information.

---

## Visualization Pipeline

### Colormap Strategy
- **CT**: `gray` - Shows anatomical structure
- **PET**: `hot` - Highlights metabolic activity
- **Overlay**: Alpha blending (alpha=0.4) for simultaneous viewing

### Display Optimization
- **Slice selection**: Middle slice (shape[0] // 2)
- **Color normalization**: Per slice for dynamic range
- **Background**: Removed axes for clean medical viewing

---

## Performance Considerations

### Computational Complexity
- **Registration**: O(N log N) for MI with 200 iterations
- **Resampling**: O(N) where N = voxels
- **Total time**: ~30-60 seconds per patient on CPU

### Memory Usage
- Float32 precision (efficient for medical volumes)
- Lazy loading: Images loaded on-demand
- Peak memory: ~500MB per volume

### Optimization Strategies
1. **Random sampling**: 20% of voxels for MI (50x speedup)
2. **Coarse-to-fine**: Multi-resolution (potential extension)
3. **GPU acceleration**: Optional SimpleITK-SimpleElastix integration

---

## Scientific Foundation

### Mutual Information Theory
Mutual Information measures statistical dependence between two images:
```
MI(I1, I2) = H(I1) + H(I2) - H(I1, I2)
```
where H is entropy. Maximizing MI aligns similar anatomical structures across modalities.

### Rigid Transform Parameter Space
6 parameters optimized:
- **Translations**: tx, ty, tz
- **Rotations**: θx, θy, θz (Euler angles)

Initialized from geometric image centers for fast convergence.

### Multi-Modal Challenges Addressed
1. **Intensity mismatch**: MI ignores absolute values
2. **Contrast differences**: Statistical correlation used
3. **Noise robustness**: Random sampling reduces sensitivity
4. **Limited overlap**: Geometry-aware initialization

---

## Extensions and Future Work

### Planned Enhancements
1. **Non-rigid registration**: B-spline deformable transforms
2. **Deep learning**: VoxelMorph-based unsupervised registration
3. **Multi-resolution**: Pyramid-based coarse-to-fine approach
4. **Validation metrics**: Dice coefficient, Hausdorff distance
5. **Batch processing**: Multi-patient automation

### Integration Opportunities
- PACS integration for clinical workflow
- Tumor segmentation on registered images
- Quantitative SUV correlation analysis
- Longitudinal registration for treatment monitoring

---

## References

1. SimpleITK Framework: https://simpleitk.org
2. Mattes Mutual Information: Mattes et al. (2003), "PET-CT Image Registration in the Chest"
3. NIfTI Format: https://nifti.nimh.nih.gov
4. Medical Image Registration: Zitová & Flusser (2003)

---

## License

Academic/Research Use

---

## Contact

For questions or contributions, refer to project documentation.


