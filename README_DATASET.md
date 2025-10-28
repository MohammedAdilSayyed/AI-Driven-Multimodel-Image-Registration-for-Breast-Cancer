# Dataset Information and Usage Statistics

## Dataset Overview

This project utilizes a clinical breast imaging dataset containing paired CT and PET volumes for multi-modal image registration research.

---

## Dataset Statistics

### **Total Images**
- **Total Volumes**: 8 volumes (4 CT + 4 PET)
- **Total Slices**: 664 slices (332 CT + 332 PET)
- **Patients**: 4 patients (patient01-patient04)

### **Per-Patient Breakdown**
| Patient ID | CT Slices | PET Slices | Total Slices |
|------------|-----------|------------|--------------|
| patient01  | 83        | 83         | 166          |
| patient02  | 83        | 83         | 166          |
| patient03  | 83        | 83         | 166          |
| patient04  | 83        | 83         | 166          |
| **TOTAL**  | **332**   | **332**    | **664**      |

---

## Image Distribution by Modality

### **CT Images**
- **Count**: 4 volumes, 332 slices
- **Purpose**: Anatomical reference images
- **Usage**: Fixed reference space for registration
- **Format**: NIfTI compressed (.nii.gz)
- **Acquisition**: GE Discovery STE, prone breast position

### **PET Images**
- **Count**: 4 volumes, 332 slices  
- **Purpose**: Functional/metabolic images
- **Usage**: Moving images to be registered to CT
- **Format**: NIfTI compressed (.nii.gz)
- **Tracer**: FDG (Fluorodeoxyglucose)
- **Acquisition**: GE Discovery STE, prone breast position

---

## Dataset Usage in Project

### **Primary Usage (All 4 Patients)**
- **Training/Development**: All 4 patients used for algorithm development
- **Testing**: All 4 patients used for validation
- **Demonstration**: patient04 used as primary example in code files

### **File-Specific Usage**

#### `og.py` - Original Images
- **Patient**: patient04 (demonstration)
- **Images**: 1 CT volume + 1 PET volume
- **Slices**: 83 CT + 83 PET = 166 slices visualized
- **Purpose**: Baseline misalignment analysis

#### `ogov.py` - Resampled Overlay  
- **Patient**: patient04 (demonstration)
- **Images**: 1 CT volume + 1 resampled PET volume
- **Slices**: 83 CT + 83 PET = 166 slices processed
- **Purpose**: Geometric resampling validation

#### `ogovint.py` - Interactive Viewer
- **Patient**: patient04 (demonstration)
- **Images**: 1 CT volume + 1 resampled PET volume
- **Slices**: 83 slices browsable via slider
- **Purpose**: Interactive slice-by-slice analysis

#### `reg.py` - Registration Pipeline
- **Patient**: patient04 (demonstration)
- **Images**: 1 CT volume + 1 PET volume
- **Slices**: 83 CT + 83 PET = 166 slices registered
- **Purpose**: Rigid registration demonstration

---

## Validation Strategy

### **No Formal Train/Test Split**
- **Reason**: Small dataset (4 patients) insufficient for statistical validation
- **Approach**: All patients used for algorithm development and testing
- **Validation**: Visual inspection and qualitative assessment

### **Cross-Patient Validation**
- **Method**: Apply registration pipeline to each patient individually
- **Assessment**: Visual inspection of registration quality
- **Metrics**: Qualitative evaluation of anatomical alignment

### **Single-Patient Demonstration**
- **Primary**: patient04 used for all code demonstrations
- **Rationale**: Consistent example across all visualization tools
- **Coverage**: Complete pipeline demonstrated on one patient

---

## Dataset Characteristics

### **Spatial Properties**
- **Slice Thickness**: 3.27 mm (PET), variable (CT)
- **Orientation**: Axial slices
- **Patient Position**: Head First Prone (HFP)
- **Coordinate System**: DICOM standard

### **Temporal Properties**
- **Acquisition**: Same imaging session
- **Time Gap**: ~13-15 minutes between CT and PET
- **Motion**: Minimal (prone positioning reduces motion)

### **Quality Assurance**
- **Deidentification**: Properly anonymized per DICOM standards
- **Format**: Standardized NIfTI conversion from DICOM
- **Metadata**: Complete JSON sidecar files preserved
- **Validation**: All volumes successfully loaded and processed

---

## Future Dataset Expansion

### **Current Limitations**
- **Size**: Only 4 patients (insufficient for deep learning)
- **Validation**: No ground truth transformations available
- **Diversity**: Single scanner, single protocol

### **Recommended Expansion**
- **Target**: 50+ patients for robust validation
- **Split**: 70% training, 15% validation, 15% testing
- **Metrics**: Quantitative registration accuracy measures
- **Diversity**: Multiple scanners, protocols, patient positions

---

## File Size Information

### **Storage Requirements**
- **Format**: NIfTI compressed (.nii.gz)
- **Estimated Size**: ~50-100 MB per volume
- **Total Dataset**: ~400-800 MB
- **Metadata**: Additional ~1 KB per JSON file

### **Processing Requirements**
- **Memory**: ~500 MB per volume during processing
- **CPU**: Single-threaded SimpleITK processing
- **Time**: ~30-60 seconds per registration

---

## Data Access and Usage

### **File Locations**
```
data/patients/
├── patient01/
│   ├── ct.nii.gz (83 slices)
│   ├── ct.json (metadata)
│   ├── pet.nii.gz (83 slices)
│   └── pet.json (metadata)
├── patient02/ (83 CT + 83 PET slices)
├── patient03/ (83 CT + 83 PET slices)
└── patient04/ (83 CT + 83 PET slices)
```

### **Usage Rights**
- **Academic Use**: Research and educational purposes
- **Clinical Use**: Not validated for clinical decision-making
- **Redistribution**: Subject to original data provider terms

---

## Summary

**Total Dataset**: 8 volumes (4 CT + 4 PET) containing 664 slices
**Primary Usage**: Algorithm development and demonstration
**Validation**: Visual inspection across all patients
**Demonstration**: patient04 used for all code examples
**Future**: Requires expansion for robust statistical validation

