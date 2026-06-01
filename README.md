# STN-DBS Localization Pipeline (Intraoperative MER Analysis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![SciPy](https://img.shields.io/badge/SciPy-1.11.3-lightgrey)

## 📌 Clinical Overview
Deep Brain Stimulation (DBS) is a highly effective surgical treatment for Parkinson's Disease. However, clinical outcomes are strictly dependent on the sub-millimetric placement of electrodes within the Subthalamic Nucleus (STN). While preoperative MRI provides anatomical targeting, intraoperative Micro-Electrode Recordings (MERs) are required for precise physiological mapping.

This repository provides an automated, real-time-oriented Deep Learning and DSP pipeline designed to analyze MER signals, extract neurophysiological features, and assist neurosurgeons in distinguishing STN from non-STN regions during the surgical descent.

## 🏗️ Architecture & Pipeline
Unlike batch-processing academic scripts, this project is engineered as a modular, object-oriented software pipeline suitable for System Integration in MedTech environments.

### 1. Data Ingestion (`data_loader.py`)
* Abstracts the decoding of proprietary `.mat` structures.
* Safely extracts continuous MER streams (sampled at 20 kHz) for both right and left hemispheres across multiple surgical depths.

### 2. Signal Pre-processing (`preprocessor.py`)
Optimized for O(N) time complexity to support low-latency inference:
1. **Artifact Rejection:** Dynamic, symmetric clipping (mean ± 3 standard deviations) to remove excessive transients and high-amplitude surgical noise.
2. **Frequency Isolation:** Zero-phase 4th-order Butterworth bandpass filter (200-5000 Hz) to isolate the neural spiking band without introducing phase distortion.

### 3. Feature Engineering (`features.py`)
To minimize computational overhead and preserve battery/latency in embedded operative environments, the feature set was strictly reduced to three highly discriminative, mathematically stable metrics calculated over 2-second non-overlapping windows:
* **Root Mean Square (RMS):** Quantifies the overall neural signal energy.
* **Curve Length:** Captures high-frequency morphological variations.
* **Number of Spikes:** Computes discrete neural firing events using threshold-based crossing detection.

### 4. Classification Engine (`model.py`)
* **Architecture:** CART Decision Tree Classifier.
* **Interpretability:** Chosen specifically over black-box architectures (like deep Neural Networks) to provide clinical explainability for surgical decision-making.

---

## 📊 Clinical Validation (LOSO Cross-Validation)

The system was evaluated using a rigorous Leave-One-Subject-Out (LOSO) cross-validation strategy on bilateral STN-DBS data from three patients. Metrics were selected to reflect clinical risk (prioritizing Sensitivity and Specificity over naive Accuracy).

| Test Subject | Accuracy | Sensitivity (Recall) | Specificity | False Positives | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subj20** | 88.52% | 99.42% | 54.13% | 50 | 2 |
| **Subj33** | 92.73% | 96.12% | 86.21% | 24 | 13 |
| **Subj38** | 75.24% | 46.02% | 98.26% | 5 | 122 |
| **Global Avg.** | **85.50%** | **80.52%** | **79.53%** | - | - |

### 🔍 Architectural Diagnostics & Next Steps
The variance observed across subjects (e.g., low Specificity in Subj20, low Sensitivity in Subj38) directly exposes the inherent limitations of training a classifier on **absolute feature values** within a constrained dataset (N=3). 

Neural signal properties (like RMS and amplitude) are highly subject-dependent, varying with electrode impedance, tissue density, and individual anatomy. For instance, Subj38 likely presented an intrinsically lower baseline impedance, causing the model (trained on higher-amplitude subjects) to misclassify STN regions as background noise (high False Negatives).

**Roadmap to Production:**
To scale this proof-of-concept into a robust intraoperative tool, the next architectural milestone involves transitioning from absolute features to **Relative Baseline Calibration**. By implementing a dynamic buffer that calculates a statistical baseline during the initial cortical descent (non-STN), the pipeline will compute delta features (e.g., ΔRMS, Z-scores). This normalizes impedance variations across patients, stabilizing Sensitivity and Specificity independent of the hardware or specific patient anatomy.

---

## 🚀 Usage & Reproducibility

### 1. Environment Setup
```bash
git clone [https://github.com/cristian-assumma/stn-dbs-localization-pipeline.git](https://github.com/cristian-assumma/stn-dbs-localization-pipeline.git)
cd stn-dbs-localization-pipeline
pip install -r requirements.txt
```
### 2. Clinical Data
Due to privacy and MedTech compliance (GDPR/HIPAA), raw `.mat` files are strictly excluded from this repository via `.gitignore`. To run the pipeline, place your authorized clinical data inside the `data/` directory.
### 3. Executing the Pipeline
Run the fully orchestrated validation cycle with:
```bash
python main.py
```
---

## 📬 Contact

**Ing. Cristian Assumma**
*MSc Biomedical Engineer | AI Healthcare & MedTech*

* [LinkedIn](https://www.linkedin.com/in/cristian-assumma-08890b224)
* [GitHub](https://github.com/cristian-assumma)

---
