# STN-DBS Localization Pipeline (Intraoperative MER Analysis)

![Python](https://img.shields.io/badge/Python-3.10--slim-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![SciPy](https://img.shields.io/badge/SciPy-1.11.3-lightgrey)
![Docker](https://img.shields.io/badge/Docker-Edge%20Ready-2496ED)
![Testing](https://img.shields.io/badge/PyTest-Passing-brightgreen)

## 📌 Clinical Overview
Deep Brain Stimulation (DBS) is a highly effective surgical treatment for Parkinson's Disease. However, clinical outcomes are strictly dependent on the sub-millimetric placement of electrodes within the Subthalamic Nucleus (STN). While preoperative MRI provides anatomical targeting, intraoperative Micro-Electrode Recordings (MERs) are required for precise physiological mapping.

This repository provides an automated, real-time-oriented Machine Learning and DSP pipeline designed to analyze MER signals, extract neurophysiological features, and assist neurosurgeons in distinguishing STN from non-STN regions during the surgical descent.

## 🏗️ Architecture & Pipeline
Unlike batch-processing academic scripts, this project is engineered as a modular, object-oriented software pipeline suitable for System Integration in MedTech environments (Edge deployment).

### 1. Data Ingestion (`data_loader.py`)
* Abstracts the decoding of proprietary `.mat` structures.
* Safely extracts continuous MER streams (sampled at 20 kHz) for both right and left hemispheres across multiple surgical depths.

### 2. Signal Pre-processing (`preprocessor.py`)
Optimized for $O(N)$ time complexity to support low-latency inference:
1. **Artifact Rejection:** Dynamic, symmetric clipping (mean ± 3 standard deviations) to remove excessive transients and high-amplitude surgical noise.
2. **Frequency Isolation:** Zero-phase 4th-order Butterworth bandpass filter (200-5000 Hz) to isolate the neural spiking band without introducing phase distortion.

### 3. Feature Engineering (`features.py`)
To minimize computational overhead and preserve latency in embedded operative environments, the feature set was strictly reduced to three highly discriminative, mathematically stable metrics calculated over 2-second non-overlapping windows:
* **Root Mean Square (RMS):** Quantifies the overall neural signal energy.
* **Curve Length:** Captures high-frequency morphological variations.
* **Number of Spikes:** Computes discrete neural firing events using threshold-based crossing detection.

### 4. Classification Engine (`model.py`)
* **Architecture:** CART Decision Tree Classifier.
* **Interpretability:** Chosen specifically over black-box architectures (like deep Neural Networks) to provide clinical explainability for surgical decision-making.

---

## 🛡️ Enterprise Software Engineering

To ensure clinical safety and maintainability, the pipeline strictly adheres to MLOps standards:

* **Mathematical Quality Assurance:** Digital Signal Processing (DSP) logic (e.g., RMS calculation, spike detection algorithms) is mathematically verified against known synthetic signals via automated unit tests (`pytest`).
* **CLI Engine:** All hyperparameters (e.g., subjects targeting, maximum tree depth) are abstracted behind an `argparse` Command Line Interface, preventing hardcoded logic and allowing dynamic tuning.
* **Edge Containerization:** The environment is encapsulated in a lightweight `python:3.10-slim` Docker container (~200MB), ensuring rapid deployment and deterministic execution on intraoperative hardware.

---

## 📊 Clinical Validation (LOSO Cross-Validation)

The system was evaluated using a rigorous Leave-One-Subject-Out (LOSO) cross-validation strategy on bilateral STN-DBS data from three patients. The model was dynamically constrained (`max_depth=4`) to prevent overfitting. Metrics prioritize clinical risk assessment (Sensitivity and Specificity over naive Accuracy).

| Test Subject | Accuracy | Sensitivity (Recall) | Specificity | False Positives | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subj20** | 88.52% | 99.42% | 54.13% | 50 | 2 |
| **Subj33** | 93.12% | 97.61% | 84.48% | 27 | 8 |
| **Subj38** | 86.55% | 73.01% | 97.21% | 8 | 61 |
| **Global Avg.** | **89.40%** | **90.01%** | **78.61%** | - | - |

## 🔍 Architectural Diagnostics & Next Steps
The variance observed across subjects exposes the inherent limitations of training a classifier on **absolute feature values** within a constrained dataset. 

Neural signal properties (like RMS and amplitude) are highly subject-dependent, varying with electrode impedance, tissue density, and individual anatomy. For instance, Subj38 likely presented an intrinsically lower baseline impedance, causing the model (trained on higher-amplitude subjects) to misclassify STN regions as background noise (high False Negatives).

**Roadmap to Production:**
To scale this proof-of-concept into a robust intraoperative tool, the next architectural milestone involves transitioning from absolute features to **Relative Baseline Calibration**. By implementing a dynamic buffer that calculates a statistical baseline during the initial cortical descent (non-STN), the pipeline will compute delta features (e.g., ΔRMS, Z-scores). This normalizes impedance variations across patients, stabilizing Sensitivity and Specificity independent of the hardware or specific patient anatomy.
---

## 🚀 Usage & Reproducibility (Docker Deployment)

### 1. Clinical Data Setup
Due to privacy and MedTech compliance (GDPR/HIPAA), raw `.mat` files are strictly excluded from this repository via `.dockerignore`. To run the pipeline, place your authorized clinical data inside the `data/` directory.

### 2. Build the Edge Container
```bash
docker build -t stn-dbs-pipeline:v1 .
```

### 3. Run the Inference Engine
Execute the fully orchestrated validation cycle dynamically via volume mounting. You can override targeted subjects and tree depth seamlessly:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  stn-dbs-pipeline:v1 \
  --subjects Subj20 Subj33 Subj38 \
  --max-depth 4
```
*(Windows PowerShell users: replace `$(pwd)` with `${PWD}`).*

---

## 📬 Contact

**Ing. Cristian Assumma**
*MSc Biomedical Engineer | AI Healthcare & MedTech*

* [LinkedIn](https://www.linkedin.com/in/cristian-assumma-08890b224)
* [GitHub](https://github.com/cristian-assumma)

---
