# RedDino: A Foundation Model for Red Blood Cell Analysis

**RedDino** is a state-of-the-art self-supervised foundation model designed specifically for red blood cell (RBC) image analysis. This repository accompanies our MICCAI 2025 submission and provides the code, example notebook, and media assets necessary to reproduce and extend our work.
---

## Data Figure

The **Data Figure** below illustrates the composition of our training set:

- **Patient and Image Count:** Over 420 patients and 56,712 original images.
- **Extracted Samples:** More than 3 million segmented red blood cells and 1.25 million patch images.

![Data Figure](media/data.png)

*Figure: Visualization of the training dataset composition, showing the number of patients, original images, and extracted RBC patches.*

---

## Overview

RedDino leverages a modified Dinov2 framework tailored for RBC morphology analysis. Key contributions include:

- **Dedicated Foundation Model:** Optimized for RBC analysis with robust feature representations.
- **Extensive Dataset:** Trained on over 1.25 million RBC images from 18 diverse datasets spanning various imaging modalities and staining techniques.
- **Rigorous Evaluation:** Comprehensive ablation studies and benchmarking against state-of-the-art models demonstrate significant improvements in classification and shape analysis tasks.
- **Visualization Tools:** PCA and UMAP visualizations reveal the model’s ability to differentiate between healthy, pathological, and agglomerated cells.

---

---

## Example Notebook

We provide an [example notebook](usage_example.ipynb) to help you get started with RedDino. This notebook demonstrates:

- How to load the pre-trained RedDino model.
- Feature extraction from RBC images.


---

## Installation

To run the example notebook and use the provided code, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/RedDino.git
   cd RedDino
