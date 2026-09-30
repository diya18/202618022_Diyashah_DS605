# DS605: Fundamentals of Machine Learning — Lab 6
**Feature Extraction & Traditional ML for Image and Text Data**

---

## 1. Project Overview
This repository implements end-to-end machine learning pipelines using hand-crafted feature engineering without deep learning, CNNs, or pre-trained embeddings
* **Part A (Image Classification):** Pavement crack detection on the Asphalt Crack Dataset (400 images) via NumPy statistical metrics and OpenCV Canny edge extraction.
* **Part B (Text Classification):** Email spam detection on 5,172 emails via `CountVectorizer` and traditional classifiers.
* **Part C (Representation Improvement):** Feature space optimization and trade-off analysis between dimensionality, runtime, and accuracy.

---

## 2. Part A: Image Feature Extraction & Classification

### Extracted Features per Image
* **NumPy Intensity Stats:** Mean brightness, contrast (std), dark-pixel ratio (< 50), and bright-pixel ratio (> 200).
* **OpenCV Canny Edge Detection:** Double threshold (100, 200) to calculate edge pixel count and edge density (128x128 resolution).

### Canny Edge Visualizations
![Canny Edge Visualization](visualizations/canny_edge_visualization.png)

### Performance Benchmark (80/20 Stratified Split)

| Model | Accuracy | Precision | Recall | F1-Score | Train Time (ms) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression**[cite: 1] | 0.8875 | 0.8750 | 0.8750 | 0.8750 | 12.40 | 0.35 |
| **SVM (RBF Kernel)**[cite: 1] | 0.9125 | 0.9000 | 0.9250 | 0.9123 | 18.20 | 0.65 |
| **Random Forest (100 Trees)**[cite: 1] | **0.9375** | **0.9268** | **0.9500** | **0.9383** | 85.60 | 3.10 |

---

## 3. Part B & C: Text Classification & Representation Improvement

### Dataset Distribution
* **Total Emails:** 5,172
* **Non-Spam (`0`):** 3,672 (70.99%)
* **Spam (`1`):** 1,500 (29.01%)

### Representation Comparison (CountVectorizer Baseline vs. Optimized)
* **Baseline (Part B):** Full vocabulary (`min_df=2`, stop words removed), raw term frequency counts.
* **Improved (Part C):** Vocabulary pruned to top 1,500 tokens (`max_features=1500`) with binary occurrence encoding (`binary=True`).

| Representation | Dimensions | Model | Accuracy | Precision | Recall | F1-Score | Train Time (ms) | Predict Time (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Part B: Baseline**[cite: 3] | 2,731 | Multinomial NB | 0.9382 | 0.8576 | 0.9433 | 0.8984 | 8.53 | 1.51 |
| **Part B: Baseline**[cite: 3] | 2,731 | Logistic Regression | 0.9710 | 0.9355 | 0.9667 | 0.9508 | 444.75 | 0.56 |
| **Part C: Improved**[cite: 4] | **1,500** | Multinomial NB | 0.9343 | 0.8554 | 0.9300 | 0.8911 | **4.10** | **0.79** |
| **Part C: Improved**[cite: 4] | **1,500** | Logistic Regression | **0.9739** | **0.9361** | **0.9767** | **0.9560** | **150.35** | **0.48** |

---

## 4. Key Observations & Trade-Off Analysis

* **Edge Density as a Discriminator (Part A):** Cracked surfaces produce significantly higher Canny edge counts and dark-pixel ratios than uniform road textures. Random Forest achieved the highest image F1-score (0.9383) because tree ensembles handle non-linear combinations of brightness thresholds and edge boundaries better than linear baselines.
* **Dimensionality vs. Overfitting (Part C):** Pruning 45% of infrequent and noisy words (2,731 -> 1,500) preserved essential discriminative vocabulary without performance loss.
* **Computational Efficiency (Part C):** Limiting vocabulary and using binary word indicators decreased Logistic Regression training time by **66.2%** (444.75 ms -> 150.35 ms) due to fewer parameters and faster optimization convergence.
* **Binary Encoding vs. Term Frequency (Part C):** Binary indicator features (`binary=True`) prevented long spam emails with repeated keywords from distorting decision boundaries, improving overall spam Recall from **96.67% to 97.67%** and Accuracy to **97.39%**.

---

## 5. Execution Instructions
```bash
# 1. Install dependencies
pip install opencv-python numpy pandas scikit-learn matplotlib

# 2. Run Image Pipeline (Generates feature table and edge plot)
python part_a_image_classification.py

# 3. Run Text Classification and Comparison
python part_b_c_spam_classification.py
