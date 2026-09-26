# Garment Worker Productivity Analysis and Prediction

An end-to-end machine learning study analyzing garment worker efficiency using the `garments_worker_productivity.csv` dataset. This project evaluates baseline linear models on both continuous productivity estimation and binary target attainment.

---

## 1. Project Overview

The objective of this project is to model and evaluate manufacturing floor performance through two complementary tasks:

1. **Continuous Target Estimation (Regression)**: Predicts the numerical value of worker output (`actual_productivity`) using ordinary least squares linear regression.
2. **Target Attainment Classification**: Formulates a decision boundary to predict whether a team meets or exceeds its targeted threshold (`Meets_Target`) using logistic regression.

---

## 2. Dataset & Preprocessing Pipeline

The workflow ingests tabular worker data and executes structured preprocessing steps using Scikit-Learn:

### Data Cleaning
* **Whitespace Trimming**: Standardized the categorical strings in the `department` feature via `df['department'].str.strip()` to eliminate duplicated levels caused by trailing spaces (e.g., `'finishing '` vs. `'finishing'`).
* **Missing Value Imputation**: Imputed missing values in `wip` (Work-in-Progress) with `0`, representing stages with no pending inventory backlog.

### Target Definition
* **Regression**: `y_reg = df['actual_productivity']`
* **Classification**: Binary flag defined as `Meets_Target = (actual_productivity >= targeted_productivity).astype(int)`

### Feature Transformation
Features are partitioned using `ColumnTransformer` prior to fitting to eliminate data leakage:
* **Numerical Features**: Scaled using `StandardScaler` to bring variance to unit scale.
* **Categorical Features**: Encoded using `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` to mitigate collinearity (dummy variable trap) and avoid inference failure on unseen labels.
* **Train/Test Partition**: 80% train and 20% test split via `train_test_split(..., test_size=0.2, random_state=42).

---

## 3. Modeling & Evaluation

### Task 1: Linear Regression (`actual_productivity`)

Models the direct productivity score using Scikit-Learn's `LinearRegression`.

| Metric | Measured Value | Description / Context |
| :--- | :--- | :--- |
| **MAE** | `0.1088` | Average absolute difference between predicted and actual productivity units. |
| **RMSE** | `0.1481` | Penalizes larger deviation errors across individual samples. |
| **$R^2$ Score** | `0.1736` | The linear features account for approximately 17.36% of the variance in worker productivity. |
| **Training Time** | `~2.14 ms` | Model training latency via `time.perf_counter()`. |
| **Inference Time**| `~0.17 ms`| Evaluation prediction latency on test batch. |

> **Key Takeaway**: The low $R^2$ score demonstrates that continuous worker productivity has non-linear relationships and hidden workplace variances that simple linear combinations struggle to capture fully.

---

### Task 2: Logistic Regression (`Meets_Target`)

Models the binary probability of meeting productivity goals using `LogisticRegression(max_iter=1000, random_state=42)`.

| Metric | Measured Value | Description / Context |
| :--- | :--- | :--- |
| **Accuracy** | `74.58%` | Overall proportion of correct binary predictions on the test set. |
| **Precision** | `76.61%`| Correctness rate when the model predicts that a team will hit its quota. |
| **Recall** | `94.35%` | Successfully captures 94.35% of all teams that actually met their quota. |
| **F1-Score** | `0.8456` | Harmonic mean of precision and recall. |
| **Training Time** | `~10.52 ms` | Model training latency via `time.perf_counter()`. |
| **Inference Time**| `~0.20 ms` | Evaluation prediction latency on test batch. |

> **Key Takeaway**: Framing target attainment as a binary classification problem is significantly more actionable; the high recall indicates an exceptionally low false-negative rate for identifying successful production lines.

---

## 4. Environment Setup & Execution

### Prerequisites

```bash
pip install numpy pandas scikit-learn tabulate
