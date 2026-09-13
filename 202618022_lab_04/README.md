https://202618022diyashahds605-gaartlqgqvn782qckirhty.streamlit.app/
<img width="939" height="912" alt="image" src="https://github.com/user-attachments/assets/5f7d3859-84d7-4765-be6d-76c04b9f4c5a" />
## 1. Project Overview & Workflow

The project follows a systematic 4-stage machine learning workflow:
* **Task 1 — Data Cleaning & Feature Engineering**: Handled missing values, filtered non-positive price anomalies, trimmed extreme price outliers via the Interquartile Range (IQR) technique, and mitigated target leakage.
* **Task 2 — Model Training, Benchmarking & Hyperparameter Tuning**: Trained Linear Regression (baseline), Random Forest Regressor, and Gradient Boosting Regressor under cross-validation. Selected and tuned the best-performing ensemble tree.
* **Task 3 — Web Application Development**: Developed an interactive UI using Streamlit allowing end-users to input property specifics (borough, room type, coordinates, availability, review metrics) and receive real-time price estimations.
* **Task 4 — Model Serialization & Cloud Deployment**: Serialized preprocessing statistics and model weights into portable artifacts and deployed the application to Streamlit Community Cloud.

---

## 2. Main Data Analysis & Preprocessing Findings

* **Target Skewness & Outliers**: Raw listing prices ranged from $0 to $10,000 with extreme right-skewness. Non-positive prices ($0) were removed. Prices above the $Q_3 + 1.5 \times \text{IQR}$ threshold (~$334) were filtered to keep the training data focused on standard listings rather than luxury penthouses.
* **Feature Leakage Elimination**: High-correlation synthetic features (such as `log_price` or redundant target columns) were excluded to prevent artificial metric inflation ($R^2 \approx 1.0$).
* **Location & Room Impact**: Geospatial features (`latitude`, `longitude`) along with categorical dimensions (`neighbourhood_group`, `room_type`) served as the strongest non-linear predictors of price variance.
* **High-Cardinality Management**: Arbitrary high-cardinality columns (`name`, `host_id`, `host_name`, `last_review`, `neighbourhood`) were dropped to prevent dimensionality explosion and maintain stable real-time inference.

---

## 3. Model Comparison Benchmark

Models were benchmarked on an 80/20 train/test split using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$):

| Model | Train $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) | Key Findings / Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 0.402 | 0.395 | $39.42 | $52.85 | Underfits; unable to map non-linear spatial interactions between latitude and longitude coordinates. |
| **Gradient Boosting Regressor** | 0.615 | 0.562 | $30.40 | $45.18 | Generalizes well, capturing sequential residual errors with moderate tree depth. |
| **Random Forest Regressor (Tuned)** | **0.884** | **0.589** | **$28.95** | **$43.70** | **Best Overall**: Captures complex borough interactions and availability thresholds with minimal variance error. |

### Final Selected Model Performance
* **Algorithm**: Random Forest Regressor (`n_estimators=80`, `max_depth=12`, `min_samples_leaf=4`)
* **Test MAE**: **$28.95** (average prediction error within standard nightly booking boundaries)
* **Test RMSE**: **$43.70**
* **Test $R^2$**: **~0.59**

---

## 4. Web Application (Streamlit)

The interactive application allows users to configure rental parameters and estimate pricing:
* **Location & Category Inputs**: Borough (`neighbourhood_group`), Room Type (`Entire home/apt`, `Private room`, `Shared room`), Coordinates (`latitude`, `longitude`).
* **Listing Mechanics**: Minimum booking nights, 365-day availability window, total review count, and host listing density.
* **Preprocessing Decoupling**: Input data is scaled using pre-calculated training distribution parameters (`mean_`, `scale_`) and category matrices, eliminating version mismatch issues across environments.

---

## 5. System Limitations & Future Improvements

1. **Temporal & Seasonal Invariance**: The dataset represents a static snapshot (2019) and does not capture dynamic pricing fluctuations driven by day-of-week demand, local events, holidays, or seasonal shifts.
2. **Omission of Amenities**: Critical price drivers such as air conditioning, pools, elevators, balconies, and parking are not captured in the tabular schema.
3. **Unstructured Review Sentiment**: Textual reviews and host descriptions were omitted; leveraging NLP sentiment analysis on reviews could provide further predictive signal.
4. **Extreme Luxury Truncation**: Because IQR filtering pruned prices >$334, the model is calibrated for mid-market listings and underpredicts luxury properties.

---

## 6. Repository Architecture

```text
202618022_Diyashah_DS605/
│
├── 202618022_lab_04/
│   ├── data/
│   │   └── AB_NYC_2019.csv               # Dataset
│   ├── 202618022_lab04.ipynb             # Complete Jupyter Notebook (EDA, Modeling, Tuning)
│   ├── app.py                            # Streamlit web application
│   ├── airbnb_model_artifacts.pkl        # Serialized model and preprocessing parameters
│   └── requirements.txt                  # Python dependencies
│
└── README.md                             # Project documentation and summary
