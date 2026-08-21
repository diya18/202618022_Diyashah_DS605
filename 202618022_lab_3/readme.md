# Hotel Booking Demand Classification

## Student Information

* **Name:** Diya Shah
* **Student ID:** 202618022
* **Assignment:** Hotel Booking Demand — Preprocessing Pipelines and Classification

## Dataset

**Dataset:** Kaggle Hotel Booking Demand (`hotel_bookings.csv`)

**Dataset Link:** [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

## Objective

The objective of this assignment is to build and compare two Scikit-learn preprocessing pipelines and evaluate two classification models for predicting whether a hotel booking will be canceled.

## Preprocessing Choices

* The dataset was split into **80% training and 20% testing data** using stratification and `random_state=42`.
* Columns with very high missingness, such as `company` and `agent`, were removed because a large proportion of their values were missing.
* `reservation_status` and `reservation_status_date` were removed to prevent **data leakage**.
* Numerical missing values were handled using **KNNImputer(n_neighbors=5)**.
* Categorical missing values were handled using **SimpleImputer(strategy="most_frequent")**.
* Categorical features were converted using **OneHotEncoder(handle_unknown="ignore")**.

### Pipeline A

**KNNImputer → StandardScaler → OneHotEncoder**

### Pipeline B

**KNNImputer → MinMaxScaler → OneHotEncoder**

Only clear and invalid outliers were removed after checking selected numerical features using the IQR method and boxplots.

## Models

Two classification algorithms were trained with both preprocessing pipelines:

1. Logistic Regression (`max_iter=1000`)
2. Decision Tree (`random_state=42`)

This resulted in four model-pipeline combinations.

## Evaluation

The models were evaluated using:

* Training Accuracy
* Testing Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## Final Observations

* The best overall combination was selected based on the **highest testing F1-score**, along with good accuracy, precision, and recall.
* Scaling can have a greater effect on **Logistic Regression** because it is sensitive to the scale of numerical features.
* **Decision Tree** performance is generally less affected by StandardScaler or MinMaxScaler because decision trees use feature thresholds rather than feature distances.
* The confusion matrices were used to compare correct and incorrect predictions for canceled and non-canceled bookings.
* The difference between training and testing accuracy was examined to identify possible overfitting. A smaller difference indicates better generalization.
