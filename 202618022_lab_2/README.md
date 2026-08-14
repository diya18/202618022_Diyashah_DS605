# LAB 02 – Vectorized Programming with NumPy and Data Analysis with Pandas

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Diya Shah |
| **Student ID** | 202618022 |
| **Assignment** | Lab 02 |

---

## 1. Project Overview

This assignment demonstrates the use of **NumPy and Pandas** for numerical computing, data analysis, data cleaning, feature engineering, and visualization.

The project is divided into two parts:

- **Part A – Vectorized Programming with NumPy**
- **Part B – Data Analysis with Pandas**

The work covers array operations, statistics, linear algebra, normal distribution, Titanic dataset analysis, filtering, grouping, missing-value handling, outlier detection, feature creation, pivot tables, and visualization.

---

## 2. Dataset

### Dataset Name
`train.csv`

### Dataset
**Titanic Passenger Dataset**

### Dataset Size
- **Rows:** 891
- **Columns:** 12

### Main Columns

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger ID |
| `Survived` | 0 = Did not survive, 1 = Survived |
| `Pclass` | Passenger class |
| `Name` | Passenger name |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation |

---

## 3. Objectives

- Perform vectorized operations using NumPy.
- Calculate statistical measures and perform array indexing and slicing.
- Perform matrix and linear algebra operations.
- Generate and analyze a normal distribution.
- Load and inspect a real-world dataset using Pandas.
- Filter data using Boolean indexing.
- Perform grouping and aggregation.
- Handle missing values using different imputation methods.
- Detect Fare outliers using the IQR method.
- Create new features and pivot tables.
- Visualize relationships and interpret the results.

---

## 4. Technologies and Libraries Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

---

# 5. Project Details

## Part A – Vectorized Programming with NumPy

### Task 1 – Arrays, Statistics, and Indexing

Generated random arrays and calculated minimum, maximum, median, mean, and standard deviation. Also used `arange()`, `zeros()`, `ones()`, `linspace()`, indexing, slicing, `reshape()`, and `flatten()`.

### Task 2 – Vectorized Arithmetic and Linear Algebra

Performed matrix addition, element-wise multiplication, matrix multiplication, transpose, determinant, inverse, and verified the inverse using `np.allclose()`.

### Task 3 – Normal Distribution and Histogram

Generated **1,000 values** from a normal distribution using a chosen mean of **50** and standard deviation of **10**. The sample mean and sample standard deviation were calculated and a histogram was plotted.

---

## Part B – Data Analysis with Pandas

### Task 4 – Load and Inspect Data

Loaded `train.csv` and used `head()`, `tail()`, `shape`, `columns`, `info()`, `describe()`, `loc`, and `iloc` to inspect and select data.

### Task 5 – Filtering and Querying

Used Boolean indexing to analyze passengers based on gender, age, class, fare, survival status, travelling status, and embarkation point.

### Task 6 – Groupby and Aggregation

Used `groupby()` to calculate:
- Survival rate by Sex
- Survival rate by Pclass
- Average Age and Fare by Pclass
- Passenger count and survival rate by Sex-Pclass
- Passenger count, average Fare, and survival rate by Embarked

### Task 7 – Missing Values and Fare Outliers

Checked missing values and their percentages, plotted missing-value counts, filled missing Age values, tested mean/median/mode/random-value imputation, and detected Fare outliers using the IQR method.

### Task 8 – Features and Pivot Table

Created:
- `FamilySize = SibSp + Parch + 1`
- `IsAlone = 1` when `FamilySize = 1`

A pivot table was also created using **Sex as rows, Pclass as columns, and mean Survived as values**.

### Task 9 – Visualizations and Observations

Created:
- Correlation heatmap
- Survival rate by Sex bar chart
- Age vs Fare scatter plot

The visualizations were used to identify relationships between passenger characteristics and survival.

---

## 6. Key Findings

- Female passengers had a considerably higher survival rate than male passengers.
- Passenger class showed a negative relationship with survival, meaning passengers in better classes generally had higher survival rates.
- Higher fares were associated with a better chance of survival.
- The analysis of Age and Fare showed passengers across different age groups, with both survivors and non-survivors present.
- The correlation heatmap helped identify relationships among numerical variables such as Pclass, Age, SibSp, Parch, Fare, and Survived.

---

## 7. Overall Conclusion

This assignment provided practical experience in using **NumPy for vectorized numerical operations** and **Pandas for data analysis**.

The Titanic dataset was inspected, filtered, grouped, cleaned, and analyzed. Missing values and Fare outliers were handled, new features were created, and different visualizations were used to understand survival patterns.

Overall, the project demonstrates a complete basic workflow for **data preprocessing, analysis, visualization, and interpretation using Python**.

---

## 8. Project Structure

```text
202618022_Lab_02/
│
├── README.md
├── 202618022_lab_02.ipynb
└── train.csv
```

---

## Author

**Diya Shah**  
**Student ID: 202618022**
