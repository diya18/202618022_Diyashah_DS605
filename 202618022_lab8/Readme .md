# DS605 Lab 8: Decision Trees and Support Vector Machines

## Overview
This assignment focuses on implementing Decision Tree and Linear Support Vector Machine (SVM) classifiers from scratch and comparing them with Scikit-learn implementations using the UCI Banknote Authentication dataset.

## Dataset
- **Dataset:** UCI Banknote Authentication
- **Samples:** 1,372
- **Features:** 4
- **Classes:** 2
- **Input features:** Variance, Skewness, Curtosis, and Entropy

## Part A: Decision Tree from Scratch
- Implemented Gini impurity and entropy.
- Developed recursive tree construction with stopping conditions.
- Implemented prediction and feature-threshold split rules.
- Visualized the manual tree structure and decision boundaries.
- Compared the manual implementation with Scikit-learn Decision Tree.

## Part B: Linear SVM from Scratch
- Implemented a linear soft-margin SVM using hinge loss and subgradient descent.
- Scaled features using training-set statistics.
- Visualized the decision boundary, margins, and margin-violating samples.
- Compared the manual implementation with Scikit-learn Linear SVM.
- Explored regularization (C), learning rate, and iterations.

## Part C: Kernel SVM
- Generated a nonlinear dataset using `make_moons`.
- Trained SVM models using Linear, Polynomial, and RBF kernels.
- Visualized and compared their decision boundaries.
- Explored how nonlinear kernels can model complex patterns.

## Part D: Final Model Comparison
The following models were evaluated using all four features of the Banknote Authentication dataset:
- Manual Decision Tree
- Scikit-learn Decision Tree
- Manual Linear SVM
- Scikit-learn Linear SVM
- Scikit-learn RBF SVM

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Prediction time

## Observations
- Decision Trees provide interpretable predictions through feature-threshold split rules.
- Gini impurity and entropy help determine suitable tree splits.
- Tree depth influences model complexity and generalization.
- Feature scaling is important for SVM models.
- Nonlinear kernels can learn complex decision boundaries.
- Evaluation metrics help compare classification performance.
- Manual implementations provide insight into the underlying algorithms.

## Conclusion
- Implemented Decision Tree and Linear SVM algorithms from scratch.
- Compared manual implementations with Scikit-learn models.
- Visualized decision boundaries, tree structures, and SVM margins.
- Explored hyperparameter tuning and nonlinear SVM kernels.
- Evaluated models using classification metrics and execution time.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

