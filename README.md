# Breast Cancer Diagnosis using k-Nearest Neighbors (k-NN) in R

This project demonstrates the implementation of the k-Nearest Neighbors (k-NN) algorithm to classify breast cancer tumors as either benign or malignant based on diagnostic measurements. The analysis is performed entirely in R using a dataset from the Wisconsin Diagnostic Breast Cancer (WDBC) collection.

---

## 📋 Table of Contents
* [Dataset](#-dataset)
* [Project Workflow](#-project-workflow)
* [Dependencies](#-dependencies)
* [How to Run](#-how-to-run)
* [Results & Evaluation](#-results--evaluation)
* [Finding the Optimal K-Value](#-finding-the-optimal-k-value)

---

## 📦 Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, which can be found in the `wisc_bc_data.csv` file.

* **Source**: This dataset is a classic and publicly available collection from the UCI Machine Learning Repository.
* **Content**: It contains 569 instances with 32 attributes.
    * One `ID` column (which is removed during preprocessing).
    * One `diagnosis` column, which is our target variable (`M` = Malignant, `B` = Benign).
    * Thirty real-valued numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

The initial distribution of the diagnosis is:
* **Benign**: 357 cases (62.7%)
* **Malignant**: 212 cases (37.3%)

---

## ⚙️ Project Workflow

The project follows these key steps:

1.  **Data Loading & Exploration**: The dataset is loaded into R, and its structure is examined using functions like `str()` and `summary()`.
2.  **Data Preprocessing**:
    * The non-informative `ID` column is removed.
    * The target variable `diagnosis` is converted into a factor with clear labels ("Benign", "Malignant").
    * All 30 numeric features are **normalized** to a scale of 0-1. This is a critical step for distance-based algorithms like k-NN, as it prevents features with larger scales from dominating the distance calculation.
3.  **Data Splitting**: The normalized dataset is split into a training set (first 469 instances) and a testing set (the remaining 100 instances). The class labels are stored separately.
4.  **Model Training & Prediction**: The k-NN model is built using the `knn()` function from the `class` package. An initial value of **k=21** is chosen.
5.  **Performance Evaluation**: The model's predictions on the test set are evaluated against the actual labels using a confusion matrix (`CrossTable`).
6.  **Hyperparameter Tuning**: The **Elbow Method** is used to find an optimal value for `k` by iterating from k=1 to k=50 and plotting the error rate for each value.

---

## 🧩 Dependencies

To run this project, you need R and the following packages:
* `class`: Contains the `knn()` function for the k-NN algorithm.
* `gmodels`: Used for creating the `CrossTable` (confusion matrix).
* `ggplot2`: Used for visualizing the error rate in the Elbow Method.

You can install them with the following command in R:

install.packages(c("class", "gmodels", "ggplot2"))

## ▶️ How to Run

1.  Clone this repository to your local machine.
2.  Make sure the `wisc_bc_data.csv` file is in the same directory as your R script or notebook.
3.  Open the R script (`.R` or `.Rmd`) in RStudio or your preferred R environment.
4.  Run the script from top to bottom to execute the analysis.

---

## 📊 Results & Evaluation

The model was initially tested with **k=21**. The performance on the 100-instance test set was excellent.

The confusion matrix below summarizes the results:

|                | **Predicted: Benign** | **Predicted: Malignant** |
| :------------- | :-------------------: | :----------------------: |
| **Actual: Benign** |          61           |            0             |
| **Actual: Malignant**|           2           |            37            |

**Key Performance Metrics:**
* **Total Predictions**: 100
* **Correct Predictions**: 98 (61 + 37)
* **Incorrect Predictions**: 2
* **Accuracy**: **98.0%**

The model correctly identified all 61 benign cases and misclassified only 2 malignant cases as benign. This high level of accuracy demonstrates the effectiveness of the k-NN algorithm for this classification task.

---

## 📈 Finding the Optimal K-Value

To ensure `k=21` was a good choice, the **Elbow Method** was used to test k-values from 1 to 50. The error rate was calculated for each `k` and plotted.

The goal is to find the "elbow" of the curve, which represents a point where the error rate becomes stable and adding more neighbors doesn't significantly improve the model's performance.

```r
# Code to generate the elbow plot data
error.rate <- NULL
for(i in 1:50){
  set.seed(101)
  df_pred = knn(train = df_train, test = df_test, cl = df_train_labels, k = i)
  error.rate[i] = mean(df_test_labels != df_pred)
}

# Plotting the error rate vs. k-values
library(ggplot2)
k_values <- 1:50
error.df <- data.frame(error.rate, k_values)
ggplot(error.df, aes(x=k_values, y=error.rate)) + 
  geom_point() + 
  geom_line(lty="dotted", color='red') +
  labs(title = "Elbow Method for Optimal k", x = "K-Value", y = "Error Rate")
