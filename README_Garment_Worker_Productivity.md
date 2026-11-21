# Garment Worker Productivity Prediction

## Project Overview

This project uses the **Garments Worker Productivity** dataset to
explore production-related information and build machine learning
regression models that estimate `actual_productivity`.

The project is completed in a Jupyter Notebook and covers data
inspection, cleaning, exploratory data analysis, feature engineering,
feature selection, model training, and performance evaluation.

## Project Objective

-   Explore the factors and production characteristics recorded in the
    garment worker dataset.
-   Prepare the data for machine learning.
-   Train and compare regression models to predict actual worker
    productivity.
-   Identify the best-performing model based on the evaluation results.

## Dataset

**File required:** `garments_worker_productivity.csv`

The notebook loads the CSV file using:

``` python
data = pd.read_csv("garments_worker_productivity.csv")
```

Place the CSV file in the **same folder** as the Jupyter Notebook before
running it.

The target column is:

-   `actual_productivity` --- the productivity value the models are
    trained to predict.

The dataset includes production and workforce information such as date,
quarter, department, team, targeted productivity, standard minute value,
work in progress, overtime, incentives, idle time, number of workers,
and actual productivity.

## Technologies and Libraries

-   Python
-   Jupyter Notebook
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn

## Project Workflow

1.  **Import libraries** --- load the Python packages used for data
    handling, visualization, and machine learning.
2.  **Load the dataset** --- read `garments_worker_productivity.csv`
    into a Pandas DataFrame.
3.  **Explore the data** --- inspect the first and last rows, shape,
    columns, data types, and descriptive statistics.
4.  **Clean the data** --- check missing values and duplicate rows,
    remove duplicates, convert the date column, fill missing numeric and
    categorical values, and remove rows without the target value.
5.  **Detect and treat outliers** --- visualize numeric columns with
    boxplots and apply the IQR clipping method to numeric input columns.
6.  **Feature engineering** --- extract month, day of month, and day of
    week from the date column.
7.  **Encode categorical data** --- convert categorical columns into
    numeric dummy variables using `pandas.get_dummies()`.
8.  **Separate features and target** --- use all remaining columns as
    `X` and `actual_productivity` as `y`.
9.  **Split the data** --- create training and testing sets using an
    80:20 split with `random_state=42`.
10. **Scale features** --- apply `StandardScaler` to the training and
    testing feature sets.
11. **Select features** --- use `SelectKBest` with `f_regression` to
    identify up to 10 useful features.
12. **Train regression models** --- fit the selected algorithms and
    generate predictions.
13. **Evaluate and compare models** --- calculate MAE, MSE, RMSE, and R²
    score.
14. **Review predictions** --- compare actual and predicted
    productivity, inspect residuals, and view Random Forest feature
    importance.

## Machine Learning Algorithms

The notebook trains and compares these four regression models:

  -----------------------------------------------------------------------
  Model                               Purpose
  ----------------------------------- -----------------------------------
  Linear Regression                   Models a linear relationship
                                      between input features and
                                      productivity.

  Decision Tree Regressor             Learns decision rules from the
                                      input features.

  Random Forest Regressor             Combines multiple decision trees to
                                      produce predictions.

  K-Neighbors Regressor (KNN)         Predicts using nearby observations
                                      in feature space.
  -----------------------------------------------------------------------

## Evaluation Metrics

  -----------------------------------------------------------------------
  Metric                  Meaning                 Preferred result
  ----------------------- ----------------------- -----------------------
  MAE                     Mean Absolute Error;    Lower
                          average absolute        
                          difference between      
                          actual and predicted    
                          values.                 

  MSE                     Mean Squared Error;     Lower
                          average squared         
                          prediction error.       

  RMSE                    Root Mean Squared       Lower
                          Error; square root of   
                          MSE, expressed in the   
                          target's units.         

  R² Score                Indicates how much      Higher
                          variation in the target 
                          is explained by the     
                          model.                  
  -----------------------------------------------------------------------

The notebook selects the model with the **lowest RMSE** as the best
model for the test set. Actual scores are produced when the notebook is
executed.

## Visualizations and Analysis

The notebook includes: - Numeric-feature boxplots for outlier
detection. - Model performance comparison using R² score. - Actual
versus predicted productivity scatter plot. - Residual distribution
histogram. - Top 10 Random Forest feature-importance chart.

## How to Run the Project

### 1. Download the project files

Keep these two files together in one folder:

-   `Untitled (5) (1).ipynb` (or rename it to
    `Garment_Worker_Productivity.ipynb`)
-   `garments_worker_productivity.csv`

### 2. Install the required packages

Run this command in a terminal or notebook cell if the libraries are not
already installed:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Open Jupyter Notebook

Start Jupyter Notebook:

``` bash
jupyter notebook
```

Open the project notebook in the browser.

### 4. Run the notebook

Run the cells **from top to bottom**. Make sure the CSV file remains in
the same folder as the notebook so the dataset can be loaded correctly.

## Expected Output

After running the notebook, you will obtain: - A cleaned and prepared
dataset. - Exploratory data analysis and visualizations. - Selected
feature scores. - Predictions from four regression models. - A model
comparison table containing MAE, MSE, RMSE, and R² score. - The best
model based on RMSE. - Actual-versus-predicted and residual
visualizations. - Random Forest feature-importance results.

## Notes and Limitations

-   The notebook reports the model performance based on the supplied
    dataset and the selected train-test split. Results may vary if the
    data or settings change.
-   Feature selection is performed and displayed as an analysis step.
    The model-training loop in the notebook uses the scaled full feature
    set rather than only the selected features.
-   Categorical encoding and outlier treatment are performed before the
    train-test split in the current notebook. For a stricter evaluation
    in a production workflow, these preprocessing steps should be fitted
    using training data only, then applied to the test data, to reduce
    the risk of data leakage.
-   Feature importance and correlations show patterns in this dataset;
    they do not establish that a variable causes productivity to change.

## Conclusion

This project demonstrates a complete introductory machine learning
workflow for garment worker productivity prediction. It explores the
dataset, prepares features, trains four regression algorithms, and
compares their predictions using standard regression metrics. The model
with the lowest RMSE is reported as the best-performing model for this
experiment.
