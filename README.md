# Car Price Prediction

Machine Learning project based on a used cars dataset.
The project focuses on predicting car prices, classifying cars into price categories, and applying clustering to find similar groups of cars.

## Dataset

The dataset contains 4,009 cars with the following features:

* Brand
* Model
* Model Year
* Mileage
* Fuel Type
* Engine
* Transmission
* Exterior Color
* Interior Color
* Accident
* Clean Title
* Price

## Data Preprocessing

The data was cleaned before training the models.

The main steps were:

* Handling missing values
* Converting price and mileage to numeric values
* Encoding categorical columns
* Scaling numerical features
* Splitting the data into training and testing sets
* Using Pipeline and ColumnTransformer

## Regression

The goal of this part is to predict the actual car price.

Models used:

* Linear Regression
* SVR
* Decision Tree
* Random Forest
* Bagging
* Boosting

The models were evaluated using:

* MAE
* RMSE
* R²

One of the results from the regression models:

| Model   |      MAE |      RMSE |     R² |
| ------- | -------: | --------: | -----: |
| Bagging | 19038.52 | 134817.44 | 0.1108 |

## Classification

The car prices were divided into three categories using price percentiles:

* Budget
* Medium
* Premium

Models used:

* Logistic Regression
* KNN
* SVM
* Naive Bayes
* Random Forest

Evaluation metrics:

* Accuracy
* Precision
* Recall
* ROC-AUC

Example results:

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |   0.8092 |  0.9418 |
| KNN                 |   0.7419 |  0.8785 |
| SVM                 |   0.7918 |  0.9313 |
| Naive Bayes         |   0.7095 |  0.8458 |
| Random Forest       |   0.7269 |  0.8948 |

## Clustering

Three clustering methods were tested:

* K-Means
* Hierarchical Clustering
* DBSCAN

K-Means was tested with 3 clusters.

Silhouette Score:

0.06445


Cluster sizes:


Cluster 0: 1678
Cluster 1: 784
Cluster 2: 1547


DBSCAN classified all observations as noise in the current setup, so a meaningful silhouette score was not calculated for it.

## PCA

PCA was used to reduce the feature space for visualization.

The first two components explained:


PC1: 20.56%
PC2: 5.04%


## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

