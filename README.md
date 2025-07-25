
---

##  How to Use

1. Clone this repo  
   ```bash
   git clone https://github.com/your-username/ML-Algorithms.git
   cd ML-Algorithms
Requirements
Install dependencies using:

pip install -r requirements.txt

Or manually:

pip install pandas numpy scikit-learn matplotlib seaborn


**Machine Learning Algorithms Summary**

This document summarizes ten essential machine learning algorithms you have completed, with key characteristics, their type (supervised/unsupervised), primary use cases, and common dataset applications. Each algorithm also includes its respective training code and brief explanation.

### Algorithm Overview Table

| S.No | Algorithm                    | Type         | Use Case / When to Use                                                                | Dataset Example Used          |
| ---- | ---------------------------- | ------------ | ------------------------------------------------------------------------------------- | ----------------------------- |
| 1    | Linear Regression            | Supervised   | Predicts continuous numeric values (e.g., housing prices, sales forecasting).         | Boston Housing Dataset        |
| 2    | Logistic Regression          | Supervised   | Used for binary/multiclass classification (e.g., spam detection, disease prediction). | Breast Cancer Dataset         |
| 3    | Decision Trees               | Supervised   | Used for both classification and regression; interpretable tree structure.            | Custom CSV Dataset            |
| 4    | Random Forest                | Supervised   | Reduces overfitting by combining multiple decision trees (e.g., feature importance).  | Wine Quality Dataset          |
| 5    | K-Means Clustering           | Unsupervised | Used for clustering unlabeled data (e.g., customer segmentation).                     | Custom CSV Dataset            |
| 6    | k-Nearest Neighbors (kNN)    | Supervised   | Classifies data based on closest neighbors (e.g., handwriting recognition).           | Custom CSV Dataset            |
| 7    | Lasso Regression             | Supervised   | L1 regularization; feature selection by zeroing less important features.              | Boston Housing Dataset        |
| 8    | Ridge Regression             | Supervised   | L2 regularization; shrinks coefficients to prevent overfitting.                       | Boston Housing Dataset        |
| 9    | Elastic Net Regression       | Supervised   | Combines L1 and L2; useful when features are correlated.                              | Boston Housing Dataset        |
| 10   | Support Vector Machine (SVM) | Supervised   | Maximizes class separation margin; effective in high-dimensional spaces.              | Cancer Classification Dataset |

---

### 1. Linear Regression

Linear Regression is used to predict continuous target variables. It assumes a linear relationship between input and output.

**Example**: Predicting house prices from square footage.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 2. Logistic Regression

Logistic Regression is used for classification problems with categorical output. It outputs probabilities and maps them to classes.

**Example**: Classifying tumor as benign or malignant.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 3. Decision Trees

Decision Trees split the data on feature values and build a tree for prediction. Very interpretable and used for both tasks.

**Example**: Approving loans based on income and credit score.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### 4. Random Forest

Random Forest uses an ensemble of decision trees to improve accuracy and reduce overfitting. It's robust and widely used.

**Example**: Predicting customer churn based on behavior data.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 5. K-Means Clustering

K-Means is an unsupervised algorithm that divides data into K clusters based on feature similarity.

**Example**: Grouping customers by spending behavior.

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X_scaled)
```

### 6. k-Nearest Neighbors (kNN)

kNN predicts class labels by majority vote of nearest neighbors. It's simple and works best on small datasets.

**Example**: Classifying handwritten digits (MNIST dataset).

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
```

### 7. Lasso Regression

Lasso Regression adds L1 penalty to eliminate unnecessary features. Helps with feature selection.

**Example**: Predicting house prices while automatically selecting relevant features.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

### 8. Ridge Regression

Ridge Regression adds L2 penalty and shrinks coefficients without eliminating them. Useful when many features have small effects.

**Example**: Forecasting sales with many small predictors.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### 9. Elastic Net Regression

Elastic Net combines both L1 and L2 regularization to balance feature selection and shrinkage.

**Example**: Modeling gene expression data with many correlated features.

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
```

### 10. Support Vector Machine (SVM)

SVM is used for classification and regression. It finds the optimal hyperplane that separates classes with the largest margin.

**Example**: Email spam detection or facial recognition.

```python
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)
```

---

All models were implemented practically using `scikit-learn`, trained on real datasets, and evaluated using accuracy, R², or other appropriate metrics. This README provides a strong summary of the foundational ML algorithms and their training in real-world applications.

