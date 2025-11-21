# 🎓 ULTIMATE MACHINE LEARNING LAB EXAM GUIDE
**Complete Offline Reference | All Algorithms | Ready to Use Code**

---

## 📦 INSTALLATION REQUIREMENTS

### Step 1: Install Python Packages
Copy and run these commands in your terminal/command prompt:

```bash
# Core Data Science Libraries
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Machine Learning & Deep Learning
pip install tensorflow keras torch torchvision

# Additional Utilities
pip install jupyter notebook ipython

# All at once
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow keras torch torchvision jupyter notebook ipython
```

### Step 2: Verify Installation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import torch

print("✓ All packages installed successfully!")
```

---

## 📚 TABLE OF CONTENTS

1. [Essential Imports](#imports)
2. [Data Loading & Preprocessing](#preprocessing)
3. [Exploratory Data Analysis](#eda)
4. [Linear & Polynomial Regression](#regression)
5. [Ridge & Lasso Regression](#regularization)
6. [Decision Trees](#decision-trees)
7. [Ensemble Methods](#ensemble)
8. [Evaluation Metrics](#metrics)
9. [Neural Networks](#neural-networks)
10. [CNNs & Advanced Models](#cnn)
11. [RNNs & Sequence Models](#rnn)
12. [Concept Learning](#concept-learning)
13. [Quick Reference](#quick-reference)

---

## 1️⃣ ESSENTIAL IMPORTS {#imports}

```python
# Data Processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics & Evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, roc_curve, roc_auc_score)

# Models - Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

# Models - Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim

# Datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
import seaborn as sns

# Random Seeds (for reproducibility)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
```

---

## 2️⃣ DATA LOADING & PREPROCESSING {#preprocessing}

### Load Built-in Datasets
```python
# Iris
iris = load_iris()
X, y = iris.data, iris.target

# Breast Cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Digits (handwritten)
digits = load_digits()
X, y = digits.data, digits.target

# Titanic (from seaborn)
df = sns.load_dataset('titanic')

# Custom CSV
df = pd.read_csv('filename.csv')
```

### Handle Missing Values
```python
# Drop rows with missing values
df = df.dropna()

# Fill with mean (numerical)
df['column'] = df['column'].fillna(df['column'].mean())

# Fill with median
df['column'] = df['column'].fillna(df['column'].median())

# Fill with mode (categorical)
df['column'] = df['column'].fillna(df['column'].mode()[0])

# Forward fill
df = df.fillna(method='ffill')
```

### Encode Categorical Variables
```python
# Label Encoding (for binary/ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])

# One-Hot Encoding (for nominal)
df = pd.get_dummies(df, columns=['categorical_col'], drop_first=True)

# Manual encoding
mapping = {'cat1': 0, 'cat2': 1, 'cat3': 2}
df['column'] = df['column'].map(mapping)
```

### Feature Scaling
```python
# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler (0 to 1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check scaling
print(f"Mean: {X_train_scaled.mean()}, Std: {X_train_scaled.std()}")
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # for reproducibility
)

# Stratified split (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,         # maintains class ratio
    random_state=42
)
```

---

## 3️⃣ EXPLORATORY DATA ANALYSIS {#eda}

### Basic Info
```python
print(df.shape)              # (rows, columns)
print(df.head())             # first 5 rows
print(df.info())             # data types, nulls
print(df.describe())         # statistics
print(df.isnull().sum())     # missing values
print(df.dtypes)             # column types
```

### Univariate Analysis
```python
# Histogram
plt.hist(df['column'], bins=30, edgecolor='black')
plt.show()

# Boxplot
plt.boxplot(df['column'])
plt.show()

# Count plot (categorical)
sns.countplot(data=df, x='column')
plt.show()

# Distribution plot
sns.histplot(data=df, x='column', kde=True)
plt.show()
```

### Bivariate Analysis
```python
# Scatter plot
plt.scatter(df['x'], df['y'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Correlation coefficient
correlation = df['col1'].corr(df['col2'])
print(f"Correlation: {correlation}")
```

### Summary Stats
```python
print(df.describe())           # mean, std, min, max
print(df.skew())               # skewness
print(df.kurtosis())           # kurtosis
print(df['col'].value_counts()) # frequency
```

---

## 4️⃣ LINEAR & POLYNOMIAL REGRESSION {#regression}

### Linear Regression (Sklearn)
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### Linear Regression from Scratch
```python
class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Errors
            error = y_pred - y
            
            # Gradients
            dw = (2/n_samples) * np.dot(X.T, error)
            db = (2/n_samples) * np.sum(error)
            
            # Updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Usage
model = LinearRegressionFromScratch(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train linear regression on polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### Normal Equation (Closed Form)
```python
# θ = (X^T X)^-1 X^T y
def normal_equation(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

theta = normal_equation(X_train, y_train)
print(f"Coefficients: {theta}")
```

---

## 5️⃣ RIDGE & LASSO REGRESSION {#regularization}

### Ridge Regression (L2 Penalty)
```python
from sklearn.linear_model import Ridge

# Train
ridge = Ridge(alpha=1.0)  # alpha is regularization strength
ridge.fit(X_train, y_train)

# Predict
y_pred = ridge.predict(X_test)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Coefficients: {ridge.coef_}")

# Ridge closed form: θ = (X^T X + αI)^-1 X^T y
def ridge_closed_form(X, y, alpha=1.0):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    I = np.eye(X_b.shape[1])
    I[0,0] = 0  # Don't regularize bias
    theta = np.linalg.inv(X_b.T.dot(X_b) + alpha * I).dot(X_b.T).dot(y)
    return theta
```

### Lasso Regression (L1 Penalty)
```python
from sklearn.linear_model import Lasso

# Train
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
print(f"Zero coefficients: {np.sum(lasso.coef_ == 0)}")
```

### Compare Ridge vs Lasso
```python
alphas = np.logspace(-3, 2, 50)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X, y)
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(X, y)
    lasso_coefs.append(lasso.coef_)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_coefs)
plt.xscale('log')
plt.title('Ridge: Coefficient Paths')
plt.xlabel('Alpha (log scale)')

plt.subplot(1, 2, 2)
plt.plot(alphas, lasso_coefs)
plt.xscale('log')
plt.title('Lasso: Coefficient Paths')
plt.xlabel('Alpha (log scale)')

plt.tight_layout()
plt.show()
```

---

## 6️⃣ DECISION TREES {#decision-trees}

### Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Create and train
dt = DecisionTreeClassifier(
    max_depth=5,              # control depth
    min_samples_split=2,      # min samples to split
    min_samples_leaf=1,       # min samples in leaf
    random_state=42
)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, rounded=True, fontsize=10)
plt.show()

# Feature importance
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.title('Feature Importances')
plt.show()
```

### Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train, y_train)

y_pred = dt_reg.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### Gini & Information Gain
```python
# Gini Index: 1 - Σ(p_i)² where p_i is proportion of class i
# Entropy: -Σ(p_i * log2(p_i))
# Information Gain = Parent Entropy - Weighted Child Entropy
```

---

## 7️⃣ ENSEMBLE METHODS {#ensemble}

### Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=None,          # max depth of each tree
    min_samples_split=2,
    random_state=42,
    n_jobs=-1                # use all CPU cores
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Feature Importances: {rf.feature_importances_}")

# Out-of-Bag Error
print(f"OOB Score: {rf.oob_score_}")
```

### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### AdaBoost Classifier
```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Estimator Weights: {ada.estimator_weights_}")
print(f"Estimator Errors: {ada.estimator_errors_}")
```

### Gradient Boosting Classifier
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Feature Importances: {gb.feature_importances_}")
```

### Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=42, max_iter=1000)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = KNeighborsClassifier()

voting = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('knn', clf3)],
    voting='soft'  # 'soft' for probability voting, 'hard' for majority
)
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 8️⃣ EVALUATION METRICS {#metrics}

### Classification Metrics
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Individual scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Comprehensive report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### ROC Curve & AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC Score: {auc:.4f}")
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Residual plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, y_test - y_pred)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5,                    # 5-fold
    scoring='accuracy'       # metric
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

### Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training')
plt.plot(train_sizes, val_mean, label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid()
plt.show()
```

---

## 9️⃣ NEURAL NETWORKS {#neural-networks}

### Neural Network with TensorFlow/Keras
```python
from tensorflow.keras import layers, models

# Build model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
```

### Neural Network with PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork(784, 128, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loader
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.LongTensor(y_test).to(device)).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
```

### Activation Functions
```python
# ReLU: max(0, x)
# Sigmoid: 1 / (1 + e^-x)
# Tanh: (e^x - e^-x) / (e^x + e^-x)
# Softmax: e^x_i / Σ(e^x_j)

# In PyTorch
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)

# In Keras
layers.Activation('relu')
layers.Activation('sigmoid')
layers.Activation('tanh')
layers.Activation('softmax')
```

### Optimizers
```python
# PyTorch
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=0.001)
optim.RMSprop(model.parameters(), lr=0.001)
optim.Adagrad(model.parameters(), lr=0.01)

# Keras
'sgd'
'adam'
'rmsprop'
'adagrad'
```

---

## 🔟 CNNs & ADVANCED MODELS {#cnn}

### Simple CNN with Keras
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### CNN with PyTorch
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = datagen.flow(X_train, y_train, batch_size=32)
model.fit(train_generator, epochs=10, steps_per_epoch=len(