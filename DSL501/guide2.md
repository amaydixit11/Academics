# 🎯 ULTIMATE ML LAB EXAM GUIDE
### Complete Reference Document - No Internet Required

---

## 📑 TABLE OF CONTENTS

1. [Essential Imports & Setup](#1-essential-imports--setup)
2. [Data Loading & Preprocessing](#2-data-loading--preprocessing)
3. [Bias-Variance Tradeoff](#3-bias-variance-tradeoff)
4. [Linear & Polynomial Regression](#4-linear--polynomial-regression)
5. [Ridge & Lasso Regression](#5-ridge--lasso-regression)
6. [Decision Trees](#6-decision-trees)
7. [Ensemble Methods](#7-ensemble-methods)
8. [Neural Networks (Deep Learning)](#8-neural-networks-deep-learning)
9. [CNNs (Convolutional Neural Networks)](#9-cnns-convolutional-neural-networks)
10. [RNNs (Recurrent Neural Networks)](#10-rnns-recurrent-neural-networks)
11. [Model Evaluation Metrics](#11-model-evaluation-metrics)
12. [Visualization Helpers](#12-visualization-helpers)
13. [Common Troubleshooting](#13-common-troubleshooting)

---

## 1. ESSENTIAL IMPORTS & SETUP

```python
# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Sklearn - Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve, ShuffleSplit

# Sklearn - Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# Sklearn - Metrics
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                              ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, 
                              r2_score, roc_curve, roc_auc_score)

# Sklearn - Datasets
from sklearn.datasets import (load_iris, load_breast_cancer, load_wine, load_digits, 
                               fetch_california_housing, make_classification, make_regression,
                               make_moons, make_blobs, make_circles)

# Deep Learning - TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, GRU, SimpleRNN, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

---

## 2. DATA LOADING & PREPROCESSING

### 2.1 Load Common Datasets

```python
# Classification Datasets
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Regression Dataset
housing = fetch_california_housing()
X_housing, y_housing = housing.data, housing.target

# Load CSV
df = pd.read_csv('data.csv')
```

### 2.2 Train-Test Split

```python
# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)
```

### 2.3 Scaling & Normalization

```python
# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler (0 to 1)
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)
```

### 2.4 Handling Missing Values

```python
# Fill with mean/median/mode
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

### 2.5 Encoding Categorical Variables

```python
# Label Encoding (for target)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encoding (for features)
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(X_categorical)
```

### 2.6 Creating Polynomial Features

```python
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## 3. BIAS-VARIANCE TRADEOFF

### 3.1 Compute Bias & Variance

```python
def compute_bias_variance(clf, X, y, n_repeat=40):
    """Compute bias and variance for a classifier"""
    shuffle_split = ShuffleSplit(test_size=0.33, n_splits=n_repeat, random_state=0)
    y_all_pred = [[] for _ in range(len(y))]
    
    for train_index, test_index in shuffle_split.split(X):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        
        for j, index in enumerate(test_index):
            y_all_pred[index].append(y_pred[j])
    
    bias_sq = sum([(1 - x.count(y[i])/len(x))**2 * len(x)/n_repeat 
                   for i, x in enumerate(y_all_pred)])
    var = sum([((1 - ((x.count(0)/len(x))**2 + (x.count(1)/len(x))**2))/2) * len(x)/n_repeat
               for i, x in enumerate(y_all_pred)])
    error = sum([(1 - x.count(y[i])/len(x)) * len(x)/n_repeat
                 for i, x in enumerate(y_all_pred)])
    
    return np.sqrt(bias_sq), var, error
```

### 3.2 Plot Bias-Variance vs Model Complexity

```python
def plot_bias_variance_complexity(model_class, X, y, param_name, param_range):
    """Plot bias-variance tradeoff"""
    bias_scores, var_scores, err_scores = [], [], []
    
    for param in param_range:
        model = model_class(**{param_name: param})
        b, v, e = compute_bias_variance(model, X, y)
        bias_scores.append(b)
        var_scores.append(v)
        err_scores.append(e)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, bias_scores, label='Bias²', marker='o')
    plt.plot(param_range, var_scores, label='Variance', marker='o')
    plt.plot(param_range, err_scores, label='Error', marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
from sklearn.tree import DecisionTreeClassifier
plot_bias_variance_complexity(
    DecisionTreeClassifier, 
    X_cancer, y_cancer, 
    'max_depth', 
    range(1, 15)
)
```

---

## 4. LINEAR & POLYNOMIAL REGRESSION

### 4.1 Linear Regression from Scratch

```python
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Usage
model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 4.2 Linear Regression with sklearn

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print(f"Coefficients: {lin_reg.coef_}")
print(f"Intercept: {lin_reg.intercept_}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### 4.3 Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create polynomial features
poly_reg = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

poly_reg.fit(X_train, y_train)
y_pred = poly_reg.predict(X_test)

# Visualize
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = poly_reg.predict(X_plot)

plt.scatter(X, y, alpha=0.5)
plt.plot(X_plot, y_plot, 'r-', linewidth=2)
plt.show()
```

### 4.4 Compare Different Polynomial Degrees

```python
degrees = [1, 2, 3, 5, 10, 15]
train_errors = []
test_errors = []

for d in degrees:
    poly_reg = make_pipeline(PolynomialFeatures(d), LinearRegression())
    poly_reg.fit(X_train, y_train)
    
    train_pred = poly_reg.predict(X_train)
    test_pred = poly_reg.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

plt.plot(degrees, train_errors, label='Train Error', marker='o')
plt.plot(degrees, test_errors, label='Test Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.show()
```

---

## 5. RIDGE & LASSO REGRESSION

### 5.1 Ridge Regression (L2)

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha controls regularization strength
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print(f"Ridge R² Score: {r2_score(y_test, y_pred):.4f}")
```

### 5.2 Lasso Regression (L1)

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Count non-zero coefficients
n_nonzero = np.sum(lasso.coef_ != 0)
print(f"Lasso R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Non-zero coefficients: {n_nonzero}/{len(lasso.coef_)}")
```

### 5.3 Ridge from Scratch

```python
class RidgeScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
    
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Closed form: (X^T X + αI)^-1 X^T y
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0  # Don't regularize bias
        
        self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights
```

### 5.4 Regularization Path Visualization

```python
alphas = np.logspace(-3, 3, 100)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_train, y_train)
    lasso_coefs.append(lasso.coef_)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_coefs)
plt.xscale('log')
plt.title('Ridge: Coefficient Paths')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')

plt.subplot(1, 2, 2)
plt.plot(alphas, lasso_coefs)
plt.xscale('log')
plt.title('Lasso: Coefficient Paths')
plt.xlabel('Alpha')

plt.tight_layout()
plt.show()
```

---

## 6. DECISION TREES

### 6.1 Classification Tree

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree_clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
tree_clf.fit(X_train, y_train)

# Predictions
y_pred = tree_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()
```

### 6.2 Regression Tree

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### 6.3 Feature Importance

```python
importances = tree_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

---

## 7. ENSEMBLE METHODS

### 7.1 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# OOB Score (if oob_score=True)
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_:.4f}")
```

### 7.2 AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 7.3 Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 7.4 Voting Classifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svc', SVC(probability=True))
    ],
    voting='soft'  # or 'hard'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print(f"Voting Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 7.5 Bagging

```python
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 8. NEURAL NETWORKS (DEEP LEARNING)

### 8.1 Simple Neural Network (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Classification
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')  # or 'sigmoid' for binary
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # or 'binary_crossentropy'
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ],
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### 8.2 Neural Network (PyTorch)

```python
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleNN(input_size=X_train.shape[1], hidden_size=64, num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    
    # Forward
    outputs = model(torch.tensor(X_train, dtype=torch.float32).to(device))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long).to(device))
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted.cpu().numpy() == y_test).mean()
    print(f'Test Accuracy: {accuracy:.4f}')
```

### 8.3 Weight Initialization Strategies

```python
# He Initialization (for ReLU)
model = Sequential([
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(10, activation='softmax')
])

# Glorot (Xavier) Initialization (for tanh/sigmoid)
model = Sequential([
    Dense(64, activation='tanh', kernel_initializer='glorot_uniform'),
    Dense(32, activation='tanh', kernel_initializer='glorot_uniform'),
    Dense(10, activation='softmax')
])
```

### 8.4 Different Optimizers

```python
# SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# RMSprop
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 9. CNNs (CONVOLUTIONAL NEURAL NETWORKS)

### 9.1 Simple CNN (Keras)

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {acc:.4f}')
```

### 9.2 CNN with Data Augmentation (CIFAR-10)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)
```

### 9.3 CNN (PyTorch)

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')
```

### 9.4 AlexNet Implementation (PyTorch)

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.eLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### 9.5 Transfer Learning (PyTorch)

```python
import torchvision.models as models

# Load pretrained model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes

model = model.to(device)

# Only train the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

---

## 10. RNNs (RECURRENT NEURAL NETWORKS)

### 10.1 Simple RNN from Scratch

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, h_prev):
        """Forward pass through time"""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        for t in range(len(inputs)):
            xs[t] = inputs[t].reshape(-1, 1)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        return xs, hs, ps
    
    def backward(self, xs, hs, ps, targets):
        """Backpropagation through time"""
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # Clip gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # Update weights
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby
```

### 10.2 LSTM (Keras)

```python
from tensorflow.keras.layers import LSTM

# For sequence classification
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 10.3 GRU (Keras)

```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    GRU(32),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 10.4 LSTM (PyTorch)

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Usage
model = LSTMModel(input_size=28, hidden_size=128, num_layers=2, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 10.5 Bidirectional LSTM

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, features)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

### 10.6 Text Classification with RNN

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this movie", "This is terrible", "Great film", "Awful experience"]
labels = [1, 0, 1, 0]

# Tokenization
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)

# Model
model = Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, np.array(labels), epochs=10, batch_size=2)
```

---

## 11. MODEL EVALUATION METRICS

### 11.1 Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, classification_report, confusion_matrix)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.show()
```

### 11.2 Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

# MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
```

### 11.3 ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc

# For binary classification
y_pred_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### 11.4 Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# K-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Different scoring metrics
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    scores = cross_val_score(model, X, y, cv=5, scoring=metric)
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 11.5 Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 12. VISUALIZATION HELPERS

### 12.1 Plot Training History (Keras)

```python
def plot_history(history):
    """Plot training and validation loss/accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage
plot_history(history)
```

### 12.2 Visualize CNN Filters

```python
def visualize_filters(model, layer_idx):
    """Visualize convolutional filters"""
    layer = model.layers[layer_idx]
    filters, biases = layer.get_weights()
    
    # Normalize filters
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    n_filters = filters.shape[3]
    n_cols = 8
    n_rows = n_filters // n_cols + (1 if n_filters % n_cols else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    
    for i in range(n_filters):
        f = filters[:, :, :, i]
        if f.shape[2] == 1:
            axes[i].imshow(f[:, :, 0], cmap='gray')
        else:
            axes[i].imshow(f)
        axes[i].axis('off')
    
    plt.suptitle(f'Filters of Layer {layer_idx}')
    plt.tight_layout()
    plt.show()
```

### 12.3 Visualize Feature Maps

```python
def visualize_feature_maps(model, layer_names, img):
    """Visualize intermediate activations"""
    from tensorflow.keras.models import Model
    
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(img.reshape(1, *img.shape))
    
    for layer_name, activation in zip(layer_names, activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        
        n_cols = 8
        n_rows = n_features // n_cols + (1 if n_features % n_cols else 0)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].imshow(activation[0, :, :, i], cmap='viridis')
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps: {layer_name}')
        plt.tight_layout()
        plt.show()
```

### 12.4 Plot Decision Boundary (2D)

```python
def plot_decision_boundary(model, X, y, resolution=0.02):
    """Plot decision boundary for 2D data"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()
```

### 12.5 PCA Visualization

```python
from sklearn.decomposition import PCA

def plot_pca(X, y, n_components=2):
    """Visualize data using PCA"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization')
    plt.colorbar(scatter, label='Class')
    plt.show()
    
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 13. COMMON TROUBLESHOOTING

### 13.1 Check for NaN/Inf Values

```python
# Check for NaN
print(f"NaN in X_train: {np.isnan(X_train).any()}")
print(f"NaN in y_train: {np.isnan(y_train).any()}")

# Check for Inf
print(f"Inf in X_train: {np.isinf(X_train).any()}")

# Replace NaN with mean
X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train))
```

### 13.2 Fix Imbalanced Classes

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Use in model training
model.fit(X_train, y_train, class_weight=class_weight_dict)
```

### 13.3 Gradient Clipping (PyTorch)

```python
# Clip gradients to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 13.4 Learning Rate Scheduling

```python
# Keras
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

model.fit(X_train, y_train, callbacks=[lr_scheduler])

# PyTorch
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

### 13.5 Save and Load Models

```python
# Keras
model.save('my_model.h5')
loaded_model = keras.models.load_model('my_model.h5')

# PyTorch
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Scikit-learn
import joblib
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')
```

---

## 14. HYPERPARAMETER TUNING

### 14.1 Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

### 14.2 Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

---

## 15. QUICK REFERENCE FORMULAS

### 15.1 Common Loss Functions

```python
# Mean Squared Error (MSE) - Regression
mse = np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error (MAE) - Regression
mae = np.mean(np.abs(y_true - y_pred))

# Binary Cross-Entropy - Binary Classification
bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross-Entropy - Multi-class Classification
cce = -np.sum(y_true * np.log(y_pred))
```

### 15.2 Activation Functions

```python
# ReLU
def relu(x):
    return np.maximum(0, x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)
```

### 15.3 Gradient Descent Update

```python
# Vanilla SGD
weights = weights - learning_rate * gradients

# SGD with Momentum
velocity = momentum * velocity - learning_rate * gradients
weights = weights + velocity

# Adam
m = beta1 * m + (1 - beta1) * gradients
v = beta2 * v + (1 - beta2) * (gradients ** 2)
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

---

## 16. COMPLETE PIPELINE TEMPLATE

```python
# Complete ML Pipeline Template

# 1. Load Data
X, y = load_data()

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 5. Train Model
model.fit(X_train_scaled, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# 7. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 8. Cross-Validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 9. Hyperparameter Tuning (optional)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# 10. Save Model
joblib.dump(best_model, 'final_model.pkl')
```

---

## 17. CHEAT CODES FOR QUICK FIXES

### 17.1 Model Not Learning

```python
# Check learning rate
# Too high: oscillating loss
# Too low: very slow convergence

# Try different learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

# Check if data is scaled
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check if labels are correct
print(np.unique(y_train))
```

### 17.2 Overfitting

```python
# Add regularization
model = Ridge(alpha=10.0)  # for regression
model = LogisticRegression(C=0.1)  # for classification (smaller C = more regularization)

# Add dropout (neural networks)
model.add(Dropout(0.5))

# Reduce model complexity
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)

# Use more data augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)

# Early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
```

### 17.3 Underfitting

```python
# Increase model complexity
tree = DecisionTreeClassifier(max_depth=20)

# Add more features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Train longer
model.fit(X, y, epochs=100)

# Reduce regularization
model = Ridge(alpha=0.01)
```

### 17.4 Class Imbalance

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Use class weights
model.fit(X_train, y_train, class_weight='balanced')

# In neural networks
class_weight_dict = {0: 1.0, 1: 5.0}  # give more weight to minority class
model.fit(X_train, y_train, class_weight=class_weight_dict)
```

---

## 18. EXAM DAY QUICK START

### Step 1: Understand the Problem
```python
# Read the question carefully
# Identify: Classification or Regression?
# Check: How many classes? Binary or Multi-class?
# Note: What's the evaluation metric? (Accuracy, MSE, F1, etc.)
```

### Step 2: Load and Explore Data
```python
# Load data
df = pd.read_csv('data.csv')  # or use sklearn datasets

# Quick exploration
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['target'].value_counts())
```

### Step 3: Preprocess
```python
# Handle missing values
df.fillna(df.mean(), inplace=True)

# Split features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale if needed
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 4: Choose and Train Model
```python
# For Classification: RandomForest is usually a safe bet
model = RandomForestClassifier(