"""
═══════════════════════════════════════════════════════════════
    ULTIMATE ML LAB EXAM GUIDE - ALL CODE SNIPPETS
═══════════════════════════════════════════════════════════════
Complete reference with all algorithms, implementations, and utilities
Author: Comprehensive ML Guide
Last Updated: 2025
"""

# ═══════════════════════════════════════════════════════════════
# 1. ESSENTIAL IMPORTS
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                              mean_squared_error, r2_score, roc_curve, roc_auc_score,
                              mean_absolute_error, ConfusionMatrixDisplay)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                               AdaBoostClassifier, GradientBoostingClassifier,
                               GradientBoostingRegressor, BaggingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import (load_iris, load_breast_cancer, load_digits, 
                               load_wine, make_classification, make_regression, make_moons)
import warnings
warnings.filterwarnings('ignore')

# For Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# ═══════════════════════════════════════════════════════════════
# 2. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════

# --- Load Common Datasets ---
def load_dataset(dataset_name='iris'):
    """Load common ML datasets"""
    datasets = {
        'iris': load_iris(),
        'breast_cancer': load_breast_cancer(),
        'digits': load_digits(),
        'wine': load_wine(),
        'titanic': sns.load_dataset('titanic')
    }
    return datasets.get(dataset_name)

# --- Load Titanic Dataset (for exam) ---
def load_titanic_clean():
    """Load and clean titanic dataset"""
    df = sns.load_dataset('titanic')[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
    return df

# --- Handle Missing Values ---
def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values
    strategy: 'drop', 'mean', 'median', 'mode', 'forward_fill'
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'forward_fill':
        return df.fillna(method='ffill')
    return df

# --- Label Encoding ---
def encode_categorical(df, columns):
    """Label encode categorical columns"""
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# --- One-Hot Encoding ---
def one_hot_encode(df, columns):
    """One-hot encode categorical columns"""
    return pd.get_dummies(df, columns=columns, drop_first=True)

# --- Feature Scaling ---
def scale_features(X_train, X_test, method='standard'):
    """Scale features using StandardScaler or MinMaxScaler"""
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# ═══════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════

def quick_eda(df):
    """Perform quick EDA on dataframe"""
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print(f"\nFirst 5 Rows:\n{df.head()}")
    
def plot_distributions(df, numerical_cols, categorical_cols):
    """Plot distributions of features"""
    # Numerical distributions
    if numerical_cols:
        fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(12, 4*len(numerical_cols)))
        for i, col in enumerate(numerical_cols):
            axes[i, 0].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[i, 0].set_title(f'{col} - Histogram')
            axes[i, 1].boxplot(df[col].dropna())
            axes[i, 1].set_title(f'{col} - Boxplot')
        plt.tight_layout()
        plt.show()
    
    # Categorical distributions
    if categorical_cols:
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(5*len(categorical_cols), 4))
        if len(categorical_cols) == 1:
            axes = [axes]
        for i, col in enumerate(categorical_cols):
            df[col].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{col} - Distribution')
        plt.tight_layout()
        plt.show()

def correlation_heatmap(df):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 4. LINEAR & POLYNOMIAL REGRESSION
# ═══════════════════════════════════════════════════════════════

# --- Linear Regression from Scratch ---
class ScratchLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# --- Using scikit-learn ---
def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate linear regression"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    
    return model

# --- Polynomial Regression ---
def train_polynomial_regression(X_train, X_test, y_train, y_test, degree=2):
    """Train polynomial regression"""
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    
    print(f"Polynomial Degree: {degree}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    
    return model, poly

# --- Closed Form Solution (Normal Equation) ---
def normal_equation(X, y):
    """Solve linear regression using normal equation: θ = (X^T X)^-1 X^T y"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

# ═══════════════════════════════════════════════════════════════
# 5. RIDGE & LASSO REGRESSION
# ═══════════════════════════════════════════════════════════════

def train_ridge_lasso(X_train, X_test, y_train, y_test, alpha=1.0):
    """Train and compare Ridge and Lasso regression"""
    
    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    # Lasso Regression
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    
    print(f"Ridge R²: {r2_score(y_test, y_pred_ridge):.4f}")
    print(f"Ridge MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}")
    print(f"Lasso R²: {r2_score(y_test, y_pred_lasso):.4f}")
    print(f"Lasso MSE: {mean_squared_error(y_test, y_pred_lasso):.4f}")
    
    # Compare coefficients
    print(f"\nRidge coefficients: {ridge.coef_}")
    print(f"Lasso coefficients: {lasso.coef_}")
    print(f"Number of zero coefficients in Lasso: {np.sum(lasso.coef_ == 0)}")
    
    return ridge, lasso

# --- Ridge Regression from Scratch ---
def ridge_closed_form(X, y, alpha=1.0):
    """Ridge regression using closed form: θ = (X^T X + αI)^-1 X^T y"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    identity = np.eye(X_b.shape[1])
    identity[0, 0] = 0  # Don't regularize bias
    theta = np.linalg.inv(X_b.T.dot(X_b) + alpha * identity).dot(X_b.T).dot(y)
    return theta

# --- Coefficient Path Visualization ---
def plot_regularization_path(X, y, alphas=None):
    """Plot coefficient paths for Ridge and Lasso"""
    if alphas is None:
        alphas = np.logspace(-3, 2, 50)
    
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(X, y)
        ridge_coefs.append(ridge.coef_)
        lasso = Lasso(alpha=alpha, max_iter=5000).fit(X, y)
        lasso_coefs.append(lasso.coef_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(alphas, ridge_coefs)
    ax1.set_xscale('log')
    ax1.set_title('Ridge: Coefficient Paths')
    ax1.set_xlabel('Alpha (log scale)')
    ax1.set_ylabel('Coefficient value')
    
    ax2.plot(alphas, lasso_coefs)
    ax2.set_xscale('log')
    ax2.set_title('Lasso: Coefficient Paths')
    ax2.set_xlabel('Alpha (log scale)')
    
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 6. DECISION TREES
# ═══════════════════════════════════════════════════════════════

def train_decision_tree_classifier(X_train, X_test, y_train, y_test, max_depth=None):
    """Train decision tree classifier"""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Max Depth: {max_depth}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return model

def train_decision_tree_regressor(X_train, X_test, y_train, y_test, max_depth=None):
    """Train decision tree regressor"""
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Max Depth: {max_depth}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    
    return model

def plot_decision_tree(model, feature_names=None, class_names=None):
    """Visualize decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, 
              filled=True, rounded=True, fontsize=10)
    plt.show()

def plot_feature_importances(model, feature_names):
    """Plot feature importances from tree-based model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 7. ENSEMBLE METHODS
# ═══════════════════════════════════════════════════════════════

# --- Random Forest ---
def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, task='classification'):
    """Train Random Forest"""
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    
    return model

# --- AdaBoost ---
def train_adaboost(X_train, X_test, y_train, y_test, n_estimators=50):
    """Train AdaBoost classifier"""
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    
    return model

# --- Gradient Boosting ---
def train_gradient_boosting(X_train, X_test, y_train, y_test, n_estimators=100, task='classification'):
    """Train Gradient Boosting"""
    if task == 'classification':
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    else:
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    return model

# --- Voting Classifier ---
def train_voting_classifier(X_train, X_test, y_train, y_test):
    """Train voting classifier with multiple models"""
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = KNeighborsClassifier()
    
    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('dt', clf2), ('knn', clf3)],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    print(f"Voting Classifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    return voting_clf

# ═══════════════════════════════════════════════════════════════
# 8. MODEL EVALUATION & METRICS
# ═══════════════════════════════════════════════════════════════

def evaluate_classifier(y_test, y_pred, y_pred_proba=None):
    """Complete evaluation of classifier"""
    print("="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve (for binary classification)
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

def evaluate_regressor(y_test, y_pred):
    """Complete evaluation of regressor"""
    print("="*60)
    print("REGRESSION METRICS")
    print("="*60)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    
    plt.tight_layout()
    plt.show()

# --- Cross-Validation ---
def perform_cross_validation(model, X, y, cv=5):
    """Perform k-fold cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores

# --- Learning Curves ---
def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves to diagnose bias/variance"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid()
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 9. NEURAL NETWORKS - PYTORCH
# ═══════════════════════════════════════════════════════════════

# --- Simple Linear Model (PyTorch) ---
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# --- Deep Neural Network (PyTorch) ---
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# --- Training Loop (PyTorch) ---
def train_pytorch_model(model, train_loader, criterion, optimizer, epochs=10, device='cpu'):
    """Training loop for PyTorch model"""
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

# --- Evaluation (PyTorch) ---
def evaluate_pytorch_model(model, test_loader, device='cpu'):
    """Evaluate PyTorch model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# ═══════════════════════════════════════════════════════════════
# 10. CONVOLUTIONAL NEURAL NETWORKS (CNN)
# ═══════════════════════════════════════════════════════════════

# --- Simple CNN (PyTorch) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- AlexNet Implementation (PyTorch) ---
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
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
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- CNN with Keras/TensorFlow ---
def build_keras_cnn(input_shape=(28, 28, 1), num_classes=10):
    """Build simple CNN with Keras"""
    model = models.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ═══════════════════════════════════════════════════════════════
# 11. RECURRENT NEURAL NETWORKS (RNN, LSTM, GRU)
# ═══════════════════════════════════════════════════════════════

# --- Simple RNN from Scratch (NumPy) ---
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, x, h_prev):
        """Forward pass"""
        self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        y = np.dot(self.Why, self.h) + self.by
        return y, self.h

# --- LSTM (PyTorch) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- GRU (PyTorch) ---
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# --- Bidirectional LSTM ---
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ═══════════════════════════════════════════════════════════════
# 12. CONCEPT LEARNING (Find-S & Candidate-Elimination)
# ═══════════════════════════════════════════════════════════════

def find_s_algorithm(data):
    """
    Find-S Algorithm: Finds most specific hypothesis
    data: DataFrame with features and target (last column)
    """
    features = np.array(data.iloc[:, :-1])
    target = np.array(data.iloc[:, -1])
    
    # Initialize with first positive example
    for i, val in enumerate(target):
        if val == 'Yes':
            specific_h = features[i].copy()
            break
    
    print(f"Initial hypothesis: {specific_h}\n")
    
    # Iterate through training examples
    for i, instance in enumerate(features):
        if target[i] == 'Yes':
            print(f"Processing positive instance {i+1}: {instance}")
            for j in range(len(specific_h)):
                if instance[j] != specific_h[j]:
                    specific_h[j] = '?'
            print(f"Updated hypothesis: {specific_h}\n")
        else:
            print(f"Ignoring negative instance {i+1}: {instance}\n")
    
    return specific_h

def candidate_elimination(data):
    """
    Candidate-Elimination Algorithm
    Returns specific and general boundary
    """
    features = np.array(data.iloc[:, :-1])
    target = np.array(data.iloc[:, -1])
    num_attributes = len(features[0])
    
    # Initialize boundaries
    specific_h = ['0'] * num_attributes
    general_h = [['?'] * num_attributes]
    
    print(f"Initial S: {specific_h}")
    print(f"Initial G: {general_h}\n")
    
    for i, instance in enumerate(features):
        print(f"--- Instance {i+1}: {instance} ({target[i]}) ---")
        
        if target[i] == "Yes":  # Positive example
            # Remove inconsistent hypotheses from G
            general_h = [g for g in general_h 
                        if all(g[j] == '?' or g[j] == instance[j] 
                              for j in range(num_attributes))]
            
            # Generalize S
            for j in range(num_attributes):
                if specific_h[j] == '0':
                    specific_h[j] = instance[j]
                elif specific_h[j] != instance[j]:
                    specific_h[j] = '?'
        
        else:  # Negative example
            # Specialize G
            new_general_h = []
            for g in general_h:
                if all(g[j] == '?' or g[j] == instance[j] 
                      for j in range(num_attributes)):
                    for j in range(num_attributes):
                        if g[j] == '?':
                            for val in np.unique(features[:, j]):
                                if val != instance[j]:
                                    temp_h = g[:j] + [val] + g[j+1:]
                                    new_general_h.append(temp_h)
                else:
                    new_general_h.append(g)
            general_h = new_general_h
        
        print(f"S[{i+1}]: {specific_h}")
        print(f"G[{i+1}]: {general_h}\n")
    
    return specific_h, general_h

# ═══════════════════════════════════════════════════════════════
# 13. OPTIMIZATION & INITIALIZATION
# ═══════════════════════════════════════════════════════════════

# --- SGD from Scratch ---
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, params, grads):
        return params - self.lr * grads

# --- SGD with Momentum ---
class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v

# --- Adam Optimizer ---
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# --- Weight Initialization ---
def initialize_weights(shape, method='xavier'):
    """
    Initialize weights with different methods
    method: 'xavier', 'he', 'normal', 'zeros'
    """
    if method == 'xavier':
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)
    elif method == 'he':
        # He initialization
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    elif method == 'normal':
        return np.random.randn(*shape) * 0.01
    elif method == 'zeros':
        return np.zeros(shape)
    else:
        return np.random.randn(*shape)

# ═══════════════════════════════════════════════════════════════
# 14. COMMON UTILITIES & HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# --- Train-Test Split Helper ---
def split_data(X, y, test_size=0.2, random_state=42):
    """Quick train-test split"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# --- Stratified Split for Imbalanced Data ---
def stratified_split(X, y, test_size=0.2, random_state=42):
    """Stratified split to maintain class distribution"""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# --- Grid Search Helper ---
def perform_grid_search(model, param_grid, X_train, y_train, cv=5):
    """Perform hyperparameter tuning with grid search"""
    grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_

# --- Save and Load Models ---
import joblib

def save_model(model, filename):
    """Save trained model"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load saved model"""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# --- Bias-Variance Computation ---
def compute_bias_variance(model, X, y, n_bootstraps=50):
    """Compute bias and variance of a model using bootstrap"""
    from sklearn.model_selection import ShuffleSplit
    
    shuffle_split = ShuffleSplit(test_size=0.33, n_splits=n_bootstraps, random_state=0)
    predictions = [[] for _ in range(len(y))]
    
    for train_idx, test_idx in shuffle_split.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        
        for j, idx in enumerate(test_idx):
            predictions[idx].append(y_pred[j])
    
    # Compute bias and variance
    bias_sq = sum([(1 - pred.count(y[i])/len(pred))**2 * len(pred)/n_bootstraps 
                   for i, pred in enumerate(predictions) if len(pred) > 0])
    
    var = sum([((1 - ((pred.count(0)/len(pred))**2 + 
                      (pred.count(1)/len(pred))**2))/2) * len(pred)/n_bootstraps
               for pred in predictions if len(pred) > 0])
    
    return np.sqrt(bias_sq), var

# --- Plot Decision Boundaries (2D) ---
def plot_decision_boundary(model, X, y, h=0.02):
    """Plot decision boundary for 2D data"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# --- Quick Model Comparison ---
def compare_models(models_dict, X_train, X_test, y_train, y_test):
    """Compare multiple models"""
    results = {}
    
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return results

# ═══════════════════════════════════════════════════════════════
# 15. COMPLETE PIPELINE EXAMPLES
# ═══════════════════════════════════════════════════════════════

# --- Example 1: Titanic Classification Pipeline ---
def titanic_pipeline():
    """Complete pipeline for Titanic dataset"""
    # Load data
    df = load_titanic_clean()
    
    # EDA
    quick_eda(df)
    
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df.dropna(inplace=True)
    
    # Encode categorical
    df = encode_categorical(df, ['sex', 'embarked'])
    
    # Split features and target
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = compare_models(models, X_train_scaled, X_test_scaled, y_train, y_test)
    
    return results

# --- Example 2: MNIST with Neural Network ---
def mnist_nn_pipeline():
    """Complete pipeline for MNIST with neural network"""
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build model
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train
    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=128, 
                        validation_split=0.1,
                        verbose=1)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    return model, history

# --- Example 3: Regression Pipeline ---
def regression_pipeline():
    """Complete regression pipeline"""
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    # Train models
    print("="*60)
    print("LINEAR REGRESSION")
    print("="*60)
    lin_model = train_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\n" + "="*60)
    print("RIDGE REGRESSION")
    print("="*60)
    ridge_model, lasso_model = train_ridge_lasso(X_train_scaled, X_test_scaled, 
                                                   y_train, y_test, alpha=1.0)
    
    print("\n" + "="*60)
    print("RANDOM FOREST REGRESSION")
    print("="*60)
    rf_model = train_random_forest(X_train_scaled, X_test_scaled, 
                                    y_train, y_test, task='regression')
    
    return lin_model, ridge_model, rf_model

# ═══════════════════════════════════════════════════════════════
# 16. QUICK REFERENCE FORMULAS
# ═══════════════════════════════════════════════════════════════

"""
KEY FORMULAS TO REMEMBER:

1. Linear Regression:
   - Normal Equation: θ = (X^T X)^-1 X^T y
   - Gradient: ∇θ = (1/m) X^T (Xθ - y)
   - Cost: J(θ) = (1/2m) Σ(hθ(x) - y)²

2. Ridge Regression (L2):
   - Cost: J(θ) = (1/2m) Σ(hθ(x) - y)² + λ Σθ²
   - Closed form: θ = (X^T X + λI)^-1 X^T y

3. Lasso Regression (L1):
   - Cost: J(θ) = (1/2m) Σ(hθ(x) - y)² + λ Σ|θ|

4. Logistic Regression:
   - Sigmoid: σ(z) = 1 / (1 + e^-z)
   - Cost: J(θ) = -(1/m) Σ[y log(hθ(x)) + (1-y) log(1-hθ(x))]

5. Neural Network:
   - Forward: a = σ(Wx + b)
   - Backward: δ = (a - y) * σ'(z)

6. Evaluation Metrics:
   - Accuracy = (TP + TN) / (TP + TN + FP + FN)
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
   - MSE = (1/n) Σ(y - ŷ)²
   - R² = 1 - (SS_res / SS_tot)

7. Gradient Descent:
   - Update: θ = θ - α ∇J(θ)
   - With momentum: v = βv + α∇J(θ); θ = θ - v
   - Adam: combines momentum + RMSProp

8. Convolution Output Size:
   - out = floor((in + 2*padding - kernel_size) / stride) + 1
"""

# ═══════════════════════════════════════════════════════════════
# END OF ULTIMATE ML GUIDE
# ═══════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          ML LAB EXAM ULTIMATE GUIDE LOADED                   ║
║                                                              ║
║  All functions, algorithms, and utilities are ready to use! ║
║                                                              ║
║  Quick Start:                                                ║
║  - Load dataset: load_titanic_clean()                        ║
║  - Quick EDA: quick_eda(df)                                  ║
║  - Train models: Use train_* functions                       ║
║  - Evaluate: Use evaluate_* functions                        ║
║                                                              ║
║  Good luck on your exam! 🚀                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")