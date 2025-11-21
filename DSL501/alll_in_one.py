# # 1. Introduction
# 
# It’s important to understand prediction errors (bias and variance) wh. There is a tradeoff between a model’s ability to minimize bias and variance. Gaining a proper understanding of these errors would help us not only to build accurate models but also to avoid the mistake of overfitting and underfitting. If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it's going to have high variance and low bias.
# 

# You can find this notebook on this [link](https://colab.research.google.com/drive/188aN4MYDokbppYTAhywQ2EGVDyCgXo67?usp=sharing)
# 
# the textual information for bias and variance is taken from [here](https://www.kaggle.com/code/azminetoushikwasi/mastering-bias-variance-tradeoff/notebook#11.1.-Calculate-Bias)
# 
# 


# # 2. Understanding Bias
# ### What is **Bias**?
# > Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data. - *Seema Singh*
# 
# ### Bias Definition in Statistics
# In statistics, bias is a term which defines the tendency of the measurement process. It means that it evaluates the over or underestimation of the value of the population parameter. Let us consider an example, in case you have the rule to evaluate the mean of the population. Hopefully, you might have found an estimation using the rule, which is the true reflection of the population. Now, by using the biased estimator, it is easy to find the difference between the true value and the statistically expected value of the population parameter.
# 
# ### Why it is so important in Data Science?
# Due to society's culture and history, historical data might be discriminatory against certain minority groups. Cognitive biases are systematic errors in thinking, usually inherited by cultural and personal experiences, that lead to distortions of perceptions when making decisions. And while data might seem objective, data is collected and analyzed by humans, and thus can be biased. Because of this, it's highly important to check assumptions over the data to avoid future algorithmic bias.
# 
# ##### Sometimes it can be helpful too!
# ***The idea of having bias was about model giving importance to some of the features in order to generalize better for the larger dataset with various other attributes. Bias in ML does help us generalize better and make our model less sensitive to some single data point.***
# 
# *Information : Wikipedia, bmc.com, Investopedia, towardsdatascience.com*
# 
# 


# # 3. Understanding Variance
# ### What is "Variance"
# 
# Variance is a measurement of the spread between numbers in a data set. In probability theory and statistics, variance is the expectation of the squared deviation of a random variable from its population mean or sample mean. Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers is spread out from their average value.
# 
# 
# ### Variance in terms of DS-ML
# > Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data. - *Seema Singh*
# 
# Also, variance refers to the changes in the model when using different portions of the training data set. Simply stated, variance is the variability in the model prediction—how much the ML function can adjust depending on the given data set.
# 
# 
# ## Measuring Variance
# In statistics, variance measures variability from the average or mean. It is calculated by taking the differences between each number in the data set and the mean, then squaring the differences to make them positive, and finally dividing the sum of the squares by the number of values in the data set. [Learn More](https://en.wikipedia.org/wiki/Variance)
# 
# Bias and variance are used in supervised machine learning, in which an algorithm learns from training data or a sample data set of known quantities. The correct balance of bias and variance is vital to building machine-learning algorithms that create accurate results from their models.
# *Information : Wikipedia, bmc.com, Investopedia, towardsdatascience.com*
# 
# 
# 


# # 6. Bias and variance using bulls-eye diagram


# In the above diagram, center of the target is a model that perfectly predicts correct values. As we move away from the bulls-eye our predictions become get worse and worse. We can repeat our process of model building to get separate hits on the target.
# 
# ### underfitting
# **In supervised learning, underfitting happens when a model unable to capture the underlying pattern of the data. These models usually have high bias and low variance. It happens when we have very less amount of data to build an accurate model or when we try to build a linear model with a nonlinear data. Also, these kind of models are very simple to capture the complex patterns in data like Linear and logistic regression.**
# 
# ### overfitting
# **In supervised learning, overfitting happens when our model captures the noise along with the underlying pattern in data. It happens when we train our model a lot over noisy dataset. These models have low bias and high variance. These models are very complex like Decision trees which are prone to overfitting.**
# 
# 
# *credit - GeekforGeeks*

# 
# ## Why Bias Variance Tradeoff?
# 
# If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.
# 
# This tradeoff in complexity is why there is a tradeoff between bias and variance. An algorithm can’t be more complex and less complex at the same time.
# 

# # Step 1: Imports and Setup

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(style="whitegrid")
np.random.seed(42)


# # Step 2: Generate Data (Ground Truth is Quadratic)

def true_function(x):
    return np.sin(1.5*x)

X = np.sort(np.random.rand(100) * 5)
y = true_function(X) + np.random.randn(100) * 0.3  # Add noise

X = X.reshape(-1, 1)

plt.scatter(X, y, color="blue", label="Noisy data")
x_plot = np.linspace(0, 5, 100).reshape(-1, 1)
plt.plot(x_plot, true_function(x_plot), color="green", label="True function")
plt.title("Ground Truth vs Noisy Observations")
plt.legend()
plt.show()


# # Step 3: Fit Different Degree Polynomial Models

degrees = [1, 4, 15]
colors = ["r", "g", "b"]

plt.figure(figsize=(15, 4))

for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X, y)

    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color="lightgray", label="Training data")
    plt.plot(x_plot, true_function(x_plot), "g--", label="True function")
    plt.plot(x_plot, model.predict(x_plot), color=colors[i], label=f"Degree {d}")
    plt.title(f"Model Degree {d}")
    plt.legend()

plt.tight_layout()
plt.show()


# Degree 1: High bias
# 
# Degree 4: Just right
# 
# Degree 15: High variance

# # Step 4: Plot Training vs Test Error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_errors = []
test_errors = []
degrees = range(1, 21)

for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.plot(degrees, train_errors, label="Training Error")
plt.plot(degrees, test_errors, label="Test Error")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("MSE")
plt.title("complexity vs error")
plt.legend()
plt.show()


from ipywidgets import interact

@interact(degree=(1, 20))
def interactive_model(degree=1):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(x_plot)

    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, color="lightgray")
    plt.plot(x_plot, true_function(x_plot), "g--", label="True function")
    plt.plot(x_plot, y_pred, "r-", label=f"Degree {degree}")
    plt.title(f"Interactive Polynomial Model (Degree={degree})")
    plt.legend()
    plt.show()


def get_bias(predicted_values, true_values):
    return np.round(np.mean((predicted_values - true_values) ** 2), 4)

def get_variance(values):
    return np.round(np.var(values), 4)

def get_metrics(target_train, target_test, model_train_predictions, model_test_predictions):
    training_mse = mean_squared_error(target_train, model_train_predictions)
    test_mse = mean_squared_error(target_test, model_test_predictions)
    bias = get_bias(model_test_predictions, target_test)
    variance = get_variance(model_test_predictions)

    return [training_mse, test_mse, bias, variance]


degrees = range(1, 16)
metrics = {
    'degree': [],
    'train_mse': [],
    'test_mse': [],
    'bias': [],
    'variance': []
}

for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_mse, test_mse, bias, variance = get_metrics(y_train, y_test, pred_train, pred_test)

    metrics['degree'].append(d)
    metrics['train_mse'].append(train_mse)
    metrics['test_mse'].append(test_mse)
    metrics['bias'].append(bias)
    metrics['variance'].append(variance)


plt.figure(figsize=(12, 6))

plt.plot(metrics['degree'], metrics['bias'], label="Bias²", marker='o')
plt.plot(metrics['degree'], metrics['variance'], label="Variance", marker='o')

plt.xlabel("Polynomial Degree (Model Complexity)")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ## interpreting the bias-variance plot

# 🟦 Bias²
# * Starts high at degree 1 (linear → poor approximation of sine)
# 
# * Drops sharply at degree 2 or 3 (captures curvature well)
# 
# * Stays low for all higher degrees → because polynomials can easily approximate sine on a short range
# 
# ✅ Interpretation:
# 
# Your model is able to approximate the function very well across degrees >2.
# 
# Hence, bias stays low — the model is expressive enough.
# 
# 🟧 Variance
# * Low for degree 1–2 → predictions don’t change much across samples
# 
# * Rises sharply from degree 3 → model starts overfitting noise
# 
# * Keeps increasing or fluctuates at high values (degree ≥10) → unstable fits, high sensitivity to input
# 
# ✅ Interpretation:
# 
# As model complexity increases, variance increases due to overfitting the training data.
# 
# This is typical: flexible models fit noise and differ a lot run-to-run.

# "Even a simple model like a degree-2 polynomial can approximate a function well if the function is smooth and the domain is limited."

# # VC Dimensions

# ## VC Dimension and VC Lines
# 
# - The **VC dimension** of a model is the largest number of points it can "shatter."
# - To **shatter** a set of points means: the model can correctly classify **every possible labeling** of those points.
# - For example:
#   - A straight line in 2D has VC dimension **3**
#   - It can shatter any arrangement of **3 points**
#   - It cannot shatter **some** arrangements of 4 points (e.g., XOR-type configuration)
# 
# This gives a formal way to understand model **capacity** and **overfitting risk**.
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create 10 2D points and labels
np.random.seed(42)

class1 = np.random.rand(5, 2) * 2   # spread around [0, 2]
class2 = np.random.rand(5, 2) * 2 + 2  # spread around [2, 4]

X = np.vstack((class1, class2))
# y = np.array([0]*5+[1]*5)
y = np.array([0]*4 + [1]*1+[0]*1 + [1]*4)

# Train a linear classifier (Logistic Regression)
clf = LogisticRegression(solver='liblinear')
clf.fit(X, y)
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(acc)

def plot_decision_boundary(X, y, model, title="VC Lines on 10 Points"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100, edgecolors='k')
    for i in range(len(X)):
        plt.text(X[i, 0] + 0.05, X[i, 1], f"{i}", fontsize=9)

    plt.title(f"{title}\nAccuracy: {acc*100:.1f}%")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, clf)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def generate_data(n_samples=200, n_features=2, seed=42):
    """
    Generate binary classification data with linear separability.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    logits = X @ true_weights
    y = (logits > 0).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


dimensions = [2, 3, 6, 10, 20, 50, 100]
train_accs = []
test_accs = []
generalization_gaps = []

for d in dimensions:
    X_train, X_test, y_train, y_test = generate_data(n_samples=200, n_features=d, seed=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    generalization_gaps.append(gap)

    print(f"Dim {d:>3}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}, Gap = {gap:.3f}")


plt.figure(figsize=(10, 6))
plt.plot(dimensions, train_accs, label='Train Accuracy', marker='o')
plt.plot(dimensions, test_accs, label='Test Accuracy', marker='s')
plt.plot(dimensions, generalization_gaps, label='Generalization Gap', linestyle='--', marker='x')
plt.xlabel('Number of Features (Dimensionality)')
plt.ylabel('Accuracy')
plt.title('VC Dimension Effect: Accuracy vs Dimensionality')
plt.legend()
plt.grid(True)
plt.show()





# # 1. Introduction
# 
# It’s important to understand prediction errors (bias and variance) wh. There is a tradeoff between a model’s ability to minimize bias and variance. Gaining a proper understanding of these errors would help us not only to build accurate models but also to avoid the mistake of overfitting and underfitting. If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it's going to have high variance and low bias.
# 

# You can find this notebook on this [link](https://colab.research.google.com/drive/188aN4MYDokbppYTAhywQ2EGVDyCgXo67?usp=sharing)
# 
# the textual information for bias and variance is taken from [here](https://www.kaggle.com/code/azminetoushikwasi/mastering-bias-variance-tradeoff/notebook#11.1.-Calculate-Bias)
# 
# 


# # 2. Understanding Bias
# ### What is **Bias**?
# > Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data. - *Seema Singh*
# 
# ### Bias Definition in Statistics
# In statistics, bias is a term which defines the tendency of the measurement process. It means that it evaluates the over or underestimation of the value of the population parameter. Let us consider an example, in case you have the rule to evaluate the mean of the population. Hopefully, you might have found an estimation using the rule, which is the true reflection of the population. Now, by using the biased estimator, it is easy to find the difference between the true value and the statistically expected value of the population parameter.
# 
# ### Why it is so important in Data Science?
# Due to society's culture and history, historical data might be discriminatory against certain minority groups. Cognitive biases are systematic errors in thinking, usually inherited by cultural and personal experiences, that lead to distortions of perceptions when making decisions. And while data might seem objective, data is collected and analyzed by humans, and thus can be biased. Because of this, it's highly important to check assumptions over the data to avoid future algorithmic bias.
# 
# ##### Sometimes it can be helpful too!
# ***The idea of having bias was about model giving importance to some of the features in order to generalize better for the larger dataset with various other attributes. Bias in ML does help us generalize better and make our model less sensitive to some single data point.***
# 
# *Information : Wikipedia, bmc.com, Investopedia, towardsdatascience.com*
# 
# 


# # 3. Understanding Variance
# ### What is "Variance"
# 
# Variance is a measurement of the spread between numbers in a data set. In probability theory and statistics, variance is the expectation of the squared deviation of a random variable from its population mean or sample mean. Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers is spread out from their average value.
# 
# 
# ### Variance in terms of DS-ML
# > Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data. - *Seema Singh*
# 
# Also, variance refers to the changes in the model when using different portions of the training data set. Simply stated, variance is the variability in the model prediction—how much the ML function can adjust depending on the given data set.
# 
# 
# ## Measuring Variance
# In statistics, variance measures variability from the average or mean. It is calculated by taking the differences between each number in the data set and the mean, then squaring the differences to make them positive, and finally dividing the sum of the squares by the number of values in the data set. [Learn More](https://en.wikipedia.org/wiki/Variance)
# 
# Bias and variance are used in supervised machine learning, in which an algorithm learns from training data or a sample data set of known quantities. The correct balance of bias and variance is vital to building machine-learning algorithms that create accurate results from their models.
# *Information : Wikipedia, bmc.com, Investopedia, towardsdatascience.com*
# 
# 
# 


# # 6. Bias and variance using bulls-eye diagram


# In the above diagram, center of the target is a model that perfectly predicts correct values. As we move away from the bulls-eye our predictions become get worse and worse. We can repeat our process of model building to get separate hits on the target.
# 
# ### underfitting
# **In supervised learning, underfitting happens when a model unable to capture the underlying pattern of the data. These models usually have high bias and low variance. It happens when we have very less amount of data to build an accurate model or when we try to build a linear model with a nonlinear data. Also, these kind of models are very simple to capture the complex patterns in data like Linear and logistic regression.**
# 
# ### overfitting
# **In supervised learning, overfitting happens when our model captures the noise along with the underlying pattern in data. It happens when we train our model a lot over noisy dataset. These models have low bias and high variance. These models are very complex like Decision trees which are prone to overfitting.**
# 
# 
# *credit - GeekforGeeks*

# 
# ## Why Bias Variance Tradeoff?
# 
# If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.
# 
# This tradeoff in complexity is why there is a tradeoff between bias and variance. An algorithm can’t be more complex and less complex at the same time.
# 

# # Step 1: Imports and Setup

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(style="whitegrid")
np.random.seed(42)


# # Step 2: Generate Data (Ground Truth is Quadratic)

def true_function(x):
    return np.sin(1.5*x)

X = np.sort(np.random.rand(100) * 5)
y = true_function(X) + np.random.randn(100) * 0.3  # Add noise

X = X.reshape(-1, 1)

plt.scatter(X, y, color="blue", label="Noisy data")
x_plot = np.linspace(0, 5, 100).reshape(-1, 1)
plt.plot(x_plot, true_function(x_plot), color="green", label="True function")
plt.title("Ground Truth vs Noisy Observations")
plt.legend()
plt.show()


# # Step 3: Fit Different Degree Polynomial Models

degrees = [1, 4, 15]
colors = ["r", "g", "b"]

plt.figure(figsize=(15, 4))

for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X, y)

    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color="lightgray", label="Training data")
    plt.plot(x_plot, true_function(x_plot), "g--", label="True function")
    plt.plot(x_plot, model.predict(x_plot), color=colors[i], label=f"Degree {d}")
    plt.title(f"Model Degree {d}")
    plt.legend()

plt.tight_layout()
plt.show()


# Degree 1: High bias
# 
# Degree 4: Just right
# 
# Degree 15: High variance

# # Step 4: Plot Training vs Test Error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_errors = []
test_errors = []
degrees = range(1, 21)

for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.plot(degrees, train_errors, label="Training Error")
plt.plot(degrees, test_errors, label="Test Error")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("MSE")
plt.title("complexity vs error")
plt.legend()
plt.show()


from ipywidgets import interact

@interact(degree=(1, 20))
def interactive_model(degree=1):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(x_plot)

    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, color="lightgray")
    plt.plot(x_plot, true_function(x_plot), "g--", label="True function")
    plt.plot(x_plot, y_pred, "r-", label=f"Degree {degree}")
    plt.title(f"Interactive Polynomial Model (Degree={degree})")
    plt.legend()
    plt.show()


def get_bias(predicted_values, true_values):
    return np.round(np.mean((predicted_values - true_values) ** 2), 4)

def get_variance(values):
    return np.round(np.var(values), 4)

def get_metrics(target_train, target_test, model_train_predictions, model_test_predictions):
    training_mse = mean_squared_error(target_train, model_train_predictions)
    test_mse = mean_squared_error(target_test, model_test_predictions)
    bias = get_bias(model_test_predictions, target_test)
    variance = get_variance(model_test_predictions)

    return [training_mse, test_mse, bias, variance]


degrees = range(1, 16)
metrics = {
    'degree': [],
    'train_mse': [],
    'test_mse': [],
    'bias': [],
    'variance': []
}

for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_mse, test_mse, bias, variance = get_metrics(y_train, y_test, pred_train, pred_test)

    metrics['degree'].append(d)
    metrics['train_mse'].append(train_mse)
    metrics['test_mse'].append(test_mse)
    metrics['bias'].append(bias)
    metrics['variance'].append(variance)


plt.figure(figsize=(12, 6))

plt.plot(metrics['degree'], metrics['bias'], label="Bias²", marker='o')
plt.plot(metrics['degree'], metrics['variance'], label="Variance", marker='o')

plt.xlabel("Polynomial Degree (Model Complexity)")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ## interpreting the bias-variance plot

# 🟦 Bias²
# * Starts high at degree 1 (linear → poor approximation of sine)
# 
# * Drops sharply at degree 2 or 3 (captures curvature well)
# 
# * Stays low for all higher degrees → because polynomials can easily approximate sine on a short range
# 
# ✅ Interpretation:
# 
# Your model is able to approximate the function very well across degrees >2.
# 
# Hence, bias stays low — the model is expressive enough.
# 
# 🟧 Variance
# * Low for degree 1–2 → predictions don’t change much across samples
# 
# * Rises sharply from degree 3 → model starts overfitting noise
# 
# * Keeps increasing or fluctuates at high values (degree ≥10) → unstable fits, high sensitivity to input
# 
# ✅ Interpretation:
# 
# As model complexity increases, variance increases due to overfitting the training data.
# 
# This is typical: flexible models fit noise and differ a lot run-to-run.

# "Even a simple model like a degree-2 polynomial can approximate a function well if the function is smooth and the domain is limited."

# # VC Dimensions

# ## VC Dimension and VC Lines
# 
# - The **VC dimension** of a model is the largest number of points it can "shatter."
# - To **shatter** a set of points means: the model can correctly classify **every possible labeling** of those points.
# - For example:
#   - A straight line in 2D has VC dimension **3**
#   - It can shatter any arrangement of **3 points**
#   - It cannot shatter **some** arrangements of 4 points (e.g., XOR-type configuration)
# 
# This gives a formal way to understand model **capacity** and **overfitting risk**.
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create 10 2D points and labels
np.random.seed(42)

class1 = np.random.rand(5, 2) * 2   # spread around [0, 2]
class2 = np.random.rand(5, 2) * 2 + 2  # spread around [2, 4]

X = np.vstack((class1, class2))
# y = np.array([0]*5+[1]*5)
y = np.array([0]*4 + [1]*1+[0]*1 + [1]*4)

# Train a linear classifier (Logistic Regression)
clf = LogisticRegression(solver='liblinear')
clf.fit(X, y)
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(acc)

def plot_decision_boundary(X, y, model, title="VC Lines on 10 Points"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100, edgecolors='k')
    for i in range(len(X)):
        plt.text(X[i, 0] + 0.05, X[i, 1], f"{i}", fontsize=9)

    plt.title(f"{title}\nAccuracy: {acc*100:.1f}%")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, clf)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def generate_data(n_samples=200, n_features=2, seed=42):
    """
    Generate binary classification data with linear separability.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    logits = X @ true_weights
    y = (logits > 0).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


dimensions = [2, 3, 6, 10, 20, 50, 100]
train_accs = []
test_accs = []
generalization_gaps = []

for d in dimensions:
    X_train, X_test, y_train, y_test = generate_data(n_samples=200, n_features=d, seed=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    generalization_gaps.append(gap)

    print(f"Dim {d:>3}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}, Gap = {gap:.3f}")


plt.figure(figsize=(10, 6))
plt.plot(dimensions, train_accs, label='Train Accuracy', marker='o')
plt.plot(dimensions, test_accs, label='Test Accuracy', marker='s')
plt.plot(dimensions, generalization_gaps, label='Generalization Gap', linestyle='--', marker='x')
plt.xlabel('Number of Features (Dimensionality)')
plt.ylabel('Accuracy')
plt.title('VC Dimension Effect: Accuracy vs Dimensionality')
plt.legend()
plt.grid(True)
plt.show()





# # A Comprehensive Tutorial on Scikit-Learn
# 
# 
# Welcome to this in-depth guide to Scikit-Learn, the most popular and powerful machine learning library for Python. This notebook will walk you through the essential concepts and functionalities of sklearn, from the basics of data handling to building, evaluating, and deploying machine learning models.

# ## 📘 1. Introduction to Scikit-Learn

# ### What is Scikit-Learn?
# 
# Scikit-learn (often stylized as `sklearn`) is an open-source Python library that provides a wide array of tools for machine learning tasks. It's built upon other fundamental scientific Python libraries like NumPy, SciPy, and Matplotlib.
# 
# The core philosophy of scikit-learn is to offer **simple and efficient tools for data mining and data analysis** that are **accessible to everybody, and reusable in various contexts**.
# 
# ### Features of Scikit-Learn
# 
# Scikit-learn provides a consistent and unified API for various machine learning tasks:
# 
# - **Classification**: Identifying which category an object belongs to (e.g., spam vs. not spam).
# - **Regression**: Predicting a continuous-valued attribute associated with an object (e.g., predicting house prices).
# - **Clustering**: Automatic grouping of similar objects into sets (e.g., customer segmentation).
# - **Dimensionality Reduction**: Reducing the number of random variables to consider (e.g., PCA, feature selection).
# - **Model Selection**: Comparing, validating, and choosing parameters and models (e.g., grid search, cross-validation).
# - **Preprocessing**: Feature extraction and normalization (e.g., scaling, encoding).

# ### Installation and Imports
# 
# You can install scikit-learn using pip. It's also included in the Anaconda distribution.
# 
# ```bash
# pip install -U scikit-learn
# ```
# 
# Let's start by importing the essential libraries we'll use throughout this tutorial.

# Standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent style for plots
sns.set_style("whitegrid")

print("Libraries imported successfully!")

# ### Dataset Structure in Scikit-Learn
# 
# Scikit-learn comes with several toy datasets. These datasets are stored in a `Bunch` object, which is similar to a dictionary. The most important attributes are:
# - `data`: The feature matrix, usually a NumPy array. By convention, this is named `X`.
# - `target`: The target vector (labels or values), usually a NumPy array. By convention, this is named `y`.
# - `feature_names`: The names of the columns in `X`.
# - `target_names`: The names of the classes in `y` (for classification).
# - `DESCR`: A full description of the dataset.

# Load the Iris dataset as an example
from sklearn.datasets import load_iris

iris = load_iris()

# Explore the Bunch object
print(f"Keys of the iris Bunch object: {iris.keys()}\n")

# Print the dataset description
# print(iris.DESCR)

# Assign features to X and target to y
X, y = iris.data, iris.target

print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}\n")

print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")

# ---

# ## 🧹 2. Data Preparation
# 
# Data preparation (or preprocessing) is a critical step in any machine learning pipeline. Scikit-learn provides excellent tools to get your data ready for modeling.

# ### Loading Built-in Datasets

from sklearn.datasets import load_digits, load_wine, load_breast_cancer

digits = load_digits()
wine = load_wine()
cancer = load_breast_cancer()

print(f"Digits dataset has {digits.data.shape[0]} samples and {digits.data.shape[1]} features.")
print(f"Wine dataset has {wine.data.shape[0]} samples and {wine.data.shape[1]} features.")
print(f"Breast Cancer dataset has {cancer.data.shape[0]} samples and {cancer.data.shape[1]} features.")

# ### Splitting Data: `train_test_split`
# 
# We must split our data into a training set and a testing set. The model learns from the training set and its performance is evaluated on the unseen testing set.
# 
# - `test_size`: The proportion of the dataset to allocate to the test split.
# - `random_state`: A seed for the random number generator to ensure reproducibility.
# - `stratify`: Ensures that the class distribution in the original dataset is preserved in both the train and test sets (very important for classification).

from sklearn.model_selection import train_test_split

# Using the breast cancer dataset for this example
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}\n")

print(f"Original class distribution: {np.bincount(y) / len(y)}")
print(f"Training class distribution: {np.bincount(y_train) / len(y_train)}")
print(f"Test class distribution: {np.bincount(y_test) / len(y_test)}")

# ### Feature Scaling
# 
# Many algorithms (like SVMs, KNN, and Logistic Regression) perform better when features are on a similar scale. **Important:** We fit the scaler on the training data *only* and then use it to transform both the training and test data.
# 
# - **`StandardScaler`**: Scales features to have a mean of 0 and a standard deviation of 1.
# - **`MinMaxScaler`**: Scales features to a given range, typically [0, 1].

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Using the wine dataset as it has features with different scales
X_wine, y_wine = load_wine(return_X_y=True)
X_train_w, X_test_w, _, _ = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_w)
X_test_scaled = scaler.transform(X_test_w) # Use the same scaler fitted on training data

print("StandardScaler Example:")
print(f"Original training data mean: {X_train_w.mean():.2f}")
print(f"Scaled training data mean: {X_train_scaled.mean():.2f}\n")

# MinMaxScaler
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train_w)
X_test_minmax = min_max_scaler.transform(X_test_w)

print("MinMaxScaler Example:")
print(f"Scaled training data min: {X_train_minmax.min():.2f}")
print(f"Scaled training data max: {X_train_minmax.max():.2f}")

# ### Encoding Categorical Features
# 
# Machine learning models require numerical input. We need to convert categorical text data into numbers.
# 
# - **`LabelEncoder`**: Converts each category into an integer. Best used for the target variable (`y`).
# - **`OneHotEncoder`**: Creates a new binary column for each category. This prevents the model from assuming an ordinal relationship between categories. Best used for features (`X`).

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Example data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'green', 'red'],
    'size': ['S', 'M', 'L', 'S', 'M'],
    'label': ['class1', 'class2', 'class1', 'class2', 'class1']
})

# LabelEncoder for the target variable
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
print("LabelEncoder Output:")
print(df[['label', 'label_encoded']])
print(f"Classes found by LabelEncoder: {le.classes_}\n")

# OneHotEncoder for features
ohe = OneHotEncoder(sparse_output=False) # sparse_output=False returns a numpy array
encoded_features = ohe.fit_transform(df[['color', 'size']])
print("OneHotEncoder Output Shape:")
print(encoded_features.shape)
print("\nOneHotEncoder Categories:")
print(ohe.categories_)
print("\nOneHotEncoder Example Output (first row):")
print(encoded_features[0])

# ### Pipelines
# 
# A `Pipeline` allows you to chain multiple processing steps (like scaling and modeling) into a single estimator. This is extremely useful because:
# 1.  It simplifies your code.
# 2.  It prevents data leakage by ensuring that steps like scaling are fitted only on the training data during cross-validation.
# 
# `make_pipeline` is a convenient function to create a pipeline without having to name each step.

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

# We'll use the cancer dataset again
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with a scaler and a classifier
# Method 1: Using Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Scale the data
    ('svm', SVC(random_state=42))       # Step 2: Apply the SVM classifier
])

# Method 2: Using make_pipeline (more concise)
pipe_simple = make_pipeline(StandardScaler(), SVC(random_state=42))

# Now, we can treat the entire pipeline as a single estimator
pipe.fit(X_train, y_train)

# Evaluate the pipeline on the test data
score = pipe.score(X_test, y_test)
print(f"Pipeline score on test data: {score:.3f}")

# ---

# ## 🔍 3. Exploratory Data Analysis (EDA)
# 
# Before building models, it's crucial to understand your data. EDA helps you discover patterns, spot anomalies, test hypotheses, and check assumptions with the help of summary statistics and graphical representations.

# ### Summary Statistics
# 
# Pandas DataFrames are excellent for this. The `.describe()` method gives a great overview of the numerical features.

# Load the wine dataset into a pandas DataFrame for easier EDA
wine_data = load_wine()
df_wine = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df_wine['target'] = wine_data.target

print("First 5 rows of the Wine DataFrame:")
display(df_wine.head())

print("\nSummary statistics:")
display(df_wine.describe())

# ### Visualizations
# 
# Visualizations are key to understanding relationships in the data.

# #### `seaborn.pairplot()`
# 
# This function creates a grid of scatterplots for each pair of features and histograms for each individual feature. It's a fantastic way to spot relationships at a glance.

# Using a subset of columns for clarity
cols_to_plot = ['alcohol', 'malic_acid', 'ash', 'flavanoids', 'target']
sns.pairplot(df_wine[cols_to_plot], hue='target', palette='viridis')
plt.suptitle('Pairplot of Wine Dataset Features', y=1.02)
plt.show()

# #### Correlation Heatmap
# 
# A heatmap helps visualize the correlation matrix, showing which features are correlated with each other and with the target variable.

plt.figure(figsize=(12, 10))
correlation_matrix = df_wine.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Wine Dataset Features')
plt.show()

# #### Class Distribution
# 
# It's important to know if your dataset is balanced or imbalanced.

plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df_wine, palette='rocket', hue='target')
plt.title('Class Distribution of Wine Dataset')
plt.xticks(ticks=[0, 1, 2], labels=wine_data.target_names)
plt.show()

# #### Feature Importance
# 
# Tree-based models (like Random Forest) can provide feature importances, which tell you which features were most influential in the model's predictions. This is a form of post-modeling EDA.

from sklearn.ensemble import RandomForestClassifier

# Using the cancer dataset
X_cancer, y_cancer = cancer.data, cancer.target
feature_names_cancer = cancer.feature_names

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_cancer, y_cancer)

importances = forest.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names_cancer, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='plasma', hue='feature')
plt.title('Feature Importances from Random Forest (Cancer Dataset)')
plt.show()

# ---

# ## 🧠 4. Supervised Learning
# 
# Supervised learning involves learning a function that maps an input to an output based on example input-output pairs. We'll cover both classification and regression.
# 
# The standard scikit-learn API for estimators is:
# 1. **Choose a model**: `from sklearn.family import Model`
# 2. **Instantiate the model**: `model = Model(hyperparameters)`
# 3. **Fit the model to data**: `model.fit(X_train, y_train)`
# 4. **Predict on new data**: `y_pred = model.predict(X_test)`
# 5. **Evaluate the model**: `score = model.score(X_test, y_test)`

# ### 🟢 Classification
# 
# Classification is the task of predicting a discrete class label. We'll use the **Breast Cancer dataset** for these examples.

# Prepare data for classification models
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)

# We'll use a pipeline to ensure data is scaled for each model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer
)

scaler = StandardScaler()
X_train_scaled_c = scaler.fit_transform(X_train_c)
X_test_scaled_c = scaler.transform(X_test_c)

# #### Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42, max_iter=10000)
log_reg.fit(X_train_scaled_c, y_train_c)
y_pred_lr = log_reg.predict(X_test_scaled_c)
print(f"Logistic Regression Accuracy: {log_reg.score(X_test_scaled_c, y_test_c):.3f}")

# #### K-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled_c, y_train_c)
y_pred_knn = knn.predict(X_test_scaled_c)
print(f"KNN Accuracy: {knn.score(X_test_scaled_c, y_test_c):.3f}")

# #### Decision Trees
# Note: Decision Trees (and Random Forests) do not require feature scaling.

from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_c, y_train_c) # Using unscaled data
y_pred_tree = tree_clf.predict(X_test_c)
print(f"Decision Tree Accuracy: {tree_clf.score(X_test_c, y_test_c):.3f}")

# #### Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c) # Using unscaled data
y_pred_rf = rf_clf.predict(X_test_c)
print(f"Random Forest Accuracy: {rf_clf.score(X_test_c, y_test_c):.3f}")

# #### Support Vector Machines (SVM)

from sklearn.svm import SVC

svm_clf = SVC(random_state=42, probability=True) # probability=True for ROC curve
svm_clf.fit(X_train_scaled_c, y_train_c)
y_pred_svm = svm_clf.predict(X_test_scaled_c)
print(f"SVM Accuracy: {svm_clf.score(X_test_scaled_c, y_test_c):.3f}")

# #### Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train_scaled_c, y_train_c)
y_pred_nb = nb_clf.predict(X_test_scaled_c)
print(f"Naive Bayes Accuracy: {nb_clf.score(X_test_scaled_c, y_test_c):.3f}")

# ### ✅ Evaluation Metrics (Classification)

# Accuracy isn't always the best metric, especially with imbalanced datasets. Here are some more robust alternatives.
# 
# - **Accuracy**: (TP+TN) / (TP+TN+FP+FN). The percentage of correct predictions.
# - **Precision**: TP / (TP+FP). Of all positive predictions, how many were actually positive? (Minimizes false positives).
# - **Recall (Sensitivity)**: TP / (TP+FN). Of all actual positives, how many did we correctly identify? (Minimizes false negatives).
# - **F1-score**: The harmonic mean of precision and recall. A good measure for imbalanced classes.

from sklearn.metrics import classification_report

# We'll use the Random Forest predictions for this example
print("Classification Report for Random Forest:")
print(classification_report(y_test_c, y_pred_rf, target_names=cancer.target_names))

# #### Confusion Matrix
# 
# A confusion matrix gives a detailed breakdown of correct and incorrect classifications for each class.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test_c, y_pred_rf)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest')
plt.show()

# #### ROC Curve and AUC
# 
# The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings. The Area Under the Curve (AUC) is a single number summary of the curve's performance. An AUC of 1 is perfect, while 0.5 is no better than random guessing.

from sklearn.metrics import roc_curve, roc_auc_score

# Get prediction probabilities for the positive class (class 1)
y_pred_proba_svm = svm_clf.predict_proba(X_test_scaled_c)[:, 1]
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled_c)[:, 1]

fpr_svm, tpr_svm, _ = roc_curve(y_test_c, y_pred_proba_svm)
fpr_lr, tpr_lr, _ = roc_curve(y_test_c, y_pred_proba_lr)

auc_svm = roc_auc_score(y_test_c, y_pred_proba_svm)
auc_lr = roc_auc_score(y_test_c, y_pred_proba_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ---

# ### 🔵 Regression
# 
# Regression is the task of predicting a continuous value. For these examples, we'll use the California Housing dataset.

from sklearn.datasets import fetch_california_housing

# Prepare data for regression models
housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.3, random_state=42
)

scaler_h = StandardScaler()
X_train_scaled_h = scaler_h.fit_transform(X_train_h)
X_test_scaled_h = scaler_h.transform(X_test_h)

# #### Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled_h, y_train_h)
y_pred_lin_reg = lin_reg.predict(X_test_scaled_h)

# #### Ridge & Lasso Regression
# 
# These are regularized versions of Linear Regression that help prevent overfitting.
# - **Ridge (L2 Regularization)**: Adds a penalty equal to the square of the magnitude of coefficients.
# - **Lasso (L1 Regularization)**: Adds a penalty equal to the absolute value of the magnitude of coefficients. It can shrink some coefficients to exactly zero, effectively performing feature selection.

from sklearn.linear_model import Ridge, Lasso

# Ridge Regression
ridge_reg = Ridge(alpha=1.0, random_state=42)
ridge_reg.fit(X_train_scaled_h, y_train_h)

# Lasso Regression
lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X_train_scaled_h, y_train_h)

# #### Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_h, y_train_h) # No scaling needed

# #### Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_h, y_train_h) # No scaling needed

# ### ✅ Evaluation Metrics (Regression)
# 
# - **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values. Penalizes larger errors more.
# - **Mean Absolute Error (MAE)**: The average of the absolute differences. Easier to interpret as it's in the same units as the target.
# - **R² Score (Coefficient of Determination)**: The proportion of the variance in the dependent variable that is predictable from the independent variable(s). An R² of 1 indicates perfect prediction, while 0 means the model is no better than just predicting the mean of the target.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate Linear Regression
y_pred_lin = lin_reg.predict(X_test_scaled_h)
mse_lin = mean_squared_error(y_test_h, y_pred_lin)
r2_lin = r2_score(y_test_h, y_pred_lin)

print("Linear Regression Metrics:")
print(f"  R² Score: {r2_lin:.3f}")
print(f"  MSE: {mse_lin:.3f}\n")

# Evaluate Random Forest Regressor
y_pred_rf_reg = rf_reg.predict(X_test_h)
mse_rf = mean_squared_error(y_test_h, y_pred_rf_reg)
r2_rf = r2_score(y_test_h, y_pred_rf_reg)

print("Random Forest Regressor Metrics:")
print(f"  R² Score: {r2_rf:.3f}")
print(f"  MSE: {mse_rf:.3f}\n")

# ---

# ## 🔄 5. Model Selection
# 
# How do we choose the best model and the best hyperparameters? Scikit-learn provides tools to automate this process and ensure our model generalizes well to new data.

# ### Cross-Validation (`cross_val_score`)
# 
# A single train-test split can be lucky or unlucky. K-Fold Cross-Validation splits the data into K 'folds', then trains the model K times, each time using a different fold as the test set and the remaining K-1 as the training set. The final score is the average of the K scores.
# 
# This gives a much more robust estimate of the model's performance.

from sklearn.model_selection import cross_val_score

# Use the Random Forest Classifier and the full (unsplit) cancer dataset
rf_clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
# cv=5 means 5 folds
# scoring='accuracy' is the metric to use
scores = cross_val_score(rf_clf_cv, X_cancer, y_cancer, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Average score: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")

# ### Grid Search (`GridSearchCV`)
# 
# Most models have hyperparameters that need to be tuned (e.g., `n_neighbors` in KNN, `C` and `gamma` in SVM). `GridSearchCV` exhaustively searches over a specified parameter grid to find the combination that gives the best cross-validated performance.

from sklearn.model_selection import GridSearchCV

# We will tune an SVM classifier
# We must use a pipeline to ensure scaling is done correctly within each CV fold
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

# Define the parameter grid to search
param_grid = {
    'svm__C': [0.1, 1, 10, 100],            # Note the __ syntax: 'estimator__parameter'
    'svm__gamma': [1, 0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}

# Instantiate GridSearchCV
# n_jobs=-1 uses all available CPU cores
grid_search = GridSearchCV(pipe_svm, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# Fit it to the data (this can take some time)
grid_search.fit(X_train_c, y_train_c)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# The grid_search object is now a trained model with the best parameters
print(f"Test set score with best parameters: {grid_search.score(X_test_c, y_test_c):.3f}")

# ### Random Search (`RandomizedSearchCV`)
# 
# `RandomizedSearchCV` is similar to Grid Search, but instead of trying every combination, it samples a fixed number of parameter settings from specified distributions. This is often more efficient, especially with a large hyperparameter space.

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter distributions to sample from
param_dist = {
    'svm__C': uniform(loc=0.1, scale=100), # Uniform distribution from 0.1 to 100.1
    'svm__gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)), # List of options
    'svm__kernel': ['rbf', 'poly', 'sigmoid']
}

# n_iter specifies how many parameter settings are sampled
random_search = RandomizedSearchCV(pipe_svm, param_distributions=param_dist, n_iter=50, 
                                 cv=5, n_jobs=-1, verbose=1, random_state=42, scoring='accuracy')

random_search.fit(X_train_c, y_train_c)

print(f"\nBest parameters found: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.3f}")
print(f"Test set score with best parameters: {random_search.score(X_test_c, y_test_c):.3f}")

# ### Overfitting vs Underfitting
# 
# - **Overfitting (High Variance)**: The model learns the training data too well, including its noise. It performs great on training data but poorly on unseen test data.
# - **Underfitting (High Bias)**: The model is too simple to capture the underlying patterns in the data. It performs poorly on both training and test data.
# 
# **Learning Curves** help diagnose this. They plot the model's performance on the training and validation sets as a function of the training set size.

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# A simple model (underfitting example)
estimator_underfit = LogisticRegression(max_iter=10000, random_state=42)
plot_learning_curve(estimator_underfit, "Learning Curve (Logistic Regression)", X_cancer, y_cancer, cv=5)
plt.show()

# A complex model (potential for overfitting)
estimator_overfit = DecisionTreeClassifier(max_depth=10, random_state=42)
plot_learning_curve(estimator_overfit, "Learning Curve (Deep Decision Tree)", X_cancer, y_cancer, cv=5)
plt.show()

# **Interpretation**:
# - **Logistic Regression (High Bias)**: Both training and validation scores are low and converge. The model is too simple; adding more data won't help.
# - **Decision Tree (High Variance)**: The training score is very high, but the validation score is much lower. There is a large gap. The model is overfitting. Adding more data could help the scores converge.

# ---

# ## 🧮 6. Unsupervised Learning
# 
# Unsupervised learning finds patterns in data without pre-existing labels. We'll explore clustering and dimensionality reduction.

# Generate some sample data for clustering
from sklearn.datasets import make_blobs, make_moons

X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

# Scale the data
scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)
X_moons_scaled = scaler.fit_transform(X_moons)

# ### K-Means Clustering
# 
# K-Means partitions data into *K* distinct, non-overlapping clusters. It works well when clusters are spherical and evenly sized.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_blobs_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering')
plt.show()

# ### Hierarchical Clustering
# 
# Builds a hierarchy of clusters, which can be visualized as a dendrogram. It doesn't require specifying the number of clusters beforehand.

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Plot the dendrogram to find the optimal number of clusters
plt.figure(figsize=(12, 7))
dendrogram = sch.dendrogram(sch.linkage(X_blobs_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()

# Perform clustering
agg_cluster = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
y_agg = agg_cluster.fit_predict(X_blobs_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_agg, s=50, cmap='plasma')
plt.title('Agglomerative Hierarchical Clustering')
plt.show()

# ### DBSCAN
# 
# Density-Based Spatial Clustering of Applications with Noise. It groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It can find arbitrarily shaped clusters and doesn't require the number of clusters to be specified.

from sklearn.cluster import DBSCAN

# DBSCAN is great for non-spherical clusters
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_moons_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_dbscan, s=50, cmap='cividis')
plt.title('DBSCAN Clustering on Moons Dataset')
plt.show()
# Note: Cluster label -1 represents noise/outliers

# ### Principal Component Analysis (PCA)
# 
# PCA is a linear dimensionality reduction technique. It transforms the data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (the first principal component), the second greatest variance on the second coordinate, and so on. It's used for visualization, noise filtering, and feature extraction.

from sklearn.decomposition import PCA

# Using the digits dataset for high-dimensional data
X_digits, y_digits = load_digits(return_X_y=True)
X_digits_scaled = StandardScaler().fit_transform(X_digits)

# Reduce from 64 dimensions to 2
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_digits_scaled)

print(f"Original shape: {X_digits_scaled.shape}")
print(f"Shape after PCA: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='jet', alpha=0.7)
plt.title('PCA of Digits Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
plt.show()

# ### t-SNE (t-Distributed Stochastic Neighbor Embedding)
# 
# t-SNE is a non-linear technique primarily used for **data visualization**. It is excellent at revealing the underlying structure of data, such as clusters, but it should **not** be used for clustering itself, as the distances between clusters in a t-SNE plot are not meaningful.

from sklearn.manifold import TSNE

# It's often a good idea to run PCA before t-SNE on high-dimensional data
pca_50 = PCA(n_components=50, random_state=42)
X_pca_50 = pca_50.fit_transform(X_digits_scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca_50)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='jet', alpha=0.7)
plt.title('t-SNE of Digits Dataset')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
plt.show()

# ---

# ## 🔧 7. Model Deployment Basics
# 
# Once you have a trained model, you need to save it so you can use it later for making predictions on new data. This process is called serialization.
# 
# `joblib` is generally preferred over `pickle` for scikit-learn objects because it is more efficient for objects that carry large NumPy arrays.

import joblib

# Let's use our best model from GridSearchCV
best_model = grid_search.best_estimator_

# Save the model to a file
model_filename = 'final_cancer_classifier.joblib'
joblib.dump(best_model, model_filename)

print(f"Model saved to {model_filename}")

# ### Inference Example with Saved Model
# 
# Now, let's pretend we are in a new script or application. We can load our saved model and use it to make predictions.

# Load the model from the file
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# Let's take one sample from our original test set to simulate new data
new_data_point = X_test_c[0].reshape(1, -1) # Reshape to be a 2D array
actual_label = y_test_c[0]

# The loaded model is a pipeline, so it will handle scaling automatically
prediction = loaded_model.predict(new_data_point)
prediction_proba = loaded_model.predict_proba(new_data_point)

predicted_class_name = cancer.target_names[prediction[0]]
actual_class_name = cancer.target_names[actual_label]

print(f"\nNew Data Point Shape: {new_data_point.shape}")
print(f"Actual Label: {actual_class_name} (Class {actual_label})")
print(f"Predicted Label: {predicted_class_name} (Class {prediction[0]})")
print(f"Prediction Probabilities: {prediction_proba}")

# ---

# ## 📁 8. Real-world Dataset Example: Titanic
# 
# Let's put everything together in a mini-project. We'll use the famous Titanic dataset to predict passenger survival. This involves data loading, cleaning, preprocessing with a full pipeline, model training, and evaluation.

# ### Step 1: Load and Inspect Data

# Load data from seaborn's repository for convenience
df_titanic = sns.load_dataset('titanic')

print("Titanic DataFrame Info:")
df_titanic.info()

print("\nFirst 5 rows:")
display(df_titanic.head())

# ### Step 2: Exploratory Data Analysis (EDA)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(x='survived', data=df_titanic, ax=axes[0])
axes[0].set_title('Survival Count')

sns.countplot(x='survived', hue='sex', data=df_titanic, ax=axes[1])
axes[1].set_title('Survival by Gender')

sns.countplot(x='survived', hue='pclass', data=df_titanic, ax=axes[2])
axes[2].set_title('Survival by Passenger Class')

plt.tight_layout()
plt.show()

# ### Step 3: Preprocessing and Pipeline Building
# 
# This is the most complex part. We will build a robust preprocessing pipeline using `ColumnTransformer` to handle different data types and missing values.
# 
# - **Numerical features**: Impute missing values (e.g., 'age') with the median, then scale them.
# - **Categorical features**: Impute missing values (e.g., 'embarked') with the most frequent value, then one-hot encode them.

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define features (X) and target (y)
X = df_titanic.drop('survived', axis=1)
y = df_titanic['survived']

# We will drop 'deck', 'embark_town', 'alive' as they are redundant or have too many missing values
X = X.drop(['deck', 'embark_town', 'alive'], axis=1)

# Split data before any processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numerical and categorical columns
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'who', 'adult_male', 'alone']

# Create preprocessing pipelines for both data types
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

print("Preprocessing pipeline created successfully.")

# ### Step 4: Full Pipeline, Model Training, and Evaluation

# Create the full pipeline by adding a classifier
titanic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
titanic_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Make predictions
y_pred = titanic_pipeline.predict(X_test)

# Evaluate the model
accuracy = titanic_pipeline.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did not survive', 'Survived'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Titanic Survival Prediction')
plt.show()

# ## 🎉 Congratulations!
# 
# You have completed this comprehensive tour of the Scikit-Learn library. You've learned how to:
# 
# 1.  Handle and prepare data.
# 2.  Perform exploratory data analysis.
# 3.  Train and evaluate a wide range of supervised and unsupervised models.
# 4.  Perform robust model selection with cross-validation and hyperparameter tuning.
# 5.  Diagnose model performance with learning curves.
# 6.  Save models for deployment.
# 7.  Apply these skills to a real-world problem from start to finish.
# 
# The next step is to practice these techniques on new datasets. Happy coding!


# # Concept Learning in Machine Learning
# 
# **Objective:** This notebook introduces the fundamental concepts of concept learning in machine learning. We will explore and implement two classic algorithms: Find-S and Candidate-Elimination. This notebook is designed to be an interactive lab session for a machine learning course.
# 
# **Author:** Raghav Borikar
# 
# ---

# ## 1. Introduction to Concept Learning
# 
# Concept learning is a fundamental area of machine learning that involves inferring a Boolean-valued function from training examples of its input and output. In simpler terms, it's about learning a general concept or category from a set of labeled examples. For instance, we might want to learn the concept of "a car that I would like to buy" based on examples of cars we have liked or disliked in the past.
# 
# ### Key Terminology
# 
# *   **Concept:** A category or a subset of objects or events defined by a set of common features. For example, the concept of a "bird" includes features like having feathers, wings, and the ability to fly.
# *   **Instances:** The individual examples from which we learn the concept. Each instance is described by a set of attributes.
# *   **Attributes:** The features that describe an instance. For a car, attributes could be 'color', 'year', 'engine size', etc.
# *   **Target Concept:** The specific concept that we are trying to learn. It is a function that maps instances to a Boolean value (True/False or Yes/No).
# *   **Hypothesis (h):** A potential definition of the target concept. Our goal is to find a hypothesis that is identical to the target concept over the entire set of instances.
# *   **Hypothesis Space (H):** The set of all possible hypotheses that the learning algorithm can consider.

# ## 2. The "Enjoy Sport" Dataset
# 
# To illustrate the concept learning algorithms, we will use the classic "Enjoy Sport" dataset. This dataset consists of several examples of weather conditions, and the target concept is whether or not it's a good day to play a sport.
# 
# The attributes are:
# *   `Sky`: Sunny, Cloudy, Rainy
# *   `AirTemp`: Warm, Cold
# *   `Humidity`: Normal, High
# *   `Wind`: Strong, Weak
# *   `Water`: Warm, Cool
# *   `Forecast`: Same, Change
# 
# And the target concept `EnjoySport` can be `Yes` or `No`.

import pandas as pd
import numpy as np

# Create the dataset as a pandas DataFrame
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("The 'Enjoy Sport' Dataset:")
df

# ## 3. The Find-S Algorithm
# 
# The Find-S algorithm is a simple concept learning algorithm that finds the most specific hypothesis that is consistent with the positive training examples. It starts with the most specific possible hypothesis and generalizes it as it encounters positive examples that are not covered by the current hypothesis. Negative examples are ignored in this algorithm.
# 
# ### Algorithm Steps:
# 
# 1.  Initialize the hypothesis `h` to the most specific hypothesis in the hypothesis space `H`.
# 2.  For each positive training instance `x`:
#     *   For each attribute `a_i` in `h`:
#         *   If the constraint `a_i` in `h` is not satisfied by `x`:
#             *   Replace `a_i` in `h` with the next more general constraint that is satisfied by `x`.
# 3.  Output the final hypothesis `h`.

def find_s_algorithm(data):
    """
    Implements the Find-S algorithm.
    """
    # Separate features and target
    features = np.array(data.iloc[:, :-1])
    target = np.array(data.iloc[:, -1])

    # Initialize with the most specific hypothesis
    # Get the first positive example to initialize the hypothesis
    for i, val in enumerate(target):
        if val == 'Yes':
            specific_h = features[i].copy()
            break
            
    print(f"Initial hypothesis: {specific_h}\n")
    
    # Iterate through the training examples
    for i, instance in enumerate(features):
        if target[i] == 'Yes':
            print(f"Processing instance {i+1}: {instance}")
            for j in range(len(specific_h)):
                if instance[j] != specific_h[j]:
                    specific_h[j] = '?'
            print(f"Updated hypothesis: {specific_h}\n")
        else:
            print(f"Ignoring negative instance {i+1}: {instance}\n")

    return specific_h

# Run the Find-S algorithm on our dataset
final_hypothesis = find_s_algorithm(df)

print(f"\nThe final maximally specific hypothesis is: {final_hypothesis}")

# ## 4. The Candidate-Elimination Algorithm
# 
# The Candidate-Elimination algorithm is a more sophisticated approach to concept learning. Unlike Find-S, it finds the set of all hypotheses that are consistent with the training examples. This set of consistent hypotheses is called the **version space**.
# 
# The algorithm maintains two sets of hypotheses:
# 
# *   **G (General Boundary):** The set of the most general hypotheses consistent with the training data.
# *   **S (Specific Boundary):** The set of the most specific hypotheses consistent with the training data.
# 
# ### Algorithm Steps:
# 
# 1.  Initialize `G` to the set containing the most general hypothesis in the hypothesis space `H`.
# 2.  Initialize `S` to the set containing the most specific hypothesis in `H`.
# 3.  For each training example `d`:
#     *   If `d` is a **positive example**:
#         *   Remove from `G` any hypothesis inconsistent with `d`.
#         *   For each hypothesis `s` in `S` that is not consistent with `d`:
#             *   Remove `s` from `S`.
#             *   Add to `S` all minimal generalizations `h` of `s` such that `h` is consistent with `d`, and some member of `G` is more general than `h`.
#             *   Remove from `S` any hypothesis that is more general than another hypothesis in `S`.
#     *   If `d` is a **negative example**:
#         *   Remove from `S` any hypothesis inconsistent with `d`.
#         *   For each hypothesis `g` in `G` that is not consistent with `d`:
#             *   Remove `g` from `G`.
#             *   Add to `G` all minimal specializations `h` of `g` such that `h` is consistent with `d`, and some member of `S` is more specific than `h`.
#             *   Remove from `G` any hypothesis that is more specific than another hypothesis in `G`.

def candidate_elimination(data):
    """
    Implements the Candidate-Elimination algorithm.
    """
    features = np.array(data.iloc[:, :-1])
    target = np.array(data.iloc[:, -1])
    
    num_attributes = len(features[0])
    
    # Initialize G and S boundaries
    specific_h = ['0'] * num_attributes
    general_h = [['?'] * num_attributes]
    
    print(f"Initial Specific Boundary (S0): {specific_h}")
    print(f"Initial General Boundary (G0): {general_h}\n")

    for i, instance in enumerate(features):
        print(f"--- Instance {i+1} ---: {instance}")
        if target[i] == "Yes":
            # Remove inconsistent hypotheses from G
            general_h = [g for g in general_h if all(g[j] == '?' or g[j] == instance[j] for j in range(num_attributes))]
            
            # Generalize S
            for j in range(num_attributes):
                if specific_h[j] == '0':
                    specific_h[j] = instance[j]
                elif specific_h[j] != instance[j]:
                    specific_h[j] = '?'

        else: # Negative example
            new_general_h = []
            for g in general_h:
                if all(g[j] == '?' or g[j] == instance[j] for j in range(num_attributes)):
                     for j in range(num_attributes):
                        if g[j] == '?':
                            for val in np.unique(features[:, j]):
                                if val != instance[j]:
                                    temp_h = g[:j] + [val] + g[j+1:]
                                    if any(all(s_val == '?' or s_val == temp_h[k] for k, s_val in enumerate(specific_h)) for k in range(num_attributes)):
                                        new_general_h.append(temp_h)
                else:
                    new_general_h.append(g)
            general_h = new_general_h
            
        print(f"S[{i+1}]: {specific_h}")
        print(f"G[{i+1}]: {general_h}\n")
        
    # Remove redundant hypotheses from G
    final_general_h = []
    for g in general_h:
        is_subsumed = False
        for g2 in general_h:
            if g != g2 and all(g2[j] == '?' or g2[j] == g[j] for j in range(num_attributes)):
                is_subsumed = True
                break
        if not is_subsumed:
            final_general_h.append(g)
    
    return specific_h, final_general_h

s_final, g_final = candidate_elimination(df)
print(f"Final Specific Boundary (S): {s_final}")
print(f"Final General Boundary (G): {g_final}")

# ## 5. Visualizations
# 
# Visualizing the hypothesis space for concept learning can be challenging, especially with multiple attributes. However, we can create visualizations that illustrate the process of the algorithms.
# 
# ### Visualizing the Find-S Algorithm's Progression

import matplotlib.pyplot as plt

def visualize_find_s(data):
    features = np.array(data.iloc[:, :-1])
    target = np.array(data.iloc[:, -1])
    
    for i, val in enumerate(target):
        if val == 'Yes':
            specific_h = features[i].copy()
            break
            
    hypotheses = [list(specific_h)]
    
    for i, instance in enumerate(features):
        if target[i] == 'Yes':
            for j in range(len(specific_h)):
                if instance[j] != specific_h[j]:
                    specific_h[j] = '?'
            hypotheses.append(list(specific_h))
            
    # Visualization
    fig, ax = plt.subplots(figsize=(10, len(hypotheses) * 0.5))
    ax.set_yticks(np.arange(len(hypotheses)))
    ax.set_yticklabels([f'h{i}' for i in range(len(hypotheses))])
    ax.set_xticks(np.arange(len(data.columns) - 1))
    ax.set_xticklabels(data.columns[:-1])
    
    for i in range(len(hypotheses)):
        for j in range(len(hypotheses[i])):
            ax.text(j, i, hypotheses[i][j], ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='b', lw=1))

    ax.set_title("Progression of Hypothesis in Find-S Algorithm")
    plt.tight_layout()
    plt.show()

visualize_find_s(df)

# ## 6. Student Activity
# 
# Now it's your turn to apply what you've learned! Here is a new dataset about whether to wait for a table at a restaurant.
# 
# ### New Dataset: Restaurant Waiting

restaurant_data = {
    'Alternate': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Bar': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Fri/Sat': ['No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
    'Hungry': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Patrons': ['Some', 'Full', 'Some', 'Full', 'Full', 'Some', 'None', 'Some', 'Full', 'Full'],
    'Price': ['$$$', '$', '$', '$', '$$$', '$', '$', '$$', '$', '$$$'],
    'Raining': ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
    'Reservation': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Type': ['French', 'Thai', 'Burger', 'Thai', 'French', 'Italian', 'Burger', 'Thai', 'Burger', 'Italian'],
    'WaitEstimate': ['0-10', '30-60', '0-10', '10-30', '>60', '0-10', '0-10', '0-10', '>60', '10-30'],
    'WillWait': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

restaurant_df = pd.DataFrame(restaurant_data)
print("Restaurant Waiting Dataset:")
restaurant_df

# ### Your Tasks:
# 
# 1.  **Manual Tracing:**
#     *   Manually trace the Find-S algorithm on the `Restaurant Waiting` dataset. Write down the hypothesis after each positive instance.
#     *   Manually trace the Candidate-Elimination algorithm for the first 4 instances of the `Restaurant Waiting` dataset. Write down the `S` and `G` boundaries after each instance.
# 
# 2.  **Implementation:**
#     *   Modify and run the `find_s_algorithm` function on the `restaurant_df` DataFrame. What is the final hypothesis?
#     *   Modify and run the `candidate_elimination` function on the `restaurant_df` DataFrame. What are the final `S` and `G` boundaries?
# 
# 3.  **Conceptual Questions:**
#     *   What happens to the Find-S algorithm if the first positive example is not representative of the target concept?
#     *   What are the limitations of the Candidate-Elimination algorithm when dealing with noisy data (i.e., incorrectly labeled examples)?
#     *   If the version space collapses to an empty set, what can you conclude about the training data or the hypothesis space?

# ### Solution for Task 2 (Implementation)

# Run Find-S on the restaurant dataset
print("--- Running Find-S on Restaurant Data ---")
restaurant_find_s = find_s_algorithm(restaurant_df)
print(f"\nFinal hypothesis from Find-S: {restaurant_find_s}\n")

# Run Candidate-Elimination on the restaurant dataset
print("--- Running Candidate-Elimination on Restaurant Data ---")
s_final_restaurant, g_final_restaurant = candidate_elimination(restaurant_df)
print(f"\nFinal Specific Boundary (S): {s_final_restaurant}")
print(f"Final General Boundary (G): {g_final_restaurant}")




import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('titanic')[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]

# In this assignment, you will work with a dataset loaded above.
# You will follow the given steps and complete the task using the concepts and we have learned.
# 
# ### **Steps to Complete**
# 
# 1. **Exploratory Data Analysis (EDA)**  
#    - Display the first few rows of the dataset.  
#    - Check the dataset’s shape, data types, and basic statistics.  
#    - Identify numerical and categorical features.  
#    - Visualize feature distributions using plots (e.g., histograms, boxplots, countplots).  
# 
# 2. **Handle Missing Values (if any)**  
#    - Check for missing values.  
#    - Decide whether to fill them (mean/median/mode) or drop rows/columns.  
# 
# 3. **Label Encode Categorical Columns**  
#    - Use `LabelEncoder` to convert categorical variables into numerical form.  
# 
# 4. **Split Data**  
#    - Split the dataset into **train** and **test** sets (e.g., 70%-30%).  
# 
# 5. **Train Two or More Decision Trees**  
#    - Train Decision Tree classifiers with **different depths** and/or **different feature subsets**.  
#    - Ensure you clearly label each model (e.g., `DT_depth3`, `DT_depth5`).  
# 
# 6. **Evaluation & Comparison**  
#    - Evaluate each Decision Tree on the **test set** using accuracy and confusion matrix.  
#    - Plot all Decision Trees you created and visually compare them.  
#    - Write your observation on which tree performs better and why.
# 
# ---
# 




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

x=a.values[:,1:5]
y=a.values[:,5]
clf = tree.DecisionTreeClassifier()
results = clf.fit(x, y)
clf.predict([[1,1,1,1]])

a=pd.read_csv('Iris.csv')
a

x

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(results,feature_names = fn,class_names=cn,filled=True)
plt.plot()

clf.score(x,y)


# # Ensemble Learning
# 
# **Crowd intelligence**
# 
# Credit-Joaquin Vanschoren

pip install preamble

# Auto-setup when running on Google Colab
import os
if 'google.colab' in str(get_ipython()) and not os.path.exists('/content/master'):
    !git clone -q https://github.com/ML-course/master.git /content/master
    !pip --quiet install -r /content/master/requirements_colab.txt
    %cd master/notebooks

# Global imports and settings
%matplotlib inline
from preamble import *
interactive = True # Set to True for interactive plots
if interactive:
    fig_scale = 0.9
    plt.rcParams.update(print_config)
else: # For printing
    fig_scale = 0.3
    plt.rcParams.update(print_config)

# ## Ensemble learning
# * If different models make different mistakes, can we simply average the predictions?
# * Voting Classifier: gives every model a _vote_ on the class label
#     * Hard vote: majority class wins (class order breaks ties)
#     * Soft vote: sum class probabilities $p_{m,c}$ over $M$ models: $\underset{c}{\operatorname{argmax}} \sum_{m=1}^{M} w_c p_{m,c}$
#     * Classes can get different weights $w_c$ (default: $w_c=1$)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Toy data
X, y = make_moons(noise=.2, random_state=18) # carefully picked random state for illustration
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Plot grid
x_lin = np.linspace(X_train[:, 0].min() - .5, X_train[:, 0].max() + .5, 100)
y_lin = np.linspace(X_train[:, 1].min() - .5, X_train[:, 1].max() + .5, 100)
x_grid, y_grid = np.meshgrid(x_lin, y_lin)
X_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
models = [LogisticRegression(C=100),
          DecisionTreeClassifier(max_depth=3, random_state=0),
          KNeighborsClassifier(n_neighbors=1),
          KNeighborsClassifier(n_neighbors=30)]

@interact
def combine_voters(model1=models, model2=models):
    # Voting Classifier and components
    voting = VotingClassifier([('model1', model1),('model2', model2)],voting='soft')
    voting.fit(X_train, y_train)

    # transform produces individual probabilities
    y_probs =  voting.transform(X_grid)

    fig, axes = plt.subplots(1, 3, subplot_kw={'xticks': (()), 'yticks': (())}, figsize=(11*fig_scale, 3*fig_scale))
    scores = [voting.estimators_[0].score(X_test, y_test),
             voting.estimators_[1].score(X_test, y_test),
             voting.score(X_test, y_test)]
    titles = [model1.__class__.__name__, model2.__class__.__name__, 'VotingClassifier']
    for prob, score, title, ax in zip([y_probs[:, 1], y_probs[:, 3], y_probs[:, 1::2].sum(axis=1)], scores, titles, axes.ravel()):
        ax.contourf(x_grid, y_grid, prob.reshape(x_grid.shape), alpha=.4, cmap='bwr')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', s=7*fig_scale)
        ax.set_title(title + f" \n acc={score:.2f}", pad=0, fontsize=9)

if not interactive:
    combine_voters(models[0],models[1])

# * Why does this work?
#     * Different models may be good at different 'parts' of data (even if they underfit)
#     * Individual mistakes can be 'averaged out' (especially if models overfit)
# * Which models should be combined?
# * Bias-variance analysis teaches us that we have two options:
#     * If model underfits (high bias, low variance): combine with other low-variance models
#         * Need to be different: 'experts' on different parts of the data
#         * Bias reduction. Can be done with **_Boosting_**
#     * If model overfits (low bias, high variance): combine with other low-bias models
#         * Need to be different: individual mistakes must be different
#         * Variance reduction. Can be done with **_Bagging_**
# * Models must be uncorrelated but good enough (otherwise the ensemble is worse)
# * We can also _learn_ how to combine the predictions of different models: **_Stacking_**

# ## Decision trees (recap)
# * Representation: Tree that splits data points into leaves based on tests
# * Evaluation (loss): Heuristic for purity of leaves (Gini index, entropy,...)
# * Optimization: Recursive, heuristic greedy search (Hunt's algorithm)
#     * Consider all splits (thresholds) between adjacent data points, for every feature
#     * Choose the one that yields the purest leafs, repeat

import graphviz

@interact
def plot_depth(depth=(1,5,1)):
    X, y = make_moons(noise=.2, random_state=18) # carefully picked random state for illustration
    fig, ax = plt.subplots(1, 2, figsize=(12*fig_scale, 4*fig_scale),
                           subplot_kw={'xticks': (), 'yticks': ()})

    tree = mglearn.plots.plot_tree(X, y, max_depth=depth)
    ax[0].imshow(mglearn.plots.tree_image(tree))
    ax[0].set_axis_off()

if not interactive:
    plot_depth(depth=3)

# ### Evaluation (loss function for classification)
# * Every leaf predicts a class probability $\hat{p}_c$ = the relative frequency of class $c$
# * Leaf impurity measures (splitting criteria) for $L$ leafs, leaf $l$ has data $X_l$:
#     - Gini-Index: $Gini(X_{l}) = \sum_{c\neq c'} \hat{p}_c \hat{p}_{c'}$
#     - Entropy (more expensive): $E(X_{l}) = -\sum_{c\neq c'} \hat{p}_c \log_{2}\hat{p}_c$
#     - Best split maximizes _information gain_ (idem for Gini index) $$ Gain(X,X_i) = E(X) - \sum_{l=1}^L \frac{|X_{i=l}|}{|X_{i}|} E(X_{i=l}) $$

def gini(p):
   return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

def entropy(p):
   return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def classification_error(p):
   return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
scaled_ent = [e*0.5 if e else None for e in ent]
c_err = [classification_error(i) for i in x]

fig = plt.figure(figsize=(5*fig_scale, 2.5*fig_scale))
ax = plt.subplot(111)

for j, lab, ls, c, in zip(
      [ent, scaled_ent, gini(x), c_err],
      ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],
      ['-', '-', '--', '-.'],
      ['lightgray', 'red', 'green', 'blue']):
   line = ax.plot(x, j, label=lab, linestyle=ls, lw=2*fig_scale, color=c)

ax.legend(loc='upper left', ncol=1, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ylim([0, 1.1])
plt.xlabel('p(j=1)',labelpad=0)
plt.ylabel('Impurity Index')
plt.show()

# ### Regression trees
# * Every leaf predicts the _mean_ target value $\mu$ of all points in that leaf
# * Choose the split that minimizes squared error of the leaves: $\sum_{x_{i} \in L} (y_i - \mu)^2$
# * Yields non-smooth step-wise predictions, cannot extrapolate

from sklearn.tree import DecisionTreeRegressor

def plot_decision_tree_regression(regr_1, regr_2):
    # Create a random dataset
    rng = np.random.RandomState(5)
    X = np.sort(7 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))
    split = 65

    # Fit regression model of first 60 points
    regr_1.fit(X[:split], y[:split])
    regr_2.fit(X[:split], y[:split])

    # Predict
    X_test = np.arange(0.0, 7.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure(figsize=(8*fig_scale,5*fig_scale))
    plt.scatter(X[:split], y[:split], c="darkorange", label="training data")
    plt.scatter(X[split:], y[split:], c="blue", label="test data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2*fig_scale)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2*fig_scale)
    plt.xlabel("data", fontsize=9)
    plt.ylabel("target", fontsize=9)
    plt.title("Decision Tree Regression", fontsize=9)
    plt.legend()
    plt.show()

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)

plot_decision_tree_regression(regr_1,regr_2)

# ### Impurity/Entropy-based feature importance
# * We can measure the importance of features (to the model) based on
#     - Which features we split on
#     - How high up in the tree we split on them (first splits ar emore important)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0).fit(Xc_train, yc_train)

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.figure(figsize=(7*fig_scale,5.4*fig_scale))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names, fontsize=7)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)

# ### Under- and overfitting
# * We can easily control the (maximum) depth of the trees as a hyperparameter
# * Bias-variance analysis:
#     * Shallow trees have high bias but very low variance (underfitting)
#     * Deep trees have high variance but low bias (overfitting)
# * Because we can easily control their complexity, they are ideal for ensembling
#     * Deep trees: keep low bias, reduce variance with **_Bagging_**
#     * Shallow trees: keep low variance, reduce bias with **_Boosting_**

from sklearn.model_selection import ShuffleSplit, train_test_split

# Bias-Variance Computation
def compute_bias_variance(clf, X, y):
    # Bootstraps
    n_repeat = 40 # 40 is on the low side to get a good estimate. 100 is better.
    shuffle_split = ShuffleSplit(test_size=0.33, n_splits=n_repeat, random_state=0)

    # Store sample predictions
    y_all_pred = [[] for _ in range(len(y))]

    # Train classifier on each bootstrap and score predictions
    for i, (train_index, test_index) in enumerate(shuffle_split.split(X)):
        # Train and predict
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])

        # Store predictions
        for j,index in enumerate(test_index):
            y_all_pred[index].append(y_pred[j])

    # Compute bias, variance, error
    bias_sq = sum([ (1 - x.count(y[i])/len(x))**2 * len(x)/n_repeat
                for i,x in enumerate(y_all_pred)])
    var = sum([((1 - ((x.count(0)/len(x))**2 + (x.count(1)/len(x))**2))/2) * len(x)/n_repeat
               for i,x in enumerate(y_all_pred)])
    error = sum([ (1 - x.count(y[i])/len(x)) * len(x)/n_repeat
            for i,x in enumerate(y_all_pred)])

    return np.sqrt(bias_sq), var, error

def plot_bias_variance(clf, X, y):
    bias_scores = []
    var_scores = []
    err_scores = []
    max_depth= range(2,11)

    for i in max_depth:
        b,v,e = compute_bias_variance(clf.set_params(random_state=0,max_depth=i),X,y)
        bias_scores.append(b)
        var_scores.append(v)
        err_scores.append(e)

    plt.figure(figsize=(8*fig_scale,3*fig_scale))
    plt.suptitle(clf.__class__.__name__, fontsize=9)
    plt.plot(max_depth, var_scores,label ="variance", lw=2*fig_scale )
    plt.plot(max_depth, np.square(bias_scores),label ="bias^2", lw=2*fig_scale )
    plt.plot(max_depth, err_scores,label ="error", lw=2*fig_scale)
    plt.xlabel("max_depth", fontsize=7)
    plt.legend(loc="best", fontsize=7)
    plt.show()

dt = DecisionTreeClassifier()
plot_bias_variance(dt, cancer.data, cancer.target)

# ## Bagging (Bootstrap Aggregating)
# 
# * Obtain different models by training the _same_ model on _different training samples_
#     * Reduce overfitting by averaging out individual predictions (variance reduction)
# * In practice: take $I$ bootstrap samples of your data, train a model on each bootstrap
#    * Higher $I$: more models, more smoothing (but slower training and prediction)    
# * Base models should be unstable: different training samples yield different models
#     * E.g. very deep decision trees, or even randomized decision trees
#     * Deep Neural Networks can also benefit from bagging (deep ensembles)
# * Prediction by averaging predictions of base models
#     * Soft voting for classification (possibly weighted)
#     * Mean value for regression
# * Can produce uncertainty estimates as well
#     * By combining class probabilities of individual models (or variances for regression)

# ### Random Forests
# * Uses _randomized trees_ to make models even less correlated (more unstable)
#     * At every split, only consider `max_features` features, randomly selected
# * Extremely randomized trees: considers 1 random threshold for random set of features (faster)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

models=[RandomForestClassifier(n_estimators=5, random_state=7, n_jobs=-1),ExtraTreesClassifier(n_estimators=5, random_state=2, n_jobs=-1)]

@interact
def run_forest_run(model=models):
    forest = model.fit(X_train, y_train)
    fig, axes = plt.subplots(2, 3, figsize=(12*fig_scale, 6*fig_scale))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i), pad=0, fontsize=9)
        mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

    mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                                    alpha=.4)
    axes[-1, -1].set_title(model.__class__.__name__, pad=0, fontsize=9)
    axes[-1, -1].set_xticks(())
    axes[-1, -1].set_yticks(())
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, s=10*fig_scale);

if not interactive:
    run_forest_run(model=models[0])

# ### Effect on bias and variance
# * Increasing the number of models (trees) decreases variance (less overfitting)
# * Bias is mostly unaffected, but will increase if the forest becomes too large (oversmoothing)

from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
cancer = load_breast_cancer()

# Faster version of plot_bias_variance that uses warm-starting
def plot_bias_variance_rf(model, X, y, warm_start=False):
    bias_scores = []
    var_scores = []
    err_scores = []

    # Bootstraps
    n_repeat = 40 # 40 is on the low side to get a good estimate. 100 is better.
    shuffle_split = ShuffleSplit(test_size=0.33, n_splits=n_repeat, random_state=0)

    # Ensemble sizes
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # Store sample predictions. One per n_estimators
    # n_estimators : [predictions]
    predictions = {}
    for nr_trees in n_estimators:
        predictions[nr_trees] = [[] for _ in range(len(y))]

    # Train classifier on each bootstrap and score predictions
    for i, (train_index, test_index) in enumerate(shuffle_split.split(X)):

        # Initialize
        clf = model(random_state=0)
        if model.__class__.__name__ == 'RandomForestClassifier':
            clf.n_jobs = -1
        if model.__class__.__name__ != 'AdaBoostClassifier':
            clf.warm_start = warm_start

        prev_n_estimators = 0

        # Train incrementally
        for nr_trees in n_estimators:
            if model.__class__.__name__ == 'HistGradientBoostingClassifier':
                clf.max_iter = nr_trees
            else:
                clf.n_estimators = nr_trees

            # Fit and predict
            clf.fit(X[train_index], y[train_index])
            y_pred = clf.predict(X[test_index])
            for j,index in enumerate(test_index):
                predictions[nr_trees][index].append(y_pred[j])

    for nr_trees in n_estimators:
        # Compute bias, variance, error
        bias_sq = sum([ (1 - x.count(y[i])/len(x))**2 * len(x)/n_repeat
                       for i,x in enumerate(predictions[nr_trees])])
        var = sum([((1 - ((x.count(0)/len(x))**2 + (x.count(1)/len(x))**2))/2) * len(x)/n_repeat
                   for i,x in enumerate(predictions[nr_trees])])
        error = sum([ (1 - x.count(y[i])/len(x)) * len(x)/n_repeat
                     for i,x in enumerate(predictions[nr_trees])])

        bias_scores.append(bias_sq)
        var_scores.append(var)
        err_scores.append(error)

    plt.figure(figsize=(8*fig_scale,3*fig_scale))
    plt.suptitle(clf.__class__.__name__, fontsize=9)
    plt.plot(n_estimators, var_scores,label = "variance", lw=2*fig_scale )
    plt.plot(n_estimators, bias_scores,label = "bias^2", lw=2*fig_scale )
    plt.plot(n_estimators, err_scores,label = "error", lw=2*fig_scale  )
    plt.xscale('log',base=2)
    plt.xlabel("n_estimators", fontsize=9)
    plt.legend(loc="best", fontsize=7)
    plt.show()

plot_bias_variance_rf(RandomForestClassifier, cancer.data, cancer.target, warm_start=True)

# #### In practice
# 
# * Different implementations can be used. E.g. in scikit-learn:
#     * `BaggingClassifier`: Choose your own base model and sampling procedure
#     * `RandomForestClassifier`: Default implementation, many options
#     * `ExtraTreesClassifier`: Uses extremely randomized trees
# 
# * Most important parameters:
#     * `n_estimators` (>100, higher is better, but diminishing returns)
#         * Will start to underfit (bias error component increases slightly)
#     * `max_features`
#         * Defaults: $sqrt(p)$ for classification, $log2(p)$ for regression
#         * Set smaller to reduce space/time requirements
#     * parameters of trees, e.g. `max_depth`, `min_samples_split`,...
#         * Prepruning useful to reduce model size, but don't overdo it
# 
# * Easy to parallelize (set `n_jobs` to -1)
# * Fix `random_state` (bootstrap samples) for reproducibility

# ### Out-of-bag error
# * RandomForests don't need cross-validation: you can use the out-of-bag (OOB) error
# * For each tree grown, about 33% of samples are out-of-bag (OOB)
#     - Remember which are OOB samples for every model, do voting over these
# * OOB error estimates are great to speed up model selection
#     - As good as CV estimates, althought slightly pessimistic
# * In scikit-learn: `oob_error = 1 - clf.oob_score_`

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = make_classification(n_samples=500, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt", n_jobs=-1,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True, n_jobs=-1,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True, n_jobs=-1,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 175

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
plt.figure(figsize=(8*fig_scale,3*fig_scale))
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label, lw=2*fig_scale)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators", fontsize=7)
plt.ylabel("OOB error rate", fontsize=7)
plt.legend(loc="upper right", fontsize=7)
plt.show()

# ### Feature importance
# * RandomForests provide more reliable feature importances, based on many alternative hypotheses (trees)

forest = RandomForestClassifier(random_state=0, n_estimators=512, n_jobs=-1)
forest.fit(Xc_train, yc_train)
plot_feature_importances_cancer(forest)

# ### Other tips
# * Model calibration
#     * RandomForests are poorly calibrated.
#     * Calibrate afterwards (e.g. isotonic regression) if you aim to use probabilities
# * Warm starting
#     * Given an ensemble trained for $I$ iterations, you can simply add more models later
#     * You _warm start_ from the existing model instead of re-starting from scratch
#     * Can be useful to train models on new, closely related data
#         * Not ideal if the data batches change over time (concept drift)
#         * Boosting is more robust against this (see later)

# ### Strength and weaknesses
# 
# * RandomForest are among most widely used algorithms:
#     * Don't require a lot of tuning
#     * Typically very accurate
#     * Handles heterogeneous features well (trees)
#     * Implictly selects most relevant features
# * Downsides:
#     * less interpretable, slower to train (but parallellizable)
#     * don't work well on high dimensional sparse data (e.g. text)

# ## Adaptive Boosting (AdaBoost)
# 
# * Obtain different models by _reweighting_ the training data every iteration
#     * Reduce underfitting by focusing on the 'hard' training examples
# * Increase weights of instances misclassified by the ensemble, and vice versa
# * Base models should be simple so that different instance weights lead to different models
#     * Underfitting models: decision stumps (or very shallow trees)
#     * Each is an 'expert' on some parts of the data
# * Additive model: Predictions at iteration $I$ are sum of base model predictions
#     * In Adaboost, also the models each get a unique weight $w_i$
# $$f_I(\mathbf{x}) = \sum_{i=1}^I w_i g_i(\mathbf{x})$$
# * Adaboost minimizes exponential loss. For instance-weighted error $\varepsilon$:
# $$\mathcal{L}_{Exp} = \sum_{n=1}^N e^{\varepsilon(f_I(\mathbf{x}))}$$
# * By deriving $\frac{\partial \mathcal{L}}{\partial w_i}$ you can find that optimal $w_{i} =  \frac{1}{2}\log(\frac{1-\varepsilon}{\varepsilon})$

# ### AdaBoost algorithm
# * Initialize sample weights: $s_{n,0} = \frac{1}{N}$
# * Build a model (e.g. decision stumps) using these sample weights
# * Give the _model_ a weight $w_i$ related to its weighted error rate $\varepsilon$
# $$w_{i} =  \lambda\log(\frac{1-\varepsilon}{\varepsilon})$$
#     * Good trees get more weight than bad trees
#     * Logit function maps error $\varepsilon$ from [0,1] to weight in [-Inf,Inf] (use small minimum error)
#     * Learning rate $\lambda$ (shrinkage) decreases impact of individual classifiers
#         * Small updates are often better but requires more iterations
# * Update the sample weights
#     * Increase weight of incorrectly predicted samples:
# $s_{n,i+1} = s_{n,i}e^{w_i}$
#     * Decrease weight of correctly predicted samples:
# $s_{n,i+1} = s_{n,i}e^{-w_i}$
#     * Normalize weights to add up to 1
# * Repeat for $I$ iterations

# ### AdaBoost variants
# * Discrete Adaboost: error rate $\varepsilon$ is simply the error rate (1-Accuracy)
# * Real Adaboost: $\varepsilon$ is based on predicted class probabilities $\hat{p}_c$ (better)
# * AdaBoost for regression: $\varepsilon$ is either linear ($|y_i-\hat{y}_i|$), squared ($(y_i-\hat{y}_i)^2$), or exponential loss
# * GentleBoost: adds a bound on model weights $w_i$
# * LogitBoost: Minimizes logistic loss instead of exponential loss
# $$\mathcal{L}_{Logistic} = \sum_{n=1}^N log(1+e^{\varepsilon(f_I(\mathbf{x}))})$$

# ### Adaboost in action
# * Size of the samples represents sample weight
# * Background shows the latest tree's predictions

from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from sklearn.preprocessing import normalize

# Code adapted from https://xavierbourretsicotte.github.io/AdaBoost.html
def AdaBoost_scratch(X,y, M=10, learning_rate = 0.5):
    #Initialization of utility variables
    N = len(y)
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [],[],[],[],[]

    #Initialize the sample weights
    sample_weight = np.ones(N) / N
    sample_weight_list.append(sample_weight.copy())

    #For m = 1 to M
    for m in range(M):

        #Fit a classifier
        estimator = DecisionTreeClassifier(max_depth = 1, max_leaf_nodes=2)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)

        #Misclassifications
        incorrect = (y_predict != y)

        #Estimator error
        estimator_error = np.mean( np.average(incorrect, weights=sample_weight, axis=0))

        #Boost estimator weights
        estimator_weight =  learning_rate * np.log((1. - estimator_error) / estimator_error)

        #Boost sample weights
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))
        sample_weight *= np.exp(-estimator_weight * np.invert(incorrect * ((sample_weight > 0) | (estimator_weight < 0))))
        sample_weight /= np.linalg.norm(sample_weight)

        #Save iteration values
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    #Convert to np array for convenience
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    #Predictions
    preds = (np.array([np.sign((y_predict_list[:,point] * estimator_weight_list).sum()) for point in range(N)]))
    #print('Accuracy = ', (preds == y).sum() / N)

    return estimator_list, estimator_weight_list, sample_weight_list, estimator_error_list

def plot_decision_boundary(classifier, X, y, N = 10, scatter_weights = np.ones(len(y)) , ax = None, title=None ):
    '''Utility function to plot decision boundary and scatter plot of data'''
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

    # Get current axis and plot
    if ax is None:
        ax = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(X[:,0],X[:,1], c = y, cmap = cm_bright, s = scatter_weights * 1000, edgecolors='none')
    ax.set_xticks(())
    ax.set_yticks(())
    if title:
        ax.set_title(title, pad=1)

    # Plot classifier background
    if classifier is not None:
        xx, yy = np.meshgrid( np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))

        #Check what methods are available
        if hasattr(classifier, "decision_function"):
            zz = np.array( [classifier.decision_function(np.array([xi,yi]).reshape(1,-1)) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
        elif hasattr(classifier, "predict_proba"):
            zz = np.array( [classifier.predict_proba(np.array([xi,yi]).reshape(1,-1))[:,1] for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
        else:
            zz = np.array( [classifier(np.array([xi,yi]).reshape(1,-1)) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )

        # reshape result and plot
        Z = zz.reshape(xx.shape)

        ax.contourf(xx, yy, Z, 2, cmap='RdBu', alpha=.5, levels=[0,0.5,1])
        #ax.contour(xx, yy, Z, 2, cmap='RdBu', levels=[0,0.5,1])


from sklearn.datasets import make_circles
Xa, ya = make_circles(n_samples=400, noise=0.15, factor=0.5, random_state=1)
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

estimator_list, estimator_weight_list, sample_weight_list, estimator_error_list = AdaBoost_scratch(Xa, ya, M=60, learning_rate = 0.5)
current_ax = None
weight_scale = 1

@interact
def plot_adaboost(iteration=(0,60,1)):
    if iteration == 0:
        s_weights = (sample_weight_list[0,:] / sample_weight_list[0,:].sum() ) * weight_scale
        plot_decision_boundary(None, Xa, ya, N = 20, scatter_weights =s_weights)
    else:
        s_weights = (sample_weight_list[iteration,:] / sample_weight_list[iteration,:].sum() ) * weight_scale
        title = "Base model {}, error: {:.2f}, weight: {:.2f}".format(
            iteration,estimator_error_list[iteration-1],estimator_weight_list[iteration-1])
        plot_decision_boundary(estimator_list[iteration-1], Xa, ya, N = 20, scatter_weights =s_weights, ax=current_ax, title=title )


if not interactive:
    fig, axes = plt.subplots(2, 2, subplot_kw={'xticks': (()), 'yticks': (())}, figsize=(11*fig_scale, 6*fig_scale))
    weight_scale = 0.5
    for iteration, ax in zip([1, 5, 37, 59],axes.flatten()):
        current_ax = ax
        plot_adaboost(iteration)

# ### Examples

from sklearn.ensemble import AdaBoostClassifier
names = ["AdaBoost 1 tree", "AdaBoost 3 trees", "AdaBoost 100 trees"]

classifiers = [
    AdaBoostClassifier(n_estimators=1, random_state=0, learning_rate=0.5),
    AdaBoostClassifier(n_estimators=3, random_state=0, learning_rate=0.5),
    AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=0.5)
    ]

mglearn.plots.plot_classifiers(names, classifiers, figuresize=(6*fig_scale,3*fig_scale))

# ### Bias-Variance analysis
# * AdaBoost reduces bias (and a little variance)
#     * Boosting is a _bias reduction_ technique
# * Boosting too much will eventually increase variance

plot_bias_variance_rf(AdaBoostClassifier, cancer.data, cancer.target, warm_start=False)

# ## Gradient Boosting
# * Ensemble of models, each fixing the remaining mistakes of the previous ones
#     * Each iteration, the task is to predict the _residual error_ of the ensemble
# * Additive model: Predictions at iteration $I$ are sum of base model predictions
#     * Learning rate (or _shrinkage_ ) $\eta$: small updates work better (reduces variance)
# $$f_I(\mathbf{x}) = g_0(\mathbf{x}) + \sum_{i=1}^I \eta \cdot g_i(\mathbf{x}) = f_{I-1}(\mathbf{x}) + \eta \cdot g_I(\mathbf{x})$$
# * The _pseudo-residuals_ $r_i$ are computed according to differentiable loss function
#     * E.g. least squares loss for regression and log loss for classification
#     * Gradient descent: _predictions_ get updated step by step until convergence
# $$g_i(\mathbf{x}) \approx r_{i} = - \frac{\partial \mathcal{L}(y_i,f_{i-1}(x_i))}{\partial f_{i-1}(x_i)}$$
# * Base models $g_i$ should be low variance, but flexible enough to predict residuals accurately
#     * E.g. decision trees of depth 2-5

# ### Gradient Boosting Trees (Regression)
# 
# * Base models are regression trees, loss function is square loss: $\mathcal{L} = \frac{1}{2}(y_i - \hat{y}_i)^2$
# * The pseudo-residuals are simply the prediction errors for every sample:
# $$r_i = -\frac{\partial \mathcal{L}}{\partial \hat{y}} = -2 * \frac{1}{2}(y_i - \hat{y}_i) * (-1) =  y_i - \hat{y}_i$$
# * Initial model $g_0$ simply predicts the mean of $y$
# * For iteration $m=1..M$:
#     * For all samples i=1..n, compute pseudo-residuals $r_i = y_i - \hat{y}_i$
#     * Fit a new regression tree model $g_m(\mathbf{x})$ to $r_{i}$
#         * In $g_m(\mathbf{x})$, each leaf predicts the mean of all its values
#     * Update ensemble predictions $\hat{y} = g_0(\mathbf{x}) + \sum_{m=1}^M \eta \cdot g_m(\mathbf{x})$
# 
# * Early stopping (optional): stop when performance on validation set does not improve for $nr$ iterations
# 

# #### Gradient Boosting Regression in action
# * Residuals quickly drop to (near) zero

# Example adapted from Andreas Mueller
from sklearn.ensemble import GradientBoostingRegressor

# Make some toy data
def make_poly(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (x) ** 3
    y = (y_no_noise + rnd.normal(scale=3, size=len(x))) / 2
    return x.reshape(-1, 1), y
Xp, yp = make_poly()

# Train gradient booster and get predictions
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, random_state=0)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=61, learning_rate=.3, random_state=0).fit(Xp_train, yp_train)
gbrt.score(Xp_test, yp_test)

line = np.linspace(Xp.min(), Xp.max(), 1000)
preds = list(gbrt.staged_predict(line[:, np.newaxis]))
preds_train = [np.zeros(len(yp_train))] + list(gbrt.staged_predict(Xp_train))

# Plot
def plot_gradient_boosting_step(step, axes):
    axes[0].plot(Xp_train[:, 0], yp_train - preds_train[step], 'o', alpha=0.5, markersize=10*fig_scale)
    axes[0].plot(line, gbrt.estimators_[step, 0].predict(line[:, np.newaxis]), linestyle='-', lw=3*fig_scale)
    axes[0].plot(line, [0]*len(line), c='k', linestyle='-', lw=1*fig_scale)
    axes[1].plot(Xp_train[:, 0], yp_train, 'o',  alpha=0.5, markersize=10*fig_scale)
    axes[1].plot(line, preds[step], linestyle='-', lw=3*fig_scale)
    axes[1].vlines(Xp_train[:, 0], yp_train, preds_train[step+1])

    axes[0].set_title("Residual prediction step {}".format(step + 1), fontsize=9)
    axes[1].set_title("Total prediction step {}".format(step + 1), fontsize=9)
    axes[0].set_ylim(yp.min(), yp.max())
    axes[1].set_ylim(yp.min(), yp.max())
    plt.tight_layout();

@interact
def plot_gradient_boosting(step = (0, 60, 1)):
    fig, axes = plt.subplots(1, 2, subplot_kw={'xticks': (()), 'yticks': (())}, figsize=(10*fig_scale, 4*fig_scale))
    plot_gradient_boosting_step(step, axes)


if not interactive:
    fig, all_axes = plt.subplots(3, 2, subplot_kw={'xticks': (()), 'yticks': (())}, figsize=(10*fig_scale, 5*fig_scale))
    for i, s in enumerate([0,3,9]):
        axes = all_axes[i,:]
        plot_gradient_boosting_step(s, axes)

# ### GradientBoosting Algorithm (Classification)
# 
# * Base models are _regression_ trees, predict probability of positive class $p$
#     * For multi-class problems, train one tree per class
# * Use (binary) log loss, with true class $y_i \in {0,1}$: $\mathcal{L_{log}} = - \sum_{i=1}^{N} \big[ y_i log(p_i) + (1-y_i) log(1-p_i) \big] $
# * The pseudo-residuals are simply the difference between true class and predicted $p$:
# $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\partial \mathcal{L}}{\partial log(p_i)} = y_i - p_i$$
# * Initial model $g_0$ predicts $p = log(\frac{\#positives}{\#negatives})$
# * For iteration $m=1..M$:
#     * For all samples i=1..n, compute pseudo-residuals $r_i = y_i - p_i$
#     * Fit a new regression tree model $g_m(\mathbf{x})$ to $r_{i}$
#         * In $g_m(\mathbf{x})$, each leaf predicts $\frac{\sum_{i} r_i}{\sum_{i} p_i(1-p_i)}$
#     * Update ensemble predictions $\hat{y} = g_0(\mathbf{x}) + \sum_{m=1}^M \eta \cdot g_m(\mathbf{x})$
# * Early stopping (optional): stop when performance on validation set does not improve for $nr$ iterations

# #### Gradient Boosting Classification in action
# * Size of the samples represents the residual weights: most quickly drop to (near) zero

from sklearn.ensemble import GradientBoostingClassifier

Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya, random_state=0)
gbct = GradientBoostingClassifier(max_depth=2, n_estimators=60, learning_rate=.3, random_state=0).fit(Xa_train, ya_train)
gbct.score(Xa_test, ya_test)
preds_train_cl = [np.zeros(len(ya_train))] + list(gbct.staged_predict_proba(Xa_train))
current_gb_ax = None
weight_scale = 1

def plot_gb_decision_boundary(gbmodel, step, X, y, N = 10, scatter_weights = np.ones(len(y)) , ax = None, title = None ):
    '''Utility function to plot decision boundary and scatter plot of data'''
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

    # Get current axis and plot
    if ax is None:
        ax = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(X[:,0],X[:,1], c = y, cmap = cm_bright, s = scatter_weights * 40, edgecolors='none')
    ax.set_xticks(())
    ax.set_yticks(())
    if title:
        ax.set_title(title, pad='0.5')

    # Plot classifier background
    if gbmodel is not None:
        xx, yy = np.meshgrid( np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        zz = np.array( [list(gbmodel.staged_predict_proba(np.array([xi,yi]).reshape(1,-1)))[step][:,1] for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
        Z = zz.reshape(xx.shape)
        ax.contourf(xx, yy, Z, 2, cmap='RdBu', alpha=.5, levels=[0,0.5,1])


@interact
def plot_gboost(iteration=(1,60,1)):
    pseudo_residuals = np.abs(ya_train - preds_train_cl[iteration][:,1])
    title = "Base model {}, error: {:.2f}".format(iteration,np.sum(pseudo_residuals))
    plot_gb_decision_boundary(gbct, (iteration-1), Xa_train, ya_train, N = 20, scatter_weights =pseudo_residuals * weight_scale, ax=current_gb_ax, title=title )


if not interactive:
    fig, axes = plt.subplots(2, 2, subplot_kw={'xticks': (()), 'yticks': (())}, figsize=(10*fig_scale, 6*fig_scale))
    weight_scale = 0.3
    for iteration, ax in zip([1, 5, 17, 59],axes.flatten()):
        current_gb_ax = ax
        plot_gboost(iteration)

# #### Examples

names = ["GradientBoosting 1 tree", "GradientBoosting 3 trees", "GradientBoosting 100 trees"]

classifiers = [
    GradientBoostingClassifier(n_estimators=1, random_state=0, learning_rate=0.5),
    GradientBoostingClassifier(n_estimators=3, random_state=0, learning_rate=0.5),
    GradientBoostingClassifier(n_estimators=100, random_state=0, learning_rate=0.5)
    ]

mglearn.plots.plot_classifiers(names, classifiers, figuresize=(6*fig_scale,3*fig_scale))

# #### Bias-variance analysis
# * Gradient Boosting is very effective at reducing bias error
# * Boosting too much will eventually increase variance

# Note: I tried if HistGradientBoostingClassifier is faster. It's not.
# We're training many small models here and the thread spawning likely causes too much overhead
plot_bias_variance_rf(GradientBoostingClassifier, cancer.data, cancer.target, warm_start=True)

# #### Feature importance
# * Gradient Boosting also provide feature importances, based on many trees
# * Compared to RandomForests, the trees are smaller, hence more features have zero importance

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(Xc_train, yc_train)
plot_feature_importances_cancer(gbrt)

# #### Gradient Boosting: strengths and weaknesses
# * Among the most powerful and widely used models
# * Work well on heterogeneous features and different scales
# * Typically better than random forests, but requires more tuning, longer training
# * Does not work well on high-dimensional sparse data
# 
# Main hyperparameters:
# 
# * `n_estimators`: Higher is better, but will start to overfit
# * `learning_rate`: Lower rates mean more trees are needed to get more complex models
#     * Set `n_estimators` as high as possible, then tune `learning_rate`
#     * Or, choose a `learning_rate` and use early stopping to avoid overfitting
# * `max_depth`: typically kept low (<5), reduce when overfitting
# * `max_features`: can also be tuned, similar to random forests
# * `n_iter_no_change`: early stopping: algorithm stops if improvement is less than a certain tolerance `tol` for more than `n_iter_no_change` iterations.

# ### Extreme Gradient Boosting (XGBoost)
# 
# - Faster version of gradient boosting: allows more iterations on larger datasets
# - Normal regression trees: split to minimize squared loss of leaf predictions
#     - XGBoost trees only fit residuals: split so that residuals in leaf are more _similar_
# - Don't evaluate every split point, only $q$ _quantiles_ per feature (binning)
#     - $q$ is hyperparameter (`sketch_eps`, default 0.03)
# - For large datasets, XGBoost uses _approximate quantiles_
#     - Can be parallelized (multicore) by chunking the data and combining histograms of data
#     - For classification, the quantiles are weighted by $p(1-p)$
# - Gradient descent sped up by using the second derivative of the loss function
# - Strong regularization by pre-pruning the trees
# - Column and row are randomly subsampled when computing splits
# - Support for out-of-core computation (data compression in RAM, sharding,...)

# #### XGBoost in practice
# * Not part of scikit-learn, but `HistGradientBoostingClassifier` is similar
#     * binning, multicore,...
# * The `xgboost` python package is sklearn-compatible
#     * Install separately, `conda install -c conda-forge xgboost`
#     * Allows learning curve plotting and warm-starting
# * Further reading:
#     * [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster)
#     * [Paper](http://arxiv.org/abs/1603.02754)  
#     * [Video](https://www.youtube.com/watch?v=oRrKeUCEbq8)

# ### LightGBM
# Another fast boosting technique
# 
# * Uses _gradient-based sampling_
#     - use all instances with large gradients/residuals (e.g. 10% largest)
#     - randomly sample instances with small gradients, ignore the rest
#     - intuition: samples with small gradients are already well-trained.
#     - requires adapted information gain criterion
# * Does smarter encoding of categorical features

# ### CatBoost
# Another fast boosting technique
# 
# * Optimized for categorical variables
#     * Uses bagged and smoothed version of target encoding
# * Uses symmetric trees: same split for all nodes on a given level
#     * Can be much faster
# * Allows monotonicity constraints for numeric features
#     * Model must be be a non-decreasing function of these features
# * Lots of tooling (e.g. GPU training)

# ## Stacking
# 
# - Choose $M$ different base-models, generate predictions
# - Stacker (meta-model) learns mapping between predictions and correct label
#     - Can also be repeated: multi-level stacking
#     - Popular stackers: linear models (fast) and gradient boosting (accurate)
# - Cascade stacking: adds base-model predictions as extra features
# - Models need to be sufficiently different, be experts at different parts of the data
# - Can be _very_ accurate, but also very slow to predict
#     
# <img src="https://raw.githubusercontent.com/ML-course/master/master/notebooks/images/stacking.png" alt="ml" style="width: 500px;"/>

# ## Other ensembling techniques
# 
# - Hyper-ensembles: same basic model but with different hyperparameter settings
#     - Can combine overfitted and underfitted models
# - Deep ensembles: ensembles of deep learning models
# - Bayes optimal classifier: ensemble of all possible models (largely theoretic)
# - Bayesian model averaging: weighted average of probabilistic models, weighted by their posterior probabilities
# - Cross-validation selection: does internal cross-validation to select best of $M$ models
# - Any combination of different ensembling techniques

%%HTML
<style>
td {font-size: 20px}
th {font-size: 20px}
.rendered_html table, .rendered_html td, .rendered_html th {
    font-size: 20px;
}
</style>

# ### Algorithm overview
# 
# | Name | Representation | Loss function | Optimization | Regularization |
# |---|---|---|---|---|
# | Classification trees | Decision tree | Entropy / Gini index | Hunt's algorithm | Tree depth,... |
# | Regression trees | Decision tree | Square loss | Hunt's algorithm | Tree depth,... |
# | RandomForest | Ensemble of randomized trees | Entropy / Gini / Square | (Bagging) |  Number/depth of trees,... |
# | AdaBoost | Ensemble of stumps | Exponential loss | Greedy search |  Number/depth of trees,... |
# | GradientBoostingRegression | Ensemble of regression trees | Square loss | Gradient descent |  Number/depth of trees,... |
# | GradientBoostingClassification | Ensemble of regression trees | Log loss | Gradient descent |  Number/depth of trees,... |
# | XGBoost, LightGBM, CatBoost | Ensemble of XGBoost trees | Square/log loss | 2nd order gradients |  Number/depth of trees,... |
# | Stacking | Ensemble of heterogeneous models | / | / |  Number of models,... |

# ### Summary
# - Ensembles of voting classifiers improve performance
#     - Which models to choose? Consider bias-variance tradeoffs!
# - Bagging / RandomForest is a variance-reduction technique
#     - Build many high-variance (overfitting) models on random data samples
#         - The more different the models, the better
#     - Aggregation (soft voting) over many models reduces variance
#         - Diminishing returns, over-smoothing may increase bias error
#     - Parallellizes easily, doesn't require much tuning
# - Boosting is a bias-reduction technique
#     - Build low-variance models that correct each other's mistakes
#         - By reweighting misclassified samples: AdaBoost
#         - By predicting the residual error: Gradient Boosting
#     - Additive models: predictions are sum of base-model predictions
#         - Can drive the error to zero, but risk overfitting
#     - Doesn't parallelize easily. Slower to train, much faster to predict.
#         - XGBoost,LightGBM,... are fast and offer some parallellization
# - Stacking: learn how to combine base-model predictions
#     - Base-models still have to be sufficiently different




# # Comprehensive Tutorial on Linear and Polynomial Regression

# ## 1. Introduction: Solving Linear Equations with the Matrix Inverse Method

# At the heart of linear regression is the fundamental task of solving a system of linear equations. A system of linear equations can be represented in the matrix form **AX = B**, where **A** is a matrix of coefficients, **X** is a vector of unknown variables, and **B** is a vector of constants.

# If **A** is a square and invertible matrix, we can find a unique solution for **X** by pre-multiplying both sides of the equation by the inverse of **A** (**A⁻¹**):
# 
# **A⁻¹AX = A⁻¹B**
# 
# Since **A⁻¹A** is the identity matrix (**I**), we get:
# 
# **IX = A⁻¹B**
# 
# **X = A⁻¹B**

import numpy as np

# Define matrix A and vector B
A = np.array([[2, 1],
              [1, 3]])
B = np.array([8, 11])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Solve for X
X = np.dot(A_inv, B)

print(f"The solution is: {X}")

# ## 2. The Best Fit Line: Moore-Penrose Pseudoinverse for Non-Invertible Matrices

# In real-world datasets, the matrix **A** (our feature matrix) is often not square. It's typically a "tall" matrix, meaning it has more rows (observations) than columns (features). In such cases, the matrix is not invertible, and a unique solution to **AX = B** might not exist. This is where the concept of a "best fit" line comes in.

# We aim to find a solution that minimizes the error between our predictions (**AX**) and the actual values (**B**). This is achieved using the **Moore-Penrose Pseudoinverse** (often denoted as **A⁺**). The solution for the best fit line is given by:
# 
# **X = A⁺B**

# This method, also known as the Normal Equation, provides a closed-form solution to the linear regression problem. While it gives an exact solution, it can be computationally expensive. The calculation of the pseudoinverse often involves matrix multiplication and inversion, which has a time complexity of approximately **O(n³)**, where 'n' is the number of features. For datasets with a large number of features, this becomes a significant bottleneck. [2, 7, 10]

# Example with a tall matrix
A = np.array([[1, 1],
              [1, 2],
              [1, 3]])
B = np.array([2, 2.5, 3.5])

# Calculate the pseudoinverse of A
A_pseudo_inv = np.linalg.pinv(A)

# Solve for X
X = np.dot(A_pseudo_inv, B)

print(f"The coefficients for the best fit line are: {X}")

# ## 3. The Motivation for Gradient Descent

# Given the computational cost of the closed-form solution, we often turn to an iterative optimization algorithm called **Gradient Descent**. Instead of calculating the solution in a single step, Gradient Descent starts with random values for the coefficients and iteratively adjusts them to minimize a cost function (typically the Mean Squared Error in linear regression). [2, 16]

# The core idea is to take steps in the direction of the negative gradient of the cost function. The size of these steps is controlled by a parameter called the **learning rate**.

# ### Interactive Visualization of Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance

learning_rate = 0.1
n_iterations = 100

theta = np.random.randn(2,1)

fig, ax = plt.subplots()
ax.scatter(X, y)
line, = ax.plot(X, X_b.dot(theta), "r-")

def animate(i):
    global theta
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    line.set_ydata(X_b.dot(theta))
    return line,

animation = FuncAnimation(fig, animate, frames=n_iterations, blit=True)
plt.close()
HTML(animation.to_jshtml())

# ## 4. Linear Regression with `statsmodels` and `sklearn`

# Now, let's move from the theoretical underpinnings to practical implementation using popular Python libraries.

# ### 4.1. `statsmodels`

# `statsmodels` is a powerful library for statistical modeling in Python. It provides detailed statistical summaries of the models. [8, 9]

import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing)

X = housing.data
print(X)

y = housing.target
print(y)

# statsmodels requires the addition of a constant to the feature matrix
X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const)
results = model.fit()

print(results.summary())

# ### 4.2. `scikit-learn`

# `scikit-learn` is the go-to library for machine learning in Python. It offers a simple and consistent API for a wide range of models. [1, 3, 4, 5, 6]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficients: {lin_reg.coef_}")

y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# ## 5. Polynomial Regression: When Linear is Not Enough

# Sometimes, the relationship between features and the target variable is not linear. In such cases, linear regression can give suboptimal results. Polynomial regression can capture these non-linear relationships by adding polynomial features (e.g., x², x³, etc.) to the model. [13, 19, 22, 23]

import numpy as np
import matplotlib.pyplot as plt

# Generate some non-linear data
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)

# Reshape for sklearn
X = X[:, np.newaxis]

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Plot the results
X_new = np.linspace(-5, 5, 200).reshape(200, 1)
X_new_poly = poly_features.transform(X_new)
y_new_poly = poly_reg.predict(X_new_poly)
y_new_lin = lin_reg.predict(X_new)

plt.scatter(X, y, color='blue')
plt.plot(X_new, y_new_lin, color='red', linewidth=2, label='Linear Regression')
plt.plot(X_new, y_new_poly, color='green', linewidth=2, label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# ## 6. Exercises

# ### Exercise 1: Implement Linear Regression from Scratch

# 
# ### Define a class for our Linear Regression model
# CLASS ScratchLinearRegression:
# 
#     # The constructor initializes the learning rate and number of iterations.
#     # It also prepares variables for weights and bias.
#     INITIALIZE(learning_rate, n_iterations):
#         SET self.learning_rate = learning_rate
#         SET self.n_iterations = n_iterations
#         SET self.weights = None
#         SET self.bias = None
# 
#     # The fit method trains the model.
#     METHOD fit(X, y):
#         # Get the number of samples (rows) and features (columns) from the input data X.
#         n_samples, n_features = GET_SHAPE(X)
# 
#         # Initialize weights as a vector of zeros with a size equal to the number of features.
#         self.weights = CREATE_ZERO_VECTOR(n_features)
#         # Initialize bias to 0.
#         self.bias = 0
# 
#         # --- Gradient Descent Loop ---
#         # Loop for the specified number of iterations.
#         FOR i FROM 0 TO self.n_iterations:
#             # 1. Calculate the current predictions using the formula: y_pred = X * weights + bias
#             y_predicted = DOT_PRODUCT(X, self.weights) + self.bias
# 
#             # 2. Calculate the gradients (partial derivatives) of the cost function (MSE)
#             #    with respect to weights (dw) and bias (db).
#             #    Gradient for weights (dw) = (1/n_samples) * SUM((y_pred - y) * X)
#             #    Gradient for bias (db) = (1/n_samples) * SUM(y_pred - y)
#             dw = (1 / n_samples) * DOT_PRODUCT(TRANSPOSE(X), (y_predicted - y))
#             db = (1 / n_samples) * SUM(y_predicted - y)
# 
#             # 3. Update the weights and bias by taking a step in the opposite direction of the gradient.
#             #    The step size is controlled by the learning rate.
#             self.weights = self.weights - self.learning_rate * dw
#             self.bias = self.bias - self.learning_rate * db
# 
#     # The predict method uses the trained weights and bias to make :
#         # Calculate and return the predictions for new data lues.ll your model's predictions match the actual values.

# How to use the class
# 1. Create an instance of the ScratchLinearRegression model.
# 2. Call the fit() method with your training data (X_train, y_train).
# 3. Call the predict() method with your test data (X_test).
# 4. (Optional but recommended) Create a function to calculate the Mean Squared Error (MSE) to evaluate how well your model's predictions match the actual values.

# ### Exercise 2: Comparing Computational Complexity

# Q1.Compare the no. of oeprations needed to solve a linear regression problem using the closed-form (Moore-Penrose Pseudoinverse) and Gradient Descent methods for varying numbers of features. [2, 5, 10, 15, 20] under various settings of Gradient Descent(SGD, Mini-Batch, Batch).
# 

# Q2. The GD has to be performed till convergence i.e. updates are not greater than the order of 10^-3.


# 
# # 📘 Ridge and Lasso Regression Tutorial
# 
# In this tutorial, we will study **Ridge Regression** and **Lasso Regression**, two widely used techniques that extend Linear Regression by adding **regularization**.
# 
# Regularization is a method to prevent **overfitting** by penalizing large model coefficients. It adds a penalty term to the loss function of linear regression:
# 
# - **Linear Regression Loss (RSS):**
#   
#   \[ L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]
# 
# - **Ridge Regression (L2 penalty):**
#   
#   \[ L_{ridge} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2 \]
# 
# - **Lasso Regression (L1 penalty):**
#   
#   \[ L_{lasso} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j| \]
# 
# where:
# - $\alpha$ is the **regularization strength**
# - Ridge shrinks coefficients **towards zero** but never exactly zero
# - Lasso can shrink coefficients **exactly to zero**, thus performing **feature selection**
# 
# We will explore these methods step-by-step with examples and visualizations.
# 

# # **Introduction to Regularization**
# 
# In machine learning, especially in regression, we often face the problem of **overfitting**. This happens when our model learns the training data too well, including its noise, and fails to generalize to new, unseen data.
# 
# **Regularization** is a set of techniques used to combat overfitting by adding a penalty term to the model's loss function. This penalty discourages the model from learning overly complex patterns or assigning too much importance to any single feature. The two most common types of regularization for linear models are Ridge (L2) and Lasso (L1).
# 

# 
# ## 🔍 Geometric Intuition
# 
# - **Ridge Regression (L2 penalty)** adds a penalty proportional to the square of coefficients. 
#   - Geometrically, it constrains the coefficients to lie within a **circle (L2 ball)**. 
#   - This leads to smooth shrinkage of all coefficients.
# 
# - **Lasso Regression (L1 penalty)** adds a penalty proportional to the absolute value of coefficients.
#   - Geometrically, it constrains the coefficients to lie within a **diamond (L1 ball)**. 
#   - Because of the shape, solutions often lie on the **axes**, meaning some coefficients are exactly zero → **sparse models**.
# 
# 📌 In practice:
# - Use **Ridge** when you believe all features are relevant but want to reduce their impact to avoid overfitting.  
# - Use **Lasso** when you suspect many features are irrelevant and want the model to automatically perform **feature selection**.
# 

#Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generating synthetic dataset
X, y = make_regression(n_samples=200, n_features=1, n_informative=1, noise=100, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training data (X_train):", X_train.shape)
print("Shape of testing data (X_test):", X_test.shape)


# Visualize the Dataset

# Let's plot our simple dataset to see the relationship between X and y.
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='royalblue', label='Training Data')
plt.xlabel("Feature (X)", fontsize=12)
plt.ylabel("Target (y)", fontsize=12)
plt.title("Synthetic Dataset", fontsize=15)
plt.legend()
plt.show()

# **Standard Linear Regression**
# 
# First train a standard Linear Regression model. This will serve as our baseline to compare against. The objective of Linear Regression is to minimize the Residual Sum of Squares (RSS).
# 
# **Loss Function (RSS):** `L = Σ(yᵢ - ŷᵢ)²`

# Initialize and train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


#  **Ridge Regression (L2 Regularization)**
# 
#  Ridge Regression adds a penalty term to the loss function that is proportional to the **square of the magnitude of the coefficients**. This is also known as the L2 norm.
# 
#  **Loss Function:** `L = Σ(yᵢ - ŷᵢ)² + α * Σ(βⱼ)²`
# 
#  - **βⱼ**: The coefficient for the j-th feature.
#  - **α (alpha)**: The regularization strength. It's a hyperparameter we need to tune.
#    - If `α = 0`, Ridge Regression is the same as Linear Regression.
#    - As `α` increases, the coefficients are pushed closer to zero, but they never become exactly zero. This is called "coefficient shrinkage."

# Initialize and train the Ridge model
ridge_reg = Ridge(alpha=200.0)
ridge_reg.fit(X_train, y_train)

# **Lasso Regression (L1 Regularization)**
# 
# Lasso (Least Absolute Shrinkage and Selection Operator) Regression adds a penalty term proportional to the **absolute value of the magnitude of the coefficients** (L1 norm).
# 
# **Loss Function:** `L = Σ(yᵢ - ŷᵢ)² + α * Σ|βⱼ|`
# 
# The key difference from Ridge is that the L1 penalty can force some coefficients to be **exactly zero**. This makes Lasso useful for **feature selection**, as it effectively removes irrelevant features from the model.
# 

# Initialize and train the Lasso model
lasso_reg = Lasso(alpha=20.0)
lasso_reg.fit(X_train, y_train)

# Comparing the Regression Lines
# Get coefficients
linear_coeffs = lin_reg.coef_
ridge_coeffs = ridge_reg.coef_
lasso_coeffs = lasso_reg.coef_

# Now, let's visualize the regression lines learned by each of our three models.

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, color='royalblue', alpha=0.8, label='Training Data')

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(x_range, lin_reg.predict(x_range), color='red', linewidth=2, label=f'Linear Regression (coef: {linear_coeffs[0]:.2f})')
plt.plot(x_range, ridge_reg.predict(x_range), color='seagreen', linewidth=2, label=f'Ridge Regression (coef: {ridge_coeffs[0]:.2f})')
plt.plot(x_range, lasso_reg.predict(x_range), color='coral', linewidth=2, label=f'Lasso Regression (coef: {lasso_coeffs[0]:.2f})')

plt.xlabel("Feature (X)", fontsize=12)
plt.ylabel("Target (y)", fontsize=12)
plt.title("Comparison of Regression Lines", fontsize=15)
plt.legend(fontsize=11)
plt.show()

print("\nCoefficients from Linear Regression:\n", np.round(linear_coeffs, 2))
print("\nCoefficients from Ridge Regression (alpha=100):\n", np.round(ridge_coeffs, 2))
print("\nCoefficients from Lasso Regression (alpha=10):\n", np.round(lasso_coeffs, 2))

#  Evaluating Model Performance
# Let's see how the models perform on the unseen test data using Mean Squared Error (MSE).

# Make predictions on the test set
y_pred_linear = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)

# Calculate Mean Squared Error
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print("\n--- Model Performance on Test Data ---")
print(f"Linear Regression MSE: {mse_linear:.2f}")
print(f"Ridge Regression MSE:  {mse_ridge:.2f}")
print(f"Lasso Regression MSE:  {mse_lasso:.2f}")

# **The Effect of Alpha**
# 
# The choice of `alpha` is critical. Let's quickly see how it affects the single coefficient in our Ridge and Lasso models.

alphas = np.logspace(-1, 2.5, 100) # Use a wider range of alphas
ridge_coefs_path = []
lasso_coefs_path = []

for a in alphas:
    ridge = Ridge(alpha=a, max_iter=100).fit(X_train, y_train)
    lasso = Lasso(alpha=a, max_iter=100).fit(X_train, y_train)
    ridge_coefs_path.append(ridge.coef_)
    lasso_coefs_path.append(lasso.coef_)

# Convert to numpy arrays for easier plotting
ridge_coefs_path = np.array(ridge_coefs_path)
lasso_coefs_path = np.array(lasso_coefs_path)

# Plotting the coefficient paths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ridge plot
for i in range(ridge_coefs_path.shape[1]):
    ax1.plot(alphas, ridge_coefs_path[:, i], 'o-')
ax1.set_xscale('log')
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Coefficients')
ax1.set_title('Ridge Coefficients as a function of Alpha')
ax1.grid(True)


# Lasso plot
for i in range(lasso_coefs_path.shape[1]):
    ax2.plot(alphas, lasso_coefs_path[:, i], 'o-')
ax2.set_xscale('log')
ax2.set_xlabel('Alpha')
ax2.set_ylabel('Coefficients')
ax2.set_title('Lasso Coefficients as a function of Alpha')
ax2.grid(True)

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression

# Generate synthetic dataset
X, y, coef = make_regression(n_samples=100, n_features=20, n_informative=10, noise=10, coef=True, random_state=42)

alphas = np.logspace(-3, 2, 50)
ridge_coefs = []
lasso_coefs = []

for a in alphas:
    ridge = Ridge(alpha=a).fit(X, y)
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=a, max_iter=5000).fit(X, y)
    lasso_coefs.append(lasso.coef_)

plt.figure(figsize=(14,6))

# Ridge coefficients path
plt.subplot(1,2,1)
plt.plot(alphas, ridge_coefs)
plt.xscale('log')
plt.title("Ridge: Coefficient Paths")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient value")

# Lasso coefficients path
plt.subplot(1,2,2)
plt.plot(alphas, lasso_coefs)
plt.xscale('log')
plt.title("Lasso: Coefficient Paths")
plt.xlabel("Alpha (log scale)")

plt.show()


# 
# # 📝 Assignments
# 
# ### **Assignment 1: Ridge Regression**
# - Implement Ridge Regression *without using scikit-learn’s Ridge class*.
# - Start from the normal equation of Linear Regression and modify it to include the L2 penalty.
# - Compare your manual implementation with `sklearn.linear_model.Ridge` on a dataset.
# 
# ---
# 
# ### **Assignment 2: Lasso Regression and Feature Selection**
# - Create a dataset with at least 50 features where only 5 are truly informative (you can use `sklearn.datasets.make_regression`).
# - Train a Lasso Regression model with different values of `alpha`.
# - Plot how many coefficients are exactly zero as alpha increases.
# - Explain how Lasso can be used for **automatic feature selection**.
# 


# # Linear Models using neural networks

# for tutorial, we will be creating our own dataset, for exercise we will use an actual one
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)

n = 100
area = np.random.uniform(800, 2500, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(18, 45, n)


price = 50 * area + 10 * bedrooms - 20 * age + np.random.normal(0, 35, n)

data = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'age': age, 'price': price})
data.head()


# A linear model tries to learn:
# 
# price=w1xarea + w2xbedrooms + w3xage+ b
# 
# Each input feature gets its own weight.
# The model's job is to find the best weights and bias that minimize prediction error.

X = torch.tensor(data[['area', 'bedrooms', 'age']].values, dtype=torch.float32)
y = torch.tensor(data[['price']].values, dtype=torch.float32)
#There are 3 features, so the input shape for each example is [3].

model = nn.Linear(in_features=3, out_features=1)
print(model)


criterion = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.1)# try different lr, like 0.01 to 2.0


epochs = 100
#try different training step and see how parmeters come close
# what is shown down is after chaning epochs and lr, in different combinations
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.2f}')


# ## lets see what it learned

for name, param in model.named_parameters():
    print(name, param.data)


with torch.no_grad():
    y_pred = model(X)
test_loss = criterion(y_pred, y)
print(f"Test MSE: {test_loss.item():.4f}")

with torch.no_grad():
    y_pred = model(X)

plt.figure(dpi=100)
plt.scatter(y, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Model Predictions")
plt.show()
# why linear and diagonal?

# # what we did above was a single layer lets make it deep

price = (
    50 * area
    + 10 * bedrooms
    - 2 * (age**2)  # nonlinear term
    + np.random.normal(0, 20, n)
)

X = torch.tensor(np.column_stack([area, bedrooms, age]), dtype=torch.float32)
y = torch.tensor(price.reshape(-1, 1), dtype=torch.float32)
data = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'age': age, 'price': price})
data.head()

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

deep_model = DeepModel()
print(deep_model)


criterion = nn.L1Loss()
optimizer = optim.Adam(deep_model.parameters(), lr=0.01)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = deep_model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.2f}')


for name, param in deep_model.named_parameters():
    print(name, param.data)


with torch.no_grad():
    y_pred = deep_model(X)
test_loss = criterion(y_pred, y)
print(f"Test MSE: {test_loss.item():.4f}")

with torch.no_grad():
    y_pred = deep_model(X)

plt.figure(dpi=100)
plt.scatter(y, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Deep Neural Network Predictions")
plt.show()


# # Exercise: Neural Network on Iris Dataset
# 1. Iris dataset is already loaded for you, it  and check the shapes.
# 2. Check if you need to preprocess the inputs (e.g., scaling).
# 3. Split the data into training and test sets.
# 4. Define a neural network appropriate for the task.
# 5. Use a suitable loss function(crossentropyloss) and an optimizer.
# 6. Train the model on the training set.
# 7. Evaluate it on the test set and report accuracy.

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target





import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-whitegrid')
print(f"Using TensorFlow version: {tf.__version__}")

# Load a subset of Fashion MNIST data for demonstration
(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
y_train = to_categorical(y_train)

# Use a small subset for quick training
x_sample, _, y_sample, _ = train_test_split(x_train, y_train, train_size=5000, stratify=np.argmax(y_train, axis=1))



def build_and_train_model(initializer, name):
    """A helper function to build, compile, and train a model."""
    model = Sequential([
        Flatten(input_shape=(784,)),
        Dense(128, activation='relu', kernel_initializer=initializer),
        Dense(64, activation='relu', kernel_initializer=initializer),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"\n--- Training with {name} Initializer ---")
    history = model.fit(x_sample, y_sample, epochs=10, batch_size=128, verbose=0, validation_split=0.2)
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {val_acc:.4f}")
    return history

# Train with Zero Initializer
history_zero = build_and_train_model(tf.keras.initializers.Zeros(), "Zeros")

def plot_activation_distributions(initializer, activation_fn):
    """Plots the mean and stddev of activations across layers."""
    input_data = np.random.randn(1000, 100) # Sample input
    
    # Use the Keras Functional API for a robust solution
    # 1. Define an explicit Input tensor
    inputs = tf.keras.Input(shape=(100,))
    
    # 2. Stack layers, connecting them manually
    x = inputs
    for _ in range(10):
        x = Dense(100, kernel_initializer=initializer, activation=activation_fn)(x)
        
    # 3. Create a model from the defined input and output tensors
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    # Now that the graph is explicitly defined, we can reliably get the outputs of each layer
    layer_outputs = [layer.output for layer in model.layers[1:]] # Skip the InputLayer
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(input_data, verbose=0)
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Activation Distributions with '{initializer.__class__.__name__}' and '{activation_fn}'", fontsize=16)
    
    # Plot Means and Std Devs
    means = [act.mean() for act in activations]
    stds = [act.std() for act in activations]
    
    plt.subplot(1, 2, 1)
    plt.plot(means, 'o-')
    plt.title('Mean of Activations per Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean')
    
    plt.subplot(1, 2, 2)
    plt.plot(stds, 'o-')
    plt.title('Std Dev of Activations per Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation')
    plt.show()

# Demonstrate with large random weights and tanh activation
large_random_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
plot_activation_distributions(large_random_init, 'tanh')

print("--- Using Glorot with tanh ---")
plot_activation_distributions(tf.keras.initializers.GlorotNormal(), 'tanh')

print("\n--- Using He with ReLU ---")
plot_activation_distributions(tf.keras.initializers.HeNormal(), 'relu')

history_glorot = build_and_train_model(tf.keras.initializers.GlorotUniform(), "Glorot Uniform")
history_he = build_and_train_model(tf.keras.initializers.HeNormal(), "He Normal")

plt.figure(figsize=(12, 6))
plt.plot(history_zero.history['val_accuracy'], label='Zeros Initializer')
plt.plot(history_glorot.history['val_accuracy'], label='Glorot Uniform Initializer')
plt.plot(history_he.history['val_accuracy'], label='He Normal Initializer')
plt.title('Impact of Weight Initialization on Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()

def beale_function(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def rosenbrock_function(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def plot_optimizer_paths(loss_fn, title, x_range, y_range, start_point):
    """A helper function to visualize optimizer paths on a 2D loss surface."""
    x = np.linspace(x_range[0], x_range[1], 300)
    y = np.linspace(y_range[0], y_range[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = loss_fn(X, Y)

    optimizers_to_test = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=0.005),
        'Momentum': tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
        'Adam': tf.keras.optimizers.Adam(learning_rate=0.01),
    }

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), norm=plt.cm.colors.LogNorm(), cmap='viridis')
    plt.title(f'Optimizer Paths on {title}')
    plt.xlabel('x')
    plt.ylabel('y')

    for name, optimizer in optimizers_to_test.items():
        x_var = tf.Variable(start_point[0], dtype=tf.float32)
        y_var = tf.Variable(start_point[1], dtype=tf.float32)
        path = [(x_var.numpy(), y_var.numpy())]

        for _ in range(150):
            with tf.GradientTape() as tape:
                loss = loss_fn(x_var, y_var)
            grads = tape.gradient(loss, [x_var, y_var])
            optimizer.apply_gradients(zip(grads, [x_var, y_var]))
            path.append((x_var.numpy(), y_var.numpy()))

        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', label=name, markersize=3, alpha=0.8)

    plt.legend()
    plt.show()

# Visualize on Rosenbrock's function
plot_optimizer_paths(rosenbrock_function, "Rosenbrock's Function", 
                     x_range=[-2, 2], y_range=[-1, 3], start_point=[-1.5, 2.5])

def train_with_optimizer(optimizer_name, optimizer_instance):
    """Helper to train a model with a specific optimizer."""
    model = Sequential([
        Flatten(input_shape=(784,)),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer_instance,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"\n--- Training with {optimizer_name} ---")
    history = model.fit(x_sample, y_sample, epochs=15, batch_size=128, verbose=0, validation_split=0.2)
    return history

# Train with different optimizers
history_sgd = train_with_optimizer('SGD', tf.keras.optimizers.SGD())
history_momentum = train_with_optimizer('Momentum', tf.keras.optimizers.SGD(momentum=0.9))
history_rmsprop = train_with_optimizer('RMSprop', tf.keras.optimizers.RMSprop())
history_adam = train_with_optimizer('Adam', tf.keras.optimizers.Adam())

# Plot the results
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(history_sgd.history['val_accuracy'], label='SGD')
plt.plot(history_momentum.history['val_accuracy'], label='SGD with Momentum')
plt.plot(history_rmsprop.history['val_accuracy'], label='RMSprop')
plt.plot(history_adam.history['val_accuracy'], label='Adam')
plt.title('Optimizer Comparison: Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['val_loss'], label='SGD')
plt.plot(history_momentum.history['val_loss'], label='SGD with Momentum')
plt.plot(history_rmsprop.history['val_loss'], label='RMSprop')
plt.plot(history_adam.history['val_loss'], label='Adam')
plt.title('Optimizer Comparison: Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()




"""
============================================================
Optimizers and Initialization in Deep Learning
============================================================

This script demonstrates:
1. Various Gradient Descent Optimizers
   - SGD
   - SGD with Momentum
   - Nesterov Accelerated Gradient (NAG)
   - Adagrad
   - RMSProp
   - Adam
   - (optional) AdamW, Nadam

2. Neuron Initialization Schemes
   - Random / Zero / Normal
   - Xavier (Glorot)
   - He
   - LeCun

"""

# =======================
# Imports
# =======================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# For reproducibility
np.random.seed(42)

# =======================
# 1. Define the Function to Optimize
# =======================
def func(x, y):
    return 0.5 * x**2 + 5 * y**2

def grad(x, y):
    return np.array([x, 10 * y])

# =======================
# 2. Optimizer Implementations
# =======================

class SGD:
    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads

class SGDMomentum:
    def __init__(self, lr=0.05, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros_like(grads := np.zeros(2))

    def step(self, params, grads):
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v

class NAG:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(2)

    def step(self, params, grads_fn):
        look_ahead = params + self.momentum * self.v
        grads = grads_fn(*look_ahead)
        self.v = self.momentum * self.v - self.lr * grads
        return params + self.v


class Adagrad:
    def __init__(self, lr=0.3, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cache = np.zeros(2)

    def step(self, params, grads):
        self.cache += grads ** 2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.eps)

class RMSProp:
    def __init__(self, lr=0.05, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = np.zeros(2)

    def step(self, params, grads):
        self.cache = self.beta * self.cache + (1 - self.beta) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.eps)

class Adam:
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def optimize(optimizer, name, steps=50, start=np.array([2.0, 2.0])):
    params = start.copy()
    trajectory = [params.copy()]
    for i in range(steps):
        if isinstance(optimizer, NAG):
            params = optimizer.step(params, grad)
        else:
            g = grad(*params)
            params = optimizer.step(params, g)
        trajectory.append(params.copy())
    trajectory = np.array(trajectory)

    # Plot
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.figure(figsize=(5, 4))
    plt.contour(X, Y, Z, levels=30, cmap=cm.viridis)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Path')
    plt.title(f"{name} Optimization Path")
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend()
    plt.show()

print("Visualizing different optimizers on an anisotropic function...\n")

optimize(SGD(lr=0.05), "SGD")
optimize(SGDMomentum(lr=0.05), "SGD + Momentum")
optimize(NAG(lr=0.05), "Nesterov Accelerated Gradient")
optimize(Adagrad(lr=0.3), "Adagrad")
optimize(RMSProp(lr=0.05), "RMSProp")
optimize(Adam(lr=0.05), "Adam")

def loss_plot(optimizer, name, steps=50, start=np.array([2.0, 2.0]), plot_path=True):
    """
    Runs optimization using the given optimizer.
    Returns the loss trajectory and optionally plots optimization path.
    """
    params = start.copy()
    trajectory = [params.copy()]
    losses = [func(*params)]

    for i in range(steps):
        if isinstance(optimizer, NAG):
            params = optimizer.step(params, grad)
        else:
            g = grad(*params)
            params = optimizer.step(params, g)

        trajectory.append(params.copy())
        losses.append(func(*params))

    trajectory = np.array(trajectory)

    # Plot optimization path (contour)
    if plot_path:
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        plt.figure(figsize=(5, 4))
        plt.contour(X, Y, Z, levels=30, cmap=cm.viridis)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Path')
        plt.title(f"{name} Optimization Path")
        plt.xlabel('x'); plt.ylabel('y')
        plt.legend()
        plt.show()

    return np.array(losses)


# Compare convergence rates
optimizers = {
    "SGD": SGD(lr=0.05),
    "Momentum": SGDMomentum(lr=0.05),
    "NAG": NAG(lr=0.05),
    "Adagrad": Adagrad(lr=0.3),
    "RMSProp": RMSProp(lr=0.05),
    "Adam": Adam(lr=0.05),
}

loss_histories = {}

# Disable contour plot for faster run when comparing
for name, opt in optimizers.items():
    print(f"Running {name}...")
    loss_histories[name] = loss_plot(opt, name, steps=60, plot_path=False)

# Plot loss convergence
plt.figure(figsize=(8, 6))
for name, losses in loss_histories.items():
    plt.plot(losses, label=name)
plt.yscale('log')
plt.title("Loss Convergence Comparison")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.grid(True)
plt.show()


# =======================
# 5. Initialization Demonstration
# =======================
def visualize_initializations():
    n_neurons = 256
    fan_in = 256
    fan_out = 128

    # Different initialization schemes
    initializations = {
        "Random Normal": np.random.randn(n_neurons),
        "Glorot (Xavier)": np.random.randn(n_neurons) * np.sqrt(2.0 / (fan_in + fan_out)),
        "He": np.random.randn(n_neurons) * np.sqrt(2.0 / fan_in),
        "LeCun": np.random.randn(n_neurons) * np.sqrt(1.0 / fan_in)
    }

    plt.figure(figsize=(10, 6))
    for name, vals in initializations.items():
        plt.hist(vals, bins=50, alpha=0.6, label=name, density=True)
    plt.title("Weight Initialization Distributions")
    plt.legend()
    plt.show()


visualize_initializations()

# =======================
# 6. Student Exercises
# =======================
print("\n" + "="*60)
print("EXERCISES:")
print("="*60)
print("""
1️⃣  Modify the learning rate and momentum values for SGD, Momentum, and NAG.
    - Observe how convergence changes. What happens if learning rate is too high or too low?

2️⃣  Implement and visualize the AdamW optimizer:
    - Hint: It's similar to Adam, but adds weight decay.

3️⃣  For the initialization part:
    - Write a small neural network layer and test different initialization schemes.
    - Plot the output variance after 1 forward pass for each initialization type.
    - Which one maintains stable activations?

(Bonus)
4️⃣  Extend the visualization to other functions.
    - How do optimizers behave on those landscapes?
""")





# # Convolutional Neural Networks (CNNs) — Tutorial
# 
# A hands-on, step-by-step Python notebook style tutorial that starts with simple CNNs and culminates with classic architectures like AlexNet, VGG and brief notes on ResNet/Inception. Contains runnable code cells (Keras + PyTorch), explanations, visualizations and exercises.

# ## Table of contents
# 
# 1. Introduction & learning goals
# 2. Prerequisites & environment
# 3. Key concepts (conv layer, kernels, stride, padding, pooling, activations, receptive field, BN, dropout)
# 4. Simple CNN on MNIST (Keras) — build, train, visualize
# 5. Deeper CNN on CIFAR-10 (Keras) — augmentation, callbacks, regularization
# 6. PyTorch: Simple CNN + training loop + debugging tips
# 7. Transfer Learning: using pretrained models (PyTorch) — feature-extractor vs fine-tune
# 8. Implementing AlexNet (PyTorch) — full architecture, tips, training on CIFAR-10 / Tiny ImageNet
# 9. Overview of later architectures: VGG, Inception, ResNet (intuition & code pointers)
# 10. Model sizing, FLOPs & parameter counting (how to think about complexity)
# 11. Common pitfalls, debugging & performance tuning
# 12. Exercises and extensions
# 13. References & further reading

# ## 1. Introduction & learning goals
# 
# By the end of this notebook you will be able to:
# 
# * Understand convolution, pooling, and how CNNs process images.
# * Build a small CNN in Keras and PyTorch and train it on MNIST/CIFAR-10.
# * Apply data augmentation and regularization to improve generalization.
# * Use pretrained networks and understand transfer learning choices.
# * Implement AlexNet in PyTorch and know how to expand toward VGG/ResNet.

# ## 2. Prerequisites & environment
# 
# * Python 3.8+
# * GPU recommended (for deeper nets like AlexNet) but CPU works for small experiments.
# * Libraries used in code examples below:
# 
#   * TensorFlow / Keras (`tensorflow>=2.10`)
#   * PyTorch (`torch`, `torchvision`)
#   * NumPy, matplotlib, seaborn (optional)
# 
# Install (example):
# 
# ```bash
# pip install numpy matplotlib seaborn tensorflow torchvision torch tqdm scikit-learn
# ```
# 
# Open a Colab or local Jupyter notebook and run cells as you go.

# ## 3. Quick conceptual primer
# 
# ### Convolutional layer
# 
# * Input: H x W x C_in feature map.
# * Kernel/filter: k x k x C_in; slides using stride `s` and padding `p`.
# * Output: H_out x W_out x C_out where C_out = number of filters.
# 
# Formula for output size (single spatial axis):
# 
# ```
# out = floor((in + 2*p - k) / s) + 1
# ```
# 
# ### Pooling
# 
# * Downsamples spatial resolution (max pooling, average pooling).
# * Reduces parameters and adds spatial invariance.
# 
# ### Activation
# 
# * ReLU commonly used: `f(x)=max(0,x)`
# * LeakyReLU, SELU, GELU also used depending on architecture.
# 
# ### BatchNorm
# 
# * Normalizes activations per-batch to stabilize and accelerate training.
# 
# ### Dropout
# 
# * Randomly zeroes activations during training for regularization.
# 
# ### Receptive field
# 
# * The region in input image that affects a particular activation in deeper layers—grows with depth and kernel sizes.
# 

# ## 4. Simple CNN on MNIST (Keras)

# Keras MNIST example
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# scale and reshape
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# One-hot labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Model
model = models.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,1)),
    layers.MaxPool2D(2),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate
print(model.evaluate(x_test, y_test, verbose=2))

# Plot training curves
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# **Notes**: This small net reaches ~99% train accuracy and ~99% test accuracy on MNIST with more epochs and minor tuning.

# Visualize filters of the first conv layer in Keras
weights = model.layers[0].get_weights()[0]
print(weights.shape) # (3,3,1,16) for first conv layer


fig, axs = plt.subplots(4,4, figsize=(6,6))
for i in range(16):
  f = weights[:,:,:,i]
  axs[i//4, i%4].imshow(f.squeeze(), cmap='gray')
  axs[i//4, i%4].axis('off')
plt.show()

# ## 5. Deeper CNN on CIFAR-10 (Keras) — augmentation, callbacks
# 
# CIFAR-10 images are 32x32x3; more challenging. Use augmentation and weight decay.

import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

# Check GPU
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Optional: enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

# Model
def make_cifar_model():
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax', dtype='float32')(x)  # force float32 output
    return models.Model(inputs, outputs)

with tf.device('/GPU:0'):
    model = make_cifar_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
]

# Train
train_gen = datagen.flow(x_train, y_train, batch_size=512, subset=None)
model.fit(train_gen, epochs=10, validation_data=(x_test, y_test), callbacks=callbacks)


# **Practical tips:** use `LearningRateScheduler` or `CosineDecay`, monitor validation accuracy, and consider SGD with momentum for better generalization.

# ## 6. PyTorch: Simple CNN + training loop
# 



# PyTorch simple CNN (CIFAR-10)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop (one epoch example)
model.train()
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, loss {running_loss/len(trainloader):.4f}')

# **Debugging tips**: start with a very small dataset, ensure loss decreases, use gradient norm clipping if exploding.

# ## 7. Transfer learning (PyTorch)
# 
# Use `torchvision.models` for pretrained AlexNet/VGG/ResNet.

import torchvision.models as models
alex = models.alexnet(pretrained=True)  # loads pretrained weights
# Option A: feature extraction (freeze backbone)
for p in alex.features.parameters():
    p.requires_grad = False
# Replace final classifier for CIFAR-10
alex.classifier[6] = nn.Linear(alex.classifier[6].in_features, 10)
alex = alex.to(device)


# * Feature-extractor: freeze backbone, train new head (fast, less data required).
# * Fine-tune: unfreeze some later layers and train with a lower LR.

# ## 8. Implementing AlexNet (PyTorch)
# 
# AlexNet (Krizhevsky et al., 2012) — 5 conv layers + 3 fully connected layers (original used local response norm and dropouts)
# 

# AlexNet implementation (PyTorch)
import torch.nn.functional as F
class AlexNetCustom(nn.Module):
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


# # Example: for CIFAR-10 (32x32) you'd adapt the first conv and pooling or upsample images to 224x224.
# 
# **Notes for training AlexNet**:
# 
# * Original AlexNet expects 224x224 inputs (ImageNet). For CIFAR-10, you can upsample to 224x224 or modify kernel/stride/pool sizes.
# * Consider using pretrained weights and fine-tuning.
# * Use data augmentation heavily and training schedules (SGD + momentum + LR decay).

from tqdm import tqdm

transform_train = transforms.Compose([
transforms.Resize(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

model = AlexNetCustom(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


for epoch in range(10):
  model.train()
  running_loss, correct, total = 0, 0, 0
  for inputs, labels in tqdm(trainloader):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()


    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()


  scheduler.step()
  print(f'Epoch {epoch+1} | Loss: {running_loss/len(trainloader):.3f} | Acc: {100*correct/total:.2f}%')

# ## 9. Quick overview: VGG, Inception, ResNet
# 
# * **VGG**: deep stacks of 3x3 convs; simple and uniform; heavy parameter count.
# * **Inception**: mixed kernel sizes in parallel (1x1, 3x3, 5x5) with dimension reduction via 1x1 convs.
# * **ResNet**: residual connections `y = F(x) + x` allow training very deep nets by solving degradation problem.
# 
# When progressing from AlexNet to these, note the evolution:
# 
# * AlexNet: larger kernels, large FC layers.
# * VGG: smaller kernels but deeper.
# * Inception: width and multi-scale processing.
# * ResNet: skip-connections to enable depth.
# 

import torchvision.models as models
vgg = models.vgg16(pretrained=True)
print(vgg.features[:])

resnet = models.resnet18(pretrained=True)
print(resnet.layer1)

inception = models.inception_v3(pretrained=True)

# ## 10. Model sizing and compute
# 
# * Parameter count roughly equals sum of (kernel_size * in_channels * out_channels) + biases.
# * FLOPs approximate multiply-adds: for each conv, `H_out * W_out * K*K * Cin * Cout * 2` (multiply + add). Tools exist to compute FLOPs automatically (e.g., `torchinfo`, `fvcore`).
# 

# ## 11. Common pitfalls & debugging checklist
# 
# * Data normalization mismatch between train and pretrained models.
# * Learning rate too high/low — watch training curves.
# * Overfitting — use augmentation, dropout, weight decay.
# * Underfitting — increase capacity or train longer.
# * Incorrect label encoding / loss mismatch.

# ## 12. Exercises & extensions
# 
# 1. Implement AlexNet and train on CIFAR-100; compare training time when upsampling to 224x224 vs modifying first layers.
# 2. Replace ReLU with GELU and observe impact.
# 3. Implement simple residual blocks and convert the simple CNN to a tiny-ResNet.
# 4. Prune channels of a trained model and measure accuracy drop.


# 
# # RNN In-Lab Assignments
# 
# ---
# 
# ## **Q 1 — Building RNN, LSTM, and GRU from Scratch**
# 
# ### Objective
# Implement fundamental recurrent architectures from scratch to understand their internal mechanics.
# 
# ### Tasks
# 1. Implement a simple RNN using NumPy/Tensorflow/Pytorch:
#    - Include forward pass and backpropagation through time.
# 2. Extend the implementation to include LSTM and GRU units.
# 3. Train all three models on a toy sequential dataset:
#    - Options: character-level text generation or sine wave prediction.
# 4. Plot and compare training loss curves.
# 5. Write short insights on which model learns faster and why.
# 6. Visualize gradient magnitudes across time steps to demonstrate vanishing/exploding gradients.(Optional)
# ---
# 
# ## **Q 2 — Training and Weight Visualization**
# 
# ### Objective
# Train RNN, LSTM, and GRU models on a real dataset and study how their weights evolve during learning.
# 
# ### Tasks
# 1. Train RNN, LSTM, and GRU models using PyTorch or TensorFlow on one of the following:
#    - Sequential MNIST
#    - IMDb Sentiment Analysis
#    - Time series dataset (e.g., stock prices, temperature)
# 2. Save model weights after each epoch.
# 3. Visualize weight distributions across epochs using histograms or kernel density plots.
# 4. Compare how weight evolution differs between RNN, LSTM, and GRU.
# 5. Discuss observations related to training stability, saturation, and convergence behavior.
# 
# ---
# 
# ## **Q 3 — Visual Question Answering (VQA) with CNN + RNN Fusion (No Training)**
# 
# ### Objective
# Understand multimodal representation fusion by combining CNN (for images) and RNN variants (for questions), without training.
# 
# ### Tasks
# 1. Use a pretrained CNN (e.g., ResNet18) to extract image feature vectors for VQA v2 dataset or COCO-QA.
# 2. Use an RNN/LSTM/GRU to encode natural language questions into hidden representations.
# 3. Visualize RNN hidden-state dynamics:
#    - Plot PCA or t-SNE trajectories of hidden states across time.
#    - Generate similarity heatmaps between hidden states of different words.
# 4. Fuse image and question embeddings:
#    - Compute cosine similarities between question embeddings and image features.
#    - Visualize similarities using heatmaps or bar charts.
# 5. Compare visualizations for RNN, LSTM, and GRU encoders and describe qualitative differences.
# 
# ---
# 
# ### **Submission Requirements**
# - .ipynb notebook
# - An explanation summarizing observations and key visualizations.
# - Notebooks or scripts implementing each question.
# - Plots and figures for analysis and discussion.
# ---
# 
# 




