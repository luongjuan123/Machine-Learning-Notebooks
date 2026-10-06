# Machine Learning Notebooks: First-Principles Implementations and Mathematical Foundations

## Overview and Pedagogical Objectives

This repository contains first-principles implementations of foundational machine learning algorithms written in Python, using raw NumPy for vectorized numerical computation, Pandas for data ingestion, and Matplotlib and Seaborn for diagnostic visualization.

Rather than relying on high-level machine learning libraries that abstract away mathematical mechanics, every model in this repository is built directly from theoretical formulations. The codebase implements the complete analytical derivation pipeline: from probabilistic priors, cost functions, and likelihood objectives to gradient derivations, Hessian matrices, and closed-form normal equations.

### Primary Goals

1. **First-Principles Understanding**: Demonstrate the inner workings of core machine learning algorithms by implementing their optimization routines and inference methods from raw matrix algebra.
2. **Mathematical Rigor**: Provide complete derivations for each model, showing how theoretical assumptions (such as the Exponential Family or Multivariate Gaussian distributions) lead directly to specific learning algorithms and decision boundaries.
3. **Vectorized Numerical Computing**: Leverage NumPy broadcasting, matrix operations, and vectorized algebra to implement efficient algorithms without unvectorized loops over samples where analytically possible.
4. **Empirical Verification and Visualization**: Train and evaluate models on controlled synthetic and real-world benchmark datasets, generating analytical convergence logs, evaluation metrics, and visual decision boundary plots.

---

## Repository Structure

```
Machine-Learning-Notebooks/
|-- README.md
|-- .gitattributes
|
|-- Gaussian Discriminant Analysis/
|   |-- GaussianDiscriminantAnalysis.ipynb
|   |-- Data/
|   |   |-- ds1_train.csv
|   |   |-- ds1_valid.csv
|   |   |-- ds2_train.csv
|   |   |-- ds2_valid.csv
|   |-- Output/
|       |-- GaussianDiscriminantAnalysis_1.png
|       |-- GaussianDiscriminantAnalysis_1.txt
|       |-- GaussianDiscriminantAnalysis_2.png
|       |-- GaussianDiscriminantAnalysis_2.txt
|
|-- Generalized Linear Model/
|   |-- GeneralizedLinearModel.ipynb
|   |-- Data/
|   |   |-- train1.csv
|   |   |-- test1.csv
|   |   |-- train2.csv
|   |   |-- test2.csv
|   |   |-- train3.csv
|   |   |-- test3.csv
|   |-- Output/
|       |-- BernoulliRegression.png
|       |-- BernoulliRegression.txt
|       |-- GaussianRegression.png
|       |-- GaussianRegression.txt
|       |-- PoissonRegression.png
|
|-- K-mean Clustering/
|   |-- K-mean Clustering.ipynb
|   |-- Data/
|   |   |-- kmeans_clustering_data.csv
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- Output/
|       |-- KMeansClustering.txt
|       |-- KmeansClusteringTrainData.png
|       |-- KmeansClusteringTrainData2.png
|       |-- KmeansClusteringEvalData.png
|       |-- KmeansClusteringEvalData2.png
|
|-- Linear Regression/
|   |-- LinearRegression.ipynb
|   |-- Data/
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- Output/
|       |-- LinearRegessionBatchGradientDescent.png
|       |-- LinearRegessionBatchGradientDescent.txt
|       |-- LinearRegressionStochaticGradientDescent.png
|       |-- LinearRegressionStochaticGradientDescent.txt
|       |-- LinearRegressionNormalEquation.png
|       |-- LinearRegressionNormalEquation.txt
|       |-- LocallyWeightedLinearRegression.png
|
|-- Logistic Regression/
|   |-- LogisticRegression.ipynb
|   |-- Data/
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- Data1/
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- Output/
|       |-- BatchGradientAscentLogisticRegression.png
|       |-- BatchGradientAscentLogisticRegression.txt
|       |-- BatchGradientDescentLogisticRegression.png
|       |-- BatchGradientDescentLogisticRegression.txt
|       |-- StochasticGradientAscentLogisticRegression.png
|       |-- StochasticGradientAscentLogisticRegression.txt
|       |-- StochaticGradientDescentLogisticRegression.png
|       |-- StochaticGradientDescentLogisticRegression.txt
|       |-- NewtonMethodLogisticRegression.png
|       |-- NewtonMethodLogisticRegression.txt
|       |-- PerceptronRegression.png
|       |-- PerceptronRegression.txt
|
|-- Naive Bayes/
|   |-- NaiveBayes.ipynb
|   |-- Data/
|   |   |-- Naive-Bayes-Classification-Data.csv
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- Data1/
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- Output/
|       |-- classification_boundary.png
|       |-- confusion_matrix.png
|
|-- Neuron Network/
|   |-- NeuronNetwork.ipynb
|
|-- Support Vector Machine/
|   |-- Support Vector Machine.ipynb
```

---

## Core Framework and Architecture

Each module follows a structured object-oriented architecture designed to ensure code clarity, modularity, and reproducible benchmarking.

### The Base Linear Model Abstraction

The foundational parent class `LinearModel` defines the standard interface shared across linear algorithms:

```python
class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-6,
                 theta_0=None, verbose=True):
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, X, y):
        """Fit linear model to training data."""
        raise NotImplementedError("Subclass must implement fit method.")

    def predict(self, X):
        """Make predictions for inputs X."""
        raise NotImplementedError("Subclass must implement predict method.")
```

#### Key Attributes

- `theta`: Parameter vector containing coefficients $\theta \in \mathbb{R}^{n+1}$ (including intercept).
- `step_size`: Learning rate $\alpha > 0$ governing the magnitude of parameter updates.
- `max_iter`: Maximum iteration threshold preventing infinite loops when gradient steps oscillate.
- `eps`: Convergence tolerance $\epsilon$. Iteration terminates when the parameter change norm $\|\theta^{(t+1)} - \theta^{(t)}\|$ drops below $\epsilon$.
- `verbose`: Logging flag indicating whether optimization progress should be printed.

### The Utility Engine (`util`)

Across all modules, a static utility class provides standard data loading, preprocessing, and plotting utilities:

- **Intercept Augmentation**: Prepends a column of ones to the design matrix $X \in \mathbb{R}^{m \times n} \mapsto \tilde{X} \in \mathbb{R}^{m \times (n+1)}$ to absorb the bias term $\theta_0$:
  $$\tilde{x}^{(i)} = [1, x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i)}]^\top$$
- **CSV Data Ingestion**: Handles UTF-8 with BOM (`utf-8-sig`) decoding, extracts features prefixed with `x`, extracts labels matching target identifiers (`y` or `t`), and enforces strict shape dimensionality.
- **Decision Boundary Rendering**: For 2D classification problems, analytically computes the decision hyperplane by solving for $\theta^\top x = 0$:
  $$x_2 = -\left(\frac{\theta_0}{\theta_2} + \frac{\theta_1}{\theta_2} x_1\right)$$
  Plots training examples categorized by label alongside the decision boundary.

---

## Detailed Module Documentation

---

### 1. Linear Regression

#### 1.1 Problem Formulation and Cost Function

Given a dataset of $m$ training pairs $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^m$, where each input feature vector is $x^{(i)} \in \mathbb{R}^n$ and each target is $y^{(i)} \in \mathbb{R}$, we define the hypothesis function as:

$$h_\theta(x) = \sum_{j=0}^n \theta_j x_j = \theta^\top x$$

where $x_0 = 1$ is the bias feature.

We construct the design matrix $X \in \mathbb{R}^{m \times (n+1)}$ and the target vector $y \in \mathbb{R}^m$:

$$X = \begin{bmatrix} — (x^{(1)})^\top — \\ — (x^{(2)})^\top — \\ \vdots \\ — (x^{(m)})^\top — \end{bmatrix}, \quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

The ordinary least squares (OLS) cost function is defined as half the sum of squared errors:

$$J(\theta) = \frac{1}{2} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 = \frac{1}{2} (X\theta - y)^\top (X\theta - y)$$

#### 1.2 Mathematical Derivation of Gradients

To find the gradient with respect to $\theta_j$:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{\partial}{\partial \theta_j} \frac{1}{2} \sum_{i=1}^m \left(\theta^\top x^{(i)} - y^{(i)}\right)^2 = \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right) x_j^{(i)}$$

In vectorized matrix form:

$$\nabla_\theta J(\theta) = X^\top (X\theta - y)$$

When normalized by the dataset size $m$, the average gradient is:

$$\nabla_\theta J(\theta) = \frac{1}{m} X^\top (X\theta - y)$$

#### 1.3 Optimization Algorithms Implemented

##### Batch Gradient Descent (BGD)

The parameter update rule applies the full dataset gradient at each step:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \frac{1}{m} X^\top \left(X\theta^{(t)} - y\right)$$

Convergence is evaluated using the $L_1$ norm:

$$\|\theta^{(t+1)} - \theta^{(t)}\|_1 < \epsilon$$

Implementation class: `LinearRegessionBatchGradientDescent`.

##### Stochastic Gradient Descent (SGD)

Instead of scanning all $m$ examples before updating, SGD updates parameters for each individual training sample $i \in \{1, \dots, m\}$:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \left(h_\theta(x^{(i)}) - y^{(i)}\right) x^{(i)}$$

Implementation class: `LinearRegressionStochaticGradientDescent`.

##### Normal Equations (Analytical Closed-Form Solution)

To derive the exact parameter vector $\theta^*$ that minimizes $J(\theta)$ analytically:

$$\nabla_\theta J(\theta) = \nabla_\theta \left[\frac{1}{2} (\theta^\top X^\top X \theta - 2 y^\top X \theta + y^\top y)\right] = X^\top X \theta - X^\top y = 0$$

$$X^\top X \theta = X^\top y \implies \theta^* = (X^\top X)^{-1} X^\top y$$

In `LinearRegressionNormalEquation`, the computation utilizes the Moore-Penrose pseudoinverse `np.linalg.pinv(X.T @ X) @ X.T @ y` to handle collinearity and rank-deficient design matrices reliably.

##### Locally Weighted Linear Regression (LWLR)

Locally weighted linear regression is a non-parametric learning algorithm that computes a tailored hypothesis for each specific query point $x$. At prediction time, LWLR solves:

$$\min_\theta \sum_{i=1}^m w^{(i)} \left(y^{(i)} - \theta^\top x^{(i)}\right)^2$$

where the sample weights $w^{(i)}$ are computed using a Gaussian kernel with bandwidth parameter $\tau$:

$$w^{(i)} = \exp\left(-\frac{\|x^{(i)} - x\|^2}{2\tau^2}\right)$$

In matrix form, defining $W = \text{diag}(w^{(1)}, \dots, w^{(m)})$, the cost function becomes:

$$J(\theta) = \frac{1}{2} (X\theta - y)^\top W (X\theta - y)$$

Setting $\nabla_\theta J(\theta) = 0$ yields:

$$\theta(x) = (X^\top W X)^{-1} X^\top W y$$

For numerical conditioning and stability, the implementation applies Tikhonov regularization:

$$\theta(x) = (X^\top W X + \lambda I)^{-1} X^\top W y \quad (\lambda = 10^{-5})$$

Implementation class: `LocallyWeightedLinearRegression`.

#### 1.4 Datasets and Execution Artifacts

- Training Data: `Data/train.csv` (699 observations, univariate input $x$ and continuous target $y$).
- Testing Data: `Data/test.csv` (300 observations).
- Visual Outputs:
  - `Output/LinearRegessionBatchGradientDescent.png`: Fitted regression line against data distribution.
  - `Output/LinearRegressionStochaticGradientDescent.png`: Solution learned via single-sample sequential updates.
  - `Output/LinearRegressionNormalEquation.png`: Closed-form optimal regression solution.
  - `Output/LocallyWeightedLinearRegression.png`: Non-linear regression curve produced by varying local bandwidth $\tau = 0.5$.
- Prediction Outputs:
  - `Output/LinearRegessionBatchGradientDescent.txt`
  - `Output/LinearRegressionStochaticGradientDescent.txt`
  - `Output/LinearRegressionNormalEquation.txt`

---

### 2. Logistic Regression

#### 2.1 Problem Formulation and Probabilistic Interpretation

In binary classification, target labels are discrete: $y^{(i)} \in \{0, 1\}$. Logistic regression maps continuous linear combinations $\theta^\top x$ to probability values in the open interval $(0, 1)$ via the sigmoid (logistic) function:

$$g(z) = \frac{1}{1 + e^{-z}}, \quad h_\theta(x) = g(\theta^\top x) = \frac{1}{1 + e^{-\theta^\top x}}$$

Assuming conditional Bernoulli distribution $y \mid x \sim \text{Bernoulli}(h_\theta(x))$:

$$p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1 - y}$$

Under independence of training samples, the likelihood function is:

$$L(\theta) = \prod_{i=1}^m p(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^m (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1 - y^{(i)}}$$

The log-likelihood $\ell(\theta)$ is:

$$\ell(\theta) = \sum_{i=1}^m \left[y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right]$$

The objective is either maximizing the log-likelihood $\ell(\theta)$ (via Gradient Ascent) or minimizing the negative log-likelihood (binary cross-entropy loss):

$$J(\theta) = -\frac{1}{m} \ell(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right]$$

#### 2.2 Gradient Derivation

Using the derivative property of the sigmoid function $g'(z) = g(z)(1 - g(z))$:

$$\frac{\partial \ell(\theta)}{\partial \theta_j} = \sum_{i=1}^m \left[y^{(i)} \frac{g'(\theta^\top x^{(i)})}{g(\theta^\top x^{(i)})} x_j^{(i)} - (1 - y^{(i)}) \frac{g'(\theta^\top x^{(i)})}{1 - g(\theta^\top x^{(i)})} x_j^{(i)}\right]$$

$$= \sum_{i=1}^m \left[y^{(i)} (1 - h_\theta(x^{(i)})) - (1 - y^{(i)}) h_\theta(x^{(i)})\right] x_j^{(i)} = \sum_{i=1}^m (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$

In vectorized notation:

$$\nabla_\theta \ell(\theta) = X^\top (y - h_\theta(X))$$

$$\nabla_\theta J(\theta) = \frac{1}{m} X^\top (h_\theta(X) - y)$$

#### 2.3 Optimization Algorithms Implemented

##### Batch Gradient Ascent and Descent

- **Batch Gradient Ascent**: Maximizes likelihood directly:
  $$\theta^{(t+1)} = \theta^{(t)} + \alpha \frac{1}{m} X^\top (y - h) + \lambda_{\text{reg}} \|\theta\|_2$$
  Implementation class: `BatchGradientAscentLogisticRegression`.
- **Batch Gradient Descent**: Minimizes binary cross-entropy loss:
  $$\theta^{(t+1)} = \theta^{(t)} - \alpha \frac{1}{m} X^\top (h - y)$$
  Implementation class: `BatchGradientDescentLogisticRegression`.

##### Stochastic Gradient Ascent (SGA)

Iterates over individual training samples to perform localized parameter updates:

$$\theta^{(t+1)} = \theta^{(t)} + \alpha (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$$

Implementation class: `StochasticGradientAscentLogisticRegression`.

##### Newton's Method (Newton-Raphson Optimization)

Newton's method is a second-order optimization procedure that achieves quadratic convergence near the optimum by taking into account the curvature of the cost function.

The multidimensional update rule is:

$$\theta^{(t+1)} = \theta^{(t)} - H^{-1} \nabla_\theta J(\theta^{(t)})$$

where $H \in \mathbb{R}^{(n+1) \times (n+1)}$ is the Hessian matrix of second partial derivatives:

$$H_{jk} = \frac{\partial^2 J(\theta)}{\partial \theta_j \partial \theta_k} = \frac{1}{m} \sum_{i=1}^m h_\theta(x^{(i)})(1 - h_\theta(x^{(i)})) x_j^{(i)} x_k^{(i)}$$

In vectorized form, letting $D = \text{diag}\left(h_\theta(x^{(1)})(1 - h_\theta(x^{(1)})), \dots, h_\theta(x^{(m)})(1 - h_\theta(x^{(m)}))\right)$:

$$H = \frac{1}{m} X^\top D X$$

For any non-zero vector $z \in \mathbb{R}^{n+1}$:

$$z^\top H z = \frac{1}{m} z^\top X^\top D X z = \frac{1}{m} (Xz)^\top D (Xz) = \frac{1}{m} \sum_{i=1}^m D_{ii} ((x^{(i)})^\top z)^2 \geq 0$$

Because $0 < h_\theta(x) < 1$, the diagonal elements $D_{ii} > 0$. Hence, $H$ is positive semi-definite (and strictly positive definite when $X$ has full column rank), guaranteeing that the negative log-likelihood is strictly convex and Newton's method converges globally.

Implementation class: `NewtonMethodLogisticRegression`.

##### The Perceptron Learning Algorithm

By replacing the smooth sigmoid function $g(z) = \frac{1}{1 + e^{-z}}$ with the Heaviside step threshold function:

$$g(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

the hypothesis becomes binary: $h_\theta(x) = g(\theta^\top x) \in \{0, 1\}$. The parameter update rule is:

$$\theta^{(t+1)} = \theta^{(t)} + \alpha (y^{(i)} - g(\theta^\top x^{(i)})) x^{(i)}$$

Implementation class: `PerceptronRegression`.

#### 2.4 Datasets and Execution Artifacts

- Primary Dataset (`Data/`): `train.csv` (800 rows, features `x_1`, `x_2`, label `y`), `test.csv` (100 rows).
- Secondary Dataset (`Data1/`): `train.csv` (799 rows, features `x_1`, `x_2`, label `y`), `test.csv` (194 rows).
- Output Visualizations:
  - `Output/BatchGradientAscentLogisticRegression.png`: Decision boundary fitted via BGA.
  - `Output/BatchGradientDescentLogisticRegression.png`: Decision boundary fitted via BGD.
  - `Output/StochasticGradientAscentLogisticRegression.png`: Decision boundary fitted via SGA.
  - `Output/NewtonMethodLogisticRegression.png`: Decision boundary fitted via Newton-Raphson on `Data1`.
  - `Output/PerceptronRegression.png`: Decision boundary derived using the discrete threshold activation.
- Text Predictions:
  - `Output/BatchGradientAscentLogisticRegression.txt`
  - `Output/BatchGradientDescentLogisticRegression.txt`
  - `Output/StochasticGradientAscentLogisticRegression.txt`
  - `Output/StochaticGradientDescentLogisticRegression.txt`
  - `Output/NewtonMethodLogisticRegression.txt`
  - `Output/PerceptronRegression.txt`

---

### 3. Generalized Linear Models (GLMs)

#### 3.1 The Exponential Family

A probability distribution belongs to the **exponential family** if its density (or probability mass function) can be expressed in canonical form:

$$p(y; \eta) = b(y) \exp\left(\eta^\top T(y) - a(\eta)\right)$$

Where:
- $\eta$: The **natural parameter** (canonical parameter).
- $T(y)$: The **sufficient statistic** (frequently $T(y) = y$).
- $a(\eta)$: The **log partition function**, acting as a normalization constant ensuring $\int p(y; \eta) dy = 1$.
- $b(y)$: The **base measure**.

Important theoretical properties of the log partition function:
- Expected value: $\mathbb{E}[T(y); \eta] = \nabla_\eta a(\eta)$
- Variance: $\text{Var}(T(y); \eta) = \nabla_\eta^2 a(\eta)$

#### 3.2 The Three Postulates of GLM Construction

To construct a Generalized Linear Model predicting target $y$ given features $x$:

1. **Exponential Family Assumption**: The conditional distribution of the target given features belongs to the exponential family:
   $$y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$$
2. **Prediction Goal**: Given input $x$, the model predicts the conditional expected value of the sufficient statistic:
   $$h(x) = \mathbb{E}[T(y) \mid x] = \mathbb{E}[y \mid x]$$
3. **Linearity of Natural Parameter**: The natural parameter $\eta$ is a linear combination of input features:
   $$\eta = \theta^\top x$$

#### 3.3 Implemented GLM Instances

##### Poisson Regression (Count Data Modeling)

Used for modeling count variables ($y \in \{0, 1, 2, \dots\}$). The Poisson distribution with mean parameter $\lambda > 0$ is:

$$p(y; \lambda) = \frac{\lambda^y e^{-\lambda}}{y!} = \frac{1}{y!} \exp\left(y \log \lambda - \lambda\right)$$

Mapping to exponential family terms:
- $b(y) = \frac{1}{y!}$
- $T(y) = y$
- Natural parameter: $\eta = \log \lambda \implies \lambda = e^\eta$
- Log partition function: $a(\eta) = \lambda = e^\eta$

Applying Postulate 3: $\eta = \theta^\top x \implies \lambda = \mathbb{E}[y \mid x] = e^{\theta^\top x}$.

The hypothesis is:

$$h_\theta(x) = e^{\theta^\top x}$$

The parameter update rule derived by maximizing log-likelihood via gradient ascent is:

$$\theta^{(t+1)} = \theta^{(t)} + \alpha \frac{1}{m} X^\top \left(y - e^{X\theta}\right)$$

Implementation class: `PoissonRegression`.

##### Gaussian Regression (Continuous Variables / Ordinary Least Squares)

The Gaussian distribution with known variance $\sigma^2 = 1$ is:

$$p(y; \mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y - \mu)^2}{2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right) \exp\left(\mu y - \frac{\mu^2}{2}\right)$$

Mapping to exponential family terms:
- $b(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right)$
- $T(y) = y$
- Natural parameter: $\eta = \mu$
- Log partition function: $a(\eta) = \frac{\eta^2}{2}$

Applying Postulate 3: $\eta = \theta^\top x \implies \mu = \mathbb{E}[y \mid x] = \theta^\top x$.

The hypothesis is:

$$h_\theta(x) = \theta^\top x$$

The gradient update rule is:

$$\theta^{(t+1)} = \theta^{(t)} + \alpha \frac{1}{m} X^\top (y - X\theta)$$

Implementation class: `GaussianRegression`.

##### Bernoulli Regression (Binary Classification)

The Bernoulli distribution with parameter $\phi \in (0, 1)$ is:

$$p(y; \phi) = \phi^y (1 - \phi)^{1 - y} = \exp\left(y \log \phi + (1 - y) \log(1 - \phi)\right) = \exp\left(y \log\frac{\phi}{1 - \phi} + \log(1 - \phi)\right)$$

Mapping to exponential family terms:
- $b(y) = 1$
- $T(y) = y$
- Natural parameter: $\eta = \log\frac{\phi}{1 - \phi} \implies \phi = \frac{1}{1 + e^{-\eta}}$
- Log partition function: $a(\eta) = -\log(1 - \phi) = \log(1 + e^\eta)$

Applying Postulate 3: $\eta = \theta^\top x \implies \phi = \mathbb{E}[y \mid x] = \frac{1}{1 + e^{-\theta^\top x}}$.

The hypothesis is:

$$h_\theta(x) = \frac{1}{1 + e^{-\theta^\top x}}$$

The gradient update rule is:

$$\theta^{(t+1)} = \theta^{(t)} + \alpha \frac{1}{m} X^\top \left(y - \frac{1}{1 + e^{-X\theta}}\right)$$

Implementation class: `BernoulliRegression`.

#### 3.4 Datasets and Execution Artifacts

- Poisson Regression Data: `Data/train1.csv` (2,500 samples, 4 features `x_1` through `x_4`, target count `y`), `Data/test1.csv` (250 samples).
- Gaussian Regression Data: `Data/train2.csv` (699 samples, feature `x`, target continuous `y`), `Data/test2.csv` (300 samples).
- Bernoulli Regression Data: `Data/train3.csv` (800 samples, features `x_1`, `x_2`, target binary label `y`), `Data/test3.csv` (100 samples).
- Visual Outputs:
  - `Output/PoissonRegression.png`: Scatter plot comparing predicted counts against actual counts.
  - `Output/GaussianRegression.png`: Fitted linear curve for Gaussian GLM.
  - `Output/BernoulliRegression.png`: Decision boundary separating classes for Bernoulli GLM.
- Text Predictions:
  - `Output/GaussianRegression.txt`
  - `Output/BernoulliRegression.txt`

---

### 4. Gaussian Discriminant Analysis (GDA)

#### 4.1 Generative vs. Discriminative Modeling

- **Discriminative Models** (e.g., Logistic Regression) directly model the conditional probability $p(y \mid x)$ or learn a direct mapping from features to labels without modeling feature generation.
- **Generative Models** (e.g., GDA, Naive Bayes) model the joint distribution $p(x, y) = p(x \mid y) p(y)$. Inference is performed by applying Bayes' Rule:
  $$p(y = 1 \mid x) = \frac{p(x \mid y = 1) p(y = 1)}{p(x)} = \frac{p(x \mid y = 1) p(y = 1)}{p(x \mid y = 0) p(y = 0) + p(x \mid y = 1) p(y = 1)}$$

#### 4.2 Mathematical Assumptions

Gaussian Discriminant Analysis assumes that input features $x \in \mathbb{R}^n$ follow a multivariate normal distribution conditioned on class label $y \in \{0, 1\}$, with shared covariance matrix $\Sigma$:

- Class prior: $y \sim \text{Bernoulli}(\phi)$
- Class 0 conditional: $x \mid y = 0 \sim \mathcal{N}(\mu_0, \Sigma)$
- Class 1 conditional: $x \mid y = 1 \sim \mathcal{N}(\mu_1, \Sigma)$

Probability density functions:

$$p(y) = \phi^y (1 - \phi)^{1 - y}$$

$$p(x \mid y = 0) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_0)^\top \Sigma^{-1} (x - \mu_0)\right)$$

$$p(x \mid y = 1) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_1)^\top \Sigma^{-1} (x - \mu_1)\right)$$

#### 4.3 Closed-Form Maximum Likelihood Estimation

The joint log-likelihood of the dataset is:

$$\ell(\phi, \mu_0, \mu_1, \Sigma) = \sum_{i=1}^m \log p(x^{(i)}, y^{(i)}) = \sum_{i=1}^m \log \left[p(x^{(i)} \mid y^{(i)}) p(y^{(i)})\right]$$

Maximizing $\ell$ with respect to each parameter yields the closed-form MLE estimators:

$$\phi = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}$$

$$\mu_0 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\}}, \quad \mu_1 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}$$

$$\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^\top$$

$$\Sigma = \frac{1}{m} \left[\sum_{i: y^{(i)}=0} (x^{(i)} - \mu_0)(x^{(i)} - \mu_0)^\top + \sum_{i: y^{(i)}=1} (x^{(i)} - \mu_1)(x^{(i)} - \mu_1)^\top\right]$$

#### 4.4 Mapping GDA to the Logistic Form

When the class conditionals share the covariance matrix $\Sigma$, the posterior probability $p(y = 1 \mid x)$ takes the exact functional form of logistic regression:

$$p(y = 1 \mid x) = \frac{1}{1 + \exp\left(-\theta^\top x\right)}$$

where the equivalent parameter vector $\theta = [\theta_0, \theta_1, \dots, \theta_n]^\top$ is computed directly from the GDA parameters:

$$\theta_0 = \frac{1}{2} \left(\mu_0^\top \Sigma^{-1} \mu_0 - \mu_1^\top \Sigma^{-1} \mu_1\right) - \log\left(\frac{1 - \phi}{\phi}\right)$$

$$\theta_{1:n} = \Sigma^{-1} (\mu_1 - \mu_0)$$

Because this transformation is exact, GDA produces a linear decision boundary in feature space.

Implementation class: `GDA`.

#### 4.5 Datasets and Execution Artifacts

- Dataset 1 (`Data/`): `ds1_train.csv` (800 rows, features `x_1`, `x_2`, label `y`), `ds1_valid.csv` (100 rows).
- Dataset 2 (`Data/`): `ds2_train.csv` (800 rows, features `x_1`, `x_2`, label `y`), `ds2_valid.csv` (100 rows).
- Visual Outputs:
  - `Output/GaussianDiscriminantAnalysis_1.png`: Fitted decision boundary for Dataset 1.
  - `Output/GaussianDiscriminantAnalysis_2.png`: Fitted decision boundary for Dataset 2.
- Text Predictions:
  - `Output/GaussianDiscriminantAnalysis_1.txt`
  - `Output/GaussianDiscriminantAnalysis_2.txt`

---

### 5. Naive Bayes Classification

#### 5.1 The Naive Bayes Assumption

In high-dimensional discrete or text classification, features represent word occurrences or attribute counts $x = (x_1, \dots, x_V)$ over vocabulary / attribute space $V$. Modeling the full joint distribution $p(x_1, \dots, x_V \mid y)$ without restrictions requires estimating $2^V - 1$ parameters for binary features.

The **Naive Bayes assumption** posits that all feature attributes are conditionally independent given the class label $y$:

$$p(x \mid y) = \prod_{j=1}^V p(x_j \mid y)$$

#### 5.2 Likelihood and Maximum Likelihood Estimation

Given training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$:

$$L(\phi_y, \phi_{k \mid y=0}, \phi_{k \mid y=1}) = \prod_{i=1}^m p(x^{(i)}, y^{(i)}) = \prod_{i=1}^m \left(\prod_{j=1}^V p(x_j^{(i)} \mid y^{(i)})\right) p(y^{(i)})$$

The class prior MLE is:

$$\phi_y = p(y = 1) = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}$$

For multinomial word/event models, the MLE for feature $k$ given class $y \in \{0, 1\}$ is:

$$\phi_{k \mid y=1} = \frac{\sum_{i=1}^m x_k^{(i)} \mathbf{1}\{y^{(i)} = 1\}}{\sum_{i=1}^m \left(\sum_{j=1}^V x_j^{(i)}\right) \mathbf{1}\{y^{(i)} = 1\}}$$

$$\phi_{k \mid y=0} = \frac{\sum_{i=1}^m x_k^{(i)} \mathbf{1}\{y^{(i)} = 0\}}{\sum_{i=1}^m \left(\sum_{j=1}^V x_j^{(i)}\right) \mathbf{1}\{y^{(i)} = 0\}}$$

#### 5.3 Laplace Smoothing

If a specific feature $k$ never appears in class $y = 1$ in the training data, the empirical estimate $\phi_{k \mid y=1} = 0$. When evaluating a query containing feature $k$, the product $\prod_j p(x_j \mid y=1)$ evaluates to zero, entirely nullifying the evidence of all other features.

To prevent zero-probability degeneracy, **Laplace smoothing** adds a pseudo-count of 1 to the numerator and $|V|$ to the denominator:

$$\phi_{k \mid y=1} = \frac{\sum_{i=1}^m x_k^{(i)} \mathbf{1}\{y^{(i)} = 1\} + 1}{\sum_{i=1}^m \left(\sum_{j=1}^V x_j^{(i)}\right) \mathbf{1}\{y^{(i)} = 1\} + |V|}$$

$$\phi_{k \mid y=0} = \frac{\sum_{i=1}^m x_k^{(i)} \mathbf{1}\{y^{(i)} = 0\} + 1}{\sum_{i=1}^m \left(\sum_{j=1}^V x_j^{(i)}\right) \mathbf{1}\{y^{(i)} = 0\} + |V|}$$

#### 5.4 Log-Space Inference for Numerical Stability

Computing products of numerous small probability values $\prod_j \phi_{k \mid y}^{x_j}$ leads to severe floating-point underflow. The implementation performs inference in log-space:

$$\log p(y = 1 \mid x) \propto \log p(y = 1) + \sum_{j=1}^V x_j \log \phi_{j \mid y=1}$$

$$\log p(y = 0 \mid x) \propto \log (1 - \phi_y) + \sum_{j=1}^V x_j \log \phi_{j \mid y=0}$$

The classification rule evaluates:

$$\hat{y} = \mathbf{1}\left\{\sum_{j=1}^V x_j \log \phi_{j \mid y=1} + \log \phi_y > \sum_{j=1}^V x_j \log \phi_{j \mid y=0} + \log (1 - \phi_y)\right\}$$

Implementation class: `NaiveBayes`.

#### 5.5 Datasets and Diagnostic Artifacts

- Raw Dataset: `Data/Naive-Bayes-Classification-Data.csv` (995 rows, features `glucose`, `bloodpressure`, label `diabetes`).
- Split Partitions: `Data/train.csv` (750 rows), `Data/test.csv` (245 rows); alternate partitions in `Data1/`.
- Diagnostic Outputs:
  - `Output/confusion_matrix.png`: Annotated 2x2 confusion matrix heatmap generated using Seaborn.
  - `Output/classification_boundary.png`: Scatter visualization of feature distribution.

---

### 6. K-Means Clustering

#### 6.1 Unsupervised Clustering Objective

Given an unlabeled dataset $\{x^{(1)}, \dots, x^{(m)}\}$ where $x^{(i)} \in \mathbb{R}^n$, the goal of K-Means clustering is to partition the $m$ data points into $K$ distinct clusters $C = \{C_1, C_2, \dots, C_K\}$.

Each cluster is represented by a centroid $\mu_k \in \mathbb{R}^n$ ($k \in \{1, \dots, K\}$). Let $c^{(i)} \in \{1, \dots, K\}$ denote the index of the centroid assigned to sample $x^{(i)}$.

The optimization problem minimizes the **distortion objective function** (inertia):

$$J(c, \mu) = \sum_{i=1}^m \|x^{(i)} - \mu_{c^{(i)}}\|^2$$

#### 6.2 The Lloyd-Forgy Coordinate Descent Algorithm

K-Means alternates between optimizing cluster assignments $c$ while holding centroids $\mu$ fixed, and optimizing centroids $\mu$ while holding assignments $c$ fixed:

1. **Centroid Initialization**: Randomly sample $K$ distinct training examples as initial centroids:
   $$\mu_k = x^{(j_k)}, \quad j_k \sim \text{Uniform}(\{1, \dots, m\})$$
2. **Assignment Step (Expectation Step)**: Assign each sample to its nearest centroid:
   $$c^{(i)} := \arg\min_{k \in \{1, \dots, K\}} \|x^{(i)} - \mu_k\|^2$$
3. **Update Step (Maximization Step)**: Recompute each centroid as the geometric mean of points assigned to it:
   $$\mu_k := \frac{\sum_{i=1}^m \mathbf{1}\{c^{(i)} = k\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{c^{(i)} = k\}}$$
4. **Convergence Check**: Repeat steps 2 and 3 until centroids cease to change between iterations:
   $$\mu_k^{(t+1)} = \mu_k^{(t)} \quad \forall k \in \{1, \dots, K\}$$

Because each step monotonically decreases $J(c, \mu)$, and there are only a finite number of possible cluster assignments ($K^m$), the algorithm is guaranteed to converge to a local optimum.

Implementation class: `KMeansClustering`.

#### 6.3 Optimal Permutation Matching for Evaluation

Because unsupervised clustering produces arbitrary label permutations (e.g., cluster index 0 might correspond to ground-truth label 2), calculating classification accuracy directly against true labels requires finding the optimal one-to-one mapping $\pi \in S_K$:

$$\text{Accuracy}^* = \max_{\pi \in \text{Permutations}(K)} \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{\pi(c^{(i)}) = y_{\text{true}}^{(i)}\}$$

The implementation in `cluster_accuracy` uses `itertools.permutations` to evaluate all $K!$ possible assignment mappings and return the maximal alignment accuracy.

#### 6.4 Datasets and Execution Artifacts

- Datasets: `Data/kmeans_clustering_data.csv` (300 rows, features `x_1`, `x_2`, cluster label `cluster_label`), `Data/train.csv` (250 rows), `Data/test.csv` (50 rows).
- Visual Outputs:
  - `Output/KmeansClusteringTrainData.png`: Training data scatter with marked final centroids.
  - `Output/KmeansClusteringTrainData2.png`: Cluster-partitioned color scatter on training set.
  - `Output/KmeansClusteringEvalData.png`: Test set evaluation points with projected centroids.
  - `Output/KmeansClusteringEvalData2.png`: Colored cluster partitions on test evaluation set.
- Text Predictions:
  - `Output/KMeansClustering.txt`

---

### 7. Support Vector Machines and Deep Learning (Roadmap)

The repository includes exploratory foundations and interface templates for additional core machine learning architectures:

- **Support Vector Machines** (`Support Vector Machine/Support Vector Machine.ipynb`):
  Initializes the `SupportVectorMachine` class parameterized with `step_size`, `lambda_param` (L2 regularization coefficient), and `max_iters`. Implements the soft-margin hinge-loss objective:
  $$\min_{\theta, b} \frac{1}{2} \|\theta\|_2^2 + C \sum_{i=1}^m \max\left(0, 1 - y^{(i)}(\theta^\top x^{(i)} + b)\right)$$
- **Neural Networks** (`Neuron Network/NeuronNetwork.ipynb`):
  Exploratory workspace for multi-layer perceptron (MLP) architectures, forward activation propagation, and backpropagation via the chain rule.

---

## Comparative Algorithmic Analysis

| Algorithm | Paradigm | Target Type | Optimization Method | Convergence Properties | Closed-Form Solution? | Computational Complexity (Train) |
|---|---|---|---|---|:---:|---|
| Linear Regression (BGD) | Discriminative | Continuous | First-Order Gradient Descent | Linear convergence rate | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Linear Regression (SGD) | Discriminative | Continuous | Stochastic Gradient Descent | Sublinear convergence rate | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Linear Regression (Normal Eq) | Discriminative | Continuous | Matrix Inversion / Pseudoinverse | Exact analytical optimum | Yes | $\mathcal{O}(m \cdot n^2 + n^3)$ |
| Locally Weighted LinReg (LWLR) | Non-parametric | Continuous | Weighted Normal Equations | Per-query solve | Yes | $\mathcal{O}(m_{\text{eval}} \cdot (m \cdot n^2 + n^3))$ |
| Logistic Regression (BGA/BGD) | Discriminative | Binary discrete | Gradient Ascent / Descent | Linear convergence rate | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Logistic Regression (Newton) | Discriminative | Binary discrete | Newton-Raphson (Second-Order) | Quadratic convergence rate | No | $\mathcal{O}(I \cdot (m \cdot n^2 + n^3))$ |
| Perceptron Algorithm | Discriminative | Binary discrete | Error-driven subgradient update | Finite steps if separable | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Poisson Regression (GLM) | Discriminative | Discrete counts | Gradient Ascent on log-likelihood | Convex, linear convergence | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Gaussian Regression (GLM) | Discriminative | Continuous | Gradient Descent on log-likelihood | Convex, linear convergence | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Bernoulli Regression (GLM) | Discriminative | Binary discrete | Gradient Ascent on log-likelihood | Convex, linear convergence | No | $\mathcal{O}(I \cdot m \cdot n)$ |
| Gaussian Discriminant Analysis | Generative | Categorical / Binary | Maximum Likelihood Estimation | Exact parameter optimum | Yes | $\mathcal{O}(m \cdot n^2 + n^3)$ |
| Naive Bayes Classification | Generative | Discrete / Categorical | Maximum Likelihood + Laplace | Exact parameter counts | Yes | $\mathcal{O}(m \cdot V)$ |
| K-Means Clustering | Unsupervised | Cluster indices | Lloyd's Coordinate Descent | Monotonic, local minimum | No | $\mathcal{O}(I \cdot m \cdot K \cdot n)$ |

---

## Mathematical Notation Reference

| Symbol | Mathematical Meaning | Dimensionality / Domain |
|---|---|---|
| $m$ | Total number of training observations | $\mathbb{N}$ |
| $n$ | Number of input feature dimensions (excluding bias) | $\mathbb{N}$ |
| $x^{(i)}$ | Feature vector of the $i$-th training observation | $\mathbb{R}^n$ or $\mathbb{R}^{n+1}$ |
| $y^{(i)}$ | Target label or response variable of the $i$-th observation | $\mathbb{R}$ (Regression) or $\{0, 1\}$ (Classification) |
| $X$ | Design matrix containing all training inputs | $\mathbb{R}^{m \times (n+1)}$ |
| $y$ | Column vector containing all training target values | $\mathbb{R}^m$ |
| $\theta$ | Model weight parameter vector | $\mathbb{R}^{n+1}$ |
| $h_\theta(x)$ | Model hypothesis function (prediction) | $\mathbb{R}$ |
| $\alpha$ | Learning rate / optimization step size | $\mathbb{R}_{> 0}$ |
| $J(\theta)$ | Cost function (Loss objective to minimize) | $\mathbb{R}$ |
| $\ell(\theta)$ | Log-likelihood objective (to maximize) | $\mathbb{R}$ |
| $\nabla_\theta$ | Gradient operator vector with respect to $\theta$ | $\mathbb{R}^{n+1}$ |
| $H$ | Hessian matrix of second partial derivatives | $\mathbb{R}^{(n+1) \times (n+1)}$ |
| $\sigma(z)$ | Sigmoid (logistic) activation function $\frac{1}{1 + e^{-z}}$ | $(0, 1)$ |
| $\tau$ | Bandwidth hyperparameter in Locally Weighted Regression | $\mathbb{R}_{> 0}$ |
| $\eta$ | Natural (canonical) parameter in Exponential Family | $\mathbb{R}^k$ |
| $a(\eta)$ | Log partition function in Exponential Family | $\mathbb{R}$ |
| $T(y)$ | Sufficient statistic vector in Exponential Family | $\mathbb{R}^k$ |
| $b(y)$ | Base measure function in Exponential Family | $\mathbb{R}_{\geq 0}$ |
| $\mu_0, \mu_1$ | Class mean vectors in Gaussian Discriminant Analysis | $\mathbb{R}^n$ |
| $\Sigma$ | Shared covariance matrix in Gaussian Discriminant Analysis | $\mathbb{R}^{n \times n}$ |
| $\phi$ | Class prior Bernoulli parameter $p(y = 1)$ | $[0, 1]$ |
| $K$ | Number of clusters in K-Means clustering | $\mathbb{N}$ |
| $\mu_k$ | Centroid coordinate vector of cluster $k$ | $\mathbb{R}^n$ |
| $c^{(i)}$ | Cluster assignment index for observation $x^{(i)}$ | $\{1, \dots, K\}$ |
| $V$ | Vocabulary size or feature attribute cardinality | $\mathbb{N}$ |

---

## Environment Setup and Installation

### Prerequisites

- Linux, macOS, or Windows WSL
- Python 3.10, 3.11, or 3.12
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/luongjuan123/Machine-Learning-Notebooks.git
cd Machine-Learning-Notebooks
```

### 2. Configure Virtual Environment

Create and activate a clean Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the core numerical and visualization dependencies:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn jupyter ipykernel
```

Register the virtual environment kernel with Jupyter:

```bash
python -m ipykernel install --user --name=ml-notebooks --display-name="Python (ML Notebooks)"
```

---

## Execution Guide

### Interactive Notebook Execution

Launch the Jupyter Notebook or JupyterLab interface:

```bash
jupyter notebook
```

Navigate to any desired directory and open the corresponding `.ipynb` notebook:

- `Gaussian Discriminant Analysis/GaussianDiscriminantAnalysis.ipynb`
- `Generalized Linear Model/GeneralizedLinearModel.ipynb`
- `K-mean Clustering/K-mean Clustering.ipynb`
- `Linear Regression/LinearRegression.ipynb`
- `Logistic Regression/LogisticRegression.ipynb`
- `Naive Bayes/NaiveBayes.ipynb`
- `Support Vector Machine/Support Vector Machine.ipynb`
- `Neuron Network/NeuronNetwork.ipynb`

Ensure that the notebook kernel is set to the virtual environment kernel (`Python (ML Notebooks)`).

### Headless / Command-Line Notebook Execution

To execute any notebook sequentially from top to bottom and update its outputs directly from the terminal:

```bash
jupyter nbconvert --to notebook --execute "Linear Regression/LinearRegression.ipynb" --inplace
```

To run all notebooks in sequence:

```bash
for nb in */*.ipynb; do
    echo "Executing $nb..."
    jupyter nbconvert --to notebook --execute "$nb" --inplace
done
```

All trained parameter weights, performance metrics, and plots will be updated and written directly into the corresponding `Output/` subdirectories.
