# Machine Learning from Scratch: Foundational Algorithms and Implementations

## Project Overview

This repository provides from-scratch implementations of fundamental machine learning algorithms using Python and NumPy. The primary focus of the project is pedagogical and theoretical: demystifying the mathematical and numerical mechanics of machine learning by building models directly from linear algebra, calculus, and probability theory, without relying on black-box abstractions or third-party estimator libraries such as Scikit-Learn.

Every major learning paradigm in the repository is developed from first principles:
- Deriving optimization objectives and loss functions.
- Formulating parameter updates using first-order and second-order calculus.
- Implementing fully vectorized matrix operations with NumPy.
- Visualizing convergence behaviors, decision boundaries, and performance metrics.

---

## Core Algorithm Families

The repository organizes classical machine learning methods into clear, self-contained modules covering regression, classification, generative modeling, exponential family distributions, and unsupervised clustering.

### 1. Regression Analysis

The regression module explores linear modeling and function approximation under squared-error loss:

- **Batch Gradient Descent (BGD)**: Evaluates the full training dataset gradient at each step, updating parameters along the steepest negative gradient direction until the L1 norm of parameter change falls below a specified tolerance.
- **Stochastic Gradient Descent (SGD)**: Updates model parameters sequentially on a per-sample basis, allowing faster initial progress and reduced memory footprint.
- **Normal Equation (Analytical Closed-Form)**: Directly computes global least-squares optimal weights by solving $X^\top X \theta = X^\top y$ using the Moore-Penrose pseudoinverse, avoiding iterative optimization when feature dimensionality allows.
- **Locally Weighted Linear Regression (LWLR)**: A non-parametric regression algorithm that fits a dedicated linear model for each query point using a Gaussian distance-weighting kernel, capable of capturing non-linear relationships without explicit feature engineering.

### 2. Logistic Regression and Linear Classification

This module focuses on binary classification through probabilistic modeling and discriminant boundaries:

- **Batch Gradient Ascent and Descent**: Optimizes the log-likelihood (or binary cross-entropy loss) via iterative first-order updates with optional L2 regularization.
- **Stochastic Gradient Ascent**: Performs rapid online updates for binary classification tasks.
- **Newton's Method (Newton-Raphson Optimization)**: Employs second-order optimization by calculating both the gradient and the Hessian matrix. Exploits quadratic convergence to reach optimum parameters in significantly fewer iterations than gradient descent.
- **Perceptron Learning Algorithm**: Implements the historical linear threshold classifier using Heaviside step activation and error-driven weight adjustments.

### 3. Generalized Linear Models (GLMs)

This module generalizes linear methods to target variables whose conditional distributions belong to the Exponential Family, unified by three core assumptions:
1. The target conditional distribution follows the Exponential Family: $p(y \mid x; \theta) \sim \text{ExponentialFamily}(\eta)$.
2. The model predicts the expected value of the sufficient statistic: $h(x) = \mathbb{E}[y \mid x]$.
3. The natural parameter is a linear combination of inputs: $\eta = \theta^\top x$.

Implemented GLM models include:
- **Poisson Regression**: Models discrete count data where $y \mid x \sim \text{Poisson}(\lambda)$, using the log link function ($\eta = \log \lambda \implies \lambda = e^{\theta^\top x}$).
- **Gaussian Regression**: Recovers ordinary least-squares linear regression under the Gaussian distribution with fixed variance.
- **Bernoulli Regression**: Recovers logistic regression under the Bernoulli distribution, mapping the linear predictor through the sigmoid function.

### 4. Generative Learning Algorithms

Unlike discriminative models that directly model $p(y \mid x)$, generative models estimate the joint probability distribution $p(x, y) = p(x \mid y)p(y)$ and apply Bayes' Rule for inference.

- **Gaussian Discriminant Analysis (GDA)**:
  - Assumes continuous feature vectors follow multivariate Gaussian distributions conditioned on class label, sharing a common covariance matrix $\Sigma$.
  - Computes exact maximum likelihood estimates (MLE) for class priors, class means, and the shared covariance matrix.
  - Demonstrates that when covariance matrices are identical across classes, the resulting posterior distribution $p(y=1 \mid x)$ yields a linear decision boundary identical in form to logistic regression.
- **Naive Bayes Classification**:
  - Operates on high-dimensional discrete or text-based features under the conditional independence assumption.
  - Computes maximum likelihood parameters for feature likelihoods given class labels.
  - Employs Laplace smoothing to handle unseen features and zero-frequency edge cases.
  - Formulates decision scoring in log-space to ensure numerical stability and prevent floating-point underflow.

### 5. Unsupervised Clustering

- **K-Means Clustering**:
  - Partitions unlabeled data into $K$ clusters by minimizing the distortion objective function (inertia).
  - Implements Lloyd's coordinate descent algorithm: alternating between the sample assignment step (E-step equivalent) and the centroid update step (M-step equivalent).
  - Incorporates optimal permutation matching to evaluate cluster assignments against ground-truth labels.

### 6. Roadmap and Extensions

The repository also includes foundational stubs for:
- **Support Vector Machines (SVM)**: Formulating maximum-margin linear classifiers with soft-margin hinge-loss optimization and L2 regularization.
- **Neural Networks**: Multi-layer perceptron (MLP) architectures and backpropagation routines.

---

## Architecture and Design Principles

The implementations throughout the repository share a consistent, modular architecture:

### 1. Vectorized Matrix Computation
All data processing and algorithmic updates are implemented using vectorized NumPy array operations. By casting operations into matrix products, outer products, and broadcasting, the code avoids explicit Python loops over samples during gradient and prediction passes.

### 2. Common Model Interface
Models inherit from a lightweight base abstraction (`LinearModel`) enforcing a clean, predictable API:
- Initialization with standard hyperparameters: `step_size` (learning rate $\alpha$), `max_iter`, and convergence tolerance `eps` ($\epsilon$).
- `fit(X, y)`: Runs the solver (gradient iteration, Newton-Raphson update, or closed-form equation) to learn parameter vector $\theta$.
- `predict(X)`: Computes predictions for new observations using the learned parameters.

### 3. Reusable Utility Suite (`util`)
A shared utility pattern provides consistent helper routines across modules:
- Adding intercept/bias columns ($x_0 = 1$) to design matrices.
- Robust dataset loading from CSV files with UTF-8 BOM handling and feature extraction.
- Decision boundary computation and matplotlib visualization for 2D classification problems.
- Diagnostic evaluation tools, including confusion matrices and cluster accuracy permutation solvers.

---

## Mathematical Summary

Below is a consolidated summary of the primary models, objective functions, and update rules:

| Model | Hypothesis $h_\theta(x)$ | Objective / Loss Function | Parameter Update Rule |
|---|---|---|---|
| Linear Regression (BGD) | $\theta^\top x$ | $J(\theta) = \frac{1}{2m} \|X\theta - y\|_2^2$ | $\theta := \theta - \alpha \frac{1}{m} X^\top (X\theta - y)$ |
| Linear Regression (Normal Eq) | $\theta^\top x$ | Least Squares Closed-Form | $\theta^* = (X^\top X)^{-1} X^\top y$ |
| Locally Weighted LinReg | $\theta(x)^\top x$ | $\frac{1}{2} (X\theta - y)^\top W (X\theta - y)$ | $\theta(x) = (X^\top W X + \lambda I)^{-1} X^\top W y$ |
| Logistic Regression (BGD) | $\frac{1}{1 + e^{-\theta^\top x}}$ | $-\frac{1}{m} \sum [y \log h + (1-y)\log(1-h)]$ | $\theta := \theta - \alpha \frac{1}{m} X^\top (h_\theta(X) - y)$ |
| Logistic Regression (Newton) | $\frac{1}{1 + e^{-\theta^\top x}}$ | Negative Log-Likelihood | $\theta := \theta - H^{-1} \nabla_\theta J(\theta)$, where $H = \frac{1}{m} X^\top D X$ |
| Perceptron Algorithm | $\mathbf{1}\{\theta^\top x \geq 0\}$ | Classification Error | $\theta := \theta + \alpha (y^{(i)} - h(x^{(i)})) x^{(i)}$ |
| Poisson Regression (GLM) | $e^{\theta^\top x}$ | Log-Likelihood under Poisson | $\theta := \theta + \alpha \frac{1}{m} X^\top (y - e^{X\theta})$ |
| Gaussian Discriminant Analysis | $\frac{1}{1 + e^{-\theta^\top x}}$ | Joint Data Likelihood $p(x, y)$ | Closed-form MLE for $\mu_0, \mu_1, \Sigma, \phi$; $\theta$ computed analytically |
| Naive Bayes | $\arg\max_y p(y) \prod p(x_j \mid y)$ | Joint Data Likelihood under independence | Closed-form MLE with Laplace smoothing |
| K-Means Clustering | $\arg\min_k \|x - \mu_k\|^2$ | $J(c, \mu) = \sum \|x^{(i)} - \mu_{c^{(i)}}\|^2$ | Alternating centroid assignment and mean update |

---

## Directory Organization

The repository is structured by algorithmic topic, with each module containing its notebook, dataset, and generated output artifacts:

```
Machine-Learning-Notebooks/
|-- Gaussian Discriminant Analysis/
|   |-- GaussianDiscriminantAnalysis.ipynb    # GDA model derivation, MLE, and boundary plot
|   |-- Data/                                 # Binary classification benchmark datasets
|   |-- Output/                               # Decision boundary plots and predictions
|
|-- Generalized Linear Model/
|   |-- GeneralizedLinearModel.ipynb          # Exponential family, Poisson, Gaussian, Bernoulli
|   |-- Data/                                 # Datasets for count, continuous, and binary targets
|   |-- Output/                               # Model output plots and prediction files
|
|-- K-mean Clustering/
|   |-- K-mean Clustering.ipynb               # Centroid clustering and permutation accuracy
|   |-- Data/                                 # Synthetic clustering datasets
|   |-- Output/                               # Cluster visual plots and cluster labels
|
|-- Linear Regression/
|   |-- LinearRegression.ipynb                # BGD, SGD, Normal Equations, Locally Weighted LinReg
|   |-- Data/                                 # Regression training and test sets
|   |-- Output/                               # Fitted line plots and text predictions
|
|-- Logistic Regression/
|   |-- LogisticRegression.ipynb              # BGD, BGA, SGA, Newton's Method, Perceptron
|   |-- Data/, Data1/                         # Linearly separable and non-separable datasets
|   |-- Output/                               # Boundary visualizations and classifications
|
|-- Naive Bayes/
|   |-- NaiveBayes.ipynb                      # Generative text/feature classification
|   |-- Data/, Data1/                         # Feature data (glucose, blood pressure, diabetes)
|   |-- Output/                               # Confusion matrix heatmaps and boundary plots
|
|-- Support Vector Machine/
|   |-- Support Vector Machine.ipynb          # SVM implementation workspace
|
|-- Neuron Network/
|   |-- NeuronNetwork.ipynb                   # Neural network implementation workspace
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Standard package manager (`pip`)

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/luongjuan123/Machine-Learning-Notebooks.git
   cd Machine-Learning-Notebooks
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install numpy pandas matplotlib seaborn jupyter ipykernel
   ```

4. **Register the Jupyter kernel** (optional):
   ```bash
   python -m ipykernel install --user --name=ml-notebooks --display-name="Python (ML Notebooks)"
   ```

### Running the Notebooks

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open any topic notebook from the file browser (for example, `Linear Regression/LinearRegression.ipynb` or `Gaussian Discriminant Analysis/GaussianDiscriminantAnalysis.ipynb`) and execute cells sequentially.

To run a notebook non-interactively from the terminal:

```bash
jupyter nbconvert --to notebook --execute "Logistic Regression/LogisticRegression.ipynb" --inplace
```
