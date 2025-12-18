# Convex Optimization for Subscriber Churn Classification
### Empirical Analysis of Numerical Solvers and Production Pipeline Engineering

## Project Overview
This repository investigates the application of **Regularized Logistic Regression** for predicting customer churn. Moving beyond standard heuristic models, this project focuses on the **mathematical rigor of convex loss surfaces**, comparing the efficiency and convergence of different numerical solvers (L-BFGS vs. SAGA) to achieve a "Glass-Box" model that balances predictive power with high interpretability.

The project is architected using a **Unified Pipeline** approach, ensuring that data preprocessing, feature engineering, and inference are encapsulated in a single, atomic artifact for production reliability.

---

## Mathematical Framework
The core objective is the minimization of the L2-regularized log-loss function:

$$
\min_{w} \mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(x_i^T w)) + (1-y_i) \log(1-\sigma(x_i^T w)) \right] + \lambda \|w\|_2^2
$$

This implementation specifically evaluates the performance of:
* **L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno):** A quasi-Newton method that approximates the Hessian to achieve second-order convergence.
* **SAGA:** A variant of the Stochastic Average Gradient descent that supports non-smooth functional components (L1 penalty) and offers a linear convergence rate.

---

## Technical Architecture
The system is built on three pillars of Machine Learning Engineering:

### 1. Robust Feature Engineering
* **Pipeline Encapsulation:** All transformations (Scaling, One-Hot Encoding) are strictly handled within a `ColumnTransformer` to prevent **Data Leakage**.
* **Production Safety:** Categorical encoders are configured with `handle_unknown='ignore'` to ensure API stability during novel real-world inputs.

### 2. High-Performance Serving (`api.py`)
* Powered by **FastAPI**, providing an asynchronous RESTful endpoint for low-latency churn scoring.
* Utilizes a single-artifact deployment model (`churn_pipeline.joblib`) to ensure **Training-Serving Alignment**.

### 3. Business Intelligence Layer (`dashboard.py`)
* An interactive **Streamlit** dashboard designed for stakeholders to simulate customer scenarios and visualize risk probabilities in real-time.

---

## Repository Structure

├── api.py # FastAPI Inference Service

├── dashboard.py # Streamlit Business Dashboard

├── 01_Model_Training.ipynb # Research & Solver Benchmarking

├── requirements.txt # Production Dependencies


---

# Getting Started

### 1. Installation

pip install -r requirements.txt

### 2. Training and Solver Analysis
Run the core research notebook to generate convergence plots and export the model:

Open 01_Model_Training.ipynb in Jupyter

### 3. Deployment

Start the API:
uvicorn api:app --reload

Start the Dashboard:
streamlit run dashboard.py

Author: Arij Belmabrouk

Focus: Numerical Optimization | Machine Learning Engineering | Systems Architecture
