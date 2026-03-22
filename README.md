# **`README.md`**

# A Distributed Method for Cooperative Transaction Cost Mitigation

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.07881-b31b1b.svg)](https://arxiv.org/abs/2603.07881)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2603.07881)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/cooperative_transaction_cost_mitigation)
[![Discipline](https://img.shields.io/badge/Discipline-Quantitative%20Finance%20%7C%20Distributed%20Optimization-00529B)](https://github.com/cooperative_transaction_cost_mitigation)
[![Data Sources](https://img.shields.io/badge/Data-LSEG%20%7C%20FRED-lightgrey)](https://www.lseg.com/)
[![Core Method](https://img.shields.io/badge/Method-ADMM%20%7C%20Convex%20Optimization-orange)](https://github.com/cooperative_transaction_cost_mitigation)
[![Analysis](https://img.shields.io/badge/Analysis-3%2F2--Power%20TC%20%7C%20VAR(1)%20Alpha-red)](https://github.com/cooperative_transaction_cost_mitigation)
[![Validation](https://img.shields.io/badge/Validation-Primal%2FDual%20Residuals-green)](https://github.com/cooperative_transaction_cost_mitigation)
[![Robustness](https://img.shields.io/badge/Robustness-Walk--Forward%20Simulation-yellow)](https://github.com/cooperative_transaction_cost_mitigation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![CVXPY](https://img.shields.io/badge/CVXPY-Optimization-brightgreen)](https://www.cvxpy.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/cooperative_transaction_cost_mitigation)

**Repository:** `https://github.com/chirindaopensource/cooperative_transaction_cost_mitigation`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"A Distributed Method for Cooperative Transaction Cost Mitigation"** by:

*   **Nikhil Devanathan**
*   **Logan Bell**
*   **Dylan Rueter**
*   **Stephen Boyd**

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, highly optimized pipeline that executes the entire research workflow: from the ingestion and cleansing of raw market data to the econometric simulation of synthetic alpha signals, culminating in the rigorous execution of a privacy-preserving, distributed convex optimization protocol that mitigates institutional market impact.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_distributed_cooperative_optimization_pipeline`](#key-callable-run_distributed_cooperative_optimization_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Devanathan, Bell, Rueter, and Boyd (2026). The core of this repository is the iPython Notebook `cooperative_transaction_cost_mitigation_draft.ipynb`, which contains a comprehensive suite of 35+ orchestrated tasks to replicate the paper's findings.

The pipeline addresses a critical "tragedy of the commons" in multi-manager quantitative hedge funds: individual Portfolio Managers (PMs) optimize their sleeves independently, but their aggregated trades incur non-linear market impact costs that erode firm-wide returns. Solving this centrally requires PMs to divulge their proprietary alpha models and constraints, which violates institutional privacy firewalls.

The codebase operationalizes the proposed solution:
-   **Simulates** a realistic market environment using a low-rank factor risk model and a VAR(1) noise process calibrated to specific Information Coefficients (IC).
-   **Models** execution friction using a rigorous 3/2-power transaction cost function.
-   **Coordinates** autonomous PMs using the Alternating Direction Method of Multipliers (ADMM), broadcasting a synthetic "tax/subsidy" signal that internalizes the firm's marginal execution costs.
-   **Evaluates** the protocol via a 25-year walk-forward backtest, demonstrating that just 5 iterations of ADMM can capture ~75% of the savings of a fully centralized (but privacy-violating) joint optimization.

## Theoretical Background

The implemented methods combine techniques from Distributed Convex Optimization, Market Microstructure, and Financial Econometrics.

**1. The Joint Firm Problem:**
The theoretical optimum minimizes the NAV-weighted sum of PM objectives plus the aggregate transaction costs on the net trade $z$:
$$ \text{minimize } \sum_{i=1}^{M} \lambda_i f_i(x_i) + \gamma_{\text{tc}}\phi_{\text{tc}(z)} \quad \text{subject to } z = \sum_{i=1}^{M} \lambda_i x_i $$

**2. Non-Linear Transaction Cost Modeling:**
The pipeline implements the 3/2-power model, accounting for both fixed bid-ask spreads and temporary market impact scaled by volatility $\nu_j$ and dollar volume $\omega_j$:
$$ \phi_{\text{tc}}(z) = \frac{1}{2}\kappa_{\text{spread}}^T |z| + \kappa_{\text{impact}}^T |z|^{3/2} \quad \text{where} \quad (\kappa_{\text{impact}})_j = \frac{b_j \nu_j}{\sqrt{\omega_j/V}} $$

**3. The ADMM Protocol (Algorithm 1):**
To decouple the joint problem, the Central Planner broadcasts a sharing signal $\ell^k$. Each PM then solves a proximally regularized local problem:
$$ \ell^k = u^k + \frac{\rho}{M} \left( -Dz_{\text{sum}}^k + D \sum_{j=1}^M \lambda_j x^{j,k} \right) $$
$$ x^{i,k+1} = \arg\min_x \left( \lambda^i f^i(x) + \lambda^i (\ell^k)^T Dx + \frac{\rho}{2} \|\lambda^i D(x - x^{i,k})\|_2^2 \right) $$
The diagonal scaling matrix $D$ is heuristically set to the Hessian of the cost approximation: $D_{jj} = \sqrt{2(\kappa_{\text{impact}})_j}$.

**4. Econometric Simulation (VAR(1) Alpha):**
To test the protocol, synthetic alphas are generated using a stationary Vector Autoregressive process, calibrated via the discrete-time Lyapunov equation:
$$ E_t = \Phi E_{t-1} + U_t \quad \text{where} \quad \Sigma_E = \Phi \Sigma_E \Phi^T + \Sigma_U $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/cooperative_transaction_cost_mitigation/blob/main/cooperative_transaction_cost_mitigation_ipo_main.png" alt="Distributed Cooperative Optimization Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`cooperative_transaction_cost_mitigation_draft.ipynb`) implements the full research pipeline, including:

-   **Disciplined Convex Programming:** Utilizes `CVXPY` to construct and solve complex PM local objectives featuring leverage, concentration, shorting, and turnover constraints.
-   **Factored Risk Constraints:** Implements the risk constraint $\|\Sigma^{1/2}w\|_2 \le \sigma_{\text{target}}$ using a highly optimized stacked vector approach $y = [F^T w; \text{diag}(\sqrt{D^{\text{idio}}}) w]$ to avoid forming dense $N \times N$ covariance matrices.
-   **Configuration-Driven Design:** All study parameters (hyperparameters, institutional limits, econometric targets) are managed in an external, cryptographically hashed `config.yaml` file.
-   **Rigorous State Isolation:** Employs deep-copying and strict object immutability to ensure that the endogenous state trajectories of the four compared protocols (Independent, Cooperative, ADMM-2, ADMM-5) never contaminate each other during the walk-forward simulation.
-   **Cryptographic Archival:** Automatically serializes all artifacts (Parquet, JSON, NPZ) and generates an immutable `.tar.gz` tarball with a master SHA-256 fingerprint for absolute reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Engineering (Tasks 1-14):** Ingests raw market data, applies structural filters, forward-fills missing prices, constructs the $N=434$ asset universe based on end-of-sample market cap, and computes daily dollar volumes and forward returns.
2.  **Risk & Alpha Modeling (Tasks 15-21):** Estimates a $J=15$ low-rank factor covariance matrix, solves the Lyapunov equation using Kronecker properties, and simulates VAR(1) noise paths to generate synthetic alphas calibrated to specific IC targets.
3.  **Parameterization (Tasks 22-23):** Dynamically computes the endogenous market impact coefficients $\kappa_{\text{impact},t}$ and the ADMM scaling matrix $D_t^{\text{scale}}$ at each time step.
4.  **Optimization Solvers (Tasks 24-29):** Defines the CVXPY closures for the independent PM problem, the massive centralized cooperative problem, and the iterative ADMM distributed updates.
5.  **Walk-Forward Simulation (Tasks 30-32):** Executes the daily rebalancing loop, rigorously accounting for post-trade weight drift, cash returns, and transaction cost attribution.
6.  **Evaluation & Archival (Tasks 33-35):** Computes annualized performance metrics, verifies the manuscript's qualitative claims regarding PM-level outcomes, and freezes the research archive.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 35 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_distributed_cooperative_optimization_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`run_distributed_cooperative_optimization_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data validation, econometric simulation, convex optimization, and deterministic state reconstruction.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `cvxpy`, `pyyaml`.
-   Recommended Solver: `mosek` (requires license) or `scs`/`ecos` (open-source alternatives supported via config).

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/cooperative_transaction_cost_mitigation.git
    cd cooperative_transaction_cost_mitigation
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy cvxpy pyyaml
    ```

## Input Data Structure

The pipeline requires four primary data structures, strictly validated at runtime:

1.  **`raw_market_df` (pd.DataFrame):** A MultiIndex `["date", "asset_id"]` panel containing `adjusted_trade_price`, `adjusted_bid_price`, `adjusted_ask_price`, `share_volume`, and `market_cap_usd`.
2.  **`risk_free_rate_series` (pd.Series):** A DatetimeIndex series containing the 3-month U.S. Treasury Bill rate.
3.  **`master_trading_calendar` (pd.DatetimeIndex):** The canonical sequence of valid business days.
4.  **`asset_identifier_map` (pd.DataFrame):** A mapping table linking internal `asset_id`s to vendor tickers.

*Note: The pipeline includes a synthetic data generator for testing purposes if access to proprietary LSEG/CRSP data is unavailable.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic data, and use the top-level orchestrator:

```python
import os
import yaml
import pandas as pd
import numpy as np

# 1. Load the master configuration from the YAML file.
# (Assumes config.yaml is in the working directory)
with open("config.yaml", "r") as f:
    study_config = yaml.safe_load(f)

# 2. Load raw datasets (Example using synthetic generator provided in the notebook)
# In production, load from Parquet: pd.read_parquet("lseg_market_data.parquet")
(
    raw_market_df, 
    risk_free_rate_series, 
    master_trading_calendar, 
    asset_identifier_map
) = generate_synthetic_market_environment()

# 3. Execute the entire replication study.
pipeline_summary = run_distributed_cooperative_optimization_pipeline(
    raw_market_df=raw_market_df,
    risk_free_rate_series=risk_free_rate_series,
    master_trading_calendar=master_trading_calendar,
    asset_identifier_map=asset_identifier_map,
    study_config=study_config,
    output_base_dir="./institutional_research_archive"
)

# 4. Access results
if pipeline_summary.get("status") == "SUCCESS":
    print("\n[*] Final Reproduction Fidelity:")
    print(f"    Classification: {pipeline_summary['fidelity_classification']}")
    
    print("\n[*] Firm-Level Performance Metrics:")
    firm_table = pipeline_summary["reproduction_package"]["performance_metrics"]["firm_table"]
    print(firm_table.to_string())
    
    print(f"\n[*] Archive frozen at: {pipeline_summary['archive_path']}")
```

## Output Structure

The pipeline returns a master dictionary containing:
-   **`manifest`**: The cryptographic provenance record, including environment versions, random seeds, input hashes, and all manuscript-unspecified placeholder assumptions.
-   **`exogenous_artifacts`**: The cleansed market panel, frozen universe IDs, risk models, and synthetic alpha vectors.
-   **`protocol_results`**: The raw historical trajectories (NAVs, trades, costs) and ADMM convergence traces for each of the four protocols.
-   **`performance_metrics`**: The finalized firm-level and PM-level tables (Return, Volatility, Sharpe) and cumulative paths.
-   **`evaluation_summary`**: The formal verification of the manuscript's qualitative claims regarding PM-level outcomes.

## Project Structure

```
cooperative_transaction_cost_mitigation/
│
├── cooperative_transaction_cost_mitigation_draft.ipynb   # Main implementation notebook
├── config.yaml                                           # Master configuration file
├── requirements.txt                                      # Python package dependencies
│
├── LICENSE                                               # MIT Project License File
└── README.md                                             # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Institutional Constraints:** Adjust leverage ($L$), concentration ($C$), and turnover ($T$) limits to reflect different fund mandates.
-   **ADMM Hyperparameters:** Tune the penalty parameter $\rho$ or test different iteration counts ($K$).
-   **Econometrics:** Alter the target Information Coefficient (IC) range or the VAR(1) temporal autocorrelation to simulate different market regimes.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Cost Models:** Replacing the 3/2-power model with piecewise-affine or quadratic transaction cost models.
-   **Asynchronous ADMM:** Implementing asynchronous updates where PMs solve their local problems at different frequencies.
-   **Live Execution Integration:** Adapting the state-transition layer to ingest real-time FIX execution reports rather than simulated walk-forward accounting.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{devanathan2026distributed,
  title={A Distributed Method for Cooperative Transaction Cost Mitigation},
  author={Devanathan, Nikhil and Bell, Logan and Rueter, Dylan and Boyd, Stephen},
  journal={arXiv preprint arXiv:2603.07881},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). A Distributed Method for Cooperative Transaction Cost Mitigation: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/cooperative_transaction_cost_mitigation
```

## Acknowledgments

-   Credit to **Nikhil Devanathan, Logan Bell, Dylan Rueter, and Stephen Boyd** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **CVXPY** contributors for enabling robust Disciplined Convex Programming.

--

*This README was generated based on the structure and content of the `cooperative_transaction_cost_mitigation_draft.ipynb` notebook and follows best practices for research software documentation.*
