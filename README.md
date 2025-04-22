# TAPTAP MCP - 기출탭탭 AI 진단 모델

This repository contains the implementation of an AI diagnostic model for the TAPTAP ("기출탭탭") educational assessment platform.

## Overview

TAPTAP MCP is a machine learning-based educational assessment system that implements various Cognitive Diagnosis Models (CDM) including Item Response Theory (IRT) to analyze and predict student performance. The system is designed to provide personalized diagnostic insights based on student responses to assessment items.

## Key Components

- **EduCDM**: Educational Cognitive Diagnosis Models framework
- **IRT**: Implementation of Item Response Theory models
  - **EM**: Expectation-Maximization algorithms for parameter estimation
  - **GD**: Gradient Descent based optimization methods

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- pandas
- tqdm
- longling
- PyBaize
- fire

## Installation

```bash
# Clone the repository
git clone https://github.com/vapene/TAPTAP_mcp.git
cd TAPTAP_mcp

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Usage

Basic usage example:

```python
from IRT.irt import irt3pl
import numpy as np

# Example: Calculate item response probability
theta = 0.5  # Student ability parameter
a = 1.2      # Item discrimination parameter
b = -0.3     # Item difficulty parameter
c = 0.2      # Item guessing parameter

# Calculate the probability of a correct response
prob = irt3pl(theta, a, b, c)
print(f"Probability of correct response: {prob}")
```
