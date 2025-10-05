# Polynomial Feature Generator by Sam Lunev

A scikit-learn compatible transformer that generates polynomial features from numeric columns.

## Overview

The `PolynomialFeatureGenerator` is a transformer that creates polynomial combinations of 
input features up to a specified degree.
It implements the scikit-learn transformer interface, making it seamless 
to integrate into machine learning pipelines.

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib
```

## Quick Start

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from polynomial_feature_generator import PolynomialFeatureGenerator

# Create sample data
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=['x1', 'x2'])

# Generate polynomial features
generator = PolynomialFeatureGenerator(degree=2, visualize=False)
poly_features = generator.fit_transform(df)
print(poly_features.shape)  # (100, 5)
```

## Parameters

|    Parameter     | Type |       Default       |                Description 
|------------------|------|---------------------|-----------------------------------------------
| `degree`         | int  | 2                   | The degree of polynomial features to generate 
| `visualize`      | bool | True                | Whether to create and save visualization 
| `output_path`    | str  | 'poly_features.png' | Path where visualization will be saved 
| `feature_prefix` | str  | 'poly'              | Prefix for generated feature names 

## Class Methods

### `__init__(degree=2, visualize=True, output_path='poly_features.png', feature_prefix='poly')`

Initialize the PolynomialFeatureGenerator.

**Parameters:**
- `degree`: The degree of polynomial features to generate (must be non-negative)
- `visualize`: Whether to create and save visualization of first two numeric columns
- `output_path`: Path where visualization will be saved
- `feature_prefix`: Prefix for generated feature names

**Raises:**
- `ValueError`: If any parameter has invalid type or value

### `fit(X, y=None)`

Fit the transformer (no fitting needed for this transformer).

**Parameters:**
- `X`: Training data
- `y`: Target values (ignored)

**Returns:**
- `self`: Returns self to allow chaining

### `transform(X)`

Generate polynomial features for the input data.

**Parameters:**
- `X`: Input data to transform

**Returns:**
- `X_poly`: DataFrame with polynomial features

**Raises:**
- `ValueError`: If input data fails validation

### `fit_transform(X, y=None)`

Fit the transformer and transform the data.

**Parameters:**
- `X`: Input data
- `y`: Target values (ignored)

**Returns:**
- `X_poly`: DataFrame with polynomial features

## Features

### Visualization Support

When `visualize=True`, the transformer creates a scatter plot of the first two numeric columns, colored by their product. 
The visualization is saved to the specified output path.

### Pipeline Integration

The transformer seamlessly integrates into scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from polynomial_feature_generator import PolynomialFeatureGenerator

# Example of pipeline usage
pipeline = Pipeline([
    ('poly_features', PolynomialFeatureGenerator(degree=2)),
    ('scaler', StandardScaler()),
])

# Use in your machine learning workflow
# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)
```

## Examples

### Basic Usage

```python
import pandas as pd
import numpy as np
from polynomial_feature_generator import PolynomialFeatureGenerator

# Create sample data
data = {
    'x1': [1, 2, 3, 4],
    'x2': [2, 4, 6, 8],
    'x3': [1, 1, 2, 2]
}
df = pd.DataFrame(data)

# Generate polynomial features
generator = PolynomialFeatureGenerator(degree=2, visualize=False)
result = generator.fit_transform(df)
print(result.head())
```

### Pipeline Integration Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from polynomial_feature_generator import PolynomialFeatureGenerator

# Create pipeline with polynomial features
pipeline = Pipeline([
    ('poly_features', PolynomialFeatureGenerator(degree=2)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Use in your machine learning workflow
# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)
```

## Input Validation

The transformer performs comprehensive input validation:
- Checks that input is a pandas DataFrame
- Ensures DataFrame is not empty
- Validates that at least two numeric columns exist
- Verifies no NaN or infinite values are present

## Output Format

The output DataFrame contains:
- Original features from the input data
- All polynomial combinations up to the specified degree
- Feature names follow the pattern: `original_feature1*original_feature2*...`

## Error Handling

The transformer raises appropriate exceptions for invalid inputs:
- `TypeError`: When input is not a pandas DataFrame
- `ValueError`: When DataFrame is empty, has insufficient columns, or contains invalid values

# All Righs Reserved