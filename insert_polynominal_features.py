"""
 ============================================================================
 Name        : insert_polynominal_features.py
 Author      : Sam Lunev
 Version     : .0
 Copyright   : All rights reserved
 Description : Block 2 DA-15 task
 ============================================================================
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def load_dataset(n_samples=1000, n_features=4, n_informative=3, n_redundant=1, 
                 n_clusters_per_class=1, random_state=42):
    """
    Load a classification dataset using sklearn's make_classification.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=4
        Total number of features
    n_informative : int, default=3
        Number of informative features
    n_redundant : int, default=1
        Number of redundant features
    n_clusters_per_class : int, default=1
        Number of clusters per class
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (X, y) where X is feature matrix and y is target vector
        
    Examples:
    ---------
    >>> X, y = load_dataset()
    >>> print(X.shape, y.shape)
    (1000, 4) (1000,)
    """
    # Generate classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state
    )
    
    return X, y


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that generates polynomial features from numeric columns.
    
    This class implements scikit-learn's transformer interface for seamless pipeline integration.
    It creates polynomial combinations of input features up to a specified degree.
    
    Parameters:
    -----------
    degree : int, default=2
        The degree of polynomial features to generate. For example, degree=2 will
        create features like x1, x2, x1*x2, x1^2, x2^2, etc.
    visualize : bool, default=True
        Whether to create and save visualization of the first two numeric columns
        before generating polynomial features. Visualization is saved as a PNG file.
    output_path : str, default='poly_features.png'
        Path where visualization will be saved if visualize=True
    feature_prefix : str, default='poly'
        Prefix for generated feature names to distinguish them from original features
    
    Attributes:
    -----------
    degree : int
        The degree of polynomial features to generate
    visualize : bool
        Whether to create and save visualization
    output_path : str
        Path where visualization will be saved
    feature_prefix : str
        Prefix for generated feature names
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> 
    >>> # Create sample data
    >>> X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
    >>> df = pd.DataFrame(X, columns=['x1', 'x2'])
    >>> 
    >>> # Generate polynomial features
    >>> generator = PolynomialFeatureGenerator(degree=2, visualize=False)
    >>> poly_features = generator.fit_transform(df)
    >>> print(poly_features.shape)
    (100, 5)
    """
    
    def __init__(self, degree=2, visualize=True, output_path='poly_features.png', 
                 feature_prefix='poly'):
        """
        Initialize the PolynomialFeatureGenerator.
        
        Parameters:
        -----------
        degree : int, default=2
            The degree of polynomial features to generate (must be non-negative)
        visualize : bool, default=True
            Whether to create and save visualization of first two numeric columns
        output_path : str, default='poly_features.png'
            Path where visualization will be saved
        feature_prefix : str, default='poly'
            Prefix for generated feature names
            
        Raises:
        -------
        ValueError
            If degree is negative or any parameter has invalid type
        """
        self.degree = degree
        self.visualize = visualize
        self.output_path = output_path
        self.feature_prefix = feature_prefix
        
        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate constructor parameters."""
        if not isinstance(self.degree, (int, float)) or self.degree < 0:
            raise ValueError("Degree must be a non-negative number.")
        if not isinstance(self.visualize, bool):
            raise ValueError("visualize must be a boolean.")
        if not isinstance(self.output_path, str):
            raise ValueError("output_path must be a string.")
        if not isinstance(self.feature_prefix, str):
            raise ValueError("feature_prefix must be a string.")

    def _validate_dataframe(self, df):
        """
        Validate input DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame to validate
            
        Raises:
        -------
        TypeError
            If input is not a pandas DataFrame
        ValueError
            If DataFrame is empty, has insufficient numeric columns, or contains invalid values
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        if df.empty:
            raise ValueError("DataFrame is empty.")
            
        # Select only numeric columns
        numeric_df = df.select_dtypes(include='number')
        
        if numeric_df.shape[1] < 2:
            raise ValueError("DataFrame must have at least two numeric columns.")
            
        # Check for invalid values (NaN, Inf)
        if numeric_df.isnull().any().any():
            raise ValueError("DataFrame contains NaN values.")
        if not np.isfinite(numeric_df.values).all():
            raise ValueError("DataFrame contains infinite or invalid numerical values.")

    def _get_numeric_columns(self, df):
        """
        Extract numeric columns from DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing only numeric columns from input
        """
        return df.select_dtypes(include='number')

    def _create_visualization(self, df, numeric_df):
        """
        Create and save visualization of first two numeric columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Original input DataFrame (used for column names)
        numeric_df : pandas.DataFrame
            DataFrame with only numeric columns
            
        Notes:
        ------
        Creates a scatter plot of the first two numeric columns colored by their product.
        Saves the visualization to self.output_path.
        """
        if len(numeric_df.columns) < 2:
            return
            
        x1_col = numeric_df.columns[0]
        x2_col = numeric_df.columns[1]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(numeric_df[x1_col], numeric_df[x2_col], alpha=0.7)
        plt.xlabel(x1_col)
        plt.ylabel(x2_col)
        plt.title(f'Scatter Plot: {x1_col} vs {x2_col}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dataframe(self, X, feature_names):
        """
        Create a DataFrame from the feature matrix.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with features and proper column names
        """
        return pd.DataFrame(X, columns=feature_names)

    def fit(self, X, y=None):
        """
        Fit the transformer (no fitting needed for this transformer).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
            
        Returns:
        --------
        self : object
            Returns self
        """
        return self

    def transform(self, X):
        """
        Generate polynomial features from the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_poly : array-like of shape (n_samples, n_output_features)
            Transformed data with polynomial features
        """
        if not isinstance(X, pd.DataFrame):
            # Convert to DataFrame if needed
            if hasattr(self, 'feature_names_'):
                feature_names = self.feature_names_
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            
        # Get numeric columns
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for polynomial feature generation")
            
        # Create polynomial features
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X_df[numeric_cols])
        
        # Get new feature names
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Create result DataFrame
        result_df = self._create_dataframe(X_poly, feature_names)
        
        if self.visualize:
            try:
                self._create_visualization(X, X_df)
            except Exception as e:
                print(f"Warning: Failed to create visualization: {e}", file=sys.stderr)
        
        return result_df

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
            
        Returns:
        --------
        X_poly : array-like of shape (n_samples, n_output_features)
            Transformed data with polynomial features
        """
        return self.transform(X)


def load_dataset_with_dataframe(n_samples=1000, n_features=4, n_informative=3, 
                               n_redundant=1, n_clusters_per_class=1, 
                               random_state=42):
    """
    Load a classification dataset and return as a DataFrame with proper column names.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=4
        Total number of features
    n_informative : int, default=3
        Number of informative features
    n_redundant : int, default=1
        Number of redundant features
    n_clusters_per_class : int, default=1
        Number of clusters per class
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features and target column 'target'
        
    Examples:
    ---------
    >>> df = load_dataset_with_dataframe()
    >>> print(df.shape)
    (1000, 5)
    """
    # Load dataset
    X, y = load_dataset(n_samples, n_features, n_informative, n_redundant, 
                       n_clusters_per_class, random_state)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add target column
    df['target'] = y
    
    return df


# Example usage

if __name__ == "__main__":
    try:
        print("Loading dataset...")
        # Using the new load_dataset function
        X, y = load_dataset()
        print(f"Dataset shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Load dataset as DataFrame
        df = load_dataset_with_dataframe()
        print(f"\nDataFrame shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())
        print("\nColumn names:", list(df.columns))
        
        # Demonstrate the transformer with the loaded data
        print("\nCreating and using PolynomialFeatureGenerator...")
        generator = PolynomialFeatureGenerator(degree=2, visualize=False)
        result = generator.fit_transform(df.iloc[:, :-1])  # Exclude target column
        
        print("Original shape:", df.iloc[:, :-1].shape)
        print("Transformed shape:", result.shape)
        print("First few transformed features:")
        print(result.head())
        
        # Test with visualization
        print("\nCreating generator with visualization...")
        generator_viz = PolynomialFeatureGenerator(degree=2, visualize=True)
        result_viz = generator_viz.fit_transform(df.iloc[:, :-1])
        print("Result with visualization created (check poly_features.png)")
        
    except Exception as e:
        print(f"Unexpected error occurred: {e}", file=sys.stderr)
