# Part 2: Time Series Modeling

This notebook implements functions for time series feature extraction and ARIMA modeling.

---

## Setup

This section sets up the plotting style for consistent visualizations throughout the notebook.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn')
sns.set_theme()
```

---

## Time Series Feature Extraction

This section implements the function to extract rolling window features from time series data.

```python
def extract_time_series_features(data, window_size=60):
    """Extract rolling window features from time series data.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
        window_size (int): Size of rolling window in seconds
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    import pandas as pd
    import numpy as np
    
    # Make a copy of the data
    df = data.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index for rolling operations
    df = df.set_index('timestamp')
    
    # Calculate window size in samples
    sampling_rate = 1 / df.index.to_series().diff().mean().total_seconds()
    window_samples = int(window_size * sampling_rate)
    
    # Features to extract
    features = {}
    signal_cols = ['heart_rate', 'eda', 'temperature']
    
    for col in signal_cols:
        if col not in df.columns:
            continue
            
        # Create rolling window
        rolling = df[col].rolling(window=window_samples, min_periods=1)
        
        # Basic statistics
        features[f'{col}_mean'] = rolling.mean()
        features[f'{col}_std'] = rolling.std()
        features[f'{col}_min'] = rolling.min()
        features[f'{col}_max'] = rolling.max()
        
        # Autocorrelation at lag 1
        features[f'{col}_autocorr'] = rolling.apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        )
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(features, index=df.index)
    
    # Add metadata columns back
    feature_df['subject_id'] = df['subject_id']
    feature_df['session'] = df['session']
    
    # Reset index to get timestamp as column
    feature_df = feature_df.reset_index()
    
    return feature_df
```

---

## ARIMA Modeling

This section implements the function to fit and evaluate ARIMA models.

```python
def build_arima_model(series, order=(1,1,1), output_dir=None):
    """Build and evaluate an ARIMA model for the given time series data.
    
    Parameters:
        series (pd.Series): Time series data to model
        order (tuple): ARIMA model order (p,d,q)
        output_dir (str): Directory to save diagnostic plots
        
    Returns:
        statsmodels.tsa.arima.model.ARIMA: The ARIMA model
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create and fit ARIMA model
    model = ARIMA(series, order=order)
    results = model.fit()
    
    if output_dir:
        # Plot 1: Original vs Fitted Values
        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series.values, 'b-', label='Original', alpha=0.7)
        plt.plot(series.index, results.fittedvalues, 'r-', label='Fitted', alpha=0.7)
        plt.title('Original vs Fitted Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'arima_fitted_values.png'))
        plt.close()
        
        # Plot 2: Residual Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residual histogram
        residuals = results.resid
        sns.histplot(residuals, kde=True, ax=axes[0,0])
        axes[0,0].set_title('Histogram of Residuals')
        axes[0,0].grid(True)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot of Residuals')
        axes[0,1].grid(True)
        
        # ACF plot
        plot_acf(residuals, ax=axes[1,0], lags=40)
        axes[1,0].set_title('ACF of Residuals')
        axes[1,0].grid(True)
        
        # PACF plot
        plot_pacf(residuals, ax=axes[1,1], lags=40)
        axes[1,1].set_title('PACF of Residuals')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'arima_residual_analysis.png'))
        plt.close()
    
    return model
```

---

## Example Usage

Here's an example of how to use these functions:

```python
# Load and preprocess data
from part1_exploration import load_data, preprocess_data

data_dir = 'data/raw'
df = load_data(data_dir)
processed_df = preprocess_data(df)

if not processed_df.empty:
    # Extract features
    features = extract_time_series_features(processed_df)
    print("Extracted features shape:", features.shape)
    print("Feature columns:", features.columns.tolist())
    
    # Build ARIMA model for heart rate from a specific subject and session
    subject_data = processed_df[
        (processed_df['subject_id'] == processed_df['subject_id'].iloc[0]) & 
        (processed_df['session'] == processed_df['session'].iloc[0])
    ]
    heart_rate_series = subject_data.set_index('timestamp')['heart_rate']
    
    # Fit ARIMA model and generate diagnostic plots
    model = build_arima_model(heart_rate_series, order=(1,1,1), output_dir='outputs/arima')
    print("\nARIMA model order:", model.order) 