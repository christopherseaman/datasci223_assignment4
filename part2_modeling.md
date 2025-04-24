---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Part 2: Time Series Modeling

This notebook implements time series modeling tasks for the wearable device stress dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import seaborn as sns
import os
from scipy.fft import fft, fftfreq

# Set plotting style
plt.style.use('seaborn')
sns.set_theme()

def check_feature_extraction(features):
    """Validate extracted time series features"""
    print("\nValidating extracted features...")
    
    # Basic checks
    assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
    assert not features.empty, "No features extracted"
    
    # Expected feature types
    expected_features = ['mean', 'std', 'min', 'max', 'autocorr']
    missing_features = [f for f in expected_features 
                       if not any(f in col.lower() for col in features.columns)]
    assert not missing_features, f"Missing expected features: {missing_features}"
    
    print("✓ Feature extraction checks passed!")
    print(f"Number of features: {len(features.columns)}")
    print("\nFeature summary:")
    print(features.describe())
    return True

def check_arima_model(model, series):
    """Validate ARIMA model fitting"""
    print("\nValidating ARIMA model...")
    
    # Basic model checks
    assert hasattr(model, 'predict'), "Model missing predict method"
    assert hasattr(model, 'fit'), "Model missing fit method"
    
    # Check predictions
    preds = model.predict(start=0, end=len(series)-1)
    assert len(preds) == len(series), "Prediction length mismatch"
    
    print("✓ ARIMA model checks passed!")
    print("\nModel summary:")
    print(model.summary())
    return True

def calculate_power_spectrum(signal, sampling_rate):
    """
    Calculate power spectrum of a signal using FFT
    
    Parameters
    ----------
    signal : array-like
        Input signal
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    tuple
        Frequencies and corresponding power values
    """
    # Remove mean and apply window
    detrended = signal - np.mean(signal)
    window = signal.windows.hann(len(detrended))
    windowed = detrended * window
    
    # Compute FFT
    yf = fft(windowed)
    xf = fftfreq(len(windowed), 1/sampling_rate)
    
    # Calculate power (magnitude squared)
    power = np.abs(yf)**2
    
    # Return only positive frequencies
    mask = xf > 0
    return xf[mask], power[mask]
```

## Feature Extraction

Implement the function to extract time series features using rolling windows.

```python
def extract_time_series_features(data, window_size=60):
    """
    Extract rolling window features from time series data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input time series data with physiological signals
    window_size : int
        Size of rolling window in seconds
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted features
    
    Implementation steps:
    1. Create rolling windows:
       - Use pandas rolling() with specified window size
       - Handle window boundaries appropriately
    2. Calculate per-window statistics:
       - Basic: mean, std, min, max
       - Handle NaN values at window edges
    3. Calculate autocorrelation:
       - Use pandas autocorr() with lag=1
       - Handle edge cases
    4. Combine features into DataFrame:
       - Use meaningful column names
       - Include signal source in names
    """
    # Make sure timestamp is the index for rolling operations
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')
    
    # List of signals to analyze
    signals = ['heart_rate', 'eda', 'temperature']
    
    # Dictionary to store features
    features = {}
    
    # Create rolling windows
    # Note: If data is not regularly sampled, you might want to use 
    # time-based windows like '60S' instead of fixed-size windows
    windows = data.rolling(window=window_size, min_periods=1)
    
    # TODO: Calculate basic statistics for each signal
    # Hint: Use windows.agg() with multiple functions
    # Example for one signal:
    # for signal in signals:
    #     signal_stats = windows[signal].agg(['mean', 'std', 'min', 'max'])
    #     features.update({
    #         f'{signal}_{stat}': values 
    #         for stat, values in signal_stats.items()
    #     })
    
    # TODO: Calculate autocorrelation
    # Hint: Use windows.apply() with a custom function
    # Be careful with NaN values
    
    # TODO: Combine all features into a DataFrame
    # Hint: Use pd.DataFrame(features)
    
    return pd.DataFrame()  # Replace with your features DataFrame
```

## ARIMA Modeling

Implement the function to build and evaluate ARIMA models.

```python
def build_arima_model(series, order=(1,1,1), output_dir='plots'):
    """
    Fit an ARIMA model to the input time series.
    
    Parameters
    ----------
    series : pandas.Series
        Input time series data
    order : tuple
        ARIMA model order (p,d,q)
    output_dir : str
        Directory to save diagnostic plots
        
    Returns
    -------
    statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMA model
    
    Implementation steps:
    1. Check stationarity:
       - Use augmented Dickey-Fuller test
       - Difference data if needed (d parameter)
    2. Fit ARIMA model:
       - Use statsmodels ARIMA
       - Handle convergence warnings
    3. Create diagnostic plots:
       - Original vs fitted values
       - Residual analysis plots
       - Save with descriptive names
    4. Return fitted model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check stationarity with Augmented Dickey-Fuller test
    adf_result = adfuller(series)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    
    # TODO: If not stationary (p > 0.05), consider differencing
    # Hint: Use np.diff() or series.diff()
    # The 'd' parameter in order controls this
    
    try:
        # Fit ARIMA model
        # Note: statsmodels will handle differencing internally based on order[1]
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        
        # Create diagnostic plots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # TODO: Plot 1 - Original vs Fitted
        # Hint: Plot original series and model_fit.fittedvalues
        # axes[0].plot(...)
        
        # TODO: Plot 2 - Residual Analysis
        # Hint: Plot model_fit.resid
        # axes[1].plot(...)
        
        # Save plots
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'arima_diagnostics.png')
        plt.savefig(plot_file)
        plt.close()
        
        # Print model summary
        print(model_fit.summary())
        
        return model_fit
        
    except Exception as e:
        print(f"Error fitting ARIMA model: {str(e)}")
        return None
```

## Testing

Here's a sample code to test your implementations:

```python
# Load preprocessed data
data_path = Path('data/processed/S1_processed.csv')
if data_path.exists():
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Test feature extraction
    print("Testing feature extraction...")
    features = extract_time_series_features(data)
    if check_feature_extraction(features):
        print("\n✓ Feature extraction successful!")
    
    # Test ARIMA modeling
    print("\nTesting ARIMA modeling...")
    if 'heart_rate' in data.columns:
        series = data.set_index('timestamp')['heart_rate']
        model = build_arima_model(series, order=(1,1,1))
        if check_arima_model(model, series):
            print("\n✓ ARIMA modeling successful!")
``` 