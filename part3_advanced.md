# Part 3: Advanced Analysis

This notebook implements advanced analysis techniques for physiological time series data.

---

## Setup

This section sets up the plotting style for consistent visualizations throughout the notebook.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
import pywt
import os

# Set plotting style
plt.style.use('seaborn')
sns.set_theme()
```

---

## Time Domain Features

This section implements the function to extract time-domain features from physiological signals.

```python
def extract_time_domain_features(data, window_size=60):
    """Extract time-domain features from physiological signals.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['heart_rate', 'eda', 'temperature']
        window_size (int): Size of rolling window in seconds
        
    Returns:
        pd.DataFrame: DataFrame with time-domain features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Convert timestamp to datetime if present and not already
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index if present
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Calculate sampling rate and window samples
    if isinstance(df.index, pd.DatetimeIndex):
        sampling_rate = 1 / df.index.to_series().diff().mean().total_seconds()
    else:
        sampling_rate = 1.0  # Default to 1 Hz if no timestamp
    window_samples = int(window_size * sampling_rate)
    
    features = {}
    
    # Process heart rate first to get required features
    if 'heart_rate' in df.columns:
        rolling_hr = df['heart_rate'].rolling(window=window_samples, min_periods=1)
        
        # Basic statistics (as per test requirements)
        features['mean_hr'] = rolling_hr.mean()
        features['std_hr'] = rolling_hr.std()
        
        # Calculate RR intervals (in seconds)
        rr_intervals = 60 / df['heart_rate']  # Convert HR to RR intervals
        
        # RMSSD (Root Mean Square of Successive Differences)
        rr_diff = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(rr_diff ** 2))
        features['rmssd'] = pd.Series(rmssd, index=df.index)
        
        # SDNN (Standard Deviation of NN intervals)
        sdnn = np.std(rr_intervals)
        features['sdnn'] = pd.Series(sdnn, index=df.index)
        
        # pNN50 (Percentage of successive RR intervals that differ by more than 50ms)
        pnn50 = 100 * np.sum(np.abs(rr_diff) > 0.05) / len(rr_diff)
        features['pnn50'] = pd.Series(pnn50, index=df.index)
    
    # Process all signals for basic features
    signals = ['heart_rate', 'eda', 'temperature']
    for signal in signals:
        if signal not in df.columns:
            continue
            
        rolling = df[signal].rolling(window=window_samples, min_periods=1)
        
        # Basic statistics (as per test requirements)
        if signal != 'heart_rate':  # Already handled for heart rate
            features[f'{signal}_mean'] = rolling.mean()
            features[f'{signal}_std'] = rolling.std()
            features[f'{signal}_min'] = rolling.min()
            features[f'{signal}_max'] = rolling.max()
        elif signal == 'heart_rate':
            # For heart rate, add min/max (mean/std already added above)
            features['min'] = rolling.min()
            features['max'] = rolling.max()
        
        # Additional statistics
        features[f'{signal}_skew'] = rolling.apply(stats.skew)
        features[f'{signal}_kurtosis'] = rolling.apply(stats.kurtosis)
        features[f'{signal}_roc'] = df[signal].diff()
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(features, index=df.index)
    
    # Add back metadata columns if they existed
    for col in ['subject_id', 'session']:
        if col in df.columns:
            feature_df[col] = df[col]
    
    # Reset index if it was a timestamp
    if isinstance(feature_df.index, pd.DatetimeIndex):
        feature_df = feature_df.reset_index()
    
    return feature_df
```

---

## Frequency Analysis

This section implements the function to perform frequency-domain analysis using Welch's method.

```python
def analyze_frequency_components(data, sampling_rate, window_size=60):
    """Perform frequency-domain analysis using Welch's method.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['heart_rate', 'eda', 'temperature']
        sampling_rate (float): Sampling rate in Hz
        window_size (int): Size of rolling window in seconds
        
    Returns:
        dict: Dictionary containing frequency components and power spectrum
    """
    results = {}
    signals = ['heart_rate', 'eda', 'temperature']
    
    # Define frequency bands
    freq_bands = {
        'vlf': (0.0033, 0.04),  # Very low frequency
        'lf': (0.04, 0.15),     # Low frequency
        'hf': (0.15, 0.4)       # High frequency
    }
    
    for sig in signals:
        if sig not in data.columns:
            continue
            
        # Get signal data
        signal_data = data[sig].dropna().values
        
        # Calculate power spectral density using Welch's method
        frequencies, psd = signal.welch(signal_data, 
                                      fs=sampling_rate,
                                      nperseg=window_size*sampling_rate,
                                      noverlap=window_size*sampling_rate//2)
        
        # Store frequencies and PSD
        results[f'{sig}_frequencies'] = frequencies
        results[f'{sig}_psd'] = psd
        
        # Calculate power in each frequency band
        for band_name, (low_freq, high_freq) in freq_bands.items():
            # Find indices for the frequency band
            band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
            
            # Calculate power in band using trapezoidal integration
            band_power = np.trapz(psd[band_mask], frequencies[band_mask])
            results[f'{sig}_{band_name}_power'] = band_power
        
        # Calculate LF/HF ratio
        if results[f'{sig}_hf_power'] > 0:
            results[f'{sig}_lf_hf_ratio'] = results[f'{sig}_lf_power'] / results[f'{sig}_hf_power']
        else:
            results[f'{sig}_lf_hf_ratio'] = np.nan
    
    # Add required outputs
    results['frequencies'] = frequencies
    results['power'] = psd
    
    return results
```

---

## Time-Frequency Analysis

This section implements the function to perform wavelet-based time-frequency analysis.

```python
def analyze_time_frequency_features(data, sampling_rate, window_size=60):
    """Perform wavelet-based time-frequency analysis.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['heart_rate', 'eda', 'temperature']
        sampling_rate (float): Sampling rate in Hz
        window_size (int): Size of rolling window in seconds
        
    Returns:
        dict: Dictionary containing wavelet coefficients and derived features
    """
    results = {}
    signals = ['heart_rate', 'eda', 'temperature']
    
    # Define wavelet parameters
    wavelet = 'morl'  # Morlet wavelet
    num_scales = window_size
    
    for sig in signals:
        if sig not in data.columns:
            continue
            
        # Get signal data
        signal_data = data[sig].dropna().values
        
        # Define scales for wavelet transform
        scales = np.arange(1, num_scales + 1)
        
        # Perform continuous wavelet transform
        coeffs, freqs = pywt.cwt(signal_data, scales, wavelet, sampling_period=1/sampling_rate)
        
        # Store wavelet coefficients and frequencies
        results[f'{sig}_wavelet_coeffs'] = coeffs
        results[f'{sig}_wavelet_freqs'] = freqs
        
        # Calculate wavelet-based features
        
        # Energy distribution
        energy = np.abs(coeffs)**2
        results[f'{sig}_wavelet_energy'] = np.sum(energy, axis=0)
        
        # Entropy
        normalized_energy = energy / np.sum(energy)
        entropy = -np.sum(normalized_energy * np.log2(normalized_energy + 1e-10), axis=0)
        results[f'{sig}_wavelet_entropy'] = entropy
        
        # Scale-averaged wavelet power
        results[f'{sig}_wavelet_power'] = np.mean(energy, axis=1)
        
        # Dominant frequency at each time point
        dominant_freqs = freqs[np.argmax(energy, axis=0)]
        results[f'{sig}_dominant_freqs'] = dominant_freqs
    
    return results
```

---

## Example Usage

Here's an example of how to use these functions:

```python
# Load example data
from part1_exploration import load_data, preprocess_data

data_dir = 'data/raw'
df = load_data(data_dir)
processed_df = preprocess_data(df)

if not processed_df.empty:
    # Calculate sampling rate
    sampling_rate = 1 / processed_df['timestamp'].diff().mean().total_seconds()
    
    # Extract features for first subject and session
    subject_data = processed_df[
        (processed_df['subject_id'] == processed_df['subject_id'].iloc[0]) & 
        (processed_df['session'] == processed_df['session'].iloc[0])
    ]
    
    # Time-domain features
    time_features = extract_time_domain_features(subject_data)
    print("Time-domain features shape:", time_features.shape)
    print("\nTime-domain feature columns:", time_features.columns.tolist())
    
    # Frequency components
    freq_features = analyze_frequency_components(subject_data, sampling_rate)
    print("\nFrequency components:", list(freq_features.keys()))
    
    # Time-frequency features
    time_freq_features = analyze_time_frequency_features(subject_data, sampling_rate)
    print("\nTime-frequency features:", list(time_freq_features.keys())) 