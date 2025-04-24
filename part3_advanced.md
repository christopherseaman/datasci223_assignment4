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

# Part 3: Advanced Analysis

This notebook implements advanced time series analysis techniques for the wearable device stress dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# Add helper functions at the top level
def calculate_rmssd(rr_intervals):
    """
    Calculate RMSSD (Root Mean Square of Successive Differences)
    
    Parameters
    ----------
    rr_intervals : array-like
        Array of RR intervals in milliseconds
        
    Returns
    -------
    float
        RMSSD value in milliseconds
    """
    rr_diffs = np.diff(rr_intervals)
    return np.sqrt(np.mean(rr_diffs**2))

def calculate_sdnn(rr_intervals):
    """
    Calculate SDNN (Standard Deviation of NN Intervals)
    
    Parameters
    ----------
    rr_intervals : array-like
        Array of RR intervals in milliseconds
        
    Returns
    -------
    float
        SDNN value in milliseconds
    """
    return np.std(rr_intervals)

def calculate_pnn50(rr_intervals):
    """
    Calculate pNN50 (Percentage of NN50)
    
    Parameters
    ----------
    rr_intervals : array-like
        Array of RR intervals in milliseconds
        
    Returns
    -------
    float
        pNN50 value as percentage (0-100)
    """
    rr_diffs = np.abs(np.diff(rr_intervals))
    nn50_count = np.sum(rr_diffs > 50)  # Count differences > 50ms
    return (nn50_count / len(rr_diffs)) * 100

def check_hrv_features(features):
    """Validate HRV feature extraction"""
    print("\nValidating HRV features...")
    
    # Check required HRV metrics
    required_metrics = ['rmssd', 'sdnn', 'pnn50']
    missing_metrics = [m for m in required_metrics 
                      if not any(m in col.lower() for col in features.columns)]
    assert not missing_metrics, f"Missing HRV metrics: {missing_metrics}"
    
    # Value range checks
    assert features['pnn50'].between(0, 100).all(), "pNN50 should be percentage between 0-100"
    assert (features['sdnn'] >= 0).all(), "SDNN should be non-negative"
    
    print("✓ HRV feature checks passed!")
    print("\nHRV metrics summary:")
    print(features[required_metrics].describe())
    return True

def check_frequency_analysis(freq_results):
    """Validate frequency analysis results"""
    print("\nValidating frequency analysis...")
    
    # Check required components
    required_keys = ['frequencies', 'power', 'dominant_freq']
    missing_keys = [k for k in required_keys if k not in freq_results]
    assert not missing_keys, f"Missing required components: {missing_keys}"
    
    print("✓ Frequency analysis checks passed!")
    print("\nFrequency analysis results:")
    print(f"Dominant frequency: {freq_results['dominant_freq']:.2f} Hz")
    return True
```

## Time Domain Features

Implement the function to extract time-domain features from physiological signals.

```python
def extract_time_domain_features(data, window_size=60):
    """
    Extract time-domain features from physiological signals.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input physiological data
    window_size : int
        Size of rolling window in seconds
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with time-domain features
    
    Implementation steps:
    1. Calculate basic statistics:
       - mean, std, min, max for each signal
       - Use rolling windows if specified
    2. Calculate heart rate statistics:
       - Convert RR intervals to heart rate
       - Calculate mean HR and std HR
    3. Calculate HRV metrics:
       - RMSSD: sqrt(mean(diff(RR)^2))
       - SDNN: std(RR)
       - pNN50: percentage of RR diffs > 50ms
    4. Combine all features with units
    """
    # Make sure we have RR intervals
    if 'RR' not in data.columns and 'IBI' in data.columns:
        data = data.rename(columns={'IBI': 'RR'})
    
    # Dictionary to store features
    features = {}
    
    # Create rolling windows if window_size > 0
    if window_size > 0:
        windows = data.rolling(window=f'{window_size}S', min_periods=1)
    else:
        windows = data
    
    # TODO: Calculate basic statistics
    # Hint: Use agg() with multiple functions
    # Example:
    # basic_stats = windows['RR'].agg(['mean', 'std', 'min', 'max'])
    # features.update({
    #     'mean_rr': basic_stats['mean'],
    #     'std_rr': basic_stats['std'],
    #     'min_rr': basic_stats['min'],
    #     'max_rr': basic_stats['max']
    # })
    
    # TODO: Calculate HRV metrics
    # Hint 1: RMSSD calculation
    # rr_diffs = np.diff(rr_intervals)
    # rmssd = np.sqrt(np.mean(rr_diffs**2))
    
    # Hint 2: SDNN calculation
    # sdnn = np.std(rr_intervals)
    
    # Hint 3: pNN50 calculation
    # nn50 = np.sum(np.abs(rr_diffs) > 50)
    # pnn50 = (nn50 / len(rr_diffs)) * 100
    
    # TODO: Convert features to DataFrame
    # Add units to column names where appropriate
    # Example: 'rmssd_ms', 'hr_bpm', etc.
    
    return pd.DataFrame()  # Replace with your features DataFrame
```

## Frequency Analysis

Implement the function to analyze frequency components of physiological signals.

```python
def analyze_frequency_components(data, sampling_rate, window_size=60):
    """
    Perform frequency analysis using FFT.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input physiological data
    sampling_rate : float
        Sampling rate in Hz
    window_size : int
        Size of analysis window in seconds
        
    Returns
    -------
    dict
        Frequency analysis results
    
    Implementation steps:
    1. Prepare signal for FFT:
       - Remove mean (detrend)
       - Apply window function
       - Ensure regular sampling
    2. Compute FFT:
       - Use scipy.fft
       - Calculate power spectrum
    3. Extract frequency bands:
       - VLF: 0.003-0.04 Hz
       - LF: 0.04-0.15 Hz
       - HF: 0.15-0.4 Hz
    4. Calculate band powers and peaks
    """
    # Define frequency bands
    freq_bands = {
        'vlf': (0.003, 0.04),
        'lf': (0.04, 0.15),
        'hf': (0.15, 0.4)
    }
    
    # Prepare the signal
    # TODO: Detrend the signal
    # Hint: Use scipy.signal.detrend
    # detrended = signal.detrend(data)
    
    # TODO: Apply window function
    # Hint: Use scipy.signal.windows.hann
    # window = signal.windows.hann(len(detrended))
    # windowed = detrended * window
    
    # Compute FFT
    # TODO: Calculate FFT
    # Hint: Use scipy.fft.fft and fftfreq
    # yf = fft(windowed)
    # xf = fftfreq(len(windowed), 1/sampling_rate)
    
    # TODO: Calculate power spectrum
    # Hint: Power is magnitude squared
    # power = np.abs(yf)**2
    
    # Initialize results dictionary
    results = {
        'frequencies': None,  # Replace with frequency array
        'power': None,       # Replace with power spectrum
        'dominant_freq': None,
        'band_powers': {}
    }
    
    # TODO: Calculate power in each frequency band
    # Hint: Use boolean indexing
    # for band_name, (low, high) in freq_bands.items():
    #     mask = (xf >= low) & (xf <= high)
    #     band_power = np.sum(power[mask])
    #     results['band_powers'][band_name] = band_power
    
    # TODO: Find dominant frequency
    # Hint: Use np.argmax on the power spectrum
    # dominant_idx = np.argmax(power)
    # results['dominant_freq'] = xf[dominant_idx]
    
    return results
```

## Time-Frequency Analysis

Implement the function to perform time-frequency analysis.

```python
def analyze_time_frequency_features(data, sampling_rate, window_size=60):
    """
    Perform time-frequency analysis using STFT.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input physiological data
    sampling_rate : float
        Sampling rate in Hz
    window_size : int
        Size of analysis window in seconds
        
    Returns
    -------
    dict
        Time-frequency analysis results
    
    Implementation steps:
    1. Prepare for STFT:
       - Select window function
       - Choose overlap size
       - Set frequency resolution
    2. Compute STFT:
       - Use scipy.signal.stft
       - Calculate spectrogram
    3. Extract features:
       - Time-varying frequency content
       - Dominant frequencies
       - Band powers over time
    4. Package results with metadata
    """
    # Initialize parameters
    nperseg = int(window_size * sampling_rate)  # Window length in samples
    noverlap = nperseg // 2  # 50% overlap
    
    # Prepare the signal
    # TODO: Detrend and normalize if needed
    # Hint: Use scipy.signal.detrend
    # detrended = signal.detrend(data)
    
    # TODO: Compute STFT
    # Hint: Use scipy.signal.stft
    # f, t, Zxx = signal.stft(detrended,
    #                         fs=sampling_rate,
    #                         nperseg=nperseg,
    #                         noverlap=noverlap)
    
    # Initialize results dictionary
    results = {
        'times': None,          # Replace with time points array
        'frequencies': None,    # Replace with frequency array
        'spectrogram': None,    # Replace with 2D spectrogram array
        'dominant_freqs': None  # Replace with time-varying dominant frequencies
    }
    
    # TODO: Calculate spectrogram magnitude
    # Hint: Use np.abs() on STFT output
    # spectrogram = np.abs(Zxx)
    
    # TODO: Find dominant frequency at each time point
    # Hint: Use np.argmax along frequency axis
    # dom_freq_idx = np.argmax(spectrogram, axis=0)
    # dominant_freqs = f[dom_freq_idx]
    
    # TODO: Calculate time-varying band powers
    # Similar to frequency analysis, but for each time point
    
    # Package results
    # results.update({
    #     'times': t,
    #     'frequencies': f,
    #     'spectrogram': spectrogram,
    #     'dominant_freqs': dominant_freqs
    # })
    
    return results
```

## Testing

Here's a sample code to test your implementations:

```python
# Load preprocessed data
data_path = Path('data/processed/S1_processed.csv')
if data_path.exists():
    data = pd.read_csv(data_path)
    
    # Test time domain features
    print("Testing time domain feature extraction...")
    time_features = extract_time_domain_features(data)
    if check_hrv_features(time_features):
        print("\n✓ Time domain analysis successful!")
    
    # Test frequency analysis
    print("\nTesting frequency analysis...")
    freq_results = analyze_frequency_components(data, sampling_rate=1.0)
    if check_frequency_analysis(freq_results):
        print("\n✓ Frequency analysis successful!")
    
    # Test time-frequency analysis
    print("\nTesting time-frequency analysis...")
    tf_results = analyze_time_frequency_features(data, sampling_rate=1.0)
    if check_frequency_analysis(tf_results):
        print("\n✓ Time-frequency analysis successful!")
``` 