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

# Part 1: Data Exploration and Preprocessing

This notebook implements the data exploration and preprocessing tasks for the wearable device stress dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import glob
import os

# Set plotting style
plt.style.use('seaborn')
sns.set_theme()

def check_data_loading(data):
    """Quick validation of loaded data structure"""
    print("\nValidating loaded data...")
    
    # Basic structure checks
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"
    assert not data.empty, "DataFrame is empty"
    
    # Required columns
    required_cols = ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
    missing_cols = [col for col in required_cols if col not in data.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    
    # Data type checks
    assert pd.api.types.is_datetime64_any_dtype(data['timestamp']), "timestamp should be datetime"
    
    # Basic quality checks
    assert data.isnull().mean().max() < 0.01, "More than 1% missing values in some columns"
    
    print("✓ Data loading checks passed!")
    print(f"Shape: {data.shape}")
    print("\nSample of loaded data:")
    print(data.head())
    return True

def check_preprocessing(data):
    """Validate preprocessed data"""
    print("\nValidating preprocessed data...")
    
    # Check for missing values
    assert data.isnull().sum().sum() == 0, "Preprocessed data contains missing values"
    
    # Check for outliers (z-score > 3.5)
    for col in ['heart_rate', 'eda', 'temperature']:
        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
        assert (z_scores <= 3.5).all(), f"Outliers remain in {col}"
    
    print("✓ Preprocessing checks passed!")
    print("\nPreprocessed data summary:")
    print(data.describe())
    return True
```

## Data Loading

Implement the function to load physiological data from all subjects and sessions.

```python
def load_data(data_dir='data/raw'):
    """
    Load all physiological data from the dataset.
    
    Parameters
    ----------
    data_dir : str
        Path to the raw data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
    
    Implementation steps:
    1. Use glob to find all subject directories (S1, S2, etc.)
    2. For each subject:
       - Find session directories (Final, Midterm 1, Midterm 2)
       - Load HR.csv, EDA.csv, TEMP.csv files
       - Parse timestamps (unix timestamps in milliseconds)
       - Merge signals on timestamp
    3. Add subject_id and session columns
    4. Combine all data into single DataFrame
    """
    # Initialize empty list to store data from each subject/session
    all_data = []
    
    # Find all subject directories
    subject_dirs = glob.glob(os.path.join(data_dir, "S*"))
    
    for subject_dir in subject_dirs:
        # Extract subject ID (e.g., "S1" from the path)
        subject_id = os.path.basename(subject_dir)
        
        # Find all session directories for this subject
        session_dirs = glob.glob(os.path.join(subject_dir, "*"))
        
        for session_dir in session_dirs:
            # Extract session name (e.g., "Midterm 1")
            session = os.path.basename(session_dir)
            
            try:
                # TODO: Load the CSV files
                # Hint: Use pd.read_csv() for each signal file
                # Remember to handle the timestamp conversion
                
                # TODO: Merge the signals on timestamp
                # Hint: Use pd.merge() with 'timestamp' as the key
                
                # TODO: Add subject_id and session columns
                
                # TODO: Append to all_data list
                
            except Exception as e:
                print(f"Error processing {subject_id}/{session}: {str(e)}")
                continue
    
    # TODO: Combine all data into a single DataFrame
    # Hint: Use pd.concat(all_data)
    
    return pd.DataFrame()  # Replace with your combined DataFrame
```

## Data Preprocessing

Implement the function to preprocess the physiological data.

```python
def preprocess_data(data, output_dir='data/processed'):
    """
    Preprocess the physiological data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Raw data from load_data()
    output_dir : str
        Directory to save processed data files
        
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame
    
    Implementation steps:
    1. Handle missing values:
       - Check percentage missing per column
       - Use appropriate interpolation method
       - Verify < 1% missing values
    2. Remove outliers:
       - Calculate z-scores for each signal
       - Use threshold of 3.5
       - Document removed outliers
    3. Resample to regular intervals:
       - Determine appropriate frequency
       - Use pandas resample() with interpolation
    4. Save processed data by subject
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the data to avoid modifying the original
    processed = data.copy()
    
    # Check missing values
    print("Checking missing values...")
    missing_pct = processed.isnull().mean() * 100
    print("Missing values (%):")
    print(missing_pct)
    
    # TODO: Handle missing values
    # Hint: Use interpolate() for time series data
    # processed = processed.interpolate(method='time')
    
    # TODO: Remove outliers
    # Hint: Calculate z-scores for each signal column
    # z_scores = np.abs((processed[col] - processed[col].mean()) / processed[col].std())
    # processed = processed[z_scores <= 3.5]
    
    # TODO: Resample to regular intervals
    # Hint: Set timestamp as index first
    # processed = processed.set_index('timestamp')
    # processed = processed.resample('1S').mean()
    
    # Save processed data by subject
    for subject_id in processed['subject_id'].unique():
        subject_data = processed[processed['subject_id'] == subject_id]
        
        # TODO: Save to file
        # Hint: Use to_csv(), to_parquet(), or to_feather()
        # output_file = os.path.join(output_dir, f"{subject_id}_processed.csv")
        # subject_data.to_csv(output_file)
    
    return processed
```

## Data Visualization

Implement the function to plot physiological signals.

```python
def plot_physiological_signals(data, subject_id, session, output_dir='plots'):
    """
    Create plots of physiological signals.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Processed data from preprocess_data()
    subject_id : str
        Subject identifier (e.g., 'S1')
    session : str
        Session identifier (e.g., 'Midterm 1')
    output_dir : str
        Directory to save plot files
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with subplots for each signal
    
    Implementation steps:
    1. Create figure with 3 subplots (one per signal)
    2. Plot each signal with appropriate:
       - Labels and titles
       - Y-axis units
       - Time formatting
    3. Add grid and legend
    4. Save plot to output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for specific subject and session
    mask = (data['subject_id'] == subject_id) & (data['session'] == session)
    plot_data = data[mask].copy()
    
    if plot_data.empty:
        raise ValueError(f"No data found for subject {subject_id}, session {session}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Physiological Signals - Subject {subject_id}, {session}")
    
    # TODO: Plot heart rate
    # Hint: Use axes[0].plot()
    # Don't forget to add labels and units
    
    # TODO: Plot EDA
    # Hint: Use axes[1].plot()
    # Don't forget to add labels and units
    
    # TODO: Plot temperature
    # Hint: Use axes[2].plot()
    # Don't forget to add labels and units
    
    # Format time axis
    plt.gcf().autofmt_xdate()  # Angle x-axis labels
    
    # Add grid to all subplots
    for ax in axes:
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f"{subject_id}_{session}_signals.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig
```

## Testing

Here's a sample code to test your implementations:

```python
# Test data loading
print("Testing data loading...")
data = load_data()
if check_data_loading(data):
    print("\n✓ Data loading successful!")

# Test preprocessing
print("\nTesting preprocessing...")
processed_data = preprocess_data(data)
if check_preprocessing(processed_data):
    print("\n✓ Preprocessing successful!")

# Test plotting
print("\nTesting visualization...")
if not processed_data.empty:
    subject_id = processed_data['subject_id'].iloc[0]
    session = processed_data['session'].iloc[0]
    fig = plot_physiological_signals(processed_data, subject_id, session)
    plt.close(fig)  # Close the figure to free memory
    print("\n✓ Visualization successful!")
```
