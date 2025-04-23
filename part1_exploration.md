# Part 1: Data Exploration and Preprocessing

This notebook implements functions for loading, preprocessing, and visualizing physiological data from the wearable exam stress dataset.

---

## Setup

This section sets up the plotting style for consistent visualizations throughout the notebook.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
from scipy import stats

# Set plotting style
plt.style.use('seaborn')
sns.set_theme()
```

---

## Data Loading

This section implements the function to load physiological data from the dataset.

```python
def load_data(data_dir='data/raw'):
    """Load physiological data from the specified directory.
    
    Parameters:
        data_dir (str): Path to the directory containing subject data
        
    Returns:
        pd.DataFrame: DataFrame containing physiological data with columns:
            - timestamp: Time of measurement
            - heart_rate: Heart rate measurement
            - eda: Electrodermal activity measurement
            - temperature: Temperature measurement
            - subject_id: Subject identifier
            - session: Session identifier
    """
    # Define required columns
    required_columns = ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
    
    # Get list of subject directories
    subject_dirs = glob.glob(os.path.join(data_dir, "S*"))
    
    # If no subject directories found, return empty DataFrame with required columns
    if not subject_dirs:
        return pd.DataFrame(columns=required_columns)
    
    all_data = []
    
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        # Get session directories
        session_dirs = glob.glob(os.path.join(subject_dir, "*"))
        
        for session_dir in session_dirs:
            session = os.path.basename(session_dir)
            
            # Load data files
            try:
                # Load HR data
                hr_data = pd.read_csv(os.path.join(session_dir, "HR.csv"))
                hr_data.columns = ['timestamp', 'heart_rate']
                
                # Load EDA data
                eda_data = pd.read_csv(os.path.join(session_dir, "EDA.csv"))
                eda_data.columns = ['timestamp', 'eda']
                
                # Load temperature data
                temp_data = pd.read_csv(os.path.join(session_dir, "TEMP.csv"))
                temp_data.columns = ['timestamp', 'temperature']
                
                # Merge data on timestamp
                session_data = pd.merge(hr_data, eda_data, on='timestamp', how='outer')
                session_data = pd.merge(session_data, temp_data, on='timestamp', how='outer')
                
                # Add subject and session info
                session_data['subject_id'] = subject_id
                session_data['session'] = session
                
                all_data.append(session_data)
                
            except Exception as e:
                print(f"Error loading data for {subject_id}/{session}: {str(e)}")
                continue
    
    # If no valid data was found, return empty DataFrame with required columns
    if not all_data:
        return pd.DataFrame(columns=required_columns)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Convert timestamp to datetime
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'], unit='s')
    
    # Ensure all required columns are present
    for col in required_columns:
        if col not in combined_data.columns:
            combined_data[col] = None
            
    return combined_data
```

---

## Data Preprocessing

This section implements the function to preprocess the physiological data.

```python
def preprocess_data(data, output_dir='data/processed'):
    """Preprocess physiological data.
    
    Parameters:
        data (pd.DataFrame): Raw physiological data
        output_dir (str): Directory to save processed data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the data
    processed_data = data.copy()
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_dtype(processed_data['timestamp']):
        processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
    
    # Handle missing values (up to 1%)
    numeric_cols = ['heart_rate', 'eda', 'temperature']
    processed_data[numeric_cols] = processed_data[numeric_cols].interpolate(
        method='time', 
        limit=int(len(processed_data)*0.01)
    )
    
    # Remove outliers using z-score method
    z_scores = np.abs(stats.zscore(processed_data[numeric_cols].fillna(method='ffill')))
    processed_data = processed_data[z_scores.max(axis=1) < 3.5]
    
    # Resample to regular intervals (1 second)
    processed_data = processed_data.set_index('timestamp')
    processed_data = processed_data.groupby(['subject_id', 'session']).resample('1S').mean()
    processed_data = processed_data.reset_index()
    
    # Save processed data by subject
    for subject_id in processed_data['subject_id'].unique():
        subject_data = processed_data[processed_data['subject_id'] == subject_id]
        subject_data.to_csv(os.path.join(output_dir, f'{subject_id}_processed.csv'), index=False)
    
    return processed_data
```

---

## Data Visualization

This section implements the function to create plots of physiological signals.

```python
def plot_physiological_signals(data, subject_id, session, output_dir='plots'):
    """Create plots of physiological signals.
    
    Parameters:
        data (pd.DataFrame): Preprocessed physiological data
        subject_id (str): Subject identifier
        session (str): Session identifier
        output_dir (str): Directory to save plots
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for specific subject and session
    plot_data = data[(data['subject_id'] == subject_id) & (data['session'] == session)]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot heart rate
    axes[0].plot(plot_data['timestamp'], plot_data['heart_rate'], 'b-')
    axes[0].set_title('Heart Rate')
    axes[0].set_ylabel('BPM')
    axes[0].grid(True)
    
    # Plot EDA
    axes[1].plot(plot_data['timestamp'], plot_data['eda'], 'g-')
    axes[1].set_title('Electrodermal Activity')
    axes[1].set_ylabel('μS')
    axes[1].grid(True)
    
    # Plot temperature
    axes[2].plot(plot_data['timestamp'], plot_data['temperature'], 'r-')
    axes[2].set_title('Temperature')
    axes[2].set_ylabel('°C')
    axes[2].set_xlabel('Time')
    axes[2].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'{subject_id}_{session}_signals.png'))
    
    return fig
```

---

## Example Usage

Here's an example of how to use these functions:

```python
# Load data
data_dir = 'data/raw'
df = load_data(data_dir)

# Preprocess data
processed_df = preprocess_data(df)

# Create plots for first subject and session
if not processed_df.empty:
    subject_id = processed_df['subject_id'].iloc[0]
    session = processed_df['session'].iloc[0]
    fig = plot_physiological_signals(processed_df, subject_id, session)
    plt.close(fig)  # Close the figure to free memory