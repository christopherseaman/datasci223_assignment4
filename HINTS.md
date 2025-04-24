# Assignment Hints

## General Tips
1. Start by exploring the data structure manually:
   - Look at the files in one subject's directory
   - Open a CSV file to understand its format
   - Print sample data at each step
2. Use the provided check functions to validate your work
3. When stuck, look at the demo notebooks for similar examples

## Part 1: Data Exploration and Preprocessing

### load_data Function
1. Use `glob.glob()` to find subject directories:
   ```python
   subject_dirs = glob.glob(os.path.join(data_dir, "S*"))
   ```
2. For each subject directory:
   ```python
   for subject_dir in subject_dirs:
       subject_id = os.path.basename(subject_dir)  # Gets "S1", "S2", etc.
       session_dirs = glob.glob(os.path.join(subject_dir, "*"))
       # Process each session...
   ```
3. Load CSV files using pandas:
   ```python
   hr_data = pd.read_csv("HR.csv")
   eda_data = pd.read_csv("EDA.csv")
   temp_data = pd.read_csv("TEMP.csv")
   ```
4. Convert timestamps:
   ```python
   # Unix timestamps are in milliseconds
   df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
   ```

### preprocess_data Function
1. Check missing values:
   ```python
   missing_pct = data.isnull().mean() * 100
   print("Missing values (%):")
   print(missing_pct)
   ```
2. Interpolate missing values:
   ```python
   # Linear interpolation for small gaps
   data = data.interpolate(method='linear')
   ```
3. Remove outliers:
   ```python
   z_scores = np.abs((data - data.mean()) / data.std())
   data = data[z_scores < 3.5]
   ```
4. Resample time series:
   ```python
   # Resample to 1-second intervals
   data = data.resample('1S').mean()
   ```

### plot_physiological_signals Function
1. Create subplots:
   ```python
   fig, axes = plt.subplots(3, 1, figsize=(12, 8))
   ```
2. Plot each signal:
   ```python
   axes[0].plot(data['timestamp'], data['heart_rate'])
   axes[0].set_ylabel('Heart Rate (bpm)')
   ```
3. Format time axis:
   ```python
   plt.gcf().autofmt_xdate()  # Angle x-axis labels
   ```

## Part 2: Time Series Modeling

### extract_time_series_features Function
1. Use rolling windows:
   ```python
   # Create 60-second rolling windows
   windows = data.rolling(window='60S')
   ```
2. Calculate features:
   ```python
   features = pd.DataFrame({
       'mean': windows.mean(),
       'std': windows.std(),
       'min': windows.min(),
       'max': windows.max()
   })
   ```

### build_arima_model Function
1. Check stationarity:
   ```python
   from statsmodels.tsa.stattools import adfuller
   result = adfuller(series)
   ```
2. Fit ARIMA model:
   ```python
   from statsmodels.tsa.arima.model import ARIMA
   model = ARIMA(series, order=(1,1,1))
   model_fit = model.fit()
   ```

## Part 3: Advanced Analysis

### extract_time_domain_features Function
1. Calculate HRV metrics:
   ```python
   # First convert RR intervals to milliseconds if needed
   rr_intervals = rr_intervals * 1000  # if in seconds
   
   # RMSSD calculation (Root Mean Square of Successive Differences)
   # 1. Calculate differences between successive RR intervals
   rr_diffs = np.diff(rr_intervals)
   # 2. Square the differences
   rr_diffs_squared = rr_diffs**2
   # 3. Calculate the mean and take the square root
   rmssd = np.sqrt(np.mean(rr_diffs_squared))
   
   # SDNN calculation (Standard Deviation of NN intervals)
   # 1. Calculate mean RR interval
   rr_mean = np.mean(rr_intervals)
   # 2. Calculate deviations from mean
   deviations = rr_intervals - rr_mean
   # 3. Calculate standard deviation
   sdnn = np.std(rr_intervals)  # or: np.sqrt(np.mean(deviations**2))
   
   # pNN50 calculation (Percentage of successive RR intervals > 50ms)
   # 1. Find differences > 50ms
   nn50 = np.sum(np.abs(rr_diffs) > 50)
   # 2. Calculate percentage
   pnn50 = (nn50 / len(rr_diffs)) * 100
   
   # Combine into features dictionary
   features = {
       'rmssd': rmssd,  # in milliseconds
       'sdnn': sdnn,    # in milliseconds
       'pnn50': pnn50   # in percentage
   }
   ```

2. Important considerations for HRV calculations:
   - Input RR intervals should be in milliseconds
   - Remove artifacts and ectopic beats before calculation
   - Use consecutive RR intervals only
   - For short-term recordings (5 min), SDNN and RMSSD are most reliable
   - Typical ranges for healthy adults:
     * RMSSD: 15-40 ms
     * SDNN: 30-60 ms
     * pNN50: 5-25%

3. Preprocessing RR intervals:
   ```python
   def preprocess_rr_intervals(rr_intervals):
       # Remove physiologically impossible values
       mask = (rr_intervals > 300) & (rr_intervals < 2000)  # ms
       
       # Remove outliers (optional)
       z_scores = np.abs(stats.zscore(rr_intervals))
       mask = mask & (z_scores < 3)
       
       return rr_intervals[mask]
   ```

### analyze_frequency_components Function
1. Prepare signal for FFT:
   ```python
   from scipy.signal import detrend
   detrended = detrend(signal)
   ```
2. Calculate FFT:
   ```python
   from scipy.fft import fft, fftfreq
   yf = fft(detrended)
   xf = fftfreq(len(detrended), 1/sampling_rate)
   ```

### analyze_time_frequency_features Function
1. Calculate STFT:
   ```python
   from scipy.signal import stft
   f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=window_size)
   ```

## Common Pitfalls to Avoid
1. Don't forget to handle edge cases:
   - Empty directories
   - Missing files
   - Invalid data values
2. Watch out for units:
   - Timestamps are in milliseconds
   - RR intervals are in milliseconds
   - Heart rate is in beats per minute
3. Memory management:
   - Process one subject at a time
   - Close matplotlib figures after saving
   - Use efficient data types

## Testing Your Work
1. Start with a single subject and session
2. Use the check functions provided
3. Visualize intermediate results
4. Compare your results with the demo notebooks 