# Time Series Analysis Assignment

## Overview

In this assignment, you will apply time series analysis techniques to real-world health data from the Wearable Device Dataset from Induced Stress and Exercise Sessions. This dataset contains physiological signals (Electrodermal Activity, Blood Volume Pulse, Heart Rate, Temperature) from healthy volunteers collected during structured acute stress induction and aerobic/anaerobic exercise sessions using the Empatica E4 wearable device.

Dataset link: https://physionet.org/content/wearable-exam-stress/1.0.0/

### Methods

Subjects were required to wear the FDA-approved Empatica E4 wristband while they take their midterm (Exam 1 and 2) and final exams. The E4 recorded skin conductance, heart rate, body temperature and movement (accelerometer). 

Each E4 device has a tag number. On coming to the exam, each participant picked up an E4 device and wrote their name on the document corresponding to the E4 device they selected. After data collection the participants returned their E4 wristbands to the study team. Finally, the course instructor provided the other members of the study team with the grades corresponding to the device numbers.

### Data Description

The data contains **electrodermal activity**, **heart rate**, **blood volume pulse**, **skin surface temperature**, **inter beat interval** and **accelerometer data** recorded during **three exam sessions**(midterm 1, midterm 2 and final) as well as their corresponding grades. All the data are the direct output of the Empatica E4 device and processing has been carried out. The duration of the midterm exam are 1.5 hrs and for final exam the durations is three hours. Some useful notes on the dataset is provided below,

- StudentGrades.txt contains the grades for each student 
- The Data.zip file contains folders for each participants named as S1, S2, etc.
- Under each of the folders corresponding to each participants, there are three folders 'Final', 'Midterm 1', and 'Midterm 2', corresponding to three exams.
- Each of the folders contains csv files: 'ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'tags.csv', 'TEMP.csv', and 'info.txt'.
- 'info.txt' contains detailed information of each of these files.
- All the unix time stamps are date shifted for de-identification but not time shifted. The date shift have been carried out such a way that it does not change the status of the day light saving settings (CT/CDT) of a day.
- All exam starts at 9:00 AM (CT or CDT depending on the date corresponding to the unix time stamp). Mid terms are 1.5 hr long and final is 3 hr long.
- Sampling frequency of the arrays are provided in 'info.txt'.
- The dataset contains two female and eight male participants, however the gender is not mentioned for the purpose of de-identification.

## Dataset Download 

1. Visit https://physionet.org/
2. Navigate to the data listings
3. Locate the "A Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings" in the "Open databases" section 
4. Extract the downloaded files to the `data` directory in this repository

## Assignment Structure

This assignment is divided into three parts:

### 1: Data Exploration and Preprocessing

In this part, you will implement the following functions in `part1_exploration.ipynb`:

1. `load_data(data_dir='data/raw')`:
   - Loads all physiological data from the dataset
   - Returns a pandas DataFrame with columns: ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
   - Timestamps can be either datetime objects or numeric values
   - Data should be organized by subject and session
   - Column names are case-insensitive

2. `preprocess_data(data, output_dir='data/processed')`:
   - Handles missing values using appropriate imputation methods (up to 1% missing values allowed)
   - Resamples irregular time series to regular intervals
   - Removes outliers using z-score method (threshold=3.5)
   - Saves the processed data to files in the output directory:
     - One file per subject: 'S1_processed.*', 'S2_processed.*', etc.
     - Supported formats: .csv, .parquet, or .feather
     - Each file should contain all sessions for that subject
   - Returns the processed DataFrame

3. `plot_physiological_signals(data, subject_id, session, output_dir='plots')`:
   - Creates a figure with subplots for each physiological signal
   - Adds appropriate labels and titles
   - Saves the plot to the output directory as 'S{subject_id}_{session}_signals.png'
   - Returns the matplotlib figure object

### 2: Time Series Modeling

In this part, you will implement the following functions in `part2_modeling.ipynb`:

1. `extract_time_series_features(data, window_size=60)`:
   - Extracts rolling window features from the time series
   - For each window, calculate: mean, std, min, max, and autocorrelation at lag 1
   - Feature names are case-insensitive and can include additional text
   - Returns a DataFrame with the extracted features

2. `build_arima_model(series, order=(1,1,1), output_dir='plots')`:
   - Fits an ARIMA model to the input time series
   - Creates and saves at least 2 diagnostic plots to the output directory:
     - Plot names should include 'arima' and end with '.png'
     - Examples: 'S1_Midterm 1_heart_rate_arima_fit.png', 'S1_Midterm 1_heart_rate_arima_residuals.png'
   - Returns the fitted model object
   - The model should have predict and fit methods

### 3: Advanced Analysis 

In this part, you will implement the following functions in `part3_advanced.ipynb`:

1. `extract_time_domain_features(data, window_size=60)`:
   - Extracts time-domain features from physiological signals
   - For each window, calculate:
     - Basic statistics: mean, std, min, max
     - Heart rate statistics: mean HR, std HR
     - Beat-to-beat variability: RMSSD, SDNN, pNN50
   - Returns a DataFrame with the extracted features
   - Feature names should be descriptive and include units where appropriate

2. `analyze_frequency_components(data, sampling_rate, window_size=60)`:
   - Performs frequency-domain analysis on the signals
   - For each window, calculate:
     - Power spectral density (PSD) using Welch's method
     - Power in physiologically relevant frequency bands:
       - Very Low Frequency (VLF): 0.003-0.04 Hz
       - Low Frequency (LF): 0.04-0.15 Hz
       - High Frequency (HF): 0.15-0.4 Hz
     - LF/HF ratio as a measure of autonomic balance
   - Returns a dictionary containing:
     - 'frequencies': array of frequency values
     - 'power': array of power spectrum values
     - 'bands': dictionary of power in different frequency bands
   - Saves the frequency analysis results to the output directory:
     - File name should include 'fft'
     - Supported formats: .npz, .npy, or .csv

3. `analyze_time_frequency_features(data, sampling_rate, window_size=60)`:
   - Applies wavelet transform to analyze time-frequency features
   - For each window, calculate:
     - Continuous wavelet transform coefficients
     - Time-frequency energy distribution
     - Dominant frequency components over time
   - Returns a dictionary containing:
     - 'scales': array of wavelet scales
     - 'coefficients': array of wavelet coefficients
     - 'time_frequency_energy': 2D array of energy distribution
   - Saves the time-frequency analysis results to the output directory:
     - File name should include 'wavelet'
     - Supported formats: .npz, .npy, or .csv

## Grading Criteria

Your assignment will be graded based on:

1. **Data Exploration and Preprocessing (Part 1)**
   - Correct implementation of data loading function
   - Proper handling of missing values and outliers
   - Correct resampling of time series
   - Quality of generated visualizations
   - Proper file organization and output

2. **Time Series Modeling (Part 2)**
   - Correct feature extraction implementation
   - Proper ARIMA model fitting
   - Appropriate model evaluation

3. **Advanced Analysis (Part 3)**
   - Correct implementation of time-domain feature extraction
   - Proper frequency-domain analysis
   - Appropriate wavelet transform analysis
   - Quality of feature interpretation and visualization

## Resources

- Lecture slides and demo notebooks
- [PhysioNet Wearable Dataset Documentation](https://physionet.org/content/wearable-exam-stress/1.0.0/)
- [Pandas Time Series Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/classes.html)
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)

## Due Date

This assignment is due on [DATE] at [TIME].

## Questions

If you have any questions about the assignment, please post them on the course forum or contact the instructor.