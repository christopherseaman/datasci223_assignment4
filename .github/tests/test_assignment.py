import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
import matplotlib.pyplot as plt

def load_notebook_functions(notebook_path):
    """Extract functions from a Jupyter notebook."""
    import nbformat
    import re
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    functions = {}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Find all function definitions in the cell
            function_defs = re.finditer(r'def\s+(\w+)\s*\(', cell.source)
            for match in function_defs:
                func_name = match.group(1)
                # Execute the cell content to get the function
                namespace = {}
                exec(cell.source, namespace)
                if func_name in namespace:
                    functions[func_name] = namespace[func_name]
    
    return functions

# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample physiological data for testing."""
    np.random.seed(42)
    n_samples = 1000
    time = np.linspace(0, 100, n_samples)
    
    # Create synthetic signals
    hr = 70 + 10 * np.sin(2 * np.pi * 0.1 * time) + np.random.normal(0, 2, n_samples)
    eda = 2 + 0.5 * np.sin(2 * np.pi * 0.05 * time) + np.random.normal(0, 0.1, n_samples)
    temp = 37 + 0.2 * np.sin(2 * np.pi * 0.02 * time) + np.random.normal(0, 0.05, n_samples)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1s'),
        'heart_rate': hr,
        'eda': eda,
        'temperature': temp,
        'subject_id': ['S1'] * n_samples,
        'session': ['Midterm 1'] * n_samples
    })
    
    return data

@pytest.fixture
def test_dirs(tmp_path):
    """Create test directories."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    filtered_dir = data_dir / "filtered"
    plots_dir = tmp_path / "plots"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    filtered_dir.mkdir(parents=True)
    plots_dir.mkdir()
    
    return {
        'raw_dir': raw_dir,
        'processed_dir': processed_dir,
        'filtered_dir': filtered_dir,
        'plots_dir': plots_dir
    }

# Part 1: Data Exploration Tests
class TestPart1:
    def test_data_loading(self, test_dirs):
        """Test data loading function."""
        try:
            functions = load_notebook_functions('part1_exploration.ipynb')
            load_data = functions.get('load_data')
        except Exception as e:
            pytest.skip(f"Could not load load_data function: {str(e)}")
            
        if load_data is None:
            pytest.skip("load_data function not found in part1_exploration.ipynb")
        
        # Test with empty data directory
        result = load_data(data_dir=str(test_dirs['raw_dir']))
        
        # Check return type and required columns
        assert isinstance(result, pd.DataFrame), "load_data should return a pandas DataFrame"
        
        # Case-insensitive column name check
        required_columns = ['timestamp', 'heart_rate', 'eda', 'temperature', 'subject_id', 'session']
        result_columns = [col.lower() for col in result.columns]
        missing_columns = [col for col in required_columns 
                         if col.lower() not in result_columns]
        assert not missing_columns, f"Missing required columns: {missing_columns}"
        
        # Flexible timestamp check
        timestamp_col = next((col for col in result.columns 
                            if col.lower() == 'timestamp'), None)
        assert timestamp_col is not None, "Could not find timestamp column"
        assert pd.api.types.is_datetime64_dtype(result[timestamp_col]) or \
               pd.api.types.is_numeric_dtype(result[timestamp_col]), \
            "timestamp column should be datetime or numeric type"

    def test_preprocessing(self, sample_data, test_dirs):
        """Test data preprocessing function."""
        try:
            functions = load_notebook_functions('part1_exploration.ipynb')
            preprocess_data = functions.get('preprocess_data')
        except Exception as e:
            pytest.skip(f"Could not load preprocess_data function: {str(e)}")
            
        if preprocess_data is None:
            pytest.skip("preprocess_data function not found in part1_exploration.ipynb")
        
        # Add some NaN values and outliers to test preprocessing
        sample_data.loc[10:15, 'heart_rate'] = np.nan
        sample_data.loc[20:25, 'heart_rate'] = 200  # Outlier
        
        # Process the data
        result = preprocess_data(sample_data, output_dir=str(test_dirs['processed_dir']))
        
        # Check return type
        assert isinstance(result, pd.DataFrame), "preprocess_data should return a pandas DataFrame"
        
        # Check for NaN values (allow small number of NaNs for edge cases)
        nan_count = result['heart_rate'].isna().sum()
        assert nan_count <= len(result) * 0.01, \
            f"Found {nan_count} NaN values in heart_rate (more than 1% of data)"
        
        # Check for outliers (allow some flexibility in threshold)
        z_scores = np.abs(stats.zscore(result['heart_rate']))
        outlier_count = (z_scores > 3.5).sum()  # Slightly more lenient threshold
        assert outlier_count <= len(result) * 0.01, \
            f"Found {outlier_count} outliers in heart_rate (more than 1% of data)"
        
        # Check output files (allow different but valid file formats)
        output_files = list(test_dirs['processed_dir'].glob('*'))
        assert len(output_files) > 0, "No output files found in processed directory"
        assert any(f.suffix in ['.csv', '.parquet', '.feather'] for f in output_files), \
            "No valid data files found (expected .csv, .parquet, or .feather)"

    def test_plotting(self, sample_data, test_dirs):
        """Test plotting function."""
        try:
            functions = load_notebook_functions('part1_exploration.ipynb')
            plot_signals = functions.get('plot_physiological_signals')
        except:
            pytest.skip("Could not load plot_physiological_signals function")
            
        if plot_signals is None:
            pytest.skip("plot_physiological_signals function not found")
        
        # Create plot
        fig = plot_signals(sample_data, 'S1', 'Midterm 1', output_dir=str(test_dirs['plots_dir']))
        
        # Check return type
        assert isinstance(fig, plt.Figure), "plot_physiological_signals should return a Figure"
        
        # Check output file
        plot_file = test_dirs['plots_dir'] / 'S1_Midterm 1_signals.png'
        assert plot_file.exists(), "Plot should be saved to file"

# Part 2: Time Series Modeling Tests
class TestPart2:
    def test_feature_extraction(self, sample_data):
        """Test time series feature extraction."""
        try:
            functions = load_notebook_functions('part2_modeling.ipynb')
            extract_features = functions.get('extract_time_series_features')
        except Exception as e:
            pytest.skip(f"Could not load extract_time_series_features function: {str(e)}")
            
        if extract_features is None:
            pytest.skip("extract_time_series_features function not found in part2_modeling.ipynb")
        
        features = extract_features(sample_data, window_size=60)
        
        # Check return type
        assert isinstance(features, pd.DataFrame), "Feature extraction should return a pandas DataFrame"
        
        # Check required features (allow different but valid feature names)
        required_features = ['mean', 'std', 'min', 'max', 'autocorr']
        feature_columns = [col.lower() for col in features.columns]
        missing_features = [feat for feat in required_features 
                          if not any(feat in col for col in feature_columns)]
        assert not missing_features, f"Missing required features: {missing_features}"

    def test_arima_modeling(self, sample_data, test_dirs):
        """Test ARIMA model building and plot generation."""
        try:
            functions = load_notebook_functions('part2_modeling.ipynb')
            build_arima = functions.get('build_arima_model')
        except Exception as e:
            pytest.skip(f"Could not load build_arima_model function: {str(e)}")
            
        if build_arima is None:
            pytest.skip("build_arima_model function not found in part2_modeling.ipynb")
        
        # Test with heart rate data
        series = sample_data['heart_rate']
        model = build_arima(series, order=(1,1,1), output_dir=str(test_dirs['plots_dir']))
        
        # Check model properties (allow different but valid model types)
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'fit'), "Model should have fit method"
        
        # Check output plots (allow different but valid plot types)
        plot_files = list(test_dirs['plots_dir'].glob('*arima*.png'))
        assert len(plot_files) >= 2, "Expected at least 2 ARIMA diagnostic plots"
        
        # Check plot content
        for plot_file in plot_files:
            try:
                img = plt.imread(plot_file)
                assert img.shape[2] in [3, 4], "Plot should be RGB or RGBA"
            except Exception as e:
                pytest.fail(f"Could not read plot file {plot_file}: {str(e)}")

# Part 3: Advanced Analysis Tests
class TestPart3:
    """Test advanced analysis functions from part3_advanced.ipynb"""
    
    def test_time_domain_features(self, sample_data):
        """Test time-domain feature extraction"""
        try:
            functions = load_notebook_functions('part3_advanced.ipynb')
            extract_features = functions.get('extract_time_domain_features')
        except Exception as e:
            pytest.skip(f"Could not load extract_time_domain_features function: {str(e)}")
            
        if extract_features is None:
            pytest.skip("extract_time_domain_features function not found in part3_advanced.ipynb")
            
        # Test with different window sizes
        for window_size in [30, 60, 120]:
            features = extract_features(sample_data, window_size=window_size)
            
            # Check required features
            required_features = [
                'mean', 'std', 'min', 'max',
                'mean_hr', 'std_hr',
                'rmssd', 'sdnn', 'pnn50'
            ]
            for feature in required_features:
                assert feature in features.columns, f"Missing required feature: {feature}"
                
            # Check feature values
            assert not features.isnull().any().any(), "Features contain NaN values"
            assert (features['rmssd'] >= 0).all(), "RMSSD should be non-negative"
            assert (features['sdnn'] >= 0).all(), "SDNN should be non-negative"
            assert (features['pnn50'] >= 0).all() and (features['pnn50'] <= 100).all(), "pNN50 should be between 0 and 100"
            
            # Check window size effect
            if window_size > 30:
                assert len(features) < len(sample_data), f"Features should be aggregated for window_size={window_size}"
        
    def test_frequency_analysis(self, sample_data):
        """Test frequency component analysis"""
        try:
            functions = load_notebook_functions('part3_advanced.ipynb')
            analyze_freq = functions.get('analyze_frequency_components')
        except Exception as e:
            pytest.skip(f"Could not load analyze_frequency_components function: {str(e)}")
            
        if analyze_freq is None:
            pytest.skip("analyze_frequency_components function not found in part3_advanced.ipynb")
            
        # Test with different window sizes
        sampling_rate = 4.0  # Hz
        for window_size in [30, 60, 120]:
            results = analyze_freq(sample_data, sampling_rate, window_size=window_size)
            
            # Check required components
            assert 'frequencies' in results, "Missing frequencies in results"
            assert 'power' in results, "Missing power spectrum in results"
            assert 'bands' in results, "Missing frequency bands in results"
            
            # Check frequency bands
            required_bands = ['VLF', 'LF', 'HF']
            for band in required_bands:
                assert band in results['bands'], f"Missing frequency band: {band}"
                
            # Check values
            assert len(results['frequencies']) == len(results['power']), "Frequency and power arrays should have same length"
            assert (results['power'] >= 0).all(), "Power values should be non-negative"
            assert 'LF/HF' in results['bands'], "Missing LF/HF ratio"
            
            # Check window size effect
            if window_size > 30:
                assert len(results['frequencies']) < len(sample_data), f"Frequencies should be aggregated for window_size={window_size}"
        
    def test_time_frequency_analysis(self, sample_data):
        """Test time-frequency feature analysis"""
        try:
            functions = load_notebook_functions('part3_advanced.ipynb')
            analyze_tf = functions.get('analyze_time_frequency_features')
        except Exception as e:
            pytest.skip(f"Could not load analyze_time_frequency_features function: {str(e)}")
            
        if analyze_tf is None:
            pytest.skip("analyze_time_frequency_features function not found in part3_advanced.ipynb")
            
        # Test with different window sizes
        sampling_rate = 4.0  # Hz
        for window_size in [30, 60, 120]:
            results = analyze_tf(sample_data, sampling_rate, window_size=window_size)
            
            # Check required components
            assert 'scales' in results, "Missing wavelet scales in results"
            assert 'coefficients' in results, "Missing wavelet coefficients in results"
            assert 'time_frequency_energy' in results, "Missing time-frequency energy in results"
            
            # Check array shapes
            assert len(results['scales']) > 0, "Should have at least one wavelet scale"
            assert results['coefficients'].ndim == 2, "Wavelet coefficients should be 2D array"
            assert results['time_frequency_energy'].ndim == 2, "Time-frequency energy should be 2D array"
            
            # Check values
            assert not np.isnan(results['coefficients']).any(), "Wavelet coefficients contain NaN values"
            assert not np.isnan(results['time_frequency_energy']).any(), "Time-frequency energy contains NaN values"
            assert (results['time_frequency_energy'] >= 0).all(), "Energy values should be non-negative"
            
            # Check window size effect
            if window_size > 30:
                assert results['coefficients'].shape[1] < len(sample_data), f"Coefficients should be aggregated for window_size={window_size}"

if __name__ == "__main__":
    pytest.main(["-v"]) 