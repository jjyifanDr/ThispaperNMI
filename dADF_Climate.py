# -*- coding: utf-8 -*-
"""
Dynamic Adaptive Thresholding for Dual-Engine Anomaly Detection Framework
Enhanced with drift-aware threshold adjustment
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
import time
import os
import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
import csv
from scipy.spatial.distance import cdist
from scipy.interpolate import NearestNDInterpolator
from scipy import stats
import traceback
import json
from datetime import datetime

# Read dataset
from ReadClimate import DataProcessor  

import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt


df = pd.read_csv('./DataSample/Climate_Label_Scientific.csv')


def pettitt_test(x, alpha=0.05):
    """
    Perform Pettitt test for change point detection.
    """
    n = len(x)
    U = np.zeros(n)
    
    # Calculate Mann-Whitney U statistic
    for t in range(1, n):
        U[t] = 0
        for i in range(t):
            for j in range(t, n):
                if x[j] > x[i]:
                    U[t] += 1
                elif x[j] < x[i]:
                    U[t] -= 1
    
    # Change point location
    K = np.argmax(np.abs(U))
    U_max = U[K]
    
    # p-value approximation
    p = 2 * np.exp(-6 * (U_max**2) / (n**3 + n**2))
    
    # Significance
    significant = p < alpha
    
    return {
        'change_point_index': K,
        'change_point_year': int(df['Year'].iloc[K]) if K < len(df) else None,
        'U_statistic': U_max,
        'p_value': p,
        'significant': significant,
        'alpha': alpha
    }


temp_series = df['Global_Temp_Anomaly'].values
result = pettitt_test(temp_series, alpha=0.05)


print("="*60)
print("PETTITT TEST RESULTS")
print("="*60)
print(f"Data range: {df['Year'].min()} - {df['Year'].max()}")
print(f"Number of samples: {len(df)}")
print(f"Detected change point year: {result['change_point_year']}")
print(f"U statistic: {result['U_statistic']:.2f}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Significant (α={result['alpha']}): {'Yes' if result['significant'] else 'No'}")

# 5. Compare with scientific consensus labels
scientific_anomalies = df[df['Label_Scientific'] == 1]
print(f"\nScientific consensus anomaly years: {sorted(scientific_anomalies['Year'].tolist())}")

# Check if detected year is among the scientific anomalies
detected_year = result['change_point_year']
is_in_scientific = detected_year in scientific_anomalies['Year'].values

print(f"\nComparison analysis:")
print(f"  Pettitt detected year {detected_year} {'is' if is_in_scientific else 'is not'} a scientific consensus anomaly year")
if is_in_scientific:
    print(f"  Matched scientific consensus anomaly: {detected_year}")
else:
    print(f"  This year is labeled as: {'Anomaly' if df[df['Year']==detected_year]['Label_Scientific'].values[0]==1 else 'Normal'} in scientific consensus")

# 6. Calculate detection rate
if len(scientific_anomalies) > 0:
    # Check if Pettitt detection falls within ±2 years of any scientific anomaly
    detected_in_window = any(
        abs(detected_year - year) <= 2 
        for year in scientific_anomalies['Year']
    )
    
    print(f"\nDetection rate analysis:")
    print(f"  Total scientific consensus anomalies: {len(scientific_anomalies)}")
    print(f"  Pettitt directly detected: {1 if is_in_scientific else 0}")
    print(f"  Pettitt detected within ±2-year window: {'Yes' if detected_in_window else 'No'}")
    
    # Find matched years within ±2 years
    if detected_in_window:
        matched_years = [
            year for year in scientific_anomalies['Year'] 
            if abs(detected_year - year) <= 2
        ]
        print(f"  Matched years in window: {matched_years}")

# 7. Visualization
plt.figure(figsize=(12, 6))

# Plot temperature series
plt.subplot(2, 1, 1)
plt.plot(df['Year'], df['Global_Temp_Anomaly'], 'b-', linewidth=1, label='Temperature Anomaly')
plt.axhline(y=df['Global_Temp_Anomaly'].mean(), color='gray', linestyle='--', alpha=0.5, label='Mean')

# Mark Pettitt change point
if result['significant']:
    plt.axvline(x=detected_year, color='red', linestyle='-', 
                linewidth=2, alpha=0.7, label=f'Pettitt Change Point ({detected_year})')

# Mark scientific consensus anomalies
scientific_years = scientific_anomalies['Year']
scientific_values = scientific_anomalies['Global_Temp_Anomaly']
plt.scatter(scientific_years, scientific_values, color='orange', 
            s=100, zorder=5, label='Scientific Anomalies', edgecolors='black')

plt.title('Global Temperature Anomaly with Pettitt Test Results')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Plot U statistic
plt.subplot(2, 1, 2)
U_values = np.zeros(len(df))
for t in range(1, len(df)):
    U = 0
    for i in range(t):
        for j in range(t, len(df)):
            if temp_series[j] > temp_series[i]:
                U += 1
            elif temp_series[j] < temp_series[i]:
                U -= 1
    U_values[t] = U

plt.plot(df['Year'], U_values, 'g-', linewidth=1)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
if result['significant']:
    plt.axvline(x=detected_year, color='red', linestyle='-', 
                linewidth=2, alpha=0.7, label=f'Max |U| at {detected_year}')

plt.title('Pettitt U-Statistic over Time')
plt.xlabel('Year')
plt.ylabel('U Statistic')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
plt.savefig('pettitt_test_results.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: pettitt_test_results.png")

# 8. Create comparison table
comparison_df = pd.DataFrame({
    'Year': scientific_anomalies['Year'],
    'Temperature_Anomaly': scientific_anomalies['Global_Temp_Anomaly'],
    'Detected_by_Pettitt': [
        'Yes' if year == detected_year else 'No' 
        for year in scientific_anomalies['Year']
    ],
    'Within_±2_years': [
        'Yes' if abs(year - detected_year) <= 2 else 'No' 
        for year in scientific_anomalies['Year']
    ]
})

print(f"\nDetailed comparison table:")
print(comparison_df.to_string(index=False))

# 9. Save comparison results
comparison_df.to_csv('pettitt_vs_scientific_comparison.csv', index=False)
print(f"Comparison results saved to: pettitt_vs_scientific_comparison.csv")

# 10. Summary statistics
print(f"\n{'='*60}")
print("Summary Statistics")
print('='*60)
print(f"Total scientific consensus anomalies: {len(scientific_anomalies)}")
print(f"Pettitt directly detected: {comparison_df['Detected_by_Pettitt'].eq('Yes').sum()}")
print(f"Pettitt detected within ±2 years: {comparison_df['Within_±2_years'].eq('Yes').sum()}")
print(f"Direct detection rate: {comparison_df['Detected_by_Pettitt'].eq('Yes').sum()/len(scientific_anomalies)*100:.1f}%")
print(f"Window detection rate: {comparison_df['Within_±2_years'].eq('Yes').sum()/len(scientific_anomalies)*100:.1f}%")


'''======================================================================================================='''
def calculate_gmean(y_true, y_pred):
    """Calculate G-mean"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    else:
        return 0.0
    
def ensure_numpy_array(data):
    """Ensure input data is a numpy array"""
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'values'):  # pandas Series or DataFrame
        return data.values
    elif torch.is_tensor(data):
        return data.cpu().numpy()
    else:
        return np.array(data)
    
    
''' ########################    data process    ############# '''
class DataCleaner:
    """Data cleaning utility"""
    
    @staticmethod
    def clean_data(X, y=None):
        """Clean data by handling NaN and outliers"""
        X_clean = X.copy()
        nan_mask = np.isnan(X_clean)
        
        if np.any(nan_mask):
            col_means = np.nanmean(X_clean, axis=0)
            for i in range(X_clean.shape[1]):
                nan_indices = np.isnan(X_clean[:, i])
                X_clean[nan_indices, i] = col_means[i]
        
        inf_mask = np.isinf(X_clean)
        if np.any(inf_mask):
            X_clean[inf_mask] = 0
        
        if np.any(np.isnan(X_clean)) or np.any(np.isinf(X_clean)):
            X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        if y is not None:
            y_clean = np.nan_to_num(y, nan=0.0)
            return X_clean, y_clean
        else:
            return X_clean
'''======================================================================================================='''        
        
        
        
        
'''=======================================================================================================''' 
"""                               dimension estimation based on Theorem 8 and  Corollary 8.1    """
    
class ImprovedDimensionEstimator:
    """Improved dimensionality estimator - fused version with Grassberger-Procaccia method"""
    
    @staticmethod
    def grassberger_procaccia(X, k=20):
        """Grassberger-Procaccia algorithm for correlation dimension"""
        try:
            from scipy.spatial.distance import cdist
            
            X = np.asarray(X)
            if len(X.shape) != 2:
                return 1
                
            n_samples = min(300, X.shape[0])
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sampled = X[indices]
            X_normalized = (X_sampled - np.mean(X_sampled, axis=0)) / (np.std(X_sampled, axis=0) + 1e-8)
            
            dist_matrix = cdist(X_normalized, X_normalized)
            np.fill_diagonal(dist_matrix, np.inf)
            r_k = np.partition(dist_matrix, k, axis=1)[:, k]
            
            r_min, r_max = np.min(r_k), np.max(r_k)
            if r_min >= r_max:
                return 1
                
            r_range = np.logspace(np.log10(r_min), np.log10(r_max), 10)
            C_r = []
            
            for r in r_range:
                C_r.append(np.mean(r_k < r))
            
            C_r = np.array(C_r)
            mask = (C_r > 0) & (C_r < 1) & (~np.isnan(C_r)) & (~np.isinf(C_r))
            
            if np.sum(mask) < 3:
                return 1
            
            log_r = np.log(r_range[mask])
            log_C = np.log(C_r[mask])
            valid_mask = (~np.isinf(log_r)) & (~np.isinf(log_C)) & (~np.isnan(log_r)) & (~np.isnan(log_C))
            
            if np.sum(valid_mask) < 3:
                return 1
                
            log_r = log_r[valid_mask]
            log_C = log_C[valid_mask]
            slope = np.polyfit(log_r, log_C, 1)[0]
            intrinsic_dim = max(1, int(round(slope)))
            
            return min(intrinsic_dim, X.shape[1])
            
        except Exception:
            return 1
'''======================================================================================================='''        
        
        


'''======================================================================================================='''        
''' ########################    Dynamic Adaptive Threshold Calculation    ############# '''
"""Enhanced dynamic adaptive threshold strategy for data drift"""

class DynamicAdaptiveThreshold:
    """Dynamic adaptive threshold calculation for handling data drift"""
    
    def __init__(self, window_size=1000, base_percentile=90.0, adaptation_rate=0.05):
        """
        Initialize dynamic threshold calculator
        
        Parameters:
        window_size: Size of sliding window for statistics
        base_percentile: Base percentile for threshold calculation
        adaptation_rate: Rate of adaptation to new data (0-1)
        """
        self.window_size = window_size
        self.base_percentile = base_percentile
        self.adaptation_rate = adaptation_rate
        self.score_history = []
        self.threshold_history = []
        self.drift_detected_history = []
        self.confidence_history = []  
        self.current_threshold = None
        self.current_mean = None
        self.current_std = None
        self.current_skew = None
             
    def update(self, new_scores, detect_drift=True):
        """
        Update threshold with new scores
        
        Parameters:
        new_scores: New anomaly scores
        detect_drift: Whether to detect and adapt to drift
        """
        new_scores = np.array(new_scores).flatten()
        
        # Add new scores to history
        self.score_history.extend(new_scores.tolist())
        
        # Maintain window size
        if len(self.score_history) > self.window_size:
            self.score_history = self.score_history[-self.window_size:]
        
        if len(self.score_history) < 50:  # Minimum samples for reliable statistics
            return self.current_threshold if self.current_threshold is not None else 0.5
        
        scores_array = np.array(self.score_history)
        
        # Calculate statistics
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        skew = stats.skew(scores_array) if len(scores_array) > 2 else 0
        
        if detect_drift:
            drift_detected = self._detect_drift(mean, std, skew)
            self.drift_detected_history.append(drift_detected)
            
            if drift_detected:
                # Adaptive adjustment for drift
                adjusted_percentile = self._adapt_to_drift(scores_array, mean, std, skew)
                threshold = np.percentile(scores_array, adjusted_percentile)
                print(f"Drift detected! Adjusted percentile: {adjusted_percentile:.2f}")
            else:
                # Normal operation
                threshold = np.percentile(scores_array, self.base_percentile)
        else:
            threshold = np.percentile(scores_array, self.base_percentile)
        
        # Apply smoothing if we have previous threshold
        if self.current_threshold is not None:
            threshold = (1 - self.adaptation_rate) * self.current_threshold + \
                       self.adaptation_rate * threshold
        
        # Update current values
        self.current_threshold = threshold
        self.current_mean = mean
        self.current_std = std
        self.current_skew = skew
        self.threshold_history.append(threshold)
        
        # Calculate and record confidence
        confidence = self.get_threshold_confidence()
        self.confidence_history.append(confidence)
        
        return threshold
    
    def _detect_drift(self, new_mean, new_std, new_skew):
        """Detect if data drift has occurred"""
        if self.current_mean is None or self.current_std is None:
            return False
        
        # Multiple drift detection criteria
        mean_change = abs(new_mean - self.current_mean) / (self.current_std + 1e-8)
        std_change = abs(new_std - self.current_std) / (self.current_std + 1e-8)
        
        # Statistical test for distribution change (simplified)
        drift_score = 0
        
        if mean_change > 1.0:  # Mean changed more than 1 std
            drift_score += 1
        
        if std_change > 0.5:  # Std changed more than 50%
            drift_score += 1
        
        if len(self.score_history) > 100:
            # Compare recent vs older statistics
            recent_scores = np.array(self.score_history[-100:])
            older_scores = np.array(self.score_history[-200:-100]) if len(self.score_history) >= 200 else recent_scores
            
            recent_mean = np.mean(recent_scores)
            recent_std = np.std(recent_scores)
            older_mean = np.mean(older_scores)
            older_std = np.std(older_scores)
            
            if abs(recent_mean - older_mean) > 0.5 * older_std:
                drift_score += 1
        
        return drift_score >= 2  # At least 2 indicators suggest drift
    
    def _adapt_to_drift(self, scores_array, mean, std, skew):
        """Adapt percentile based on drift characteristics"""
        adjusted_percentile = self.base_percentile
        
        # 1. Enhanced adjustment based on skewness
        if abs(skew) > 0.3:
            if skew > 0:  # Positive skew (long tail on right)
                adjusted_percentile += min(1.5, abs(skew) * 2.0)  # Increase more
            else:  # Negative skew
                adjusted_percentile -= min(1.0, abs(skew) * 1.5)
        
        # 2. Adjustment based on kurtosis (if there are many extreme values)
        if len(scores_array) > 10:
            from scipy import stats
            kurt = stats.kurtosis(scores_array)
            if kurt > 3:  # High kurtosis (heavy tails)
                adjusted_percentile += min(1.0, (kurt - 3) * 0.5)
            elif kurt < 2:  # Low kurtosis (light tails)
                adjusted_percentile -= min(0.5, (3 - kurt) * 0.3)
        
        # 3. Adjustment based on expected anomaly rate
        # Financial data typically has anomaly rate of 1-5%
        expected_anomaly_rate = 0.02  # Expected 2% anomaly rate
        current_high_score_ratio = np.mean(scores_array > np.percentile(scores_array, 95))
        
        if current_high_score_ratio > expected_anomaly_rate * 2:
            # Current high score ratio is too high, increase threshold
            adjusted_percentile += 0.5
        elif current_high_score_ratio < expected_anomaly_rate / 2:
            # Current high score ratio is too low, decrease threshold
            adjusted_percentile -= 0.5
        
        # Ensure threshold is within reasonable range
        adjusted_percentile = max(95.0, min(99.9, adjusted_percentile))
        
        return adjusted_percentile
    
    
    def get_threshold_confidence(self):
        """Get confidence level of current threshold"""
        if self.current_std is None or len(self.score_history) < 50:
            return 0.5
        
        # Confidence based on stability and sample size
        stability = 1.0 / (1.0 + self.current_std)  # Higher std = less confidence
        sample_sufficiency = min(1.0, len(self.score_history) / self.window_size)
        
        return stability * 0.6 + sample_sufficiency * 0.4
    

        
        
               
''' ########################    Advanced Dynamic Threshold Manager    ############# '''
"""Enhanced dynamic threshold manager with performance feedback and adaptive adjustment"""

class AdvancedDynamicThresholdManager:
    """Enhanced dynamic threshold manager supporting performance feedback and adaptive adjustment"""
    
    def __init__(self, window_size=1000, base_percentile=90.0, 
                 adaptation_rate=0.1, target_anomaly_rate=0.02):
        """
        Initialize advanced dynamic threshold manager
        
        Parameters:
        window_size: Sliding window size
        base_percentile: Base percentile
        adaptation_rate: Adaptation rate (0-1)
        target_anomaly_rate: Target anomaly rate
        """
        self.dynamic_thresholder = DynamicAdaptiveThreshold(
            window_size=window_size, 
            base_percentile=base_percentile, 
            adaptation_rate=adaptation_rate
        )
        self.target_anomaly_rate = target_anomaly_rate
        self.performance_history = []
        self.threshold_adjustment_factor = 1.0
        self.consecutive_low_performance = 0
        self.optimal_thresholds = []
        
    def update_with_feedback(self, scores, y_true=None, predictions=None):
        """
        Threshold update with feedback
        
        Parameters:
        scores: Anomaly scores
        y_true: True labels (if available)
        predictions: Model predictions (if true labels not available)
        
        Returns:
        Adjusted threshold
        """
        # Update base dynamic threshold
        base_threshold = self.dynamic_thresholder.update(scores)
        
        # Calculate statistics
        current_scores = np.array(scores).flatten()
        score_mean = np.mean(current_scores)
        score_std = np.std(current_scores)
        score_skew = stats.skew(current_scores) if len(current_scores) > 2 else 0
        
        # If true labels are available, perform performance evaluation and adjustment
        adjusted_threshold = base_threshold
        
        if y_true is not None and len(y_true) > 0:
            # Adjust y_true to match scores length
            y_true_adj = y_true[:len(current_scores)] if len(y_true) > len(current_scores) else y_true
            
            # Calculate performance at current threshold
            current_predictions = (current_scores > base_threshold).astype(int)
            if len(current_predictions) > len(y_true_adj):
                current_predictions = current_predictions[:len(y_true_adj)]
            elif len(current_predictions) < len(y_true_adj):
                y_true_adj = y_true_adj[:len(current_predictions)]
            
            current_f1 = f1_score(y_true_adj, current_predictions, zero_division=0)
            current_precision = precision_score(y_true_adj, current_predictions, zero_division=0)
            current_recall = recall_score(y_true_adj, current_predictions, zero_division=0)
            current_anomaly_rate = np.mean(current_predictions)
            
            # Store performance history
            performance_record = {
                'threshold': base_threshold,
                'f1': current_f1,
                'precision': current_precision,
                'recall': current_recall,
                'anomaly_rate': current_anomaly_rate,
                'time': len(self.performance_history),
                'score_mean': score_mean,
                'score_std': score_std,
                'score_skew': score_skew
            }
            self.performance_history.append(performance_record)
            
            # Find optimal threshold (if supervised signals are available)
            optimal_threshold = self._find_optimal_threshold(current_scores, y_true_adj)
            if optimal_threshold is not None:
                self.optimal_thresholds.append(optimal_threshold)
                # If there are multiple optimal thresholds in history, calculate average
                if len(self.optimal_thresholds) > 5:
                    optimal_avg = np.mean(self.optimal_thresholds[-5:])
                    # Incorporate optimal threshold information into adjustment
                    adjusted_threshold = 0.7 * base_threshold + 0.3 * optimal_avg
            
            # Adjust threshold factor based on performance
            if len(self.performance_history) > 10:
                recent_performance = self.performance_history[-10:]
                recent_f1 = np.mean([p['f1'] for p in recent_performance])
                recent_anomaly_rate = np.mean([p['anomaly_rate'] for p in recent_performance])
                
                # Performance evaluation
                if recent_f1 < 0.3:
                    self.consecutive_low_performance += 1
                else:
                    self.consecutive_low_performance = max(0, self.consecutive_low_performance - 1)
                
                # If consecutive low performance, perform aggressive adjustment
                if self.consecutive_low_performance > 5:
                    # Aggressive adjustment: recalculate based on statistical characteristics
                    adjusted_threshold = score_mean + 3.0 * score_std
                    print(f"Aggressive adjustment: detected consecutive low performance, threshold reset to {adjusted_threshold:.4f}")
                else:
                    # Normal adjustment: based on anomaly rate
                    if recent_anomaly_rate > self.target_anomaly_rate * 1.5:
                        # Anomaly rate too high, increase threshold
                        self.threshold_adjustment_factor *= 1.05
                    elif recent_anomaly_rate < self.target_anomaly_rate / 2:
                        # Anomaly rate too low, decrease threshold
                        self.threshold_adjustment_factor *= 0.95
                    
                    # Adjust based on score distribution
                    if score_skew > 0.5:
                        # Positive skew distribution, increase threshold to capture more anomalies
                        self.threshold_adjustment_factor *= 1.02
                    elif score_skew < -0.5:
                        # Negative skew distribution, decrease threshold
                        self.threshold_adjustment_factor *= 0.98
                    
                    # Limit adjustment range
                    self.threshold_adjustment_factor = max(0.7, min(1.3, self.threshold_adjustment_factor))
                    
                    # Apply adjustment
                    adjusted_threshold = base_threshold * self.threshold_adjustment_factor
                    
                    # Output adjustment information
                    if len(self.performance_history) % 20 == 0:
                        print(f"Threshold adjustment: base={base_threshold:.4f}, adjustment factor={self.threshold_adjustment_factor:.3f}, "
                              f"final={adjusted_threshold:.4f}, F1={current_f1:.3f}, anomaly rate={current_anomaly_rate:.3f}")
        
        # If no true labels but predictions are provided, perform simple adjustment
        elif predictions is not None and len(predictions) > 0:
            current_anomaly_rate = np.mean(predictions)
            
            # Simple adjustment based on predicted anomaly rate
            if current_anomaly_rate > self.target_anomaly_rate * 1.5:
                adjusted_threshold = base_threshold * 1.05
            elif current_anomaly_rate < self.target_anomaly_rate / 2:
                adjusted_threshold = base_threshold * 0.95
        
        # Apply smoothing (prevent sudden changes)
        if len(self.performance_history) > 1:
            last_threshold = self.performance_history[-1]['threshold']
            threshold_change = abs(adjusted_threshold - last_threshold)
            if threshold_change > 0.1:
                # Change too large, apply stronger smoothing
                adjusted_threshold = 0.3 * adjusted_threshold + 0.7 * last_threshold
        
        return adjusted_threshold
    
    
    def _find_optimal_threshold(self, scores, y_true):
        """Find supervised optimal threshold"""
        if len(scores) != len(y_true):
            return None
        
        try:
            # Prepare candidate thresholds
            score_range = np.max(scores) - np.min(scores)
            
            if score_range > 0.1:
                # Generate candidate thresholds
                if np.mean(y_true) < 0.01:  # Extremely low anomaly rate
                    candidates = np.percentile(scores, [99.0, 99.5, 99.7, 99.9])
                else:
                    candidates = np.linspace(np.percentile(scores, 80), 
                                           np.percentile(scores, 99.9), 20)
            else:
                candidates = np.linspace(np.min(scores), np.max(scores), 20)
            
            # Find optimal threshold
            best_threshold = None
            best_f1 = 0
            
            for threshold in candidates:
                predictions = (scores > threshold).astype(int)
                if np.sum(predictions) == 0:
                    continue
                f1 = f1_score(y_true, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            return best_threshold
        except Exception as e:
            print(f"Error finding optimal threshold: {e}")
            return None
    
    def get_performance_summary(self):
        """Get performance summary"""
        if not self.performance_history:
            return {"total_records": 0, "avg_f1": 0, "avg_anomaly_rate": 0}
        
        total_records = len(self.performance_history)
        recent_records = self.performance_history[-min(20, total_records):]
        
        avg_f1 = np.mean([r['f1'] for r in recent_records])
        avg_precision = np.mean([r['precision'] for r in recent_records])
        avg_recall = np.mean([r['recall'] for r in recent_records])
        avg_anomaly_rate = np.mean([r['anomaly_rate'] for r in recent_records])
        
        return {
            "total_records": total_records,
            "avg_f1": avg_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_anomaly_rate": avg_anomaly_rate,
            "current_adjustment_factor": self.threshold_adjustment_factor,
            "target_anomaly_rate": self.target_anomaly_rate,
            "consecutive_low_performance": self.consecutive_low_performance
        }
               
        
"""Enhanced Theorem 5-based threshold strategy with dynamic adaptation"""

class TheoremBasedThresholdDynamic:
    """Theorem 5-based threshold strategy with dynamic adaptation"""
    
    def __init__(self, window_size=500):
        self.dynamic_threshold_goe = DynamicAdaptiveThreshold(window_size=window_size)
        self.dynamic_threshold_mlnn = DynamicAdaptiveThreshold(window_size=window_size)
        self.dynamic_threshold_dual = DynamicAdaptiveThreshold(window_size=window_size)
        
    @staticmethod
    def enhanced_synergy_strategy(goe_scores, mlnn_scores, y_true=None, use_dynamic=True, 
                                  dynamic_thresholder=None):
        """Enhanced synergy strategy combining Theorem 5 and dynamic adaptation"""
        goe_norm = (goe_scores - np.min(goe_scores)) / (np.max(goe_scores) - np.min(goe_scores) + 1e-8)
        mlnn_norm = (mlnn_scores - np.min(mlnn_scores)) / (np.max(mlnn_scores) - np.min(mlnn_scores) + 1e-8)
        
        if np.max(goe_norm) < 0.01 and np.max(mlnn_norm) < 0.01:
            return 0.5, (goe_norm + mlnn_norm) / 2
        
        correlation = np.corrcoef(goe_norm, mlnn_norm)[0, 1]
        
        if abs(correlation) < 0.3:
            goe_weight = 0.55
            mlnn_weight = 0.45
        else:
            goe_weight = 0.7
            mlnn_weight = 0.3
        
        final_scores = goe_weight * goe_norm + mlnn_weight * mlnn_norm
        
        if y_true is not None:
            # Use supervised threshold optimization if labels available
            min_length = min(len(final_scores), len(y_true))
            final_scores = final_scores[:min_length]
            y_true = y_true[:min_length]
            
            best_threshold = 0.5
            best_f1 = 0
            
            if np.mean(y_true) < 0.01:
                thresholds = np.percentile(final_scores, [99.0, 99.5, 99.8, 99.9])
            else:
                thresholds = np.linspace(np.percentile(final_scores, 80), 
                                       np.percentile(final_scores, 99.9), 20)
            
            for threshold in thresholds:
                predictions = (final_scores > threshold).astype(int)
                if np.sum(predictions) == 0:
                    continue
                f1 = f1_score(y_true, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            print(f"Supervised optimal threshold: {best_threshold:.4f}, Best F1: {best_f1:.4f}")
            
            # Update dynamic thresholder with supervised result
            if dynamic_thresholder is not None:
                dynamic_thresholder.update([best_threshold])
            
            return best_threshold, final_scores
        else:
            # Unsupervised threshold selection
            if use_dynamic and dynamic_thresholder is not None:
                # Use dynamic thresholding
                threshold = dynamic_thresholder.update(final_scores)
                print(f"Dynamic adaptive threshold: {threshold:.4f}, "
                      f"Confidence: {dynamic_thresholder.get_threshold_confidence():.3f}")
            else:
                # Fallback to statistical threshold
                statistical_threshold = np.percentile(final_scores, 99)
                threshold = statistical_threshold
                print(f"Statistical threshold: {threshold:.4f}")
            
            return threshold, final_scores
    
    def adaptive_fusion_strategy_dynamic(self, goe_scores, mlnn_scores, y_true=None):
        """Dynamic fusion strategy with separate threshold adaptation per engine"""
        # Normalize scores
        goe_norm = (goe_scores - np.min(goe_scores)) / (np.max(goe_scores) - np.min(goe_scores) + 1e-8)
        mlnn_norm = (mlnn_scores - np.min(mlnn_scores)) / (np.max(mlnn_scores) - np.min(mlnn_scores) + 1e-8)
        
        # Update individual engine thresholds
        goe_threshold = self.dynamic_threshold_goe.update(goe_norm)
        mlnn_threshold = self.dynamic_threshold_mlnn.update(mlnn_norm)
        
        print(f"GOE dynamic threshold: {goe_threshold:.4f}, "
              f"MLNN dynamic threshold: {mlnn_threshold:.4f}")
        
        # Calculate confidence-weighted fusion
        goe_confidence = self.dynamic_threshold_goe.get_threshold_confidence()
        mlnn_confidence = self.dynamic_threshold_mlnn.get_threshold_confidence()
        
        # Dynamic weight adjustment based on confidence
        total_confidence = goe_confidence + mlnn_confidence + 1e-8
        goe_weight = goe_confidence / total_confidence
        mlnn_weight = mlnn_confidence / total_confidence
        
        print(f"GOE confidence: {goe_confidence:.3f}, MLNN confidence: {mlnn_confidence:.3f}")
        
        # Fuse scores
        final_scores = goe_weight * goe_norm + mlnn_weight * mlnn_norm
        
        # Update dual threshold
        dual_threshold = self.dynamic_threshold_dual.update(final_scores)
        
        return dual_threshold, final_scores
'''======================================================================================================='''
        
        
        

'''======================================================================================================='''        
''' ########################    GOE Engine    ############# '''
"""GOE macro-statistical engine"""
        
class GOEEngine:
    """GOE macro-statistical engine"""
    
    def __init__(self, perturbation_strength=0.1):
        self.perturbation_strength = perturbation_strength
        self.eigenvectors = None
        self.eigenvalues = None
        self.reference_mean = None
        self.reference_cov = None
        self.projection_basis = None
        self.data_covariance = None
        self.regularization_epsilon = 1e-4
        self.mahalanobis_min = 1e-8
        np.random.seed(42)
        
        
    def compute_boundary_scores(self, X):
        """Enhanced boundary-sensitive score calculation - optimized for financial data"""
        X_goe = self.transform(X)
        
        if self.reference_mean is None or self.reference_cov is None:
            raise ValueError("GOE engine are not trained")
        
        ''' Mahalanobis distance'''
        diff = X_goe - self.reference_mean
        try:
            cov_inv = np.linalg.inv(self.reference_cov + np.eye(self.reference_cov.shape[0]) * 1e-6)
            mahalanobis_dist = np.sum(diff @ cov_inv * diff, axis=1)
        except np.linalg.LinAlgError:
            ''' If matrix is not invertible, use pseudo-inverse'''
            cov_pinv = np.linalg.pinv(self.reference_cov)
            mahalanobis_dist = np.sum(diff @ cov_pinv * diff, axis=1)
        
        ''' ========== Enhanced financial data boundary detection ========== '''
        ''' 1. Multi-scale local density analysis '''
        n_samples = min(2000, X_goe.shape[0])
        if n_samples < 50:
            return mahalanobis_dist
        
        indices = np.random.choice(X_goe.shape[0], n_samples, replace=False)
        X_sampled = X_goe[indices]
        
        ''' Calculate local density using multiple k values'''
        local_density_multi = []
        for k in [5, 10, 20]:
            dist_matrix = cdist(X_sampled, X_sampled)
            np.fill_diagonal(dist_matrix, np.inf)
            kth_dist = np.partition(dist_matrix, k, axis=1)[:, k]
            local_density = 1.0 / (kth_dist + 1e-8)
            local_density_multi.append(local_density)
            
        
        ''' Combine multi-scale densities '''
        local_density_combined = np.mean(local_density_multi, axis=0)
        
        '''# 2. Directional anomaly detection (for multidimensional correlations in financial data)'''
        if hasattr(self, 'reference_cov'):
            try:
                ''' Calculate anomaly degree in principal component directions'''
                eigvals, eigvecs = np.linalg.eigh(self.reference_cov)
                ''' Select main eigen directions'''
                top_k = min(10, eigvecs.shape[1])
                ''' Eigenvectors corresponding to largest eigenvalues'''
                principal_dirs = eigvecs[:, -top_k:]   
                
                ''' Projection distance in main directions '''
                proj_dist = np.zeros((X_sampled.shape[0], top_k))
                for i in range(top_k):
                    proj = X_sampled @ principal_dirs[:, i]
                    proj_dist[:, i] = (proj - np.mean(proj)) ** 2 / (np.var(proj) + 1e-8)
                
                directional_score = np.mean(proj_dist, axis=1)
            except:
                directional_score = np.ones(X_sampled.shape[0])
        else:
            directional_score = np.ones(X_sampled.shape[0])
        

     
        ''' Interpolate local density '''
        interpolator_density = NearestNDInterpolator(X_sampled, local_density_combined)
        all_local_density = interpolator_density(X_goe)
        
        ''' Interpolate directional anomaly scores '''
        if hasattr(self, 'reference_cov'):
            interpolator_directional = NearestNDInterpolator(X_sampled, directional_score)
            all_directional = interpolator_directional(X_goe)
        else:
            all_directional = np.ones(len(X_goe))
        
        ''' ========== Boundary score calculation for financial data ========== '''
        if np.max(mahalanobis_dist) - np.min(mahalanobis_dist) > 1e-8:
            mahalanobis_norm = (mahalanobis_dist - np.min(mahalanobis_dist)) / (np.max(mahalanobis_dist) - np.min(mahalanobis_dist))
        else:
            mahalanobis_norm = mahalanobis_dist * 0
        
        if np.max(all_local_density) - np.min(all_local_density) > 1e-8:
            density_norm = (all_local_density - np.min(all_local_density)) / (np.max(all_local_density) - np.min(all_local_density))
        else:
            density_norm = all_local_density * 0
        
        if np.max(all_directional) - np.min(all_directional) > 1e-8:
            directional_norm = (all_directional - np.min(all_directional)) / (np.max(all_directional) - np.min(all_directional))
        else:
            directional_norm = all_directional * 0
        
        ''' Financial data boundary score = f(distance, density, direction) '''
        ''' Boundary samples: medium distance * (1-density) * directional anomaly '''
        boundary_scores = (mahalanobis_norm * 0.4 + (1 - density_norm) * 0.3 + directional_norm * 0.3)
        
        ''' Enhance scores in boundary regions '''
        ''' Give higher boundary scores to medium score regions (0.3-0.7) '''
        boundary_scores = boundary_scores * (1 + 0.5 * np.exp(-((boundary_scores - 0.5) ** 2) / 0.1))
        
        ''' Normalize to [0, 1] '''
        if np.max(boundary_scores) - np.min(boundary_scores) > 1e-8:
            boundary_scores = (boundary_scores - np.min(boundary_scores)) / (np.max(boundary_scores) - np.min(boundary_scores))
        
        return boundary_scores
    

    def get_boundary_samples(self, X, boundary_ratio=0.1):
        """identify boundary samples"""
        boundary_scores = self.compute_boundary_scores(X)
        threshold = np.percentile(boundary_scores, 100 * (1 - boundary_ratio))
        boundary_mask = boundary_scores > threshold
        return boundary_mask, boundary_scores
    
        
    def fit(self, X):
        """Train GOE engine with data-aware perturbation (Theorem 3)"""
        print("Training GOE macro-statistical engine...")
        
        X_clean = DataCleaner.clean_data(X)
        covariance_matrix = np.cov(X_clean.T)
        n_features = covariance_matrix.shape[0]
        
        self.data_covariance = covariance_matrix
        U_goe = self._generate_goe_basis_enhanced(n_features)
        
        G = self._generate_perturbation_matrix(n_features)
        perturbed_covariance = covariance_matrix + self.perturbation_strength * G
        
        eigenvalues, eigenvectors = np.linalg.eigh(perturbed_covariance)
        optimal_dim = self._select_optimal_dimension_enhanced(eigenvalues, X_clean.shape[0], X_clean)
        
        idx = np.argsort(eigenvalues)[::-1][:optimal_dim]
        self.projection_basis = eigenvectors[:, idx]
        self.eigenvalues = eigenvalues[idx]
        
        X_goe = self.transform(X_clean)
        self.reference_mean = np.mean(X_goe, axis=0)
        self.reference_cov = self._compute_regularized_covariance(X_goe)
        
        print(f"GOE engine training completed, selected dimension: {optimal_dim}")
        return self
    
    
    def transform(self, X):
        """Project data to GOE space"""
        if self.projection_basis is None:
            raise ValueError("GOE engine not trained")
        
        X_clean = DataCleaner.clean_data(X)
        return X_clean @ self.projection_basis
    
    
    def compute_anomaly_scores(self, X):
        """Compute Mahalanobis distance anomaly scores"""
        X_goe = self.transform(X)
        
        if self.reference_mean is None or self.reference_cov is None:
            raise ValueError("GOE engine not trained")
        
        diff = X_goe - self.reference_mean
        cov = self.reference_cov.copy()
        
        try:
            L = np.linalg.cholesky(cov)
            y = np.linalg.solve(L, diff.T)
            mahalanobis_dist = np.sum(y * y, axis=0)
            mahalanobis_dist = np.maximum(mahalanobis_dist, self.mahalanobis_min)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(cov, full_matrices=False)
            s_inv = np.zeros_like(s)
            threshold = np.max(s) * 1e-12
            mask = s > threshold
            s_inv[mask] = 1.0 / s[mask]
            cov_pinv = Vt.T @ np.diag(s_inv) @ U.T
            mahalanobis_dist = np.sum(diff @ cov_pinv * diff, axis=1)
            mahalanobis_dist = np.maximum(mahalanobis_dist, self.mahalanobis_min)
        
        return mahalanobis_dist
    
    
    def _compute_regularized_covariance(self, X):
        """Compute regularized covariance matrix"""
        cov = np.cov(X.T)
        cond_number = np.linalg.cond(cov)
        if cond_number > 1e10:
            epsilon = 1e-6 * np.trace(cov) / cov.shape[0]
            cov += epsilon * np.eye(cov.shape[0])
        return cov
    
    
    def _generate_goe_basis_enhanced(self, n):
        """Enhanced GOE basis generation with data-aware Cholesky method"""
        try:
            G_base = np.random.normal(0, 1, (n, n))
            G_symmetric = (G_base + G_base.T) / 2
            
            if hasattr(self, 'data_covariance') and self.data_covariance is not None:
                try:
                    cov_eigenvalues = np.linalg.eigvalsh(self.data_covariance)
                    scaling_factor = np.sqrt(np.abs(cov_eigenvalues) + 1e-8)
                    mean_scaling = np.mean(scaling_factor)
                    G_data_aware = G_symmetric * mean_scaling / np.sqrt(n)
                except:
                    G_data_aware = G_symmetric / np.sqrt(n)
            else:
                G_data_aware = G_symmetric / np.sqrt(n)
            
            try:
                if hasattr(self, 'data_covariance') and self.data_covariance is not None:
                    epsilon = 1e-6 * np.trace(self.data_covariance) / n
                    cov_regularized = self.data_covariance + epsilon * np.eye(n)
                    L = np.linalg.cholesky(cov_regularized)
                    Z = np.random.normal(0, 1/np.sqrt(n), (n, n))
                    G_cholesky = L @ Z @ L.T
                    G_cholesky = (G_cholesky + G_cholesky.T) / 2
                    alpha = 0.7
                    G_combined = alpha * G_data_aware + (1 - alpha) * G_cholesky
                    _, U = np.linalg.eigh(G_combined)
                    return U
            except Exception:
                pass
            
            _, U = np.linalg.eigh(G_data_aware)
            return U
            
        except Exception:
            G_standard = np.random.normal(0, 1/np.sqrt(n), (n, n))
            G_standard = (G_standard + G_standard.T) / 2
            _, U = np.linalg.eigh(G_standard)
            return U
        
    
    def _generate_perturbation_matrix(self, n):
        """Generate data-aware perturbation matrix"""
        perturbation = np.random.normal(0, 1/np.sqrt(n), (n, n))
        return (perturbation + perturbation.T) / 2
    
    
    def _select_optimal_dimension_enhanced(self, eigenvalues, n_samples, X_original=None):
        """Enhanced optimal dimension selection"""
        intrinsic_dim = self._estimate_intrinsic_dimension_ensemble_enhanced(eigenvalues, X_original)
        n_features = eigenvalues.shape[0]
        optimal_dim = int((intrinsic_dim**0.7) * (np.log(n_samples)**0.3))
        return max(10, min(optimal_dim, n_features))
    
    
    def _estimate_intrinsic_dimension_ensemble_enhanced(self, eigenvalues, X=None):
        """Enhanced ensemble dimensionality estimation"""
        estimators = []
        
        if eigenvalues is not None and len(eigenvalues) > 1:
            try:
                eigenvalues_sorted = np.sort(eigenvalues)[::-1]
                cumulative_variance = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)
                dim1 = np.argmax(cumulative_variance >= 0.95) + 1
                estimators.append(dim1)
            except Exception:
                dim1 = max(50, len(eigenvalues) // 3)
                estimators.append(dim1)
        
        if X is not None and X.shape[0] > 100:
            dim2 = ImprovedDimensionEstimator.grassberger_procaccia(X)
            estimators.append(dim2)
        
        if not estimators:
            return 50
        
        intrinsic_dim = int(np.median(estimators))
        return max(10, intrinsic_dim)
'''======================================================================================================='''
    
    


'''======================================================================================================='''    
''' ########################    MLNN Engines    ############# '''
"""             MLNN micro- dynamic engine             """

class LTCUnit(nn.Module):
    """LTC unit implementing ODE-driven network"""
    
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LTCUnit, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.time_constant = nn.Parameter(torch.randn(hidden_size) * 0.1 + 1.0)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_in = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.activation = nn.Tanh()
        
    
    def forward(self, x):
        """Forward pass using Euler method for ODE solving"""
        batch_size = x.size(1)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        hidden_states = []
        
        for t in range(x.size(0)):
            input_term = x[t] @ self.W_in
            recurrent_term = h @ self.W_rec
            activation_term = self.activation(h)
            dh_dt = -h / torch.exp(self.time_constant) + recurrent_term * activation_term + input_term + self.bias
            h = h + self.dt * dh_dt
            hidden_states.append(h)
        
        return torch.stack(hidden_states)
    

class MLNNEngine(nn.Module):
    """MLNN micro-dynamic engine with multi-granularity architecture"""
    
    def __init__(self, input_size, fine_grained_size=32, coarse_grained_size=32):  # default 32,32 for finance, 16 for credit card collaborative improvement  --0.01 16 better than 32
        super(MLNNEngine, self).__init__()
        
        nhead = 4              
        self.transformer_input_size = self._calculate_adjusted_dimension(input_size, nhead)
        
        if self.transformer_input_size != input_size:
            self.input_adjustment = nn.Linear(input_size, self.transformer_input_size)
        else:
            self.input_adjustment = nn.Identity()
        
        self.ltc_units = nn.ModuleList([
            LTCUnit(self.transformer_input_size, fine_grained_size // 2) for _ in range(2)
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_input_size, 
            nhead=nhead,  
            dim_feedforward=64,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_size = fine_grained_size + coarse_grained_size
        self.attention_context = nn.Parameter(torch.randn(self.output_size))
        self.projection = nn.Linear(2 * self.transformer_input_size, coarse_grained_size)
        self.final_projection = nn.Linear(self.output_size, 1)
        
        self.fine_grained_size = fine_grained_size
        self.coarse_grained_size = coarse_grained_size
        
        ''' boundary-aware module'''
        self.boundary_detector = nn.Sequential(
            nn.Linear(self.output_size, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.boundary_weight = nn.Parameter(torch.tensor(0.5))
        ''' output layer: output deterministic score (0-1)'''
        self.final_projection = nn.Linear(self.output_size, 1)
        ''' Use sigmoid to ensure output between 0-1'''
        self.sigmoid = nn.Sigmoid()
          
        self.financial_feature_extractor = nn.Sequential(
            nn.Linear(self.output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        ''' Financial data specific extractor '''
        self.financial_output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.training_epoch = 0
        
               
    def _calculate_adjusted_dimension(self, input_size, nhead):
        """Calculate adjusted dimension divisible by nhead"""
        if input_size % nhead == 0:
            return input_size
        
        candidate1 = ((input_size + nhead - 1) // nhead) * nhead
        candidate2 = (input_size // nhead) * nhead
        
        if abs(candidate1 - input_size) <= abs(candidate2 - input_size):
            return candidate1
        else:
            return candidate2 if candidate2 >= nhead else nhead
    
             
    def forward(self, x):
        """Forward pass """
        x = torch.nan_to_num(x, nan=0.0)
        x_adjusted = self.input_adjustment(x)
        x_transposed = x_adjusted.transpose(0, 1)
        
        fine_features = []
        for ltc_unit in self.ltc_units:
            ltc_output = ltc_unit(x_transposed)
            final_state = ltc_output[-1]
            fine_features.append(final_state)
        
        fine_combined = torch.cat(fine_features, dim=1)
        coarse_features = self._coarse_grained_processing(x_adjusted)
        fused_features = self._multi_granularity_fusion(fine_combined, coarse_features)
        
        ''' ========== Financial data specific feature extraction ========== '''
        financial_features = self.financial_feature_extractor(fused_features)
        
        if self.training and hasattr(self, 'training_epoch'):
            noise_scale = max(0.01, 0.05 * (1.0 - self.training_epoch / 100.0))
            noise = torch.randn_like(financial_features) * noise_scale
            financial_features = financial_features + noise
        
        output = self.financial_output(financial_features)
        
        return output.squeeze(-1)
        
           
    def extract_features(self, x):
        """Extract intermediate features for diagnostics"""
        with torch.no_grad():
            output = self.forward(x)
            return output.unsqueeze(-1)
        
    
    def _coarse_grained_processing(self, x):
        """Coarse-grained processing with multi-scale downsampling"""
        batch_size, seq_len, input_size = x.shape
        scales = [3, 6]
        scale_features = []
        
        for scale in scales:
            if seq_len >= scale:
                downsampled = self._temporal_downsample(x, scale)
                transformer_output = self.transformer_encoder(downsampled)
                scale_rep = transformer_output.mean(dim=1)
                scale_features.append(scale_rep)
            else:
                scale_rep = torch.zeros(batch_size, input_size, device=x.device)
                scale_features.append(scale_rep)
        
        while len(scale_features) < 2:
            scale_features.append(torch.zeros(batch_size, input_size, device=x.device))
        
        coarse_combined = torch.cat(scale_features, dim=1)
        expected_projection_input = 2 * self.transformer_input_size
        
        if coarse_combined.shape[1] != expected_projection_input:
            if coarse_combined.shape[1] < expected_projection_input:
                padding = torch.zeros(batch_size, 
                                    expected_projection_input - coarse_combined.shape[1], 
                                    device=coarse_combined.device)
                coarse_combined = torch.cat([coarse_combined, padding], dim=1)
            else:
                coarse_combined = coarse_combined[:, :expected_projection_input]
        
        return self.projection(coarse_combined)
    
    
    def _temporal_downsample(self, x, scale):
        """Temporal downsampling for multi-scale processing"""
        batch_size, seq_len, input_size = x.shape
        
        if seq_len % scale != 0:
            pad_len = scale - (seq_len % scale)
            x = torch.cat([x, torch.zeros(batch_size, pad_len, input_size, device=x.device)], dim=1)
            seq_len += pad_len
        
        downsampled = x.reshape(batch_size, seq_len // scale, scale, input_size)
        return downsampled.mean(dim=2)
    
    
    def _multi_granularity_fusion(self, fine_features, coarse_features):
        """Multi-granularity fusion with attention mechanism"""
        if fine_features.shape[1] != self.fine_grained_size:
            if fine_features.shape[1] < self.fine_grained_size:
                padding = torch.zeros(fine_features.shape[0], 
                                    self.fine_grained_size - fine_features.shape[1], 
                                    device=fine_features.device)
                fine_features = torch.cat([fine_features, padding], dim=1)
            else:
                fine_features = fine_features[:, :self.fine_grained_size]
                
        if coarse_features.shape[1] != self.coarse_grained_size:
            if coarse_features.shape[1] < self.coarse_grained_size:
                padding = torch.zeros(coarse_features.shape[0], 
                                    self.coarse_grained_size - coarse_features.shape[1], 
                                    device=coarse_features.device)
                coarse_features = torch.cat([coarse_features, padding], dim=1)
            else:
                coarse_features = coarse_features[:, :self.coarse_grained_size]
        
        combined = torch.cat([fine_features, coarse_features], dim=1)
        
        if combined.shape[1] != self.attention_context.shape[0]:
            self.attention_context = nn.Parameter(torch.randn(combined.shape[1], device=combined.device))
        
        attention_weights = torch.softmax(combined @ self.attention_context, dim=0)
        fine_weighted = attention_weights.unsqueeze(1) * fine_features
        coarse_weighted = (1 - attention_weights.unsqueeze(1)) * coarse_features
        fused = torch.cat([fine_weighted, coarse_weighted], dim=1)
        
        if fused.shape[1] != self.output_size:
            if fused.shape[1] < self.output_size:
                padding = torch.zeros(fused.shape[0], 
                                    self.output_size - fused.shape[1], 
                                    device=fused.device)
                fused = torch.cat([fused, padding], dim=1)
            else:
                fused = fused[:, :self.output_size]
        
        return fused
'''=======================================================================================================''' 
    
    


'''======================================================================================================='''     
'''                 Dual-Engines  Synergy                         '''
"""Enhanced dual-engine based on Theorem 5, Theorem 6 and Theorem 7""" 

class EnhancedDualEngineSynergy:
    """Enhanced dual-engine fusion strategy with dynamic thresholding"""
    
    def __init__(self, window_size=500):
        self.dynamic_thresholder = DynamicAdaptiveThreshold(window_size=window_size)
        
    @staticmethod
    def adaptive_fusion_strategy(goe_scores, mlnn_certainty_scores, y_true=None, use_dynamic=True, 
                                 dynamic_thresholder=None):
        
        
        min_length = min(len(goe_scores), len(mlnn_certainty_scores))
        goe_scores = goe_scores[:min_length]
        mlnn_certainty = mlnn_certainty_scores[:min_length]
        
        # GOE Normalization: Considering the Low Anomaly Rate of Financial Data
        goe_percentiles = np.percentile(goe_scores, [ 80,85,90,95, 99, 99.5])
        # Normalize using the median to the 99.5 percentile
        goe_min = goe_percentiles[0]   
        goe_max = goe_percentiles[2]   
        goe_norm = (goe_scores - goe_min) / (goe_max - goe_min + 1e-8)
        goe_norm = np.clip(goe_norm, 0, 1)
        
        # MLNN normalization: MLNN's outputs may be concentrated in a certain range
        mlnn_min = np.percentile(mlnn_certainty, 10)   
        mlnn_max = np.percentile(mlnn_certainty, 90)   
        mlnn_norm = (mlnn_certainty - mlnn_min) / (mlnn_max - mlnn_min + 1e-8)
        mlnn_norm = np.clip(mlnn_norm, 0, 1)
        
        ''' ========== Calculate the confidence level of financial data ========== '''
        # GOE Confidence: Steepness based on Score Distribution
        goe_hist, goe_bins = np.histogram(goe_norm, bins=50)
        goe_entropy = -np.sum((goe_hist / np.sum(goe_hist)) * np.log(goe_hist / np.sum(goe_hist) + 1e-8))
        goe_confidence = 1.0 - min(goe_entropy / np.log(50), 1.0)  # Normalized to 0-1
        
        # MLNN Confidence: Output based Consistency
        mlnn_std = np.std(mlnn_norm)
        mlnn_confidence = 1.0 - min(mlnn_std * 5, 1.0)
        
        ''' ========== Specific fusion strategies for financial data: GOE led, MLNN assisted ========== '''
        dual_scores = np.zeros_like(goe_norm)
        
        # Calculate the boundedness of GOE
        goe_uncertainty = 2.0 * np.abs(goe_norm - 0.5)   
        
        for i in range(len(goe_norm)):
            g_score = goe_norm[i]
            m_score = mlnn_norm[i]
            g_uncertain = goe_uncertainty[i]
            
            # Rule 1: In cases where GOE is highly determined (GOE is more reliable for macro models in financial data)
            if g_uncertain < 0.3:   
                # If GOE is determined and the score is high, it is considered abnormal
                if g_score > 0.7:
                    # GOE considers it abnormal and gives high weight to GOE
                    weight = 0.8
                    dual_scores[i] = weight * g_score + (1 - weight) * m_score
                else:
                    # GOE considers it normal and gives it high weight
                    weight = 0.7
                    dual_scores[i] = weight * g_score + (1 - weight) * m_score
                    
            # Rule 2: GOE uncertainty (boundary samples)
            elif g_uncertain > 0.7:
                # In boundary samples, if MLNN has high confidence, trust MLNN
                if mlnn_confidence > 0.6 and np.abs(m_score - 0.5) > 0.3:
                    # MLNN has a clear judgment and high weights are given to MLNN
                    weight = 0.7   
                    dual_scores[i] = weight * m_score + (1 - weight) * g_score
                else:
                    # Both are uncertain, use a conservative average
                    dual_scores[i] = 0.5 * g_score + 0.5 * m_score
            
            # Rule 3: Medium uncertainty
            else:
                # Dynamically adjust weights based on score differences
                score_diff = np.abs(g_score - m_score)
                if score_diff < 0.2:   
                    dual_scores[i] = (g_score + m_score) / 2
                else:  
                    # Both are inconsistent, there is a tendency to trust GOE (more sensitive to statistical patterns)
                    if g_score > 0.6:   
                        weight = 0.6
                    else:
                        weight = 0.4
                    dual_scores[i] = weight * g_score + (1 - weight) * m_score
        
        # Enhance the discrimination of high scoring regions
        high_score_mask = dual_scores > 0.7
        if np.sum(high_score_mask) > 0:
            dual_scores[high_score_mask] = dual_scores[high_score_mask] ** 0.8  
        
        # Smooth processing  
        if len(dual_scores) > 10:
            # Use moving average smoothing
            smoothed = np.convolve(dual_scores, np.ones(5)/5, mode='same')
            dual_scores = 0.7 * dual_scores + 0.3 * smoothed
        
        # Dynamic threshold selection
        if y_true is not None and len(y_true) >= min_length:
            y_true_adj = y_true[:min_length]
            anomaly_rate = np.mean(y_true_adj)
            
            # New addition: Supervised threshold and dynamic threshold fusion
            # Calculate supervised optimal threshold
            if anomaly_rate < 0.005:   
                threshold_candidates = np.percentile(dual_scores, [99.0, 99.5, 99.7, 99.9])
            elif anomaly_rate < 0.015:  
                threshold_candidates = np.percentile(dual_scores, [98.0, 98.5, 99.0, 99.5])
            else:  
                threshold_candidates = np.percentile(dual_scores, [95.0, 96.0, 97.0, 98.0])
            
            # Select the optimal threshold
            best_supervised_threshold = 0.5
            best_f1 = 0
            supervised_candidates = []
            
            for threshold in threshold_candidates:
                predictions = (dual_scores > threshold).astype(int)
                if np.sum(predictions) == 0:
                    continue
                f1 = f1_score(y_true_adj, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_supervised_threshold = threshold
                # Collect all candidate thresholds
                supervised_candidates.append(threshold)
            
            # If dynamic thresholder is available, perform fusion
            if use_dynamic and dynamic_thresholder is not None:
                # Get current dynamic threshold
                current_dynamic_threshold = dynamic_thresholder.current_threshold
                
                # Use EnhancedThresholdFusion for fusion
                fused_threshold = EnhancedThresholdFusion.fuse_thresholds(
                    dynamic_threshold=current_dynamic_threshold,
                    supervised_thresholds=supervised_candidates,
                    dynamic_confidence=dynamic_thresholder.get_threshold_confidence(),
                    n_samples=len(dual_scores),
                    supervised_f1=best_f1
                )
                
                # Update dynamic thresholder (update with fused threshold)
                dynamic_thresholder.update([fused_threshold])
                
                print(f"Threshold Fusion - Supervised: {best_supervised_threshold:.4f}, "
                      f"Dynamic: {current_dynamic_threshold:.4f}, "
                      f"Fused: {fused_threshold:.4f}, F1: {best_f1:.4f}")
                
                return fused_threshold, dual_scores
            else:
                # If no dynamic thresholder, directly use supervised threshold
                print(f"Supervised optimal threshold: {best_supervised_threshold:.4f}, "
                      f"F1: {best_f1:.4f}, anomaly ratio: {anomaly_rate:.4%}")
                return best_supervised_threshold, dual_scores
        else:
            # Dynamic adaptive threshold when unlabeled
            if use_dynamic and dynamic_thresholder is not None:
                threshold = dynamic_thresholder.update(dual_scores)
                print(f"Dynamic adaptive threshold: {threshold:.4f}, "
                      f"Confidence: {dynamic_thresholder.get_threshold_confidence():.3f}")
            else:
                # Fallback to financial data experience threshold
                threshold = np.percentile(dual_scores, 98.5)
                print(f"Statistical threshold: {threshold:.4f}")
            
            return threshold, dual_scores
        
        
        
'''                 Threshold Fusion Module                         '''
"""Enhanced threshold fusion strategy for combining supervised and dynamic thresholds"""

class EnhancedThresholdFusion:
    """Strategy for fusing supervised signals and unsupervised dynamic thresholds"""
    
    @staticmethod
    def fuse_thresholds(dynamic_threshold, supervised_thresholds, 
                       dynamic_confidence, n_samples, supervised_f1=None):
        """
        Fuse dynamic threshold and supervised threshold
        
        Parameters:
        dynamic_threshold: Dynamic threshold
        supervised_thresholds: List of multiple candidate thresholds obtained from supervised learning
        dynamic_confidence: Dynamic threshold confidence (0-1)
        n_samples: Number of samples
        supervised_f1: F1 score corresponding to supervised threshold (if available)
        """
        
        if not supervised_thresholds:
            return dynamic_threshold
        
        # 1. Process supervised thresholds
        supervised_thresholds_array = np.array(supervised_thresholds)
        
        # Exclude extreme values (e.g., 0.01 or 0.99)
        valid_mask = (supervised_thresholds_array > 0.1) & (supervised_thresholds_array < 0.9)
        if np.sum(valid_mask) == 0:
            # If no valid thresholds, use median
            supervised_median = np.median(supervised_thresholds_array)
        else:
            supervised_median = np.median(supervised_thresholds_array[valid_mask])
        
        # 2. Calculate fusion weights
        # Based on dynamic threshold confidence
        if dynamic_confidence > 0.8:
            dynamic_weight = 0.6
        elif dynamic_confidence > 0.6:
            dynamic_weight = 0.4
        else:
            dynamic_weight = 0.2
        
        # Adjust based on sample size
        if n_samples > 5000:
            # More samples, trust data-driven more
            dynamic_weight += 0.1
        elif n_samples < 1000:
            # Fewer samples, rely more on supervised signals
            dynamic_weight -= 0.1
        
        # Adjust based on supervised F1 score
        if supervised_f1 is not None:
            if supervised_f1 > 0.5:
                # Good supervised performance, increase supervised weight
                dynamic_weight -= 0.15
            elif supervised_f1 < 0.2:
                # Poor supervised performance, increase dynamic weight
                dynamic_weight += 0.15
        
        # Ensure weights are within reasonable range
        dynamic_weight = max(0.1, min(0.9, dynamic_weight))
        supervised_weight = 1.0 - dynamic_weight
        
        print(f"Fusion weights - Dynamic: {dynamic_weight:.2f}, Supervised: {supervised_weight:.2f}")
        
        # 3. Calculate fused threshold
        fused_threshold = (dynamic_weight * dynamic_threshold + 
                         supervised_weight * supervised_median)
        
        # 4. Apply smoothing (prevent sudden changes)
        # Calculate threshold change rate
        if hasattr(EnhancedThresholdFusion, 'last_threshold'):
            threshold_change = abs(fused_threshold - EnhancedThresholdFusion.last_threshold)
            if threshold_change > 0.2:
                # Change too large, apply stronger smoothing
                fused_threshold = 0.3 * fused_threshold + 0.7 * EnhancedThresholdFusion.last_threshold
                print(f"Large threshold change detected ({threshold_change:.3f}), applying smoothing")
        
        EnhancedThresholdFusion.last_threshold = fused_threshold
        
        return fused_threshold
'''======================================================================================================='''
    
    
    
 
'''======================================================================================================='''    
''' ########################    The MODEL- Dual-Engine Anomaly Detection Framework (dADF)   ############# '''

class DualEngineADF:
    """Dual-Engine Anomaly Detection Framework with Dynamic Thresholding"""
    
    def __init__(self, input_size, perturbation_strength=0.1, lambda_reg=0.1, window_size=500):
        self.goe_engine = GOEEngine(perturbation_strength)
        self.mlnn_engine = MLNNEngine(input_size)
        self.lambda_reg = lambda_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlnn_engine.to(self.device)
        
        # Dynamic thresholding components
        self.dynamic_thresholder = DynamicAdaptiveThreshold(window_size=window_size)
        self.theorem_thresholder = TheoremBasedThresholdDynamic(window_size=window_size)
        self.dual_synergy = EnhancedDualEngineSynergy(window_size=window_size)
        
        self.training_history = {
            'mlnn_loss': [], 'goe_loss': [], 'dual_loss': [],
            'mlnn_accuracy': [], 'mlnn_f1': [], 'goe_accuracy': [], 'goe_f1': [],
            'dual_accuracy': [], 'dual_f1': []
        }
        
        self.progressive_trainer = ProgressiveTrainingStrategy(self)
        
    
    def fit(self, X_train, y_train, X_val, y_val, use_progressive_training=True):
        """Train dual-engine framework"""
        print("Starting dual-engine anomaly detection framework training...")
        
        total_start_time = time.time()
        goe_start_time = time.time()
        
        print("... Training GOE macro-statistical engine...")
        X_train_clean, y_train_clean = DataCleaner.clean_data(X_train, y_train)
        self.goe_engine.fit(X_train_clean)
        
        goe_training_time = time.time() - goe_start_time
        self.training_history['goe_training_time'] = goe_training_time
        print(f"GOE training time: {goe_training_time:.2f} seconds")
        
        print("... Training MLNN micro-dynamic engine...")
        mlnn_start_time = time.time()
        
        if use_progressive_training:
            print("Using progressive training strategy...")
            stage_results = self.progressive_trainer.train_with_progressive_strategy(
                X_train_clean, y_train_clean, X_val, y_val
            )
            print("Progressive training completed!")
        else:
            self._train_mlnn_with_goe_guidance(X_train_clean, y_train_clean, X_val, y_val, 100, 10)
        
        mlnn_training_time = time.time() - mlnn_start_time
        self.training_history['mlnn_training_time'] = mlnn_training_time
        
        total_training_time = time.time() - total_start_time
        self.training_history['total_training_time'] = total_training_time
        
        print(f"Total training time: {total_training_time:.2f} seconds")
        print("Dual-engine training completed!")
        
    
    def _train_mlnn_with_goe_guidance(self, X_train, y_train, X_val, y_val, epochs, patience):
        """Incorporating the concept of adversarial training"""   
        
        sequence_length = 5
        X_seq = self._create_sequences(X_train, sequence_length)
        y_seq = self._create_sequence_labels(y_train, sequence_length)
        
        X_val_seq = self._create_sequences(X_val, sequence_length)
        y_val_seq = self._create_sequence_labels(y_val, sequence_length)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        
        optimizer = optim.Adam(self.mlnn_engine.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.training_epoch = epoch  
            
            self.mlnn_engine.train()
            optimizer.zero_grad()
            
            mlnn_scores = self.mlnn_engine(X_tensor)
            mlnn_loss = criterion(mlnn_scores, y_tensor)
            goe_regularization = self._compute_goe_regularization(X_tensor, mlnn_scores)
            
            # Dynamic adjustment of loss weights: focus on independent learning of MLNN in the early stage, and focus on fusion in the later stage
            progress = epoch / max(1, epochs-1)
            # Top 30%: MLNN focuses on independent learning
            if progress < 0.3:  
                dual_loss = mlnn_loss * 0.8 + self.lambda_reg * goe_regularization * 0.2
            # 30% -70%: Balanced learning
            elif progress < 0.7:   
                dual_loss = mlnn_loss * 0.5 + self.lambda_reg * goe_regularization * 0.5
            # Last 30%: GOE guidance as the main approach
            else:   
                dual_loss = mlnn_loss * 0.2 + self.lambda_reg * goe_regularization * 0.8
            
            dual_loss.backward()
            optimizer.step()
            
            if patience_counter >= patience:
                break
        
        if best_model_state is not None:
            self.mlnn_engine.load_state_dict(best_model_state)
    
    
    def _compute_goe_regularization(self, X, mlnn_scores):
        """GOE regularization for financial data - enhanced boundary sample learning"""
        try:
            batch_size, seq_len, input_size = X.shape
            
            # Obtain GOE boundary samples
            X_flat = X.reshape(-1, input_size)
            X_np = X_flat.detach().cpu().numpy()
            
            # Calculate GOE boundary score
            if hasattr(self.goe_engine, 'compute_boundary_scores'):
                goe_boundary_scores = self.goe_engine.compute_boundary_scores(X_np)
            else:
                goe_scores = self.goe_engine.compute_anomaly_scores(X_np)
                goe_boundary_scores = (goe_scores - np.min(goe_scores)) / (np.max(goe_scores) - np.min(goe_scores) + 1e-8)
            
            goe_boundary_scores_reshaped = goe_boundary_scores.reshape(batch_size, seq_len)
            goe_boundary_avg = np.mean(goe_boundary_scores_reshaped, axis=1)
            
            # Identification of boundary samples specific to financial data
            # In financial data, we need to identify more boundary samples (target: 10-20%)
            
            # Method 1: Based on percentiles
            boundary_threshold_low = np.percentile(goe_boundary_avg, 70)  
            boundary_threshold_high = np.percentile(goe_boundary_avg, 90)   
            
            # Method 2: Based on distribution patterns
            hist, bins = np.histogram(goe_boundary_avg, bins=50)
            # Find valleys in the histogram as boundary thresholds
            if len(hist) > 3:
                # Smooth histogram
                hist_smooth = np.convolve(hist, np.ones(3)/3, mode='valid')
                # Find local minimum value
                minima = []
                for i in range(1, len(hist_smooth)-1):
                    if hist_smooth[i] < hist_smooth[i-1] and hist_smooth[i] < hist_smooth[i+1]:
                        minima.append(bins[i])
                
                if minima:
                    # Use the first trough as the boundary threshold
                    boundary_threshold_dynamic = minima[0]
                else:
                    boundary_threshold_dynamic = np.percentile(goe_boundary_avg, 75)
            else:
                boundary_threshold_dynamic = np.percentile(goe_boundary_avg, 75)
            
            # Choose a lower threshold to obtain more boundary samples
            boundary_threshold = min(boundary_threshold_low, boundary_threshold_dynamic)
            
            # Identify boundary samples (Objective: Increase the proportion of boundary samples)
            boundary_mask_np = (goe_boundary_avg >= boundary_threshold) & (goe_boundary_avg < boundary_threshold_high)
            non_boundary_mask_np = goe_boundary_avg < boundary_threshold  # Low score region
            clear_anomaly_mask_np = goe_boundary_avg >= boundary_threshold_high  # Clear anomaly
            
            # Low score area
            boundary_ratio = np.sum(boundary_mask_np) / len(boundary_mask_np)
            
            # Calculate the proportion of boundary samples
            if boundary_ratio < 0.1:   
                boundary_threshold = np.percentile(goe_boundary_avg, 60)   
                boundary_mask_np = (goe_boundary_avg >= boundary_threshold) & (goe_boundary_avg < boundary_threshold_high)
                boundary_ratio = np.sum(boundary_mask_np) / len(boundary_mask_np)
            
            # Normalized GOE score
            goe_boundary_norm = (goe_boundary_avg - np.min(goe_boundary_avg)) / (np.max(goe_boundary_avg) - np.min(goe_boundary_avg) + 1e-8)
            
            # Convert GOE scores to tensor
            goe_boundary_tensor = torch.FloatTensor(goe_boundary_norm).to(self.device).unsqueeze(1)
            
            # MLNN output (deterministic score)
            mlnn_certainty = torch.sigmoid(mlnn_scores).unsqueeze(1)
            
            # MLNN output (deterministic score)
            min_dim = min(goe_boundary_tensor.size(0), mlnn_certainty.size(0))
            goe_boundary_tensor = goe_boundary_tensor[:min_dim]
            mlnn_certainty = mlnn_certainty[:min_dim]
            boundary_mask_np = boundary_mask_np[:min_dim]
            non_boundary_mask_np = non_boundary_mask_np[:min_dim] if len(non_boundary_mask_np) >= min_dim else non_boundary_mask_np
            clear_anomaly_mask_np = clear_anomaly_mask_np[:min_dim] if len(clear_anomaly_mask_np) >= min_dim else clear_anomaly_mask_np
            
            boundary_mask = torch.tensor(boundary_mask_np, device=self.device, dtype=torch.bool)
            non_boundary_mask = torch.tensor(non_boundary_mask_np, device=self.device, dtype=torch.bool)
            clear_anomaly_mask = torch.tensor(clear_anomaly_mask_np, device=self.device, dtype=torch.bool)
            
            # Specific regularization strategies
            total_regularization = torch.tensor(0.0, device=self.device)
            
            # 1. Regularization of boundary samples
            boundary_loss = torch.tensor(0.0, device=self.device)
            if torch.sum(boundary_mask) > 10:  
                # Ensure sufficient samples are available
                # For financial data boundary samples: MLNN should learn GOE patterns
                mlnn_boundary = mlnn_certainty[boundary_mask]
                goe_boundary = goe_boundary_tensor[boundary_mask]
                
                # Using mild MSE loss to encourage MLNN to approach GOE but not enforce consistency
                boundary_loss = torch.mean((mlnn_boundary - goe_boundary) ** 2)
                
                # Record the number of boundary samples
                boundary_count = torch.sum(boundary_mask).item()
            else:
                boundary_count = 0
            
            # 2. Regularization of non-boundary samples (normal samples)
            non_boundary_loss = torch.tensor(0.0, device=self.device)
            if torch.sum(non_boundary_mask) > 10:
                # For normal samples: MLNN should output a low anomaly score
                mlnn_normal = mlnn_certainty[non_boundary_mask]
                # Goal: Approaching 0 (as normal)
                target_normal = torch.zeros_like(mlnn_normal)
                non_boundary_loss = torch.mean((mlnn_normal - target_normal) ** 2)
            
            # 3. Clarify the regularization of abnormal samples
            clear_anomaly_loss = torch.tensor(0.0, device=self.device)
            if torch.sum(clear_anomaly_mask) > 5:
                # For clear anomalies: MLNN should output high anomaly scores
                mlnn_anomaly = mlnn_certainty[clear_anomaly_mask]
                # Goal: Approaching 1 (as abnormal)
                target_anomaly = torch.ones_like(mlnn_anomaly)
                clear_anomaly_loss = torch.mean((mlnn_anomaly - target_anomaly) ** 2)
            
            # Dynamic weight adjustment
            epoch = getattr(self, 'training_epoch', 0)
            
            # Early: Focus on boundary learning
            if epoch < 30:   
                boundary_weight = 0.6
                non_boundary_weight = 0.2
                anomaly_weight = 0.2
            # Mid-term: Balanced Learning
            elif epoch < 60:   
                boundary_weight = 0.4
                non_boundary_weight = 0.3
                anomaly_weight = 0.3
            # Post-production: Overall optimization
            else: 
                boundary_weight = 0.3
                non_boundary_weight = 0.4
                anomaly_weight = 0.3
            
            mlnn_std = torch.std(mlnn_certainty)
            diversity_loss = torch.exp(-mlnn_std * 5.0)  
            
            # Loss
            total_regularization = (
                boundary_weight * boundary_loss +
                non_boundary_weight * non_boundary_loss +
                anomaly_weight * clear_anomaly_loss +
                0.05 * diversity_loss
            )
            
            self._debug_info = {
                'boundary_ratio': boundary_count / min_dim,
                'boundary_count': boundary_count,
                'boundary_loss': boundary_loss.item() if torch.sum(boundary_mask) > 0 else 0,
                'non_boundary_loss': non_boundary_loss.item() if torch.sum(non_boundary_mask) > 0 else 0,
                'mlnn_std': mlnn_std.item(),
                'boundary_threshold': boundary_threshold,
                'epoch': epoch
            }
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: boundary sample ratio={self._debug_info['boundary_ratio']:.3f}, "
                      f"boundary counts ={self._debug_info['boundary_count']}")
            
            return total_regularization * 0.3  
            
        except Exception as e:
            print(f"GOE regularization error: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device)
        
          
    def _create_sequences(self, X, sequence_length):
        """Create time sequences"""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
        if not sequences:
            sequences = [X]
        return np.array(sequences)
    
    
    def _create_sequence_labels(self, y, sequence_length):
        """Create sequence labels"""
        if len(y) <= sequence_length:
            return y[-1:] if len(y) > 0 else np.array([0])
        return y[sequence_length-1:]
    
    
    def find_optimal_threshold(model, X_val, y_val, metric='f1', num_candidates=200):
        """
        Find optimal threshold on validation set that maximizes the given metric.
        
        Parameters:
            model: trained DualEngineADF model
            X_val, y_val: validation data
            metric: one of 'f1', 'gmean', 'recall', 'precision', 'accuracy'
            num_candidates: number of threshold candidates to evaluate
        
        Returns:
            best_threshold: float
            best_score: float
        """
        """
        Search for optimal threshold on validation set and return the percentile corresponding to that threshold.
        """
        scores = model.predict_anomaly_scores(X_val, use_dynamic_threshold=False)
        dual_scores = scores['dual_scores']
        min_len = min(len(dual_scores), len(y_val))
        dual_scores = dual_scores[:min_len]
        y_val = y_val[:min_len].astype(int)

        low_percentile, high_percentile = 80, 99.9
        low = np.percentile(dual_scores, low_percentile)
        high = np.percentile(dual_scores, high_percentile)
        if high <= low:
            low, high = np.min(dual_scores), np.max(dual_scores)

        thresholds = np.linspace(low, high, num_candidates)
        best_th = thresholds[0]
        best_score = 0.0

        for th in thresholds:
            pred = (dual_scores > th).astype(int)
            if metric == 'f1':
                score = f1_score(y_val, pred, zero_division=0)
            elif metric == 'gmean':
                score = calculate_gmean(y_val, pred)
            elif metric == 'recall':
                score = recall_score(y_val, pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, pred, zero_division=0)
            else:
                score = accuracy_score(y_val, pred)
            if score > best_score:
                best_score = score
                best_th = th

        # Calculate the percentile corresponding to the optimal threshold
        percentile = np.sum(dual_scores <= best_th) / len(dual_scores) * 100
        print(f"Best threshold on validation set ({metric}): {best_th:.6f} with score {best_score:.4f}, percentile={percentile:.2f}%")
        return best_th, best_score, dual_scores, percentile
    
    
    
    
    def predict_anomaly_scores(self, X, use_dynamic_threshold=True, streaming_mode=False, custom_threshold=None):
            """
            Predict anomaly scores combining dual-engine outputs with dynamic thresholding
            
            Parameters:
            X: Input data
            use_dynamic_threshold: Whether to use dynamic adaptive thresholding
            streaming_mode: Whether in streaming mode (for real-time processing)
            custom_threshold: If provided, use this fixed threshold for prediction
            """
            # Compute base scores
            goe_scores = self.goe_engine.compute_anomaly_scores(X)
            
            sequence_length = 5
            X_seq = self._create_sequences(X, sequence_length)
            if len(X_seq) == 0:
                X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
            else:
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            self.mlnn_engine.eval()
            with torch.no_grad():
                mlnn_scores = self.mlnn_engine(X_tensor)
                mlnn_scores_norm = torch.sigmoid(mlnn_scores).cpu().numpy()
            
            min_length = min(len(goe_scores), len(mlnn_scores_norm))
            goe_scores = goe_scores[:min_length]
            mlnn_scores_norm = mlnn_scores_norm[:min_length]
            
            goe_scores_norm = (goe_scores - np.min(goe_scores)) / (np.max(goe_scores) - np.min(goe_scores) + 1e-8)
            
            # First compute fusion scores (reuse existing fusion strategy but not its threshold)
            # Note: EnhancedDualEngineSynergy.adaptive_fusion_strategy returns (threshold, dual_scores)
            # We always call it to obtain dual_scores, even if we may use a custom threshold
            _, dual_scores = EnhancedDualEngineSynergy.adaptive_fusion_strategy(
                goe_scores_norm, mlnn_scores_norm, None, use_dynamic=False
            )
            
            # Determine threshold
            if custom_threshold is not None:
                threshold = custom_threshold
                threshold_type = 'fixed'
            elif use_dynamic_threshold:
                # Use dynamic adaptive threshold
                if streaming_mode:
                    # Streaming mode (keep original logic, simplified here)
                    # In practice, iterate, but for simplicity just a placeholder
                    # It is recommended to keep the original streaming code, but ensure consistent threshold handling
                    # The following is just a structural example; actually keep the original code
                    pass
                else:
                    threshold, _ = self.dual_synergy.adaptive_fusion_strategy(
                        goe_scores_norm, mlnn_scores_norm, None,
                        use_dynamic=True, dynamic_thresholder=self.dynamic_thresholder
                    )
                threshold_type = 'dynamic'
            else:
                # Traditional statistical threshold (i.e., use_dynamic=False)
                threshold, _ = EnhancedDualEngineSynergy.adaptive_fusion_strategy(
                    goe_scores_norm, mlnn_scores_norm, None, use_dynamic=False
                )
                threshold_type = 'static'
            
            # Generate predictions
            thresholds = np.full_like(dual_scores, threshold)
            predictions = (dual_scores > threshold).astype(int)
            
            return {
                'goe_scores': goe_scores_norm,
                'mlnn_scores': mlnn_scores_norm,
                'dual_scores': dual_scores,
                'threshold': threshold,
                'thresholds': thresholds,  # For compatibility with original return
                'predictions': predictions,
                'threshold_type': threshold_type,
                'streaming_mode': streaming_mode
            }
        
        
        
    
    
    
    
    def save_model(self, filepath):
        """Save model state dictionary"""
        checkpoint = {
            'goe_eigenvectors': self.goe_engine.eigenvectors,
            'goe_eigenvalues': self.goe_engine.eigenvalues,
            'goe_reference_mean': self.goe_engine.reference_mean,
            'goe_reference_cov': self.goe_engine.reference_cov,
            'goe_projection_basis': self.goe_engine.projection_basis,
            'mlnn_engine_state_dict': self.mlnn_engine.state_dict(),
            'training_history': self.training_history,
            'lambda_reg': self.lambda_reg,
            'model_type': 'DualEngineADF_Dynamic',
            'dynamic_thresholder': self.dynamic_thresholder if hasattr(self, 'dynamic_thresholder') else None
        }
        
        torch.save(checkpoint, filepath)
        print(f"\nModel saved to: {filepath}")
        
        
    def print_training_debug(self, epoch):
        if hasattr(self, '_debug_info'):
            debug = self._debug_info
            print(f"Epoch {epoch}: boundary ratio={debug['boundary_ratio']:.3f}, "
                  f"boundary loss={debug['boundary_loss']:.4f}, "
                  f"non-boundary loss={debug['non_boundary_loss']:.4f}, "
                  f"MLNN variance={debug['mlnn_std']:.4f}")
'''=======================================================================================================''' 
        
        
        

'''======================================================================================================='''         
''' ########################    Training Strategy    ############# '''

class ProgressiveTrainingStrategy:
    """Progressive training strategy as described in the paper"""
    
    def __init__(self, model):
        self.model = model
        self.stage_results = {}
        
 
    def train_with_progressive_strategy(self, X_train, y_train, X_val, y_val):
        """Execute progressive training strategy with boundary-aware focus"""
        print("Starting boundary-aware progressive training strategy...")
        
        stages_config = [
            {
                'name': 'Stage 1: Boundary Sample Identification',
                'epochs': 25,
                'lr': 0.001,
                'lambda_reg': 0.3,
                'focus': 'independent',   # Phase 1: Independent learning, identifying boundaries
                'patience': 8
            },
            {
                'name': 'Stage 2: Boundary Discrimination Enhancement', 
                'epochs': 40,
                'lr': 0.0005,
                'lambda_reg': 0.5,
                'focus': 'complementary',  # Phase 2: Complementary learning to enhance boundary discrimination
                'patience': 12
            },
            {
                'name': 'Stage 3: Whole-Space Optimization',
                'epochs': 25,
                'lr': 0.0001,
                'lambda_reg': 0.2,
                'focus': 'fine_tune', # Phase 3: Overall fine-tuning
                'patience': 6
            }
        ]
        
        overall_history = {
            'mlnn_loss': [], 'goe_loss': [], 'dual_loss': [],
            'mlnn_accuracy': [], 'mlnn_f1': [], 'goe_accuracy': [], 'goe_f1': [],
            'dual_accuracy': [], 'dual_f1': [], 'stage': []
        }
        
        total_epoch_offset = 0
        for stage_idx, stage_config in enumerate(stages_config):
            print(f"\n{stage_config['name']}")
            print(f"  Focus: {stage_config['focus']}")
            
            self.model.lambda_reg = stage_config['lambda_reg']
            stage_result = self._train_stage(
                X_train, y_train, X_val, y_val,
                epochs=stage_config['epochs'],
                lr=stage_config['lr'],
                patience=stage_config['patience'],
                focus=stage_config['focus'],
                stage_idx=stage_idx
            )
            
            self.stage_results[stage_config['name']] = stage_result
            
            for key in overall_history:
                if key in stage_result['history']:
                    overall_history[key].extend(stage_result['history'][key])
                elif key == 'stage':
                    overall_history[key].extend([stage_idx] * len(stage_result['history'].get('dual_loss', [])))
        
        self.model.training_history = overall_history
        
        print("Boundary-aware progressive training strategy completed")
        return self.stage_results
        
    
    def _train_stage(self, X_train, y_train, X_val, y_val, epochs, lr, patience, focus, stage_idx):
       
                    """Execute a single training phase - optimize training strategies for different focuses"""
                    sequence_length = 5
                    
                    y_train = ensure_numpy_array(y_train)
                    y_val = ensure_numpy_array(y_val)

                    X_seq = self._create_sequences(X_train, sequence_length)
                    y_seq = self._create_sequence_labels(y_train, sequence_length)
                    
                    X_val_seq = self._create_sequences(X_val, sequence_length)
                    y_val_seq = self._create_sequence_labels(y_val, sequence_length)
                    
                    X_tensor = torch.FloatTensor(X_seq).to(self.model.device)
                    y_tensor = torch.FloatTensor(y_seq).to(self.model.device)
                    X_val_tensor = torch.FloatTensor(X_val_seq).to(self.model.device)
                    
                    optimizer = optim.Adam(self.model.mlnn_engine.parameters(), lr=lr)
                    
                    # Adjust the loss function according to different training stages
                    if focus == 'independent':
                        # MLNN independent training phase: almost no use of GOE regularization, emphasizing MLNN autonomy
                        print(f"\n   Initial Phase: MLNN independent training stage -> emphasizes self-learning, with extremely low GOE regularization weights. \n")
                        
                        criterion = nn.BCEWithLogitsLoss()
                        
                        stage_history = {
                            'mlnn_loss': [], 'dual_loss': [],
                            'train_mlnn_accuracy': [], 'train_mlnn_precision': [], 'train_mlnn_recall': [], 'train_mlnn_f1': [], 'train_mlnn_gmean': [],
                            'train_dual_accuracy': [], 'train_dual_precision': [], 'train_dual_recall': [], 'train_dual_f1': [], 'train_dual_gmean': []
                        }
                        
                        csv_path = 'Climate_results/training_metrics_consistent.csv'
                        is_new_file = not os.path.exists(csv_path)
                        
                        for epoch in range(epochs):
                            self.model.mlnn_engine.train()
                            self.model.training_epoch = epoch   
                            optimizer.zero_grad()
                            
                            mlnn_scores = self.model.mlnn_engine(X_tensor)
                            mlnn_loss = criterion(mlnn_scores, y_tensor)
                            
                            # Independent training phase: extremely low GOE regularization (only maintains direction, does not affect primary learning)
                            try:
                                goe_regularization = self.model._compute_goe_regularization(X_tensor, mlnn_scores)
                                dual_loss = mlnn_loss + 0.01 * goe_regularization  
                            except Exception as e:
                                print(f"GOE regularization failure: {e}")
                                dual_loss = mlnn_loss
                            
                            dual_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.mlnn_engine.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            train_metrics = self._compute_training_metrics_consistent(X_train, y_train, X_tensor, y_tensor)
                            
                            goe_reg_value = goe_regularization.item() if 'goe_regularization' in locals() else 0.0
                            self._save_training_metrics_to_csv(csv_path, stage_idx, epoch, train_metrics, 
                                                              mlnn_loss.item(), goe_reg_value, dual_loss.item(),
                                                              is_new_file and epoch == 0)
                            
                            stage_history['mlnn_loss'].append(mlnn_loss.item())
                            stage_history['dual_loss'].append(dual_loss.item())
                            
                            for key in train_metrics:
                                if key in stage_history:
                                    stage_history[key].append(train_metrics[key])
                            
                            if (epoch + 1) % 5 == 0:
                                dual_accuracy = train_metrics.get('train_dual_accuracy', 0)
                                dual_f1 = train_metrics.get('train_dual_f1', 0)
                                mlnn_f1 = train_metrics.get('train_mlnn_f1', 0)
                                print(f"  Epoch {epoch+1}/{epochs}: MLNN F1={mlnn_f1:.4f}, Dual F1={dual_f1:.4f}")
                        
                        return {
                            'best_metrics': train_metrics,
                            'best_f1': train_metrics.get('train_dual_f1', 0),
                            'history': stage_history
                        }
                        
                    elif focus == 'complementary':
                        # Complementary learning stage: Moderate GOE guidance, emphasizing complementary learning
                        print(f"\n  GOE Regularization Phase: GOE guided complementary learning stage -> emphasizing the complementarity between GOE and MLNN")
                        
                        n_normal = np.sum(y_train == 0)
                        n_anomaly = np.sum(y_train == 1)
                        if n_anomaly > 0:
                            weight_for_0 = 1.0
                            weight_for_1 = (n_normal / n_anomaly) * 0.8  
                            pos_weight = torch.tensor([weight_for_1 / weight_for_0]).to(self.model.device)
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        else:
                            criterion = nn.BCEWithLogitsLoss()
                        
                        stage_history = {
                            'mlnn_loss': [], 'goe_loss': [], 'dual_loss': [],
                            'train_mlnn_accuracy': [], 'train_mlnn_precision': [], 'train_mlnn_recall': [], 'train_mlnn_f1': [], 'train_mlnn_gmean': [],
                            'train_goe_accuracy': [], 'train_goe_precision': [], 'train_goe_recall': [], 'train_goe_f1': [], 'train_goe_gmean': [],
                            'train_dual_accuracy': [], 'train_dual_precision': [], 'train_dual_recall': [], 'train_dual_f1': [], 'train_dual_gmean': []
                        }
                        
                        csv_path = 'Climate_results/training_metrics_consistent.csv'
                        is_new_file = not os.path.exists(csv_path)
                        
                        best_f1 = 0.0
                        best_model_state = None
                        best_metrics = {}
                        
                        for epoch in range(epochs):
                            self.model.mlnn_engine.train()
                            self.model.training_epoch = epoch   
                            optimizer.zero_grad()
                            
                            mlnn_scores = self.model.mlnn_engine(X_tensor)
                            mlnn_loss = criterion(mlnn_scores, y_tensor)
                            
                            # Complementary Learning Stage: Moderate GOE Regularization
                            try:
                                goe_regularization = self.model._compute_goe_regularization(X_tensor, mlnn_scores)
                                
                                # Dynamic adjustment of fusion weights: MLNN is mainly used in the early stage, and GOE guidance is added in the later stage
                                progress = epoch / max(1, epochs-1)
                                
                                # Early low GOE guidance
                                if progress < 0.5:
                                    lambda_weight = 0.3  
                                else:
                                    # Add GOE guidance in the later stage
                                    lambda_weight = 0.6   
                                
                                dual_loss = mlnn_loss + lambda_weight * goe_regularization
                                
                            except Exception as e:
                                print(f"GOE regularization failed, using MLNN loss: {e}")
                                dual_loss = mlnn_loss
                                goe_regularization = torch.tensor(0.0, device=self.model.device)
                            
                            dual_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.mlnn_engine.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            train_metrics = self._compute_training_metrics_consistent(X_train, y_train, X_tensor, y_tensor)
                            
                            goe_reg_value = goe_regularization.item() if hasattr(goe_regularization, 'item') else goe_regularization
                            self._save_training_metrics_to_csv(csv_path, stage_idx, epoch, train_metrics, mlnn_loss.item(), goe_reg_value, dual_loss.item(),is_new_file and epoch == 0)
                            
                            stage_history['mlnn_loss'].append(mlnn_loss.item())
                            stage_history['goe_loss'].append(goe_regularization.item() if hasattr(goe_regularization, 'item') else goe_regularization)
                            stage_history['dual_loss'].append(dual_loss.item())
                            
                            for key in train_metrics:
                                if key in stage_history:
                                    stage_history[key].append(train_metrics[key])
                            
                            if (epoch + 1) % 5 == 0:
                                dual_accuracy = train_metrics.get('train_dual_accuracy', 0)
                                dual_precision = train_metrics.get('train_dual_precision', 0)
                                dual_recall = train_metrics.get('train_dual_recall', 0)
                                dual_f1 = train_metrics.get('train_dual_f1', 0)
                                print(f"  Epoch {epoch+1}/{epochs}: Train Dual Accuracy={dual_accuracy:.4f}, F1={dual_f1:.4f}")
                            
                            # Early stop check (using validation set)
                            self.model.mlnn_engine.eval()
                            with torch.no_grad():
                                val_scores = self.model.predict_anomaly_scores(X_val)
                                val_dual_scores = val_scores['dual_scores']
                                val_goe_scores = val_scores['goe_scores'][:len(val_dual_scores)]
                                
                                y_val_adj = y_val[:len(val_dual_scores)]
                                y_val_int = y_val_adj.astype(int)
                                
                                val_threshold, val_final_scores = TheoremBasedThresholdDynamic.enhanced_synergy_strategy(
                                    val_goe_scores, val_dual_scores, y_val_int
                                )
                                val_predictions = (val_final_scores > val_threshold).astype(int)
                                val_f1 = f1_score(y_val_int, val_predictions, zero_division=0)
                                
                                if val_f1 > best_f1:
                                    best_f1 = val_f1
                                    best_model_state = self.model.mlnn_engine.state_dict().copy()
                                    best_metrics = train_metrics.copy()
                        
                        if best_model_state is not None:
                            self.model.mlnn_engine.load_state_dict(best_model_state)
                        
                        return {
                            'best_metrics': best_metrics,
                            'best_f1': best_f1,
                            'history': stage_history
                        }
                        
                    elif focus == 'fine_tune':
                        # Fine tuning stage: Balance two engines and optimize overall performance
                        print(f"\n Stable Alignment Phase: Dual engine fine-tuning stage -> optimizing overall performance \n")
                        
                        n_normal = np.sum(y_train == 0)
                        n_anomaly = np.sum(y_train == 1)
                        if n_anomaly > 0:
                            weight_for_0 = 1.0
                            weight_for_1 = (n_normal / n_anomaly) * 1.0  
                            pos_weight = torch.tensor([weight_for_1 / weight_for_0]).to(self.model.device)
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        else:
                            criterion = nn.BCEWithLogitsLoss()
                        
                        stage_history = {
                            'mlnn_loss': [], 'goe_loss': [], 'dual_loss': [],
                            'train_mlnn_accuracy': [], 'train_mlnn_precision': [], 'train_mlnn_recall': [], 'train_mlnn_f1': [], 'train_mlnn_gmean': [],
                            'train_goe_accuracy': [], 'train_goe_precision': [], 'train_goe_recall': [], 'train_goe_f1': [], 'train_goe_gmean': [],
                            'train_dual_accuracy': [], 'train_dual_precision': [], 'train_dual_recall': [], 'train_dual_f1': [], 'train_dual_gmean': []
                        }
                        
                        csv_path = 'Climate_results/training_metrics_consistent.csv'
                        is_new_file = not os.path.exists(csv_path)
                        
                        best_f1 = 0.0
                        best_model_state = None
                        best_metrics = {}
                        
                        for epoch in range(epochs):
                            self.model.mlnn_engine.train()
                            self.model.training_epoch = epoch  
                            optimizer.zero_grad()
                            
                            mlnn_scores = self.model.mlnn_engine(X_tensor)
                            mlnn_loss = criterion(mlnn_scores, y_tensor)
                            
                            # Fine tuning stage: Moderate GOE regularization, emphasizing fusion optimization
                            try:
                                goe_regularization = self.model._compute_goe_regularization(X_tensor, mlnn_scores)
                                dual_loss = mlnn_loss + 0.1 * goe_regularization  # Lower GOE guidance
                            except Exception as e:
                                print(f"GOE regularization failed, using MLNN loss: {e}")
                                dual_loss = mlnn_loss
                                goe_regularization = torch.tensor(0.0, device=self.model.device)
                            
                            dual_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.mlnn_engine.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            train_metrics = self._compute_training_metrics_consistent(X_train, y_train, X_tensor, y_tensor)
                            
                            goe_reg_value = goe_regularization.item() if hasattr(goe_regularization, 'item') else goe_regularization
                            self._save_training_metrics_to_csv(csv_path, stage_idx, epoch, train_metrics, 
                                                              mlnn_loss.item(), goe_reg_value, dual_loss.item(),
                                                              is_new_file and epoch == 0)
                            
                            stage_history['mlnn_loss'].append(mlnn_loss.item())
                            stage_history['goe_loss'].append(goe_regularization.item() if hasattr(goe_regularization, 'item') else goe_regularization)
                            stage_history['dual_loss'].append(dual_loss.item())
                            
                            for key in train_metrics:
                                if key in stage_history:
                                    stage_history[key].append(train_metrics[key])
                            
                            if (epoch + 1) % 5 == 0:
                                dual_f1 = train_metrics.get('train_dual_f1', 0)
                                mlnn_f1 = train_metrics.get('train_mlnn_f1', 0)
                                goe_f1 = train_metrics.get('train_goe_f1', 0)
                                print(f"  Epoch {epoch+1}/{epochs}: MLNN F1={mlnn_f1:.4f}, GOE F1={goe_f1:.4f}, Dual F1={dual_f1:.4f}")
                            
                            # Early stop check
                            self.model.mlnn_engine.eval()
                            with torch.no_grad():
                                val_scores = self.model.predict_anomaly_scores(X_val)
                                val_dual_scores = val_scores['dual_scores']
                                val_goe_scores = val_scores['goe_scores'][:len(val_dual_scores)]
                                
                                y_val_adj = y_val[:len(val_dual_scores)]
                                y_val_int = y_val_adj.astype(int)
                                
                                val_threshold, val_final_scores = TheoremBasedThresholdDynamic.enhanced_synergy_strategy(
                                    val_goe_scores, val_dual_scores, y_val_int
                                )
                                val_predictions = (val_final_scores > val_threshold).astype(int)
                                val_f1 = f1_score(y_val_int, val_predictions, zero_division=0)
                                
                                if val_f1 > best_f1:
                                    best_f1 = val_f1
                                    best_model_state = self.model.mlnn_engine.state_dict().copy()
                                    best_metrics = train_metrics.copy()
                        
                        if best_model_state is not None:
                            self.model.mlnn_engine.load_state_dict(best_model_state)
                        
                        return {
                            'best_metrics': best_metrics,
                            'best_f1': best_f1,
                            'history': stage_history
                        }
                        
                    else:
                        print("...........call backup training strategy\n................")
                        return self._train_stage_fallback(X_train, y_train, X_val, y_val, epochs, lr, patience, focus, stage_idx)
        
    
    def _train_stage_fallback(self, X_train, y_train, X_val, y_val, epochs, lr, patience, focus, stage_idx):
            """Backup Training Method - Training Loop Based on Deterministic Learning"""
            
            sequence_length = 5
            
            y_train = ensure_numpy_array(y_train)
            y_val = ensure_numpy_array(y_val)

            X_seq = self._create_sequences(X_train, sequence_length)
            y_seq = self._create_sequence_labels(y_train, sequence_length)
            
            X_val_seq = self._create_sequences(X_val, sequence_length)
            y_val_seq = self._create_sequence_labels(y_val, sequence_length)
            
            X_tensor = torch.FloatTensor(X_seq).to(self.model.device)
            y_tensor = torch.FloatTensor(y_seq).to(self.model.device)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.model.device)
            
            optimizer = optim.Adam(self.model.mlnn_engine.parameters(), lr=lr)
            
            # 1. Regularization loss: learning deterministic patterns of GOE
            # 2. Auxiliary loss: When GOE is determined, MLNN should be consistent with GOE decision
            
            stage_history = {
                'mlnn_loss': [], 'goe_loss': [], 'dual_loss': [],
                'train_mlnn_accuracy': [], 'train_mlnn_precision': [], 'train_mlnn_recall': [], 'train_mlnn_f1': [], 'train_mlnn_gmean': [],
                'train_goe_accuracy': [], 'train_goe_precision': [], 'train_goe_recall': [], 'train_goe_f1': [], 'train_goe_gmean': [],
                'train_dual_accuracy': [], 'train_dual_precision': [], 'train_dual_recall': [], 'train_dual_f1': [], 'train_dual_gmean': []
            }
            
            csv_path = 'Climate_results/training_metrics_consistent.csv'
            is_new_file = not os.path.exists(csv_path)
            
            best_f1 = 0.0
            best_model_state = None
            best_metrics = {}
            
            for epoch in range(epochs):
                self.model.mlnn_engine.train()
                optimizer.zero_grad()
                
                # MLNN outputs deterministic score (forward directly outputs determinism)
                mlnn_certainty = self.model.mlnn_engine(X_tensor)
                
                # Calculate GOE regularization loss
                goe_regularization = self.model._compute_goe_regularization(X_tensor, mlnn_certainty)
                
                # Auxiliary loss: Encourage MLNN to make correct judgments when GOE is determined
                # Obtain GOE score
                X_flat = X_tensor.reshape(-1, X_tensor.shape[-1])
                X_np = X_flat.detach().cpu().numpy()
                goe_scores = self.model.goe_engine.compute_anomaly_scores(X_np)
                goe_scores_reshaped = goe_scores.reshape(X_tensor.shape[0], -1)
                goe_scores_avg = np.mean(goe_scores_reshaped, axis=1)
                goe_norm = (goe_scores_avg - np.min(goe_scores_avg)) / (np.max(goe_scores_avg) - np.min(goe_scores_avg) + 1e-8)
                
                # Certainty of GOE
                goe_certainty = 1.0 - 2.0 * np.abs(goe_norm - 0.5)
                
                # GOE's decision (when certainty is high)
                high_certainty_mask = (goe_certainty > 0.8)
                if np.sum(high_certainty_mask) > 0:
                    goe_decisions = (goe_norm > 0.5).astype(float)
                    
                    # Select high certainty samples
                    mlnn_high_certainty = mlnn_certainty[high_certainty_mask]
                    goe_high_decisions = torch.FloatTensor(goe_decisions[high_certainty_mask]).to(self.model.device)
                    
                    # Auxiliary loss: MLNN should be consistent with GOE's high certainty decision
                    auxiliary_loss = nn.BCELoss()(mlnn_high_certainty, goe_high_decisions)
                else:
                    auxiliary_loss = torch.tensor(0.0, device=self.model.device)
                
                # Total loss: mainly using GOE regularization loss, with auxiliary loss as a guide
                # Adjust the loss weight based on the training phase
                if focus == 'boundary_identification' or focus == 'independent':
                    # Phase 1: Smaller auxiliary loss weights
                    lambda_aux = 0.05  
                elif focus == 'boundary_discrimination' or focus == 'complementary':
                    # Phase 2: Moderate auxiliary loss weights
                    lambda_aux = 0.1   
                else:  
                    # Phase 3: Higher auxiliary loss weights
                    lambda_aux = 0.15  
                
                dual_loss = goe_regularization + lambda_aux * auxiliary_loss
                
                dual_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.mlnn_engine.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_metrics = self._compute_training_metrics_consistent(X_train, y_train, X_tensor, y_tensor)
                
                self._save_training_metrics_to_csv(csv_path, stage_idx, epoch, train_metrics, 
                                                  dual_loss.item(), goe_regularization.item(), dual_loss.item(),
                                                  is_new_file and epoch == 0)
                
                stage_history['mlnn_loss'].append(dual_loss.item())
                stage_history['goe_loss'].append(goe_regularization.item())
                stage_history['dual_loss'].append(dual_loss.item())
                
                for key in train_metrics:
                    if key in stage_history:
                        stage_history[key].append(train_metrics[key])
                
                if (epoch + 1) % 5 == 0:
                    dual_f1 = train_metrics.get('train_dual_f1', 0)
                    mlnn_f1 = train_metrics.get('train_mlnn_f1', 0)
                    goe_f1 = train_metrics.get('train_goe_f1', 0)
                    print(f"  Epoch {epoch+1}/{epochs}: MLNN F1={mlnn_f1:.4f}, GOE F1={goe_f1:.4f}, Dual F1={dual_f1:.4f}")
                
                self.model.mlnn_engine.eval()
                with torch.no_grad():
                    val_scores = self.model.predict_anomaly_scores(X_val)
                    val_goe_scores = val_scores['goe_scores']
                    val_mlnn_scores = val_scores['mlnn_scores']
                    
                    min_val_length = min(len(val_goe_scores), len(val_mlnn_scores))
                    val_goe_scores = val_goe_scores[:min_val_length]
                    val_mlnn_scores = val_mlnn_scores[:min_val_length]
                    
                    y_val_adj = y_val[:min_val_length]
                    y_val_int = y_val_adj.astype(int)
                    
                    val_threshold, val_final_scores = EnhancedDualEngineSynergy.adaptive_fusion_strategy(
                        val_goe_scores, val_mlnn_scores, y_val_int
                    )
                    val_predictions = (val_final_scores > val_threshold).astype(int)
                    val_f1 = f1_score(y_val_int, val_predictions, zero_division=0)
                    
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_model_state = self.model.mlnn_engine.state_dict().copy()
                        best_metrics = train_metrics.copy()
            
            if best_model_state is not None:
                self.model.mlnn_engine.load_state_dict(best_model_state)
            
            return {
                'best_metrics': best_metrics,
                'best_f1': best_f1,
                'history': stage_history
            }
    
    
    """####### calculate training metrics#############"""
    def _compute_training_metrics_consistent(self, X_train_flat, y_train_flat, X_tensor, y_tensor):
        
        self.model.mlnn_engine.eval()
        metrics = {}
        
        with torch.no_grad():
            train_scores = self.model.predict_anomaly_scores(X_train_flat)
            
            train_dual_scores = train_scores['dual_scores']
            train_goe_scores = train_scores['goe_scores'][:len(train_dual_scores)]
            
            y_train_adj = y_train_flat[:len(train_dual_scores)]
            y_train_int = y_train_adj.astype(int)
            
            train_threshold, train_final_scores = TheoremBasedThresholdDynamic.enhanced_synergy_strategy(
                train_goe_scores, train_dual_scores, y_train_int
            )
            
            train_predictions = (train_final_scores > train_threshold).astype(int)
            
            goe_norm = (train_goe_scores - np.min(train_goe_scores)) / (np.max(train_goe_scores) - np.min(train_goe_scores) + 1e-8)
            goe_threshold, goe_final_scores = TheoremBasedThresholdDynamic.enhanced_synergy_strategy(
                goe_norm, goe_norm, y_train_int
            )
            goe_predictions = (goe_final_scores > goe_threshold).astype(int)
            
            mlnn_probabilities = train_scores['mlnn_scores'][:len(train_dual_scores)]
            mlnn_norm = (mlnn_probabilities - np.min(mlnn_probabilities)) / (np.max(mlnn_probabilities) - np.min(mlnn_probabilities) + 1e-8)
            mlnn_threshold, mlnn_final_scores = TheoremBasedThresholdDynamic.enhanced_synergy_strategy(
                mlnn_norm, mlnn_norm, y_train_int
            )
            mlnn_predictions = (mlnn_final_scores > mlnn_threshold).astype(int)

            metrics['train_dual_accuracy'] = accuracy_score(y_train_int, train_predictions)
            metrics['train_dual_precision'] = precision_score(y_train_int, train_predictions, zero_division=0)
            metrics['train_dual_recall'] = recall_score(y_train_int, train_predictions, zero_division=0)
            metrics['train_dual_f1'] = f1_score(y_train_int, train_predictions, zero_division=0)
            metrics['train_dual_gmean'] = calculate_gmean(y_train_int, train_predictions)
            metrics['train_dual_threshold'] = train_threshold
            
            metrics['train_mlnn_accuracy'] = accuracy_score(y_train_int, mlnn_predictions)
            metrics['train_mlnn_precision'] = precision_score(y_train_int, mlnn_predictions, zero_division=0)
            metrics['train_mlnn_recall'] = recall_score(y_train_int, mlnn_predictions, zero_division=0)
            metrics['train_mlnn_f1'] = f1_score(y_train_int, mlnn_predictions, zero_division=0)
            metrics['train_mlnn_gmean'] = calculate_gmean(y_train_int, mlnn_predictions)
            metrics['train_mlnn_threshold'] = mlnn_threshold
            
            metrics['train_goe_accuracy'] = accuracy_score(y_train_int, goe_predictions)
            metrics['train_goe_precision'] = precision_score(y_train_int, goe_predictions, zero_division=0)
            metrics['train_goe_recall'] = recall_score(y_train_int, goe_predictions, zero_division=0)
            metrics['train_goe_f1'] = f1_score(y_train_int, goe_predictions, zero_division=0)
            metrics['train_goe_gmean'] = calculate_gmean(y_train_int, goe_predictions)
            metrics['train_goe_threshold'] = goe_threshold
            
            metrics['train_dual_scores_mean'] = np.mean(train_final_scores)
            metrics['train_dual_scores_std'] = np.std(train_final_scores)
            metrics['train_anomaly_rate'] = np.mean(y_train_int)
        
        self.model.mlnn_engine.train()
        return metrics
        
        
    def _save_training_metrics_to_csv(self, csv_path, stage, epoch, train_metrics, mlnn_loss, goe_loss, dual_loss, is_header):
        """ ################## save training metrics to CSV file #####################"""
        engines = ['mlnn', 'goe', 'dual']
        
        mode = 'a' if not is_header else 'w'
        
        with open(csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if is_header:
                writer.writerow(['stage', 'epoch', 'engine', 'accuracy', 'precision', 'recall', 'f1', 'gmean', 'threshold', 'loss'])
            
            for engine in engines:
                row = [
                    stage,
                    epoch,
                    engine,
                    train_metrics.get(f'train_{engine}_accuracy', 0.0),
                    train_metrics.get(f'train_{engine}_precision', 0.0),
                    train_metrics.get(f'train_{engine}_recall', 0.0),
                    train_metrics.get(f'train_{engine}_f1', 0.0),
                    train_metrics.get(f'train_{engine}_gmean', 0.0),
                    train_metrics.get(f'train_{engine}_threshold', 0.0),
                    mlnn_loss if engine == 'mlnn' else (goe_loss if engine == 'goe' else dual_loss)
                ]
                writer.writerow(row)
                
                
    def _create_sequences(self, X, sequence_length):
        """Create time sequences"""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
        if not sequences:
            sequences = [X]
        return np.array(sequences)
    
    
    def _create_sequence_labels(self, y, sequence_length):
        """Create sequence labels"""
        if len(y) <= sequence_length:
            return y[-1:] if len(y) > 0 else np.array([0])
        return y[sequence_length-1:]
    
    
    def _assign_global_epochs(self, df):
        current_global_epoch = 0
        stage_epoch_combos = df[['stage', 'epoch']].drop_duplicates().sort_values(['stage', 'epoch'])
        
        # Create a mapping dictionary
        stage_epoch_to_global = {}
        for _, row in stage_epoch_combos.iterrows():
            stage, epoch = row['stage'], row['epoch']
            stage_epoch_to_global[(stage, epoch)] = current_global_epoch
            current_global_epoch += 1
        
        global_epochs = []
        for _, row in df.iterrows():
            key = (row['stage'], row['epoch'])
            global_epochs.append(stage_epoch_to_global[key])
        
        return global_epochs
    
        
    def _extract_engine_data(self, df):
        """Extract each engine's data from DataFrame"""
        engines = ['mlnn', 'goe', 'dual']
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'gmean', 'loss']
        
        # Initialize data structure - use dictionary to store lists
        engine_data = {engine: {metric: {'epochs': [], 'values': []} for metric in metrics} for engine in engines}

        # Fill data by engine and global epoch
        for engine in engines:
            engine_df = df[df['engine'] == engine]
            if len(engine_df) > 0:
                # Ensure sorted by global epoch
                engine_df = engine_df.sort_values('global_epoch')
                
                for metric in metrics:
                    if metric in engine_df.columns:
                        # Get all non-NaN values
                        metric_series = engine_df[['global_epoch', metric]].dropna()
                        
                        if len(metric_series) > 0:
                            engine_data[engine][metric]['epochs'] = metric_series['global_epoch'].tolist()
                            engine_data[engine][metric]['values'] = metric_series[metric].tolist()

        return engine_data
'''======================================================================================================='''

def evaluate_on_test(model, X_test, y_test, val_dual_scores, val_threshold,
                             val_percentile=None, adaptation='percentile_val',
                             save_path='CredictCard_results/Test/result.csv'):
        """
        Evaluate on test set using a fixed threshold.
        
        Parameters:
            model: trained DualEngineADF model
            X_test, y_test: test data
            fixed_threshold: threshold to use for binarization
            save_path: path to save CSV results
        """
        test_scores = model.predict_anomaly_scores(X_test, use_dynamic_threshold=False)
        test_dual = test_scores['dual_scores']

        if adaptation == 'percentile_val':
            if val_percentile is None:
                raise ValueError("val_percentile must be provided for percentile adaptation")
            used_threshold = np.percentile(test_dual, val_percentile)
            print(f"Percentile adaptation (p={val_percentile:.2f}%): test threshold={used_threshold:.6f}")
        else:
            used_threshold = val_threshold
            print(f"No adaptation: test threshold={used_threshold:.6f}")

        # Align lengths
        min_len = min(len(test_dual), len(y_test))
        test_dual = test_dual[:min_len]
        y_true = y_test[:min_len].astype(int)
        predictions = (test_dual > used_threshold).astype(int)

        # Calculate metrics
        from sklearn.metrics import average_precision_score
        auc = roc_auc_score(y_true, test_dual)
        pr_auc = average_precision_score(y_true, test_dual)
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        gmean = calculate_gmean(y_true, predictions)

        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df = pd.DataFrame({
            'Metric': ['AUC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'G-Mean'],
            'Value': [auc, pr_auc, accuracy, precision, recall, f1, gmean]
        })
        results_df.to_csv(save_path, index=False)

        print(f"\n=== Test Results (adaptation={adaptation}) ===")
        print(results_df.to_string(index=False))
        print(f"Used threshold: {used_threshold:.6f}")
        return results_df    
 
def find_optimal_threshold(model, X_val, y_val, metric='f1', num_candidates=100):
    """
    Find optimal threshold on validation set that maximizes the given metric.
    
    Parameters:
        model: trained DualEngineADF model
        X_val, y_val: validation data
        metric: one of 'f1', 'gmean', 'recall', 'precision', 'accuracy'
        num_candidates: number of threshold candidates to evaluate
    
    Returns:
        best_threshold: float
        best_score: float
    """
    scores = model.predict_anomaly_scores(X_val, use_dynamic_threshold=False)
    dual_scores = scores['dual_scores']
    min_len = min(len(dual_scores), len(y_val))
    dual_scores = dual_scores[:min_len]
    y_val = y_val[:min_len].astype(int)

    low_percentile, high_percentile = 80, 99.9
    low = np.percentile(dual_scores, low_percentile)
    high = np.percentile(dual_scores, high_percentile)
    if high <= low:
        low, high = np.min(dual_scores), np.max(dual_scores)

    thresholds = np.linspace(low, high, num_candidates)
    best_th = thresholds[0]
    best_score = 0.0

    for th in thresholds:
        pred = (dual_scores > th).astype(int)
        if metric == 'f1':
            score = f1_score(y_val, pred, zero_division=0)
        elif metric == 'gmean':
            score = calculate_gmean(y_val, pred)
        elif metric == 'recall':
            score = recall_score(y_val, pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, pred, zero_division=0)
        else:
            score = accuracy_score(y_val, pred)
        if score > best_score:
            best_score = score
            best_th = th

    # Calculate the percentile corresponding to the optimal threshold
    percentile = np.sum(dual_scores <= best_th) / len(dual_scores) * 100

    print(f"Best threshold on validation set ({metric}): {best_th:.6f} with score {best_score:.4f}, percentile={percentile:.2f}%")
    return best_th, best_score, dual_scores, percentile
    

def dEADF_climate_detection(model, data_path='./DataSample/Climate_Label_Scientific.csv'):
          """
          Use the trained dEADF model for climate anomaly detection.
          Returns detected anomaly years and performance metrics.
          """
          import pandas as pd
          import numpy as np
          from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

          # 1. Load climate data
          try:
              df = pd.read_csv(data_path)
          except FileNotFoundError:
              print(f"Error: Climate data file not found at '{data_path}'")
              return None

          temp_series = df['Global_Temp_Anomaly'].values
          years = df['Year'].values

          # 2. Get model input feature size
          input_size = model.goe_engine.projection_basis.shape[0]
          print(f"Model input feature size: {input_size}")

          # Assume features are the last (input_size-1) lagged values plus the current value
          window_size = input_size - 1
          if window_size < 1:
              print(f"Error: window_size ({window_size}) is too small, check input_size")
              return None

          # 3. Build feature matrix (ensure columns match input_size)
          features = []
          for i in range(window_size, len(temp_series)):
              feature_vec = list(temp_series[i-window_size:i+1])
              features.append(feature_vec)

          features = np.array(features)
          if len(features) == 0:
              print("Error: Not enough data to create features")
              return None

          valid_years = years[window_size:]          # same length as features
          y_true_full = df['Label_Scientific'].values[window_size:]  # same length as features

          print(f"Features shape: {features.shape}, valid samples: {len(features)}")

          # 4. Use dEADF for prediction
          print("\nRunning dEADF climate anomaly detection...")
          scores = model.predict_anomaly_scores(features, use_dynamic_threshold=True)

          # 5. Extract anomaly scores and dynamic threshold
          probs_mlnn = scores['mlnn_scores']
          probs_goe  = scores['goe_scores']
          probs_dual = scores['dual_scores']
          threshold  = scores['threshold']

          # 6. Generate binary predictions
          predictions_mlnn = (probs_mlnn > threshold).astype(int)
          predictions_goe  = (probs_goe  > threshold).astype(int)
          predictions_dual = (probs_dual > threshold).astype(int)

          # ========== Key modification: align all predictions with true labels ==========
          # Take the minimum common length among all predictions and true labels
          min_len = min(len(predictions_mlnn), len(predictions_goe), len(predictions_dual), len(y_true_full))
          if min_len == 0:
              print("Error: No valid predictions")
              return None

          y_true_aligned = y_true_full[:min_len]
          valid_years_aligned = valid_years[:min_len]

          predictions_mlnn_aligned = predictions_mlnn[:min_len]
          predictions_goe_aligned  = predictions_goe[:min_len]
          predictions_dual_aligned = predictions_dual[:min_len]

          # Also truncate probability values to same length (optional)
          probs_mlnn_aligned = probs_mlnn[:min_len]
          probs_goe_aligned  = probs_goe[:min_len]
          probs_dual_aligned = probs_dual[:min_len]

          print(f"Aligned sample count: {min_len} (original features: {len(features)})")

          # 7. Extract detected anomaly years (based on aligned data)
          anomaly_indices_dual = np.where(predictions_dual_aligned == 1)[0]
          detected_years_dual = valid_years_aligned[anomaly_indices_dual].tolist()

          # 8. Compute performance metrics (using aligned data)
          def calculate_metrics(y_true, y_pred, engine_name):
              if len(np.unique(y_true)) < 2:
                  print(f"Warning: {engine_name} - Only one class in y_true")
                  return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
              try:
                  accuracy = accuracy_score(y_true, y_pred)
                  precision = precision_score(y_true, y_pred, zero_division=0)
                  recall = recall_score(y_true, y_pred, zero_division=0)
                  f1 = f1_score(y_true, y_pred, zero_division=0)
              except Exception as e:
                  print(f"Error calculating metrics for {engine_name}: {e}")
                  accuracy = precision = recall = f1 = 0
              return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

          metrics_mlnn = calculate_metrics(y_true_aligned, predictions_mlnn_aligned, 'MLNN')
          metrics_goe  = calculate_metrics(y_true_aligned, predictions_goe_aligned,  'GOE')
          metrics_dual = calculate_metrics(y_true_aligned, predictions_dual_aligned, 'Dual')

          # Optional: compute AUC and PR-AUC using probability values
          try:
              auc = roc_auc_score(y_true_aligned, probs_dual_aligned)
          except:
              auc = 0.0
          metrics_dual['auc'] = auc

          # 9. Compare with scientific consensus labels (based on aligned years)
          scientific_anomalies = df[df['Label_Scientific'] == 1]['Year'].tolist()
          # Only consider scientific anomalies within the valid_years_aligned range
          scientific_anomalies_in_range = [y for y in scientific_anomalies if y in valid_years_aligned]
          directly_detected = len(set(detected_years_dual) & set(scientific_anomalies_in_range))
          direct_detection_rate = directly_detected / len(scientific_anomalies_in_range) if scientific_anomalies_in_range else 0

          window_detected = 0
          for year in scientific_anomalies_in_range:
              if any(abs(year - dy) <= 2 for dy in detected_years_dual):
                  window_detected += 1
          window_detection_rate = window_detected / len(scientific_anomalies_in_range) if scientific_anomalies_in_range else 0

          comparison_table = {
              'Year': scientific_anomalies_in_range,
              'Detected_by_dEADF': ['Yes' if year in detected_years_dual else 'No' for year in scientific_anomalies_in_range],
              'Within_±2_years': ['Yes' if any(abs(year - dy) <= 2 for dy in detected_years_dual) else 'No' for year in scientific_anomalies_in_range]
          }

          # 10. Return results
          results = {
              'scientific_anomalies': scientific_anomalies_in_range,
              'detected_years': {
                  'MLNN': valid_years_aligned[np.where(predictions_mlnn_aligned == 1)[0]].tolist(),
                  'GOE':  valid_years_aligned[np.where(predictions_goe_aligned  == 1)[0]].tolist(),
                  'Dual': detected_years_dual
              },
              'performance_metrics': {
                  'MLNN': metrics_mlnn,
                  'GOE':  metrics_goe,
                  'Dual': metrics_dual
              },
              'detection_rates': {
                  'direct': direct_detection_rate,
                  'window': window_detection_rate
              },
              'comparison_table': comparison_table,
              'valid_years': valid_years_aligned,   # actual years used for evaluation
              'scientific_anomalies_full': df[df['Label_Scientific'] == 1]['Year'].tolist()
          }

          return results      
    
    
def compare_with_pettitt(dEADF_results, pettitt_result_full, dEADF_valid_years, scientific_anomalies_full):
        """
        Compare dEADF and Pettitt detection results based on the effective evaluation range of dEADF.
        
        Parameters:
        dEADF_results: result dictionary from dEADF_climate_detection
        pettitt_result_full: full-sequence Pettitt test result dictionary (contains change_point_year)
        dEADF_valid_years: array, list of years where dEADF has valid predictions
        scientific_anomalies_full: list of all scientific consensus anomaly years (global)
        
        Returns:
        comparison_summary: dictionary of comparison results
        """
        # Get scientific anomaly years within dEADF effective range
        scientific_in_range = [y for y in scientific_anomalies_full if y in dEADF_valid_years]
        
        # Pettitt detected change point year
        pettitt_year = pettitt_result_full['change_point_year']
        
        # Check if Pettitt detection falls within dEADF effective range
        pettitt_in_range = pettitt_year in dEADF_valid_years if pettitt_year is not None else False
        
        # If Pettitt change point is within range, calculate its matching within range
        if pettitt_in_range:
            # Direct match
            pettitt_direct = 1 if pettitt_year in scientific_in_range else 0
            # Window match (±2 years)
            pettitt_window = 1 if any(abs(pettitt_year - y) <= 2 for y in scientific_in_range) else 0
        else:
            pettitt_direct = 0
            pettitt_window = 0
        
        # dEADF direct and window matches (based on effective range)
        detected_years_dual = dEADF_results['detected_years']['Dual']
        directly_detected = len(set(detected_years_dual) & set(scientific_in_range))
        direct_rate = directly_detected / len(scientific_in_range) if scientific_in_range else 0
        
        window_detected = 0
        for y in scientific_in_range:
            if any(abs(y - dy) <= 2 for dy in detected_years_dual):
                window_detected += 1
        window_rate = window_detected / len(scientific_in_range) if scientific_in_range else 0
        
        # Build comparison table (only for scientific anomalies within range)
        comparison_table = []
        for year in scientific_in_range:
            row = {
                'Year': year,
                'Detected_by_PETTITT': 'Yes' if pettitt_in_range and abs(year - pettitt_year) <= 2 else 'No',
                'Detected_by_dEADF': 'Yes' if year in detected_years_dual else 'No',
                'Within_±2_years_dEADF': 'Yes' if any(abs(year - dy) <= 2 for dy in detected_years_dual) else 'No'
            }
            comparison_table.append(row)
        
        # For scientific anomalies outside the effective range, list them separately
        out_of_range_years = [y for y in scientific_anomalies_full if y not in dEADF_valid_years]
        
        # Summary comparison results
        comparison_summary = {
            'Method': ['PETTITT Test', 'dEADF Framework'],
            'Type': ['Change Point Detection', 'Point Anomaly Detection'],
            'Effective_Range': [f"{min(dEADF_valid_years)}-{max(dEADF_valid_years)}"] * 2,
            'Direct_Detection_Rate': [
                f"{pettitt_direct / len(scientific_in_range) * 100:.1f}%" if pettitt_in_range else "N/A",
                f"{direct_rate * 100:.1f}%"
            ],
            'Window_Detection_Rate': [
                f"{pettitt_window / len(scientific_in_range) * 100:.1f}%" if pettitt_in_range else "N/A",
                f"{window_rate * 100:.1f}%"
            ],
            'Recall': [
                'N/A',
                f"{dEADF_results['performance_metrics']['Dual']['recall'] * 100:.1f}%"
            ],
            'F1_Score': [
                'N/A',
                f"{dEADF_results['performance_metrics']['Dual']['f1']:.3f}"
            ]
        }
        
        # Print or return results
        print("\nComparison based on dEADF effective range:")
        print(f"  dEADF effective years: {dEADF_valid_years[0]} - {dEADF_valid_years[-1]}")
        print(f"  Scientific anomalies in range: {scientific_in_range}")
        if out_of_range_years:
            print(f"  Scientific anomalies out of range (excluded): {out_of_range_years}")
        
        return comparison_summary, comparison_table


def main():
    """Main function for training and evaluation with dynamic thresholding"""
    print("=== Dual-Engine Anomaly Detection Framework with Dynamic Thresholding ===")
    
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)

    processor = DataProcessor()
    processed_data = processor.load_and_preprocess_data()

    if processed_data is None:
        print("Data loading failed, exiting program")
        return

    X_train, y_train, X_val, y_val, X_test, y_test = processed_data

    y_train = ensure_numpy_array(y_train)
    y_val = ensure_numpy_array(y_val)
    y_test = ensure_numpy_array(y_test)

    input_size = X_train.shape[1]
    print(f"Input dimension: {input_size}")

    # ========== Train model ==========
    print("\nTraining dEADF model with dynamic thresholding...")
    os.makedirs('Climate_results', exist_ok=True)
    dEADF_model = DualEngineADF(input_size=input_size, perturbation_strength=0.1, lambda_reg=0.1, window_size=500)
    dEADF_model.fit(X_train, y_train, X_val, y_val, use_progressive_training=True)
    dEADF_model.save_model('./Climate_results/dEADF_model_Climate.pth')
    
    # ========== Climate anomaly detection ==========
    print("\n" + "="*60)
    print("CLIMATE ANOMALY DETECTION")
    print("="*60)
    
    # Run dEADF climate detection
    dEADF_results = dEADF_climate_detection(dEADF_model)
    
    if dEADF_results:
        # Print dEADF results
        print("\n" + "="*60)
        print("dEADF CLIMATE DETECTION RESULTS")
        print("="*60)
        print(f"Scientific consensus anomalies: {dEADF_results['scientific_anomalies']}")
        print(f"dEADF detected anomalies (Dual): {dEADF_results['detected_years']['Dual']}")
        
        
 
        print(f"\nPerformance Metrics (Dual Engine):")
        
        metrics = dEADF_results['performance_metrics']['Dual']
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        print(f"\nDetection Rates:")
        print(f"  Direct detection: {dEADF_results['detection_rates']['direct']*100:.1f}%")
        print(f"  Window detection: {dEADF_results['detection_rates']['window']*100:.1f}%")
   
        
        # Run PETTITT test (using previous code)
        print("\n" + "="*60)
        print("PETTITT TEST RESULTS")
        print("="*60)
        pettitt_results =  pettitt_test(temp_series, alpha=0.05)
        
        # Compare the two methods
        print("\n" + "="*60)
        print("COMPARISON: dEADF vs PETTITT")
        print("="*60)
        valid_years_aligned = dEADF_results['valid_years']
        scientific_anomalies_full = dEADF_results['scientific_anomalies_full']
        comparison_summary, comparison_table = compare_with_pettitt(
            dEADF_results, pettitt_results, valid_years_aligned, scientific_anomalies_full
        )
        
        # Print comparison table
        import pandas as pd
        comparison_df =pd.DataFrame(comparison_summary)
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))
        
        # Save comparison results
        comparison_df.to_csv('./Climate_results/dEADF_vs_PETTITT_comparison.csv', index=False)
        print("\nComparison results saved to: ./Climate_results/dEADF_vs_PETTITT_comparison.csv")
        
        # Generate summary for paper
        print("\n" + "="*60)
        print("SUMMARY FOR PAPER")
        print("="*60)
        print(f"dEADF detected {len(dEADF_results['detected_years']['Dual'])} anomalies")
        print(f"Direct detection rate: {dEADF_results['detection_rates']['direct']*100:.1f}%")
        print(f"Window detection rate: {dEADF_results['detection_rates']['window']*100:.1f}%")
        print(f"Recall: {dEADF_results['performance_metrics']['Dual']['recall']*100:.1f}%")
        print(f"F1-Score: {dEADF_results['performance_metrics']['Dual']['f1']:.3f}")
        
        
        
        
        print(f"dEADF effective years: {valid_years_aligned[0]} - {valid_years_aligned[-1]}")
        print(f"Scientific anomalies in range: {[y for y in scientific_anomalies_full if y in valid_years_aligned]}")
        print(f"Scientific anomalies out of range: {[y for y in scientific_anomalies_full if y not in valid_years_aligned]}")
        print(f"dEADF detected anomalies (Dual): {dEADF_results['detected_years']['Dual']}")
        print(f"Direct detection rate: {dEADF_results['detection_rates']['direct']*100:.1f}%")
        print(f"Window detection rate: {dEADF_results['detection_rates']['window']*100:.1f}%")
        print(f"Recall: {dEADF_results['performance_metrics']['Dual']['recall']*100:.1f}%")
        print(f"F1-Score: {dEADF_results['performance_metrics']['Dual']['f1']:.3f}")
        print(f"Pettitt change point: {pettitt_results['change_point_year']} (p={pettitt_results['p_value']:.4f})")
    
    return dEADF_model


if __name__ == "__main__": 
    model = main()