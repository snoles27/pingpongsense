#!/usr/bin/env python3
"""
Test script for matched filter signal detection.
This script demonstrates how to use the matched filter approach
for detecting signal onset times in ping pong events.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import locate as loc
import ReceiveData as rd

def test_matched_filter_basic():
    """Test basic matched filter functionality with synthetic data."""
    print("Testing basic matched filter functionality...")
    
    # Create synthetic signal with known onset
    sample_rate = 1000  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a template (what we're looking for)
    template_start = 0.1  # seconds
    template_duration = 0.05  # seconds
    template_indices = (t >= template_start) & (t <= template_start + template_duration)
    
    # Create template with characteristic shape
    template = np.zeros_like(t)
    template[template_indices] = np.sin(2 * np.pi * 50 * (t[template_indices] - template_start)) * np.exp(-20 * (t[template_indices] - template_start))
    
    # Normalize template
    template = (template - np.mean(template)) / np.std(template)
    
    # Create signal with noise
    signal_data = np.random.normal(0, 0.1, len(t))  # Noise
    signal_onset = 0.3  # seconds
    onset_idx = np.argmin(np.abs(t - signal_onset))
    
    # Add template to signal at onset
    template_length = np.sum(template_indices)
    signal_data[onset_idx:onset_idx + template_length] += template[:template_length] * 0.5
    
    # Apply matched filter detection
    detection_idx, correlation_score, confidence = loc.matched_filter_detection(
        signal_data, template, correlation_threshold=0.3
    )
    
    detection_time = t[detection_idx]
    true_onset = signal_onset
    
    print(f"True onset time: {true_onset:.3f} s")
    print(f"Detected onset time: {detection_time:.3f} s")
    print(f"Detection error: {abs(detection_time - true_onset) * 1000:.1f} ms")
    print(f"Correlation score: {correlation_score:.3f}")
    print(f"Confidence: {confidence:.2f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Signal and template
    ax1.plot(t, signal_data, label='Signal with Noise', alpha=0.7)
    ax1.plot(t, template * 0.5, label='Template (scaled)', linewidth=2, linestyle='--')
    ax1.axvline(x=true_onset, color='green', linestyle=':', label='True Onset')
    ax1.axvline(x=detection_time, color='red', linestyle=':', label='Detected Onset')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Signal and Template')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation
    correlation = signal.correlate(signal_data, template, mode='valid')
    correlation_times = t[:len(correlation)]
    ax2.plot(correlation_times, correlation, label='Correlation')
    ax2.axhline(y=0.3, color='red', linestyle='--', label='Threshold')
    ax2.axvline(x=detection_time, color='red', linestyle=':', label='Detection Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Matched Filter Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return detection_time, true_onset, correlation_score

def test_template_generation():
    """Test template generation from real data files."""
    print("\nTesting template generation from real data...")
    
    try:
        # Try to load existing templates
        templates, template_times = loc.load_templates()
        
        if all(templates.values()):
            print("Successfully loaded existing templates")
            for channel in range(3):
                print(f"  Channel {channel}: template length = {len(templates[channel])}")
        else:
            print("No existing templates found")
            
    except Exception as e:
        print(f"Error testing template generation: {e}")

if __name__ == "__main__":
    print("Matched Filter Test Script")
    print("=" * 40)
    
    # Test basic functionality
    try:
        detection_time, true_onset, correlation = test_matched_filter_basic()
        print(f"\n✓ Basic test completed successfully")
        print(f"  Detection accuracy: {abs(detection_time - true_onset) * 1000:.1f} ms")
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
    
    # Test template loading
    try:
        test_template_generation()
        print(f"✓ Template loading test completed")
    except Exception as e:
        print(f"✗ Template loading test failed: {e}")
    
    print("\nTest script completed!")
