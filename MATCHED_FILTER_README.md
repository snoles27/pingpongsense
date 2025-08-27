# Matched Filter Signal Detection for Ping Pong Events

This implementation adds robust signal detection using matched filtering to the ping pong event analysis system.

## Overview

The matched filter approach provides several advantages over simple thresholding:

- **Robust Detection**: Less sensitive to noise and baseline variations
- **Template Learning**: Automatically learns signal characteristics from your data
- **Confidence Scoring**: Provides confidence measures for detection quality
- **Adaptive Uncertainty**: Adjusts timing uncertainty based on detection confidence

## How It Works

### 1. Template Generation
The system automatically generates signal templates by:
- Reading multiple event files
- Using existing threshold detection to find signal onset points
- Extracting signal segments around onset times
- Averaging multiple segments to create robust templates
- Normalizing templates for consistent correlation analysis

### 2. Matched Filter Detection
For each new event:
- Correlates the raw signal with the learned template
- Finds peaks in correlation that exceed a threshold
- Selects the earliest strong correlation as the detection point
- Calculates confidence based on correlation strength
- Adjusts timing uncertainty based on confidence

### 3. Fallback Mechanism
If matched filtering fails or no templates are available:
- Automatically falls back to the original threshold method
- Ensures system reliability even without templates

## Key Functions

### `generate_signal_template(events, channel, template_length)`
Generates a normalized template from multiple events for a specific channel.

**Parameters:**
- `events`: List of Event objects to use for template generation
- `channel`: Channel index (0, 1, or 2)
- `template_length`: Number of samples in the template

**Returns:**
- `template_values`: Normalized template array
- `template_times`: Corresponding time array

### `matched_filter_detection(signal_data, template, correlation_threshold, window_size)`
Detects signal onset using correlation analysis.

**Parameters:**
- `signal_data`: Raw signal values to analyze
- `template`: Normalized template to match against
- `correlation_threshold`: Minimum correlation for detection (default: 0.6)
- `window_size`: Minimum distance between detection peaks

**Returns:**
- `detection_index`: Index of detected onset
- `correlation_score`: Correlation coefficient at detection
- `confidence`: Confidence measure relative to threshold

### `calculate_signal_time_matched_filter(event, channel, template, base_range, correlation_threshold)`
Main function for determining signal onset time using matched filtering.

**Parameters:**
- `event`: Event object containing sensor data
- `channel`: Channel index to analyze
- `template`: Pre-computed template (None for fallback to threshold method)
- `base_range`: Time range for baseline calculation
- `correlation_threshold`: Minimum correlation for detection

**Returns:**
- `detection_time`: Detected onset time in microseconds
- `uncertainty`: Timing uncertainty in microseconds

## Configuration Parameters

```python
# Matched filter parameters (in locate.py)
TEMPLATE_LENGTH = 100          # Length of template in samples
CORRELATION_THRESHOLD = 0.6    # Minimum correlation for detection
TEMPLATE_WINDOW_START = 0      # Start of template window (μs)
TEMPLATE_WINDOW_END = 500      # End of template window (μs)
```

## Usage Examples

### Basic Usage
```python
import locate as loc

# Generate templates from existing events
events = [event1, event2, event3]  # Your event objects
template, times = loc.generate_signal_template(events, channel=0)

# Detect signal onset in new event
detection_time, uncertainty = loc.calculate_signal_time_matched_filter(
    new_event, channel=0, template=template
)
```

### Template Management
```python
# Save templates for future use
loc.save_templates(templates, template_times, "my_templates.npz")

# Load previously saved templates
templates, template_times = loc.load_templates("my_templates.npz")
```

### Integration with Main Analysis
The main analysis script automatically:
1. Loads existing templates if available
2. Generates new templates if needed
3. Uses matched filtering for signal detection
4. Falls back to threshold method if necessary
5. Provides comprehensive visualization

## Visualization

The enhanced analysis provides three subplots:

1. **Raw Data with Signal Times**: Shows detected onset times overlaid on raw signals
2. **Template Comparison**: Displays template alignment with detected signal
3. **Location Solution**: Shows ball impact location with uncertainty ellipse

## Performance Benefits

### Detection Accuracy
- **Threshold Method**: ~±125 μs uncertainty (fixed)
- **Matched Filter**: ~±(125/confidence) μs uncertainty (adaptive)

### Robustness
- **Threshold Method**: Sensitive to noise and baseline drift
- **Matched Filter**: Robust against noise, adapts to signal characteristics

### Confidence Scoring
- **Threshold Method**: Binary detection (detected/not detected)
- **Matched Filter**: Continuous confidence measure (0.0 to ∞)

## Troubleshooting

### No Templates Generated
- Ensure you have at least 2-3 valid event files
- Check that events contain sufficient data around onset times
- Verify file paths and data format

### Poor Detection Quality
- Adjust `CORRELATION_THRESHOLD` (lower = more sensitive)
- Increase `TEMPLATE_LENGTH` for longer signal patterns
- Use more events for template generation

### Performance Issues
- Reduce `TEMPLATE_LENGTH` for faster processing
- Use fewer events for template generation
- Consider saving/loading templates to avoid regeneration

## Advanced Customization

### Custom Template Generation
```python
def custom_template_generator(events, channel):
    """Custom template generation logic."""
    # Your custom implementation here
    pass

# Override the default function
loc.generate_signal_template = custom_template_generator
```

### Adaptive Thresholds
```python
def adaptive_correlation_threshold(signal_quality):
    """Adjust correlation threshold based on signal quality."""
    base_threshold = 0.6
    return base_threshold * (1.0 - signal_quality * 0.3)
```

### Multi-Scale Detection
```python
def multi_scale_detection(signal_data, templates):
    """Detect signals at multiple scales."""
    detections = []
    for scale in [0.5, 1.0, 2.0]:
        scaled_template = signal.resample(templates[0], int(len(templates[0]) * scale))
        detection = loc.matched_filter_detection(signal_data, scaled_template)
        detections.append(detection)
    return detections
```

## Testing

Run the test script to verify functionality:

```bash
python test_matched_filter.py
```

This will:
- Test basic matched filter functionality with synthetic data
- Verify template loading/saving
- Demonstrate correlation analysis
- Show detection accuracy

## Future Enhancements

Potential improvements include:
- **Multi-template Matching**: Use multiple templates per channel
- **Adaptive Templates**: Update templates based on new data
- **Frequency Domain Analysis**: Combine with FFT-based detection
- **Machine Learning Integration**: Use ML to optimize template selection
- **Real-time Processing**: Optimize for live event detection

## References

- **Matched Filter Theory**: Optimal detection of known signals in noise
- **Correlation Analysis**: Statistical measure of signal similarity
- **Template Matching**: Pattern recognition in time series data
- **Signal Processing**: Digital signal analysis techniques
