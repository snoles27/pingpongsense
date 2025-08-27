import ReceiveData as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from scipy.optimize import root
from scipy import signal
from scipy.stats import pearsonr
import os

### SENSOR POSITIONS ### 
SENS0_X = 51.5
SENS0_Y = 8.5

SENS1_X = 5.75
SENS1_Y = 6.38

SENS2_X = 25.25
SENS2_Y = 45.75

# wave speed through table in inches/microsecond
WAVE_SPEED = 0.019  # rough calc based on minimal data 
SPEED_UNCERT = 0.02  # estimate of speed uncertainty 

MIN_PRERANGE = -5000
MAX_PRERANGE = -2000

# Matched filter parameters
TEMPLATE_LENGTH = 100  # Length of template in samples
CORRELATION_THRESHOLD = 0.6  # Minimum correlation for detection
TEMPLATE_WINDOW_START = 0  # Start of template window relative to signal onset
TEMPLATE_WINDOW_END = 500  # End of template window in microseconds


def calculate_signal_time(event: rd.Event, channel: int, base_range: list[int] = [MIN_PRERANGE, MAX_PRERANGE], 
                        trigger_multiple: int = 5, confirmation_points: int = 10, confirmation_percentage: float = 0.5) -> tuple[int, int]:
    """
    Function to determine time when signal reaches the sensor using robust thresholding.
    
    Args:
        event: Event object containing sensor data
        channel: Channel index to analyze
        base_range: Time range [start, end] in microseconds to determine baseline
        trigger_multiple: Multiple of noise level to use as threshold
        confirmation_points: Number of points to check after initial threshold crossing
        confirmation_percentage: Percentage of confirmation points that must exceed threshold (0.0 to 1.0)
    
    Returns:
        tuple(time solution, uncertainty in time)
    """
    default_uncertainty = 125  # setting the default uncertainty at the sample rate

    # Get raw time and value data 
    sensor_data = event.get_sensor_data(channel)
    times = sensor_data.time
    vals = sensor_data.values

    index_start = get_index_first_greater(times, base_range[0])
    index_end = get_index_first_greater(times, base_range[1])

    # Get average and standard deviation of base value
    base_val = np.average(vals[index_start:index_end])
    noise_level = np.std(vals[index_start:index_end])

    # Get a set of values that is centered around 0
    zero_centered_vals = vals - base_val
    threshold = noise_level * trigger_multiple

    # Find all points that exceed the threshold
    threshold_exceeded = np.abs(zero_centered_vals) > threshold
    
    # Look for the first point where enough subsequent points also exceed threshold
    for i in range(len(threshold_exceeded) - confirmation_points):
        if threshold_exceeded[i]:
            # Check if enough of the next N points also exceed threshold
            subsequent_points = threshold_exceeded[i+1:i+1+confirmation_points]
            if np.sum(subsequent_points) >= confirmation_percentage * confirmation_points:
                return (times[i], default_uncertainty)
    
    # If no robust trigger found, fall back to original method
    index_event = get_index_first_greater(np.abs(zero_centered_vals), threshold)
    return (times[index_event], default_uncertainty)


def get_index_first_greater(numbers: list[int], thresh: int) -> int: 
    """
    Returns the index of the first instance in numbers to be greater than thresh
    """
    try: 
        index = next(x for x, val in enumerate(numbers) if val >= thresh)
    except StopIteration:
        return -1 

    return index


def zero_at_event(x: np.ndarray, sensor_loc: np.ndarray, dt: np.ndarray, speed: float) -> np.ndarray:
    """
    Multivariate function that should equal zero when the equation is solved
    x: [x_e, y_e]. 2x1 np array encoding event position
    sensor_loc: 3x2 np array with each sensor location 
    dt: 2x1 np array [dt10, dt20]
    speed: speed of sound in table with units matching x, sensor_loc and dt
    """
    rho_0 = distance(x, sensor_loc[0, :])
    rho_1 = distance(x, sensor_loc[1, :])
    rho_2 = distance(x, sensor_loc[2, :])

    return np.array([
        rho_1 - rho_0 - speed * dt[0],
        rho_2 - rho_0 - speed * dt[1]
    ])


def jac_zero_at_event(x: np.ndarray, sensor_loc: np.ndarray, dt: np.ndarray, speed: float) -> np.ndarray:
    """
    Returns jacobian of zero_at_event evaluated at x
    """
    rho_0 = distance(x, sensor_loc[0, :])
    rho_1 = distance(x, sensor_loc[1, :])
    rho_2 = distance(x, sensor_loc[2, :])

    return np.array([
        [(x[0] - sensor_loc[1, 0])/rho_1 - (x[0] - sensor_loc[0, 0])/rho_0, 
         (x[1] - sensor_loc[1, 1])/rho_1 - (x[1] - sensor_loc[0, 1])/rho_0],
        [(x[0] - sensor_loc[2, 0])/rho_2 - (x[0] - sensor_loc[0, 0])/rho_0, 
         (x[1] - sensor_loc[2, 1])/rho_2 - (x[1] - sensor_loc[0, 1])/rho_0]
    ])


def solve_location(dt: np.ndarray, sensor_loc: np.ndarray, speed: float, guess: np.ndarray = np.array([10, 10])) -> np.ndarray:
    """
    Invoke Newton's method to solve for event location.
    
    Args:
        dt: 2x1 array of time differences [dt10, dt20]
        sensor_loc: 3x2 array with sensor positions
        speed: wave speed in table
        guess: initial guess for event position [x, y]
    
    Returns:
        scipy.optimize.OptimizeResult object with solution
    """
    args = (sensor_loc, dt, speed)

    return root(
        fun=zero_at_event,
        jac=jac_zero_at_event,
        args=args,
        x0=guess,
        method='lm'
    )


def distance(xf: np.ndarray, x0: np.ndarray) -> float:
    """
    Returns distance between two points, xf and x0
    """
    return np.linalg.norm(xf - x0)


def plot_event_with_signal_times(
    event_data: rd.Event, 
    t0: float, 
    ut0: float, 
    t1: float, 
    ut1: float, 
    t2: float, 
    ut2: float,
    ax: matplotlib.axes.Axes = None,
    time_padding_frac: float = 0.2,
    show_plot: bool = True) -> matplotlib.axes.Axes:
    """
    Plot event data with calculated signal times overlaid as semi-transparent rectangles.
    The rectangle colors match the default matplotlib color cycle used in plottingTools.plotEvent.
    
    Args:
        event_data: Event object containing sensor data
        t0, ut0: Time and uncertainty for channel 0
        t1, ut1: Time and uncertainty for channel 1
        t2, ut2: Time and uncertainty for channel 2
        ax: matplotlib axes to plot on (if None, creates new figure)
        time_padding_frac: fraction of time range to add as padding
        show_plot: whether to show the plot immediately
    
    Returns:
        matplotlib axes object
    """
    # Default matplotlib color cycle (first 3 colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots()
        event_data.plot(ax)
    else:
        # Plot on existing axes
        event_data.plot(ax)
    
    # Get the y-axis limits to determine rectangle height
    y_min, y_max = ax.get_ylim()
    rect_height = y_max - y_min
    
    # Create semi-transparent rectangles centered on tx with width 2 * utx
    # Use the same colors as the plot lines
    rect0 = patches.Rectangle((t0 - ut0, y_min), 2 * ut0, rect_height, 
                             facecolor=colors[0], alpha=0.3, label="t0 ± ut0")
    rect1 = patches.Rectangle((t1 - ut1, y_min), 2 * ut1, rect_height, 
                             facecolor=colors[1], alpha=0.3, label="t1 ± ut1")
    rect2 = patches.Rectangle((t2 - ut2, y_min), 2 * ut2, rect_height, 
                             facecolor=colors[2], alpha=0.3, label="t2 ± ut2")
    
    # Add rectangles to the plot
    ax.add_patch(rect0)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Focus the plot view on the beginning of the event where the rectangles are
    # Find the earliest and latest times among the signal times and their uncertainties
    earliest_time = min(t0 - ut0, t1 - ut1, t2 - ut2)
    latest_time = max(t0 + ut0, t1 + ut1, t2 + ut2)
    
    # Add some padding around the rectangles for better visibility
    time_padding = (latest_time - earliest_time) * time_padding_frac
    x_min = earliest_time - time_padding
    x_max = latest_time + time_padding
    
    # Set the x-axis limits to focus on the signal detection region
    ax.set_xlim(x_min, x_max)
    
    ax.legend()
    
    if show_plot:
        plt.show()
    
    return ax


def plot_location_solution(sensor_loc: np.ndarray, x_solution: np.ndarray, x_cov: np.ndarray, 
                          ax: matplotlib.axes.Axes = None, data_label: str = None, 
                          plot_sensors: bool = True, color: str = None) -> matplotlib.axes.Axes:
    """
    Plot sensor positions, solution location, and covariance ellipse.
    
    Args:
        sensor_loc: 3x2 array with sensor positions [x, y]
        x_solution: 2x1 array with solution position [x, y]
        x_cov: 2x2 covariance matrix for the solution
        ax: matplotlib axes to plot on (if None, creates new figure)
        data_label: label for this solution in the legend
        plot_sensors: whether to plot sensor positions (only plot once)
        color: color for this solution (if None, uses default cycle)
    
    Returns:
        matplotlib axes object
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_sensors = True  # Always plot sensors for new figure
    
    # Plot sensor positions only once
    if plot_sensors:
        ax.scatter(sensor_loc[:, 0], sensor_loc[:, 1], c=['red', 'green', 'blue'], 
                   s=100, marker='s', label='Sensors', edgecolors='black', linewidth=2)
        
        # Add sensor labels
        for i, (x, y) in enumerate(sensor_loc):
            ax.annotate(f'S{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=12, fontweight='bold')
    
    # Use provided color or get from color cycle
    if color is None:
        color = plt.cm.tab10(len(ax.get_legend_handles_labels()[0]) % 10)
    
    # Plot solution location
    solution_label = f'Solution {data_label}' if data_label else 'Solution'
    ax.scatter(x_solution[0], x_solution[1], c=color, s=150, marker='*', 
               label=solution_label, edgecolors='black', linewidth=2)
    
    # Plot covariance ellipse
    # Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(x_cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Calculate ellipse parameters
    # Use 95% confidence interval (2-sigma)
    confidence_level = 2.0
    major_axis = confidence_level * np.sqrt(eigenvals[0])
    minor_axis = confidence_level * np.sqrt(eigenvals[1])
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    # Create ellipse
    from matplotlib.patches import Ellipse
    ellipse_label = f'95% Confidence {data_label}' if data_label else '95% Confidence'
    ellipse = Ellipse(xy=x_solution, width=2*major_axis, height=2*minor_axis, 
                      angle=np.degrees(angle), facecolor=color, alpha=0.3, 
                      edgecolor=color, linewidth=2, label=ellipse_label)
    
    ax.add_patch(ellipse)
    
    # Set equal aspect ratio and labels (only for new figures)
    if plot_sensors:
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (inches)')
        ax.set_ylabel('Y Position (inches)')
        ax.set_title('Ball Impact Location with Uncertainty')
        ax.grid(True, alpha=0.3)
    
    ax.legend()
    
    # Update axis limits to include new data
    all_x = np.concatenate([sensor_loc[:, 0], [x_solution[0]]])
    all_y = np.concatenate([sensor_loc[:, 1], [x_solution[1]]])
    
    # Get current limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Update limits to include new data
    new_x_min = min(x_min, np.min(all_x))
    new_x_max = max(x_max, np.max(all_x))
    new_y_min = min(y_min, np.min(all_y))
    new_y_max = max(y_max, np.max(all_y))
    
    # Add padding
    x_padding = (new_x_max - new_x_min) * 0.1
    y_padding = (new_y_max - new_y_min) * 0.1
    
    ax.set_xlim(new_x_min - x_padding, new_x_max + x_padding)
    ax.set_ylim(new_y_min - y_padding, new_y_max + y_padding)
    
    return ax


def generate_signal_template(events: list[rd.Event], channel: int, template_length: int = TEMPLATE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple parameterized signal template: sinusoid with exponential decay.
    
    Args:
        events: List of Event objects (not used for this simple template)
        channel: Channel index (not used for this simple template)
        template_length: Number of samples in the template
    
    Returns:
        tuple of (template_values, template_times) - normalized template and corresponding times
    """
    # Create a simple sinusoidal template with exponential decay
    # This is more robust than learning from noisy data
    
    # Template parameters
    frequency = 0.02  # Frequency in cycles per sample (adjustable)
    decay_rate = 0.015  # Exponential decay rate (adjustable)
    num_periods = 1.5  # Number of periods in the template
    
    # Create time array for template
    template_times = np.linspace(0, num_periods / frequency, template_length)
    
    # Generate template: sinusoid * exponential decay
    template_values = np.sin(2 * np.pi * frequency * template_times * template_length / num_periods)
    template_values *= np.exp(-decay_rate * template_times * template_length / num_periods)
    
    # Normalize the template
    template_values = (template_values - np.mean(template_values)) / np.std(template_values)
    
    return template_values, template_times


def create_scalable_template(base_frequency: float = 0.02, base_amplitude: float = 1.0, 
                           frequency_scale: float = 1.0, amplitude_scale: float = 1.0,
                           template_length: int = TEMPLATE_LENGTH, num_periods: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a scalable sinusoidal template with exponential decay.
    
    Args:
        base_frequency: Base frequency in cycles per sample
        base_amplitude: Base amplitude
        frequency_scale: Frequency scaling factor
        amplitude_scale: Amplitude scaling factor
        template_length: Number of samples in the template
        num_periods: Number of periods in the template
    
    Returns:
        tuple of (template_values, template_times) - scaled template and corresponding times
    """
    # Apply scaling
    frequency = base_frequency * frequency_scale
    amplitude = base_amplitude * amplitude_scale
    
    # Create time array for template
    template_times = np.linspace(0, num_periods / frequency, template_length)
    
    # Generate template: scaled sinusoid * exponential decay
    template_values = amplitude * np.sin(2 * np.pi * frequency * template_times * template_length / num_periods)
    template_values *= np.exp(-0.015 * template_times * template_length / num_periods)
    
    # Normalize the template
    template_values = (template_values - np.mean(template_values)) / np.std(template_values)
    
    return template_values, template_times


def generate_adaptive_templates(events: list[rd.Event], channel: int, 
                              template_length: int = TEMPLATE_LENGTH) -> tuple[dict, dict]:
    """
    Generate multiple templates at different scales for robust detection.
    
    Args:
        events: List of Event objects to analyze for scaling parameters
        channel: Channel index to analyze
        template_length: Number of samples in the template
    
    Returns:
        tuple of (templates, template_times) - dictionaries of templates at different scales
    """
    if not events:
        # Fall back to default template if no events available
        default_template, default_times = create_scalable_template()
        return {0: default_template}, {0: default_times}
    
    # Analyze events to determine appropriate scaling
    frequencies = []
    amplitudes = []
    
    for event in events[:3]:  # Use first 3 events for analysis
        try:
            sensor_data = event.get_sensor_data(channel)
            values = sensor_data.values
            
            # Simple frequency estimation using zero crossings
            zero_crossings = np.where(np.diff(np.signbit(values)))[0]
            if len(zero_crossings) > 1:
                # Estimate frequency from zero crossings
                time_between_crossings = np.diff(sensor_data.time[zero_crossings])
                avg_period = np.mean(time_between_crossings)
                if avg_period > 0:
                    freq = 1.0 / avg_period
                    frequencies.append(freq)
            
            # Estimate amplitude from peak-to-peak
            peak_to_peak = np.max(values) - np.min(values)
            amplitudes.append(peak_to_peak)
            
        except Exception as e:
            continue
    
    # Calculate scaling factors
    if frequencies:
        avg_freq = np.mean(frequencies)
        freq_scale = avg_freq / 0.02  # Normalize to base frequency
    else:
        freq_scale = 1.0
    
    if amplitudes:
        avg_amplitude = np.mean(amplitudes)
        amp_scale = avg_amplitude / 1000.0  # Normalize to base amplitude
    else:
        amp_scale = 1.0
    
    # Create templates at different scales
    templates = {}
    template_times = {}
    
    # Base template
    base_template, base_times = create_scalable_template(
        frequency_scale=freq_scale, amplitude_scale=amp_scale
    )
    templates[0] = base_template
    template_times[0] = base_times
    
    # Higher frequency template
    high_freq_template, high_freq_times = create_scalable_template(
        frequency_scale=freq_scale * 1.5, amplitude_scale=amp_scale
    )
    templates[1] = high_freq_template
    template_times[1] = high_freq_times
    
    # Lower frequency template
    low_freq_template, low_freq_times = create_scalable_template(
        frequency_scale=freq_scale * 0.7, amplitude_scale=amp_scale
    )
    templates[2] = low_freq_template
    template_times[2] = low_freq_times
    
    return templates, template_times


def matched_filter_detection(signal_data: np.ndarray, template: np.ndarray, 
                           correlation_threshold: float = CORRELATION_THRESHOLD,
                           window_size: int = 50) -> tuple[int, float, float]:
    """
    Detect signal onset using matched filtering with correlation analysis.
    
    Args:
        signal_data: Raw signal values to analyze
        template: Normalized template to match against
        correlation_threshold: Minimum correlation coefficient for detection
        window_size: Size of sliding window for correlation calculation
    
    Returns:
        tuple of (detection_index, correlation_score, confidence)
    """
    if len(signal_data) < len(template):
        raise ValueError("Signal data must be longer than template")
    
    # Normalize the signal data
    signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    # Use scipy's correlate function for efficient correlation
    correlation = signal.correlate(signal_normalized, template, mode='valid')
    
    # Normalize correlation values
    correlation = correlation / (len(template) * np.std(signal_normalized) * np.std(template))
    
    # Find peaks in correlation that exceed threshold
    peaks, _ = signal.find_peaks(correlation, height=correlation_threshold, distance=window_size)
    
    if len(peaks) == 0:
        # No strong correlation found, return the maximum correlation point
        max_idx = np.argmax(correlation)
        max_corr = correlation[max_idx]
        confidence = max_corr / correlation_threshold if max_corr > 0 else 0
        return max_idx, max_corr, confidence
    
    # Find the earliest strong correlation (first peak)
    detection_idx = peaks[0]
    correlation_score = correlation[detection_idx]
    
    # Calculate confidence based on how much the correlation exceeds threshold
    confidence = correlation_score / correlation_threshold
    
    return detection_idx, correlation_score, confidence


def calculate_signal_time_matched_filter(event: rd.Event, channel: int, 
                                       template: np.ndarray = None,
                                       base_range: list[int] = [MIN_PRERANGE, MAX_PRERANGE],
                                       correlation_threshold: float = CORRELATION_THRESHOLD) -> tuple[int, int]:
    """
    Determine signal onset time using matched filtering approach.
    
    Args:
        event: Event object containing sensor data
        channel: Channel index to analyze
        template: Pre-computed template (if None, will use default threshold method)
        base_range: Time range for baseline calculation
        correlation_threshold: Minimum correlation for detection
    
    Returns:
        tuple(time solution, uncertainty in time)
    """
    default_uncertainty = 125
    
    # Get sensor data
    sensor_data = event.get_sensor_data(channel)
    times = sensor_data.time
    values = sensor_data.values
    
    if template is None:
        # Fall back to threshold method if no template provided
        return calculate_signal_time(event, channel, base_range)
    
    try:
        # Apply matched filter detection
        detection_idx, correlation_score, confidence = matched_filter_detection(
            values, template, correlation_threshold
        )
        
        # Convert index to time
        detection_time = times[detection_idx]
        
        # Adjust uncertainty based on correlation confidence
        # Higher confidence = lower uncertainty
        adjusted_uncertainty = default_uncertainty / max(confidence, 0.1)
        
        print(f"    Matched filter: correlation={correlation_score:.3f}, confidence={confidence:.2f}")
        
        return detection_time, int(adjusted_uncertainty)
        
    except Exception as e:
        print(f"    Matched filter failed: {e}, falling back to threshold method")
        return calculate_signal_time(event, channel, base_range)


def plot_template_comparison(event: rd.Event, channel: int, template: np.ndarray, 
                           template_times: np.ndarray, detection_time: int,
                           ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
    """
    Plot the signal template comparison with the detected signal.
    
    Args:
        event: Event object containing sensor data
        channel: Channel index being analyzed
        template: Normalized template values
        template_times: Template time array
        detection_time: Detected onset time
        ax: matplotlib axes to plot on (if None, creates new figure)
    
    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get sensor data
    sensor_data = event.get_sensor_data(channel)
    times = sensor_data.time
    values = sensor_data.values
    
    # Plot the raw signal
    ax.plot(times, values, label=f'Channel {channel} Raw Signal', color='blue', alpha=0.7)
    
    # Plot the template aligned at detection time
    template_aligned_times = detection_time + template_times
    # Scale template to match signal amplitude for visualization
    template_scaled = template * np.std(values) + np.mean(values)
    ax.plot(template_aligned_times, template_scaled, label='Signal Template', 
            color='red', linewidth=2, linestyle='--')
    
    # Mark the detection point
    ax.axvline(x=detection_time, color='green', linestyle=':', linewidth=2, 
               label=f'Detection Time: {detection_time:.1f}')
    
    # Focus on the detection region
    detection_idx = np.argmin(np.abs(times - detection_time))
    window_size = 200
    start_idx = max(0, detection_idx - window_size // 2)
    end_idx = min(len(times), detection_idx + window_size // 2)
    
    ax.set_xlim(times[start_idx], times[end_idx])
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.set_title(f'Matched Filter Detection - Channel {channel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def save_templates(templates: dict, template_times: dict, filename: str = "signal_templates.npz"):
    """
    Save generated templates to a file for reuse.
    
    Args:
        templates: Dictionary of templates for each channel
        template_times: Dictionary of template times for each channel
        filename: Name of the file to save templates to
    """
    try:
        np.savez(filename, 
                 template_0=templates[0], template_1=templates[1], template_2=templates[2],
                 times_0=template_times[0], times_1=template_times[1], times_2=template_times[2])
        print(f"Templates saved to {filename}")
    except Exception as e:
        print(f"Error saving templates: {e}")


def load_templates(filename: str = "signal_templates.npz") -> tuple[dict, dict]:
    """
    Load previously saved templates from a file.
    
    Args:
        filename: Name of the file to load templates from
    
    Returns:
        tuple of (templates, template_times) dictionaries
    """
    try:
        data = np.load(filename)
        templates = {
            0: data['template_0'],
            1: data['template_1'], 
            2: data['template_2']
        }
        template_times = {
            0: data['times_0'],
            1: data['times_1'],
            2: data['times_2']
        }
        print(f"Templates loaded from {filename}")
        return templates, template_times
    except Exception as e:
        print(f"Error loading templates: {e}")
        return {0: None, 1: None, 2: None}, {0: None, 1: None, 2: None}


if __name__ == "__main__":
    import os
    
    trigger_multiple = 5
    sensor_loc_list = np.array([[SENS0_X, SENS0_Y],
                               [SENS1_X, SENS1_Y],
                               [SENS2_X, SENS2_Y]])

    # Directory containing the locating data files
    data_directory = "Data/RawEventData/LocatingData/"
    
    # Get all .txt files in the directory
    data_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
    data_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(data_files)} data files to process:")
    for file in data_files:
        print(f"  - {file}")
    
    # Try to load existing templates first, generate new ones if needed
    print(f"\nLoading or generating signal templates for matched filtering...")
    
    # Try to load existing templates
    templates, template_times = load_templates()
    
    if all(templates.values()):  # All templates loaded successfully
        print("  Successfully loaded existing templates")
    else:
        print("  No existing templates found, generating new ones...")
        try:
            # Read first few events to generate templates
            template_events = []
            for i, file_name in enumerate(data_files[:5]):  # Use first 5 files for templates
                try:
                    full_path = os.path.join(data_directory, file_name)
                    event_data = rd.event_file_read(full_path=full_path)
                    template_events.append(event_data)
                    print(f"  Added {file_name} to template generation")
                except Exception as e:
                    print(f"  Warning: Could not read {file_name} for template: {e}")
                    continue
        
            if template_events:
                # Generate templates for each channel
                templates = {}
                template_times = {}
                for channel in range(3):
                    try:
                        template_values, times = generate_signal_template(template_events, channel)
                        templates[channel] = template_values
                        template_times[channel] = times
                        print(f"  Generated template for channel {channel} (length: {len(template_values)})")
                    except Exception as e:
                        print(f"  Warning: Could not generate template for channel {channel}: {e}")
                        templates[channel] = None
                        template_times[channel] = None
                
                # Save templates for future use
                try:
                    save_templates(templates, template_times)
                except Exception as e:
                    print(f"  Warning: Could not save templates: {e}")
            else:
                print("  Warning: No events available for template generation")
                templates = {0: None, 1: None, 2: None}
                template_times = {0: None, 1: None, 2: None}
            
        except Exception as e:
            print(f"  Error during template generation: {e}")
            templates = {0: None, 1: None, 2: None}
            template_times = {0: None, 1: None, 2: None}
    
    # Process each file
    for file_name in data_files:
        print(f"\nProcessing: {file_name}")
        
        # Construct full file path
        full_path = os.path.join(data_directory, file_name)
        
        try:
            # Read event data
            event_data = rd.event_file_read(full_path=full_path)
            
            # Calculate signal times using matched filtering if templates available
            print(f"  Detecting signal times...")
            
            if templates[0] is not None:
                t0, ut0 = calculate_signal_time_matched_filter(event_data, channel=0, template=templates[0])
            else:
                t0, ut0 = calculate_signal_time(event_data, channel=0, trigger_multiple=trigger_multiple)
                print(f"    Using threshold method for channel 0")
            
            if templates[1] is not None:
                t1, ut1 = calculate_signal_time_matched_filter(event_data, channel=1, template=templates[1])
            else:
                t1, ut1 = calculate_signal_time(event_data, channel=1, trigger_multiple=trigger_multiple)
                print(f"    Using threshold method for channel 1")
            
            if templates[2] is not None:
                t2, ut2 = calculate_signal_time_matched_filter(event_data, channel=2, template=templates[2])
            else:
                t2, ut2 = calculate_signal_time(event_data, channel=2, trigger_multiple=trigger_multiple)
                print(f"    Using threshold method for channel 2")
            
            # Print signal times
            print(f"  Signal times: t0={t0:.1f}±{ut0}, t1={t1:.1f}±{ut1}, t2={t2:.1f}±{ut2}")
            
            # Calculate time differences for location solving
            t10 = t1 - t0
            t20 = t2 - t0
            print(f"  Time differences: t10={t10:.1f}, t20={t20:.1f}")
            
            # Create figure with three subplots: raw data, template comparison, and location
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(f'Event Analysis: {file_name}', fontsize=14, fontweight='bold')
            
            # Plot event with signal times on left subplot
            plot_event_with_signal_times(event_data, t0, ut0, t1, ut1, t2, ut2, 
                                       ax=ax1, time_padding_frac=0.5, show_plot=False)
            ax1.set_title('Raw Data with Signal Detection Times')
            
            # Plot template comparison for middle subplot (show channel 0 as example)
            if templates[0] is not None:
                plot_template_comparison(event_data, 0, templates[0], template_times[0], t0, ax=ax2)
                ax2.set_title('Matched Filter Detection (Channel 0)')
            else:
                ax2.text(0.5, 0.5, 'No Template\nAvailable', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax2.set_title('Template Comparison (No Template)')
            
            # Solve for location and plot on right subplot
            try:
                soln = solve_location(np.array([t10, t20]), sensor_loc=sensor_loc_list, speed=WAVE_SPEED)
                print(f"  Location solution: x={soln.x[0]:.2f}, y={soln.x[1]:.2f}")
                
                plot_location_solution(sensor_loc_list, soln.x, soln.cov_x, 
                                     ax=ax3, data_label=file_name.replace('.txt', ''), 
                                     plot_sensors=True, color=None)
                ax3.set_title('Ball Impact Location')
                
            except Exception as loc_error:
                print(f"  Error solving location: {loc_error}")
                ax3.text(0.5, 0.5, 'Location\nSolution\nFailed', 
                        ha='center', va='center', transform=ax3.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax3.set_title('Ball Impact Location (Failed)')
            
            plt.tight_layout()
            plt.show()
            
            
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            continue
    
    print(f"\nCompleted processing {len(data_files)} files.")



    # # sensor_loc_list = np.array([[SENS0_X, SENS0_Y],
    # #                             [SENS1_X, SENS1_Y],
    # #                             [SENS2_X, SENS2_Y]])

    # sensor_loc_list = np.array([[0, 0],
    #                             [3, 0],
    #                             [-1, 0]])
    
    # event = np.array([1, 0])
    # dt = np.array([1, 1])
    # c = 1
    # print(solve_location(dt, sensor_loc_list, c, guess=np.array([0, 0])))


    # thresh_mult = 8
    # data = rd.event_file_read("Data/RawEventData/LocatingData/6in_(9.0,6.4)_1.txt")
    # t1, ut = calculate_signal_time(data, 1, trigger_multiple=thresh_mult)
    # t0, ut = calculate_signal_time(data, 0, trigger_multiple=thresh_mult)

    # print(t1 - t0)
    