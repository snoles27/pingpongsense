import ReceiveData as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from scipy.optimize import root

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
    
    # Process each file
    for file_name in data_files:
        print(f"\nProcessing: {file_name}")
        
        # Construct full file path
        full_path = os.path.join(data_directory, file_name)
        
        try:
            # Read event data
            event_data = rd.event_file_read(full_path=full_path)
            
            # Calculate signal times for all channels
            t0, ut0 = calculate_signal_time(event_data, channel=0, trigger_multiple=trigger_multiple)
            t1, ut1 = calculate_signal_time(event_data, channel=1, trigger_multiple=trigger_multiple)
            t2, ut2 = calculate_signal_time(event_data, channel=2, trigger_multiple=trigger_multiple)
            
            # Print signal times
            print(f"  Signal times: t0={t0:.1f}±{ut0}, t1={t1:.1f}±{ut1}, t2={t2:.1f}±{ut2}")
            
            # Calculate time differences for location solving
            t10 = t1 - t0
            t20 = t2 - t0
            print(f"  Time differences: t10={t10:.1f}, t20={t20:.1f}")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Event Analysis: {file_name}', fontsize=14, fontweight='bold')
            
            # Plot event with signal times on left subplot
            plot_event_with_signal_times(event_data, t0, ut0, t1, ut1, t2, ut2, 
                                       ax=ax1, time_padding_frac=0.5, show_plot=False)
            ax1.set_title('Raw Data with Signal Detection Times')
            
            # Solve for location and plot on right subplot
            try:
                soln = solve_location(np.array([t10, t20]), sensor_loc=sensor_loc_list, speed=WAVE_SPEED)
                print(f"  Location solution: x={soln.x[0]:.2f}, y={soln.x[1]:.2f}")
                
                plot_location_solution(sensor_loc_list, soln.x, soln.cov_x, 
                                     ax=ax2, data_label=file_name.replace('.txt', ''), 
                                     plot_sensors=True, color=None)
                ax2.set_title('Ball Impact Location')
                
            except Exception as loc_error:
                print(f"  Error solving location: {loc_error}")
                ax2.text(0.5, 0.5, 'Location\nSolution\nFailed', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax2.set_title('Ball Impact Location (Failed)')
            
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
    