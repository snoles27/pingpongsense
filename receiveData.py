import serial
from inputimeout import inputimeout 
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import matplotlib.axes
import matplotlib.figure

SERIALPORT3 = "/dev/tty.usbmodem14301"
SERIALPORT1 = "/dev/tty.usbmodem14201"
SERIALPORT = "/dev/tty.usbmodem1101"
BAUDRATE = 57600  # must match that set in eventRecordandOutput
READ_ATTEMPT_TIMEOUT = 2.0

LINE_START = 2
UUID_START = 14
LINE_END = -5 
MIN_LEN = 4
EVENT_END_MARKER = "b'COMPLETE\\r\\n'"

DATA_A0_INDEX = 2
TIME_A0_INDEX = 1
DATA_A1_INDEX = 5
TIME_A1_INDEX = 4
DATA_A2_INDEX = 8
TIME_A2_INDEX = 7

UUID_ABBREV = 8

NO_LABEL_STR = "NO LABEL"

# file info
HEADER_LINES_NUM = 2
UUID_LINE = 0
LABEL_LINE = 1
CHANNEL_INDEX = 0
TIME_INDEX = 1
VALUE_INDEX = 2


class SensorData:
    """
    Class to store sensor data with time and value arrays.
    time: numpy array of timestamps (microseconds)
    values: numpy array of sensor readings (arbitrary units)
    """
    def __init__(self, time: np.ndarray, values: np.ndarray):
        assert len(time) == len(values), f"Time and values arrays must have same length. Got {len(time)} and {len(values)}"
        self.time = time.copy()
        self.values = values.copy()

    def __str__(self):
        return f"SensorData with {len(self.time)} points"

    def get_sample_rate(self) -> tuple[float, float]:
        """
        Returns: (float, float) average sample rate and standard deviation of sample rate
        """
        time_deltas = np.diff(self.time)
        avg_sample_spacing = np.mean(time_deltas)
        std_sample_spacing = np.std(time_deltas)
        return avg_sample_spacing, std_sample_spacing

    def plot(self, ax: matplotlib.axes.Axes, label: str = None, **kwargs) -> None:
        """
        Plot this sensor's data on the given axes.
        
        Args:
            ax: matplotlib axes to plot on
            label: label for the plot line
            **kwargs: additional plotting arguments passed to ax.plot()
        """
        if label is None:
            label = f"Sensor Data"
        
        ax.plot(self.time, self.values, label=label, **kwargs)

    def plot_with_derivative(self, ax: matplotlib.axes.Axes, label: str = None, show_plot: bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Plot this sensor's data and its derivative with twin axes.
        
        Args:
            ax: matplotlib axes to plot on
            label: label for the plot line
            show_plot: whether to show the plot immediately
        
        Returns:
            tuple of (figure, axes) objects
        """
        if label is None:
            label = "Sensor Data"
        
        # Calculate the derivative using finite differences
        derivative = np.zeros_like(self.values, dtype=float)
        derivative = derivative[:-1]
        times_derivative = np.zeros_like(derivative)

        # Interior points
        for i in range(0, len(self.values)-1):
            derivative[i] = (self.values[i+1] - self.values[i]) / (self.time[i+1] - self.time[i])
            times_derivative[i] = (self.time[i+1] + self.time[i]) / 2
       
        # Create twin axes for the derivative
        ax2 = ax.twinx()
        
        # Plot the original signal on the primary y-axis
        line1 = ax.plot(self.time, self.values, label=label, linewidth=2, color='blue')
        
        # Plot the derivative on the secondary y-axis
        line2 = ax2.plot(times_derivative, derivative, label=f'{label} Derivative', 
                        linestyle='--', linewidth=1.5, color='red')
        
        # Set labels and title
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Amplitude (a.u.)', color='blue')
        ax2.set_ylabel('Derivative (a.u./μs)', color='red')
        ax.set_title(f'Signal and Derivative - {label}')
        
        # Set grid on primary axes
        ax.grid(True, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Color the y-axis labels to match the data
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        if show_plot:
            plt.show()
        
        return ax.figure, ax


class Event:
    """
    A "ping pong event". All the information dumped from the microcontroller when the signal crosses a threshold
    uuid: unique tag to name the event for later reference
    label: Label for type of event. 
    data: list of SensorData objects, one for each sensor
    """
    def __init__(self, uuid: str, data: list[SensorData] = None, label: str = NO_LABEL_STR):
        self.uuid = uuid
        self.label = label
        self.data = data.copy() if data else []

    def get_short_uuid(self):
        return self.uuid[:UUID_ABBREV]
    
    def __str__(self):
        return f"UUID-{self.get_short_uuid()}___Label-{self.label}"

    def get_channel_data(self, channel_number: int) -> tuple[np.ndarray, np.ndarray]:
        """
        channel_number: (int) index of channel data is being requested for
        returns numpy arrays of times and values from channel_number 
        """
        if channel_number >= len(self.data):
            raise ValueError(f"Channel {channel_number} does not exist. Only {len(self.data)} channels available.")
        
        sensor_data = self.data[channel_number]
        return sensor_data.time, sensor_data.values
    
    def get_sensor_data(self, channel_number: int) -> SensorData:
        """
        channel_number: (int) index of channel data is being requested for
        returns SensorData object for the channel
        """
        if channel_number >= len(self.data):
            raise ValueError(f"Channel {channel_number} does not exist. Only {len(self.data)} channels available.")
        
        return self.data[channel_number]

    def plot(self, ax: matplotlib.axes.Axes = None, channels: list[int] = None, **kwargs) -> matplotlib.axes.Axes:
        """
        Plot data from this event object.
        
        Args:
            ax: matplotlib axes to plot on (creates new if None)
            channels: list of channel indices to plot (plots all if None)
            **kwargs: additional plotting arguments passed to sensor.plot()
        
        Returns:
            matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if channels is None:
            channels = list(range(len(self.data)))
        
        # Default matplotlib color cycle
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, channel in enumerate(channels):
            if channel < len(self.data):
                color = colors[i % len(colors)]
                self.data[channel].plot(ax, label=f"Sensor {channel}", color=color, **kwargs)
        
        ax.set_title(str(self))
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Response (a.u.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax

    def plot_with_derivative(self, channel: int, ax: matplotlib.axes.Axes = None, show_plot: bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Plot a specific channel with its derivative.
        
        Args:
            channel: index of channel to plot
            ax: matplotlib axes to plot on (creates new if None)
            show_plot: whether to show the plot immediately
        
        Returns:
            tuple of (figure, axes) objects
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if channel >= len(self.data):
            raise ValueError(f"Channel {channel} does not exist. Only {len(self.data)} channels available.")
        
        return self.data[channel].plot_with_derivative(ax, label=f"Channel {channel}", show_plot=show_plot)


def read_event_data(open_port, request_label: bool = False) -> Event:
    """
    open_port: open serial port object
    returns: Event object if event happens within READ_ATTEMPT_TIMEOUT of function call. None if not. 
    """
    with open_port as ser:
        line = str(ser.readline())
        if len(line) >= MIN_LEN:
            uuid = line[UUID_START:LINE_END]
            event_data = Event(uuid=uuid)
            
            # Initialize lists to collect data for each sensor
            sensor_times = [[], [], []]
            sensor_values = [[], [], []]
            
            second_line = str(ser.readline())  # do nothing with the second line (yet)
            line = str(ser.readline())  # read in the first line to process
            
            while line != EVENT_END_MARKER:
                # process the line that was read in
                line_data = process_line(line)

                # get analog timestamped data points
                a0_data = line_data[DATA_A0_INDEX]
                a0_time = line_data[TIME_A0_INDEX]
                a1_data = line_data[DATA_A1_INDEX]
                a1_time = line_data[TIME_A1_INDEX]
                a2_data = line_data[DATA_A2_INDEX]
                a2_time = line_data[TIME_A2_INDEX]
                
                # append data points to sensor arrays
                sensor_times[0].append(a0_time)
                sensor_values[0].append(a0_data)
                sensor_times[1].append(a1_time)
                sensor_values[1].append(a1_data)
                sensor_times[2].append(a2_time)
                sensor_values[2].append(a2_data)

                # read the next line
                line = str(ser.readline())

            # Convert lists to numpy arrays and create SensorData objects
            for i in range(3):
                sensor_data = SensorData(
                    time=np.array(sensor_times[i]),
                    values=np.array(sensor_values[i])
                )
                event_data.data.append(sensor_data)

            if request_label:
                _request_event_label(event_data)

            return event_data
        else:
            return None


def process_line(line: str) -> list[int]:
    """
    line: (str) comma-space delimited list of integers
    returns: list of integers
    """
    split_list = line[LINE_START:LINE_END].split(", ")
    return [int(ele) for ele in split_list]
    

def _request_event_label(event_data: Event, timeout=10) -> None:
    try:
        event_data.label = inputimeout(f"Label Data UUID: {event_data.uuid[:8]} (p = ping pong ball, n = not ping pong ball)", timeout=timeout)
    except Exception: 
        print("TIMEOUT: LABEL UNCHANGED")


def event_file_write(folder_loc: str, event_data: Event) -> None:
    file_name = str(event_data) + ".txt"
    event_file_write_generic_name(folder_loc, event_data, file_name)
    return


def event_file_write_generic_name(folder_loc: str, event_data: Event, file_name: str) -> None:
    """
    Writes data from event_data to a new file in folder_loc with event_data.uuid in the title
    """
    full_path = folder_loc + file_name

    with open(full_path, mode='x') as file:
        file.write(file_name + "\n")
        file.write(f"# UUID: {event_data.uuid}\n")
        file.write(f"# Label: {event_data.label}\n")

        # Write data for each sensor
        for sensor_idx, sensor_data in enumerate(event_data.data):
            for i in range(len(sensor_data.time)):
                file.write(f"{sensor_idx}, {sensor_data.time[i]}, {sensor_data.values[i]}\n")


def event_file_read(full_path: str) -> Event:
    """
    Reads data from full_path and returns an Event object
    """
    uuid = None
    label = NO_LABEL_STR
    sensor_data_dict = {}
    
    with open(full_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Parse header lines (starting with #)
            if line.startswith('#'):
                if line.startswith('# UUID: '):
                    uuid = line.replace('# UUID: ', '')
                elif line.startswith('# Label: '):
                    label = line.replace('# Label: ', '')
                continue
            
            # Parse data lines
            line_items = line.split(", ")
            if len(line_items) >= 3:
                channel = int(line_items[CHANNEL_INDEX])
                time_val = int(line_items[TIME_INDEX])
                value_val = int(line_items[VALUE_INDEX])
                
                if channel not in sensor_data_dict:
                    sensor_data_dict[channel] = {'times': [], 'values': []}
                
                sensor_data_dict[channel]['times'].append(time_val)
                sensor_data_dict[channel]['values'].append(value_val)
    
    if uuid is None:
        raise ValueError("No UUID found in file")
    
    # Convert to SensorData objects
    sensor_data_list = []
    for channel in sorted(sensor_data_dict.keys()):
        times = np.array(sensor_data_dict[channel]['times'])
        values = np.array(sensor_data_dict[channel]['values'])
        sensor_data = SensorData(times, values)
        sensor_data_list.append(sensor_data)
    
    return Event(uuid=uuid, data=sensor_data_list, label=label)


def read_all_events(folder_loc: str) -> list[Event]:
    """
    Returns list of all events located in the folder_loc folder
    """
    events = []
    files = os.listdir(folder_loc)
    files = [f for f in files if os.path.isfile(folder_loc + '/' + f)]  # Filtering only the files.
    
    for file in files:
        full_path = folder_loc + file
        events.append(event_file_read(full_path))

    return events
    

def read_single_event(open_port, save: bool = True, folder_name: str = "Data/RawEventData/LocatingData/", num_attempt = 10) -> Event:
    """
    Reads single event dump from the microcontroller if happens within certain number of read attempts
    Allows options for plotting and saving event with user defined name 
    HARDCODED FOLDER ENTER :(
    """
    count_miss = 0
    while True:
        event_data = read_event_data(open_port, request_label=False)
        if event_data is not None:
            if save:
                print(f"Saving to folder {folder_name}")
                name = input("File Name: ")
                event_file_write_generic_name(folder_name, event_data, name + ".txt")
            break
        else:
            count_miss = count_miss + 1
            if count_miss > num_attempt:
                leave = input("Want to stop? (Y/N)")
                if leave == "Y":
                    break
                else:
                    count_miss = 0


# Multi-event plotting functions
def overlay_single_channel_plot(events: list[Event], channel_to_plot: int, show_plot: bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Overlay data from multiple event objects for a single channel
    
    Args:
        events: list of event objects to pull data from
        channel_to_plot: Index of channel to plot on all of the events in events
        show_plot: whether to show the plot immediately
    
    Returns:
        tuple of (figure, axes) objects
    """
    # Set up plots
    title_str = f"Multiplot Channel {channel_to_plot}"
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title_str)

    # Iterate through and plot all events in events
    for index, event in enumerate(events): 
        times, vals = event.get_channel_data(channel_to_plot)
        ax.plot(times, vals, label=f"({index}) {event.get_short_uuid()}")

    # Clean up plot before showing
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Response (a.u.)")
    ax.set_title(title_str)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show_plot: 
        plt.show()
    
    return fig, ax


def many_event_subplot(events: list[Event], channels_to_plot: list[int], show_plot: bool = True, 
                      time_range_microseconds: list[int] = [-3500, 3500]) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Vertically stack plots of multiple event objects
    
    Args:
        events: list of event objects to plot
        channels_to_plot: list of channel indices to plot for each event
        show_plot: whether to show the plot immediately
        time_range_microseconds: x-axis limits for all plots
    
    Returns:
        tuple of (figure, axes) objects
    """
    num_plots = len(events)

    # Create set of vertically stacked subplots 
    fig, axes = plt.subplots(num_plots) 
    
    # Handle single event case
    if num_plots == 1:
        axes = [axes]

    # Iterate through each axes object 
    for index, ax in enumerate(axes):
        event = events[index]  # set active event 
        for channel_num in channels_to_plot:
            if channel_num < len(event.data):
                event.data[channel_num].plot(ax, label=f"Channel {channel_num}")
        
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Response (a.u.)")
        ax.set_xlim(time_range_microseconds)
        ax.set_title(event.get_short_uuid())
        ax.legend(bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3)

    if show_plot: 
        plt.show()
    
    return fig, axes


if __name__ == "__main__":
    folder_name = "Data/RawEventData/"
    ser = serial.Serial(SERIALPORT, BAUDRATE, timeout=READ_ATTEMPT_TIMEOUT)
    read_single_event(ser)