
import scipy
import numpy as np
import matplotlib.pyplot as plt
import ReceiveData as rec


def compare_fft_metric(events: list[rec.Event], metric):
    """
    Compare FFT metrics between ping pong and non-ping pong events.
    
    Args:
        events: list of event objects
        metric: function that takes two lists metric(freqs, mags) and returns some value characterizing the FFT
    
    Returns:
        tuple of (ping_pong_results, non_ping_pong_results)
    """
    # Get FFTs of ping pong events and not ping pong events
    fft_results = []
    for single_event in events:
        # Calculate FFT for each channel
        for channel in range(len(single_event.data)):
            freqs, mags = fft_select(single_event, channel, plot=False, null_avg=True)
            fft_results.append((freqs, mags, single_event.label))
    
    metric_results_p = []
    metric_results_n = []
    for freqs, mags, label in fft_results:
        if label == "p":
            metric_results_p.append(metric(freqs, mags))
        else:
            metric_results_n.append(metric(freqs, mags))
        
    return metric_results_p, metric_results_n


def fft_select(event_data: rec.Event, channel_number: int, plot=True, null_avg=True):
    """
    Calculate and optionally plot FFT for a specific channel of an event.
    
    Args:
        event_data: Event object containing sensor data
        channel_number: Index of channel to analyze (0, 1, or 2)
        plot: Whether to display the FFT plot
        null_avg: Whether to remove the mean from the signal before FFT
    
    Returns:
        tuple of (frequencies, magnitudes)
    """
    times, values = event_data.get_channel_data(channel_number)
    freqs, mags = data_fft(times, values, plot=False, null_avg=null_avg)
    
    if plot:
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(str(event_data) + ", Channel: " + str(channel_number))
        plt.plot(freqs, mags)
        plt.title(str(event_data) + ", Channel: " + str(channel_number))
        plt.xlabel("Frequency (Hz)")
        plt.xlim([0, 3000])
        plt.show()

    return freqs, mags


def fft_overlay(event_data: rec.Event):
    """
    Plot FFTs for all three channels of an event on the same axes.
    
    Args:
        event_data: Event object containing sensor data
    """
    freqs0, values0 = fft_select(event_data, 0, plot=False)
    freqs1, values1 = fft_select(event_data, 1, plot=False)
    freqs2, values2 = fft_select(event_data, 2, plot=False)

    fig = plt.gcf()
    fig.canvas.manager.set_window_title(str(event_data))
    plt.plot(freqs0, values0, label="ch0")
    plt.plot(freqs1, values1, label="ch1")
    plt.plot(freqs2, values2, label="ch2")
    plt.title(str(event_data))
    plt.xlabel("Frequency (Hz)")
    plt.xlim([0, 3000])
    plt.legend()
    plt.show()


def data_fft(times: list[int], values: list[int], null_avg=True, plot=False):
    """
    Calculate FFT of time series data.
    
    Args:
        times: List of timestamps in microseconds
        values: List of sensor values
        null_avg: Whether to remove the mean from the signal before FFT
        plot: Whether to display the FFT plot
    
    Returns:
        tuple of (frequencies, magnitudes)
    """
    assert len(times) == len(values), "Vector sizes are not equal"

    N = len(times)
    time_seconds = [us_time * 1e-6 for us_time in times]
    time_deltas = []
    for i in range(1, len(time_seconds)):
        time_deltas.append(time_seconds[i] - time_seconds[i-1])
    avg_sample_spacing = np.average(time_deltas)
    std_sample_spacing = np.std(time_deltas)
    # print("Sample Spacing: " + str(avg_sample_spacing) + " +/- " + str(std_sample_spacing))
    # print("Percent Variation: " + str(std_sample_spacing / avg_sample_spacing))

    if null_avg:
        average = np.average(values)
        amps = [value - average for value in values]
    else:
        amps = values.copy()
   
    ffts = scipy.fft.fft(amps, N, norm="forward")
    freqs = scipy.fft.fftfreq(N, avg_sample_spacing)
    mags = abs(ffts)
    phase = np.angle(ffts)

    if plot:
        plt.plot(freqs, mags)
        plt.xlim([0, 3000])
        plt.xlabel("Frequency (Hz)")
        plt.show()

    return freqs, mags


def apply_metric_to_events(events: list[rec.Event], metric):
    """
    Apply a metric function to all events, separating ping pong and non-ping pong results.
    
    Args:
        events: list of event objects
        metric: function that takes an event and returns some value characterizing the event
    
    Returns:
        tuple of (ping_pong_results, non_ping_pong_results)
    """
    metric_results_p = []
    metric_results_n = []

    for event in events:
        if event.label == "p":
            metric_results_p.append(metric(event))
        elif event.label == "n":
            metric_results_n.append(metric(event))
        
    return metric_results_p, metric_results_n


def mag_rms_range(freqs, mags, min_freq, max_freq):
    """
    Calculate the RMS ratio of a frequency range to the total signal.
    
    Args:
        freqs: frequencies in fourier transform
        mags: magnitudes in fourier transform
        min_freq: minimum frequency
        max_freq: maximum frequency
    
    Returns:
        RMS ratio of the specified frequency range to total signal
    """
    freqs_array = np.array(freqs)
    mags_array = np.array(mags)
    total_rms = np.sqrt(np.mean(mags_array**2))  # total RMS value of all the magnitudes

    min_index = np.argmin(np.absolute(freqs_array - min_freq))
    max_index = np.argmin(np.absolute(freqs_array - max_freq))

    range_rms = np.sqrt(np.mean(mags_array[min_index:max_index]**2))  # RMS of all values in the range 

    return range_rms / total_rms


def mean_channel_mag_rms_range(event: rec.Event, min_freq, max_freq):
    """
    Calculate the mean RMS ratio across all channels for a specific frequency range.
    
    Args:
        event: Event object containing sensor data
        min_freq: minimum frequency for the range
        max_freq: maximum frequency for the range
    
    Returns:
        Mean RMS ratio across all channels
    """
    rms_list = []
    
    # Calculate RMS ratio for each channel
    for channel in range(3):
        freqs, mags = fft_select(event, channel, plot=False, null_avg=True)
        rms_list.append(mag_rms_range(freqs, mags, min_freq, max_freq))

    return np.mean(rms_list)


def evaluate_metric(events: list[rec.Event], metric, cost):
    """
    Evaluate a metric using a cost function to compare ping pong vs non-ping pong events.
    
    Args:
        events: list of event objects
        metric: function that takes an event and returns a value
        cost: function that takes ping pong and non-ping pong results and returns a cost
    
    Returns:
        Cost value from the evaluation
    """
    presults, nresults = apply_metric_to_events(events, metric)
    return cost(presults, nresults)


def classify_rms(event: rec.Event, min_freq, max_freq, thresh) -> bool:
    """
    Classify an event as ping pong based on RMS ratio threshold.
    
    Args:
        event: Event object to classify
        min_freq: minimum frequency for RMS calculation
        max_freq: maximum frequency for RMS calculation
        thresh: threshold value for classification
    
    Returns:
        True if classified as ping pong, False otherwise
    """
    return mean_channel_mag_rms_range(event, min_freq, max_freq) > thresh


def calculate_classification_threshold(events: list[rec.Event], min_freq_range=(700, 1500), max_freq_range=(1200, 2000), grid_size=20):
    """
    Calculate optimal frequency range and threshold for classifying ping pong events.
    
    This function performs a grid search over frequency ranges to find the best
    separation between ping pong and non-ping pong events based on RMS ratios.
    
    Args:
        events: List of event objects to analyze
        min_freq_range: Tuple of (min, max) for minimum frequency search range
        max_freq_range: Tuple of (min, max) for maximum frequency search range  
        grid_size: Number of grid points for each frequency dimension
    
    Returns:
        tuple of (best_min_freq, best_max_freq, best_threshold, cost_matrix)
    """
    def rms_cost(presults, nresults): 
        return (1 - (np.min(presults) - np.max(nresults))/(np.max(nresults)))
    
    # Create frequency grids
    min_freqs = np.linspace(min_freq_range[0], min_freq_range[1], grid_size)
    max_freqs = np.linspace(max_freq_range[0], max_freq_range[1], grid_size)
    
    # Initialize results matrix
    results = np.zeros((grid_size, grid_size))
    
    # Grid search over frequency ranges
    for i in range(grid_size):
        for j in range(grid_size):
            min_freq = min_freqs[i]
            max_freq = max_freqs[j]
            
            # Skip invalid frequency ranges (min >= max)
            if min_freq >= max_freq - 20:
                results[i][j] = np.inf
            else:
                def eval_metric(event): 
                    return mean_channel_mag_rms_range(event, min_freq, max_freq)
                results[i][j] = evaluate_metric(events, eval_metric, rms_cost)
    
    # Find best parameters
    best_index = np.argmin(results)
    i = int(np.floor(best_index / grid_size))
    j = best_index % grid_size
    
    best_min_freq = min_freqs[i]
    best_max_freq = max_freqs[j]
    best_cost = results[i][j]
    
    # Calculate threshold using best frequency range
    def channel_rms(event): 
        return mean_channel_mag_rms_range(event, best_min_freq, best_max_freq)
    
    presults, nresults = apply_metric_to_events(events, channel_rms)
    best_threshold = (np.min(presults) + np.max(nresults)) / 2
    
    print(f"Best frequency range: {best_min_freq:.1f} - {best_max_freq:.1f} Hz")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Cost value: {best_cost:.3f}")
    
    return best_min_freq, best_max_freq, best_threshold, results


if __name__ == "__main__":
    folder_name = "Data/RawEventData/"
    event1 = rec.event_file_read(folder_name + "LocatingData/6in_(18.0,6.4)_0.txt")
    event2 = rec.event_file_read(folder_name + "LocatingData/6in_(18.0,6.4)_1.txt")

    channel = 2
    fig, ax = plt.subplots()
    event1.plot(ax)
    plt.show()
    x, t = event1.get_channel_data(channel)
    x = np.array(x)
    t = np.array(t)

    x = x[15:-250]
    t = t[15:-250]
    t_s = t * 1e-6

    tx, utx = event1.get_channel_events(channel).get_sample_rate()
    fs = 1 / tx * 1e6
    N = len(x)

    window_length = 6
    g_std = 2
    # window = scipy.signal.windows.gaussian(
    #     M=window_length,
    #     std=g_std,
    #     sym=True
    # )
    window = scipy.signal.windows.boxcar(window_length)
    sft = scipy.signal.ShortTimeFFT(
        win=window,
        hop=3,
        fs=fs,
        fft_mode='centered',
        mfft=None,
        scale_to='magnitude'
    )

    sx = sft.stft(x)
    t_lo, t_hi, f_lo, f_hi = sft.extent(
        n=N,
        axes_seq='tf'
    )

    fig1, ax1 = plt.subplots(figsize=(6., 4.)) 

    ax1.set_title(rf"STFT ({sft.m_num*sft.T:g}$\,us$ Gaussian window, " +
                rf"$\sigma_t={g_std*sft.T}\,$us)")
    ax1.set(xlabel=f"Time $t$ in microseconds ({sft.p_num(N)} slices, " +
                rf"$\Delta t = {sft.delta_t:g}\,$us)",
            ylabel=f"Freq. $f$ in MHz ({sft.f_pts} bins, " +
                rf"$\Delta f = {sft.delta_f:g}\,$MHz)",
            xlim=(t_lo, t_hi))

    im1 = ax1.imshow(abs(sx), origin='lower', aspect='auto',
                    extent=sft.extent(N), cmap='viridis')
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    # Shade areas where window slices stick out to the side:
    for t0_, t1_ in [(t_lo, sft.lower_border_end[0] * sft.T),
                    (sft.upper_border_begin(N)[0] * sft.T, t_hi)]:
        ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    for t_ in [0, N * sft.T]:  # mark signal borders with vertical line:
        ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    plt.show()


### Older things pre 8/1/2025
##### Evaluating a bunch of possible frequencies 

    # # Load events and calculate optimal parameters
    # all_events = rec.read_all_events("Data/RawEventData/")
    # min_freq, max_freq, threshold, cost_matrix = calculate_classification_threshold(all_events)

##### Plotting metric for all events and showing the boundry

    # folder_name = "Data/RawEventData/"
    # def rms_cost(presults, nresults): return (1 - (np.min(presults) - np.max(nresults))/(np.max(nresults)))
    # all_events = rec.read_all_events(folder_name)

    # # min_freq = 1415
    # # max_freq = 1663

    # ## 11/12/23 results
    # min_freq = 1163.157894736842
    # max_freq = 1915.7894736842106
    # mid = 0.31597582047520656

    # def rms(freqs, mags): return mag_rms_range(freqs, mags, min_freq, max_freq)
    # def channel_rms(event): return mean_channel_mag_rms_range(event, min_freq, max_freq)
   
    # presults, nresults = apply_metric_to_events(all_events, channel_rms)

    # plt.vlines(presults, -1, 1, colors="blue")
    # plt.vlines(nresults, -1, 1, colors="red")
    # plt.vlines(0.312, -1, 1, colors="black")
    # plt.show()