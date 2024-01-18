import warnings
from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from gaitmap.utils.array_handling import merge_intervals
from intervaltree import IntervalTree
from typing_extensions import Self, Unpack

from gaitlink.data_transform import EpflDedriftedGaitFilter, Resample
from gaitlink.gsd.base import BaseGsdDetector


def hilbert_envelop(y, Smooth_window, threshold_style, DURATION):
    """NOTE: This has been edited from the original MATLAB version to remove perceived error"""
    # Calculate the analytical signal and get the envelope
    amplitude_envelope = np.abs(scipy.signal.hilbert(y))

    # Take the moving average of analytical signal
    env = np.convolve(
        amplitude_envelope, np.ones(Smooth_window) / Smooth_window, "same"
    )  # Smooth  NOTE: Original matlab code used mode 'full', meaning the length of convolution was different to env, this has been fixed here
    env = env - np.mean(env)  # Get rid of offset
    env = env / np.max(env)  # Normalize

    """ Threshold the signal """
    # Input the threshold if needed
    if not threshold_style:
        f = plt.figure()
        plt.plot(env)
        plt.title("Select a threshold on the graph")
        threshold_sig = input("What threshold have you selected?\n")
        print("You have selected: ", threshold_sig)
        plt.close(f)
    else:
        # Threshold style
        threshold_sig = 4 * np.nanmean(env)
    noise = np.mean(env) * (1 / 3)  # Noise level
    threshold = np.mean(env)  # Signal level

    # Initialize Buffers
    thresh_buff = np.zeros(len(env) - DURATION + 1)
    noise_buff = np.zeros(len(env) - DURATION + 1)
    thr_buff = np.zeros(len(env) + 1)
    h = 1
    alarm = np.zeros(len(env) + 1)

    for i in range(len(thresh_buff)):
        # Update threshold 10% of the maximum peaks found
        if (env[i : i + DURATION] > threshold_sig).all():
            alarm[i] = max(env)
            threshold = 0.1 * np.mean(env[i : i + DURATION])
            h = h + 1
        elif np.mean(env[i : i + DURATION]) < threshold_sig:
            noise = np.mean(env[i : i + DURATION])
        else:
            if noise_buff.any():
                noise = np.mean(noise_buff)

        thresh_buff[i] = threshold
        noise_buff[i] = noise

        # Update threshold
        if h > 1:
            threshold_sig = noise + 0.50 * (abs(threshold - noise))
        thr_buff[i] = threshold_sig

    return alarm, env


def find_min_max_above_threshold(signal: np.ndarray, threshold: float) -> tuple:
    """
    Identify the indices of local minima and maxima in a 1D numpy array (signal),
    where the values are beyond a specified threshold.

    Parameters
    ----------
    signal (np.ndarray): A 1D numpy array representing the signal.
    threshold (float): A threshold value to filter the minima and maxima.

    Returns
    -------
    tuple: Two arrays containing the indices of local minima and maxima, respectively.
    """
    signal = signal.squeeze()
    diff = np.diff(signal)
    extrema_indices = np.nonzero(diff[1:] * diff[:-1] <= 0)[0] + 1

    minima = extrema_indices[diff[extrema_indices] >= 0]
    maxima = extrema_indices[diff[extrema_indices] < 0]

    minima = minima[signal[minima] < -threshold]
    maxima = maxima[signal[maxima] > threshold]

    return minima, maxima


def find_pulse_trains(x):
    walkflag = 0
    THD = 3.5 * 40
    n = 0

    start = [0]
    steps = [0]
    end = [0]

    if len(x) > 2:
        for i in range(len(x) - 1):
            if x[i + 1] - x[i] < THD:
                if walkflag == 0:
                    start[n] = x[i]
                    steps[n] = 1
                    walkflag = 1
                else:
                    steps[n] = steps[n] + 1
                    THD = 1.5 * 40 + (x[i] - start[n]) / steps[n]
            else:
                if walkflag == 1:
                    end[n] = x[i - 1]
                    n = n + 1
                    start = start + [0]
                    steps = steps + [0]
                    end = end + [0]
                    walkflag = 0
                    THD = 3.5 * 40

    if walkflag == 1:
        if x[-1] - x[-2] < THD:
            end[-1] = x[-1]
            steps[n] = steps[n] + 1
        else:
            end[-1] = x[-1]

    return np.array([start, end, steps]).T


def find_intersections(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]]) -> np.ndarray:
    """Find the intersections between two sets of intervals.

    Parameters
    ----------
    intervals_a : list of tuple of int
        The first list of intervals. Each interval is represented as a tuple of two integers.
    intervals_b : list of tuple of int
        The second list of intervals. Each interval is represented as a tuple of two integers.

    Returns
    -------
    np.ndarray
        An array of intervals that are the intersections of the intervals in `intervals_a` and `intervals_b`.
        Each interval is represented as a list of two integers.

    """    # Create Interval Trees
    intervals_a_tree = IntervalTree.from_tuples(intervals_a)
    intervals_b_tree = IntervalTree.from_tuples(intervals_b)

    overlap_intervals = []

    # Calculate TP and FP
    for interval in intervals_b_tree:
        overlaps = sorted(intervals_a_tree.overlap(interval.begin, interval.end))
        if overlaps:
            for overlap in overlaps:
                start = max(interval.begin, overlap.begin)
                end = min(interval.end, overlap.end)
                overlap_intervals.append([start, end])

    return merge_intervals(np.array(overlap_intervals)) if len(overlap_intervals) != 0 else np.array([])

class GsdLowBackAcc(BaseGsdDetector):

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        gsd_output = gsd_low_back_acc(data[["acc_x", "acc_y", "acc_z"]].to_numpy(), sampling_rate_hz, plot_results=True)
        self.gsd_list_ = gsd_output

        return self


class NoActivePeriodsDetectedError(Exception):
    pass


def find_active_period_peak_threshold(signal, sampling_rate_hz) -> float:
    # Find pre-detection of 'active' periods in order to estimate the amplitude of acceleration peaks
    alarm, _ = hilbert_envelop(
        signal, sampling_rate_hz, True, sampling_rate_hz
    )  # NOTE: This has been edited from the original MATLAB version to remove perceived error
    walkLowBack = np.array([])

    if not np.any(alarm):  # If hilbert_envelope fails to detect 'active' try version [1]
        raise NoActivePeriodsDetectedError()

    len_alarm = [
        len(list(s)) for v, s in groupby(alarm, key=lambda x: x > 0)
    ]  # Length of each consecutive stretch of nonzero values in alarm
    end_alarm = np.cumsum(len_alarm)
    start_alarm = np.concatenate([np.array([0]), end_alarm[:-1]])
    alarmed = [
        v for v, s in groupby(alarm, key=lambda x: x > 0)
    ]  # Whether each consecutive stretch of nonzero values in alarm is alarmed

    for s, e, a in zip(start_alarm, end_alarm, alarmed):  # Iterate through the consecutive periods
        if a:  # If alarmed
            if e - s <= 3 * sampling_rate_hz:  # If the length of the alarm period is too short
                alarm[s:e] = 0  # Replace this section of alarm with zeros
            else:
                walkLowBack = np.concatenate([walkLowBack, signal[s - 1 : e - 1]])

    if walkLowBack.size == 0:
        raise NoActivePeriodsDetectedError()

    peaks_p, _ = scipy.signal.find_peaks(walkLowBack)
    peaks_n, _ = scipy.signal.find_peaks(-walkLowBack)
    pksp, pksn = walkLowBack[peaks_p], -walkLowBack[peaks_n]
    pks = np.concatenate([pksp[pksp > 0], pksn[pksn > 0]])
    return np.percentile(pks, 5)  # data adaptive threshold


def gsd_low_back_acc(acc: np.ndarray, sampling_rate_hz, plot_results=True):
    """

    :param acc:
    :param sampling_rate_hz:
    :param plot_results:
    :return GSD_Output:
    """
    ALGORITHM_TARGET_SAMPLING_RATE_HZ = 40  # Sampling rate required for the algorithm

    # Signal vector magnitude
    acc_norm = np.linalg.norm(acc, axis=1)

    # Resample to algorithm_target_fs
    acc_norm_resampled = (
        Resample(ALGORITHM_TARGET_SAMPLING_RATE_HZ)
        .transform(acc_norm, sampling_rate_hz=sampling_rate_hz)
        .transformed_data_
    )
    # NOTE: accN_resampled is slightly different in length and values to accN40 in MATLAB, plots look ok though

    # Filter to enhance the acceleration signal, when low SNR, impaired, asymmetric and slow gait
    acc_filtered = scipy.signal.savgol_filter(acc_norm_resampled, polyorder=7, window_length=21)
    acc_filtered = EpflDedriftedGaitFilter().filter(acc_filtered, sampling_rate_hz=40).filtered_data_
    # NOTE: Original MATLAB code calls old version of cwt (open wavelet.internal.cwt in MATLAB to inspect) in
    #   accN_filt3=cwt(accN_filt2,10,'gaus2',1/40);
    #   Here, 10 is the scale, gaus2 is the second derivative of a Gaussian wavelet, aka a Mexican Hat or Ricker wavelet
    #   In Python, a scale of 7 matches the MATLAB scale of 10 from visual inspection of plots (likely due to how to two
    #   languages initialise their wavelets), giving the line below
    acc_filtered = scipy.signal.cwt(acc_filtered.squeeze(), scipy.signal.ricker, [7])
    acc_filtered4 = scipy.signal.savgol_filter(acc_filtered, 11, 5)
    acc_filtered = scipy.signal.cwt(acc_filtered4.squeeze(), scipy.signal.ricker, [7])  # See NOTE above
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 2
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 2
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 3
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 15 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 2
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)

    FALLBACK_THRESHOLD = 0.15

    try:
        active_peak_threshold = find_active_period_peak_threshold(acc_filtered, ALGORITHM_TARGET_SAMPLING_RATE_HZ)
        signal = acc_filtered
    except NoActivePeriodsDetectedError:
        # If we don't find the active periods, use a fallback threshold and use
        # a less filtered signal for further processing, for which we can better predict the threshold.
        warnings.warn("No active periods detected, using fallback threshold")
        active_peak_threshold = FALLBACK_THRESHOLD
        signal = acc_filtered4

    # Find extrema in signal that might represent steps
    min_peaks, max_peaks = find_min_max_above_threshold(signal, active_peak_threshold)

    # Combine steps detected by the maxima and minima
    MIN_N_STEPS = 5

    gs_from_max = find_pulse_trains(max_peaks)
    gs_from_min = find_pulse_trains(min_peaks)
    gs_from_max = gs_from_max[gs_from_max[:, 2] >= MIN_N_STEPS]
    gs_from_min = gs_from_min[gs_from_min[:, 2] >= MIN_N_STEPS]

    combined_final = find_intersections(
        gs_from_max[:, :2], gs_from_min[:, :2]
    )  # Combine the gs from the maxima and minima

    if combined_final.size == 0:  # Check if no gs detected
        return pd.DataFrame(columns=["start", "end"]).astype(int)  # Return empty df, if no gs

    # Find all max_peaks withing each final gait sequence (GS)
    steps_per_gs = [[x for x in max_peaks if gs[0] <= x <= gs[1]] for gs in combined_final]
    n_steps_per_gs = np.array([len(steps) for steps in steps_per_gs])
    mean_step_times = np.array([np.mean(np.diff(steps)) for steps in steps_per_gs])

    # Pad each gs by 0.75*step_time before and after
    combined_final[:, 0] = np.fix(combined_final[:, 0] - 0.75 * mean_step_times)
    combined_final[:, 1] = np.fix(combined_final[:, 1] + 0.75 * mean_step_times)

    # Filter again by number of steps, remove any gs with too few steps
    combined_final = combined_final[n_steps_per_gs >= MIN_N_STEPS]

    if combined_final.size == 0:  # Check if all gs removed
        return pd.DataFrame(columns=["start", "end"]).astype(int)  # Return empty df if no gs

    # Merge gs if time (in seconds) between consecutive gs is less than MAX_GAP_S
    MAX_GAP_S = 3
    combined_final = merge_intervals(combined_final, ALGORITHM_TARGET_SAMPLING_RATE_HZ * MAX_GAP_S)

    # Convert back to original sampling rate
    combined_final = combined_final * sampling_rate_hz / ALGORITHM_TARGET_SAMPLING_RATE_HZ

    # Cap the start and the end of the signal using clip, incase padding extended any gs past the signal length
    combined_final = np.clip(combined_final, 0, len(acc))

    return pd.DataFrame(combined_final, columns=["start", "end"]).astype(int)
