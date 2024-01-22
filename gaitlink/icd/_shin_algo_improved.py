from typing import Any, Literal
import numpy as np
import pandas as pd
from gaitmap.data_transform import Resample
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter
from scipy.signal import cwt, ricker, savgol_filter
from typing_extensions import Self, Unpack
from gaitlink.data_transform import EpflDedriftedGaitFilter, EpflGaitFilter
from gaitlink.ICD.base import BaseIcdDetector, base_icd_docfiller


@base_icd_docfiller
class IcdShinImproved(BaseIcdDetector):
    """
    This algorithm is designed to detect initial contacts from accelerometry signals [1, 2]. Once walking bouts are identified
    in a separate stage, this algorithm utilizes the accelerometry signals recorded during these walking bouts to
    calculate and pinpoint initial contacts within each bout. The process involves multiple stages, including signal
    processing, filtering, and zero-crossing detection, as outlined in the class documentation and references provided

    This is based on the implementation published as part of the mobilised project [3].
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    :param axis : str, optional
        selecting which part of the accelerometry signal to be used. Can be 'x', 'y', 'z', or 'norm'.

    :param signal : np.ndarray
        A numpy array of the signal values.

    :param mode : str
        A string specifying the type of zero crossings to detect.
        Can be 'positive_to_negative', 'negative_to_positive', or 'both'.
        The default is 'both'.


    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(ic_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:
    1) Structural differences: here we calculate ICs separately using the Shin algorithm,
    those ICs will be used to calculate cadence separately. On matlab, cadence is calculated
    in the same function as the ICs.
    2) Configurable accelerometry signal: on matlab, all axes are used to calculate ICs, here we provide
    the option to select which axis to use. Despite the fact that the Shin algorithm on matlab uses all axes,
    here we provide the option of selecting a single axis because other contact detection algorithms use only the
    horizontal axis.
    3) Despite all filter giving identical results in python and matlab, gaussian_filter exhibits some small differences
    4) All parameters are expressed in the units used in gaitlink, as opposed to matlab.
      Specifically, we use m/s^2 instead of g.

    References
    ----------
    [1] Shin, Seung Hyuck, and Chan Gook Park. "Adaptive step length estimation algorithm
    using optimal parameters and movement status awareness." Medical engineering & physics 33.9 (2011): 1064-1071.

    [2] Paraschiv-Ionescu, A. et al. "Real-world speed estimation using single trunk IMU:
    methodological challenges for impaired gait patterns". IEEE EMBC (2020): 4596-4599

    [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/CADB_CADC/Library/Shin_algo_improved.m

    """

    axis: Literal["x", "y", "z", "norm"]

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(self, axis: Literal["x", "y", "z", "norm"] = "norm") -> None:
        self.axis = axis

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """ %(detect_short)s

        Parameters
        ----------
        %(detect_para)s

        -------
        %(detect_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if self.axis not in ["x", "y", "z", "norm"]:
            raise ValueError("Invalid axis. Choose 'x', 'y', 'z', or 'norm'.")

        signal = (
            norm(data[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=1)
            if self.axis == "norm"
            else data[f"acc_{self.axis}"].to_numpy()
        )

        # Resample to 40Hz to process with filters
        resampler = Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
        resampler.transform(data=signal, sampling_rate_hz=sampling_rate_hz)
        signal_downsampled = resampler.transformed_data_

        # We need to intitialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        # TODO: We should evaluate, if we need the padding at all, or if the filter methods that we use handle that
        #  correctly anyway. -> filtfilt uses padding automatically and savgol allows to actiavte padding, put uses the
        #  default mode (polyinomal interpolation) might be suffiecent anyway, cwt might have some edeeffects, but
        #  usually nothing to worry about.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        # Padding to cope with short data
        len_pad = 4 * n_coefficients
        signal_downsampled_padded = np.pad(signal_downsampled, (len_pad, len_pad), "wrap")

        # Filters
        # 1
        # TODO (future): Replace svagol and cwt with class implementation to easily expose parameters.
        accN_filt1 = savgol_filter(signal_downsampled_padded.squeeze(), window_length=21, polyorder=7)
        # 2
        filter = EpflDedriftedGaitFilter()
        accN_filt2 = filter.filter(accN_filt1, sampling_rate_hz=40).filtered_data_
        # 3
        accN_filt3 = cwt(accN_filt2.squeeze(), ricker, [10])
        # 4
        accN_filt4 = savgol_filter(accN_filt3.squeeze(), window_length=11, polyorder=5)
        # 5
        accN_filt5 = cwt(accN_filt4.squeeze(), ricker, [10])
        # 6
        windowWidth = 10
        sigma = windowWidth / 5
        accN_filt6 = gaussian_filter(accN_filt5.squeeze(), sigma)
        # 7
        windowWidth = 10
        sigma = windowWidth / 5
        accN_filt7 = gaussian_filter(accN_filt6.squeeze(), sigma)
        # 8
        windowWidth = 15
        sigma = windowWidth / 5
        accN_filt8 = gaussian_filter(accN_filt7.squeeze(), sigma)
        accN_MultiFilt_rmp = accN_filt8[len_pad:-len_pad]

        # TODO (for future): Upsampling here does not seem that "smart". Maybe better to detect zero crossing in low
        #                    freq and then upsample results. Should be much more efficient.
        # Resample to 100Hz for consistency with the original data (for ICD)
        resampler = Resample(sampling_rate_hz)
        resampler.transform(data=accN_MultiFilt_rmp, sampling_rate_hz=self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
        accN_MultiFilt_rmp100 = resampler.transformed_data_

        # Initial contacts timings (heel strike events) detected as positive slopes zero-crossing in sample 120
        IC_lowSNR = find_zero_crossings(accN_MultiFilt_rmp100, "positive_to_negative")

        self.ic_list_ = pd.DataFrame({"ic": IC_lowSNR})

        return self


def find_zero_crossings(
    signal: np.ndarray, mode: Literal["positive_to_negative", "negative_to_positive", "both"] = "both"
) -> np.ndarray:
    """
    Find zero crossings in a signal.

    Parameters
    ----------
    :param signal : np.ndarray
        A numpy array of the signal values.
    :param mode : str, optional
        A string specifying the type of zero crossings to detect.
        Can be 'positive_to_negative', 'negative_to_positive', or 'both'.
        The default is 'both'.

    Returns
    -------
    np.ndarray
        A numpy array containing the indices where zero crossings occur.

    Raises
    ------
    ValueError
        If the mode is not one of the specified options.

    Examples
    --------
    >>> signal = np.array([1, -1, 1, -1, 1])
    >>> find_zero_crossings(signal, mode="both")
    array([0, 1, 2, 3])
    """
    # Compute differences between consecutive elements
    diffs = np.diff(signal)

    # Find indices where sign changes
    crossings = np.where(np.diff(np.sign(diffs)))[0]

    if mode == "positive_to_negative":
        return crossings[signal[crossings] > 0]
    elif mode == "negative_to_positive":
        return crossings[signal[crossings] < 0]
    elif mode == "both":
        return crossings
    else:
        raise ValueError("Invalid mode. Choose 'positive_to_negative', 'negative_to_positive', or 'both'.")
