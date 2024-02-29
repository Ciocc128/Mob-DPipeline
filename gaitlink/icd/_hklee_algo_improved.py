from typing import Any, Literal

import numpy as np
import pandas as pd
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from numpy.linalg import norm
from scipy.ndimage import grey_closing, grey_opening
from typing_extensions import Self, Unpack

from gaitlink.data_transform import (
    CwtFilter,
    EpflDedriftedGaitFilter,
    EpflGaitFilter,
    GaussianFilter,
    Pad,
    Resample,
    SavgolFilter,
    chain_transformers,
)
from gaitlink.icd.base import BaseIcDetector, base_icd_docfiller


@base_icd_docfiller
class IcdHKLeeImproved(BaseIcDetector):
    """Detect initial contacts using the HKLee [1]_ algorithm, with improvements by Ionescu et al. [2]_.

    This algorithm is designed to detect initial contacts from accelerometer signals within a gait sequence.
    The algorithm filters the accelerometer signal down to its primary frequency components
    and then employs morphological operations with closing and opening structural elements
    to detect signal closings and openings, respectively.
    Their difference is analyzed to identify instances where R is greater than 0.
    These regions are interpreted as initial contacts.

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    axis
        selecting which part of the accelerometer signal to be used. Can be 'x', 'y', 'z', or 'norm'.
        The default is 'norm', which is also the default in the original implementation.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(ic_list_)s
    final_filtered_signal_
        (upsampled again in HKLee)
        The downsampled signal after all filter steps.
        This might be useful for debugging.
    ic_list_internal_
        The initial contacts detected on the downsampled signal, before upsampling to the original sampling rate.
        This can be useful for debugging in combination with the `final_filtered_signal_` attribute.


    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - Configurable accelerometer signal: on matlab, all axes are used to calculate ICs, here we provide
      the option to select which axis to use. Despite the fact that the Shin algorithm on matlab uses all axes,
      here we provide the option of selecting a single axis because other contact detection algorithms use only the
      horizontal axis.
    - We use a different downsampling method, which should be "more" correct from a signal theory perspective,
      but will yield slightly different results.
    - only in case the upsampling will be removed #The matlab code upsamples to 120 Hz before the
      final morphological operations.
      #We skip the upsampling of the filtered signal and perform the morphological operations on the downsampled signal.
      #To compensate for the "loss of accuracy" due to the downsampling, we use linear interpolation to determine the
      #exact position of the 0-crossing, even when it occurs between two samples.
      #We then project the interpolated index back to the original sampling rate.
    - For CWT and gaussian filter, the actual parameter we pass to the respective functions differ from the matlab
      implementation, as the two languages use different implementations of the functions.
      However, the similarity of the output was visually confirmed.
    - All parameters are expressed in the units used in gaitlink, as opposed to matlab.
      Specifically, we use m/s^2 instead of g.
    - #Some early testing indicates, that the python version finds all ICs 5-10 samples earlier than the matlab version.
      #However, this seems to be a relatively consistent offset.
      #Hence, it might be possible to shift/tune this in the future.

    Future work:
    - The algorithm can be improved by increasing the threshold of the allowed non-zero values.
      Currently, only single non-zero sequences are removed.
      For example, we could include a threshold of the minimum duration (samples) of an initial contact.

    .. [1] Lee, H-K., et al. "Computational methods to detect step events for normal and pathological
        gait evaluation using accelerometer." Electronics letters 46.17 (2010): 1185-1187.
    .. [2] Paraschiv-Ionescu, A. et al. "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns". IEEE EMBC (2020): 4596-4599
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/CADB_CADC/Library/Shin_algo_improved.m

    """

    axis: Literal["x", "y", "z", "norm"]

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40
    _UPSAMPLED_SAMPLING_RATE_HZ: int = 120

    final_filtered_signal_: np.ndarray
    ic_list_internal_: pd.DataFrame

    def __init__(self, axis: Literal["x", "y", "z", "norm"] = "norm") -> None:
        self.axis = axis

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        %(detect_info)s

        Parameters
        ----------
        %(detect_para)s

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

        # We need to initialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        # TODO: We should evaluate, if we need the padding at all, or if the filter methods that we use handle that
        #  correctly anyway. -> filtfilt uses padding automatically and savgol allows to actiavte padding, put uses the
        #  default mode (polyinomal interpolation) might be suffiecent anyway, cwt might have some edeeffects, but
        #  usually nothing to worry about.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        # Padding to cope with short data
        len_pad_s = 4 * n_coefficients / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        padding = Pad(pad_len_s=len_pad_s, mode="wrap")

        #   CWT - Filter
        #   Original MATLAB code calls old version of cwt (open wavelet.internal.cwt in MATLAB to inspect) in
        #   accN_filt3=cwt(accN_filt2,10,'gaus2',1/40);
        #   Here, 10 is the scale, gaus2 is the second derivative of a Gaussian wavelet, aka a Mexican Hat or Ricker
        #   wavelet.
        #   This frequency this scale corresponds to depends on the sampling rate of the data.
        #   As the gaitlink cwt method uses the center frequency instead of the scale, we need to calculate the
        #   frequency that scale corresponds to at 40 Hz sampling rate.
        #   Turns out that this is 1.2 Hz
        cwt = CwtFilter(wavelet="gaus2", center_frequency_hz=1.2)

        # Savgol filters
        # The original Matlab code useses two savgol filter in the chain.
        # To replicate them with our classes we need to convert the sample-parameters of the original matlab code to
        # sampling-rate independent units used for the parameters of our classes.
        # The parameters from the matlab code are: (21, 7) and (11, 5)
        savgol_1_win_size_samples = 21
        savgol_1 = SavgolFilter(
            window_length_s=savgol_1_win_size_samples / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
            polyorder_rel=7 / savgol_1_win_size_samples,
        )
        savgol_2_win_size_samples = 11
        savgol_2 = SavgolFilter(
            window_length_s=savgol_2_win_size_samples / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
            polyorder_rel=5 / savgol_2_win_size_samples,
        )

        # Now we build everything together into one filter chain.
        filter_chain = [
            # Resample to 40Hz to process with filters
            ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding", padding),
            (
                "savgol_1",
                savgol_1,
            ),
            ("epfl_gait_filter", EpflDedriftedGaitFilter()),
            ("cwt_1", cwt),
            (
                "savol_2",
                savgol_2,
            ),
            ("cwt_2", cwt),
            ("gaussian_1", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_2", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_3", GaussianFilter(sigma_s=3 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding_remove", padding.get_inverse_transformer()),
            ("resampling_up", Resample(self._UPSAMPLED_SAMPLING_RATE_HZ)),
        ]

        final_filtered = chain_transformers(signal, filter_chain, sampling_rate_hz=sampling_rate_hz)
        self.final_filtered_signal_ = final_filtered

        # Apply morphological filters
        se_closing = np.ones(32, dtype=int)
        se_opening = np.ones(18, dtype=int)

        c = grey_closing(self.final_filtered_signal_, structure=se_closing)
        o = grey_opening(c, structure=se_opening)
        r = c - o

        detected_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="ic_id")

        if np.any(r > 0):
            non_zero = bool_array_to_start_end_array(r > 0)
            # removing single non-zero values to be more consistent with the original implementation
            non_zero = non_zero[non_zero[:, 1] - non_zero[:, 0] > 1]
            detected_ics = np.zeros(len(non_zero), dtype=float)
            for j in range(len(non_zero)):
                start_non_zero, end_non_zero = non_zero[j, 0], non_zero[j, 1]
                values_within_range = r[start_non_zero : end_non_zero + 1]
                imax = start_non_zero + np.argmax(values_within_range)

                # Assign the value to the NumPy array
                detected_ics[j] = imax

            detected_ics = pd.DataFrame({"ic": detected_ics}).rename_axis(index="ic_id")

        self.ic_list_internal_ = detected_ics

        # Downsample initial contacts to original sampling rate
        ic_downsampled = (detected_ics * sampling_rate_hz / self._UPSAMPLED_SAMPLING_RATE_HZ).round().astype(int)

        self.ic_list_ = ic_downsampled

        return self
