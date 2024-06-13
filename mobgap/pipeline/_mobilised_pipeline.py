from types import MappingProxyType
from typing import Final, Generic, Optional, TypeVar

import pandas as pd
from tpcp import Pipeline, cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from mobgap._docutils import make_filldoc
from mobgap.aggregation import MobilisedAggregator, apply_thresholds, get_mobilised_dmo_thresholds
from mobgap.aggregation.base import BaseAggregator
from mobgap.cad import CadFromIcDetector
from mobgap.cad.base import BaseCadCalculator
from mobgap.data.base import BaseGaitDataset, ParticipantMetadata
from mobgap.gsd import GsdIluz, GsdIonescu
from mobgap.gsd.base import BaseGsDetector
from mobgap.icd import IcdHKLeeImproved, IcdIonescu, IcdShinImproved, refine_gs
from mobgap.icd.base import BaseIcDetector
from mobgap.lrc import LrcUllrich, strides_list_from_ic_lr_list
from mobgap.lrc.base import BaseLRClassifier
from mobgap.pipeline._gs_iterator import FullPipelinePerGsResult, GsIterator
from mobgap.stride_length import SlZijlstra
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.turning import TdElGohary
from mobgap.turning.base import BaseTurnDetector
from mobgap.utils.array_handling import create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions
from mobgap.walking_speed import WsNaive
from mobgap.walking_speed.base import BaseWsCalculator
from mobgap.wba import StrideSelection, WbAssembly

mobilsed_pipeline_docfiller = make_filldoc(
    {
        "run_short": "Run the pipeline on the provided data.",
        "run_para": """
    datapoint
        The data to run the pipeline on.
        This needs to be a valid datapoint (i.e. a dataset with just a single row).
        The Dataset should be a child class of :class:`~mobgap.data.base.BaseGaitDataset` or implement all the same
        parameters and methods.
    """,
        "run_return": """
    Returns
    -------
    self
        The pipeline object itself with all the results stored in the attributes.
    """,
        "core_parameters": """
    gait_sequence_detection
        A valid instance of a gait sequence detection algorithm.
        This will get the entire raw data as input.
        The core output is available via the ``gs_list_`` attribute.
    initial_contact_detection
        A valid instance of an initial contact detection algorithm.
        This will run on each gait sequence individually.
        The concatenated raw ICs are available via the ``raw_ic_list_`` attribute.
    laterality_classification
        A valid instance of a laterality classification algorithm.
        This will run on each gait sequence individually, getting the predicted ICs from the IC detection algorithm as
        input.
        The concatenated raw ICs with L/R label are available via the ``raw_ic_list_`` attribute.
    cadence_calculation
        A valid instance of a cadence calculation algorithm.
        This will run on each "refined" gait sequence individually.
        This means the provided gait sequence will start and end at the first and last detected IC.
        The detected ICs (with L/R label) and all :class:`~mobgap.data.base.ParticipantMetadata` parameters are provided
        as keyword arguments.
        The concatenated raw cadence per second values are available via the ``raw_per_sec_parameters_`` attribute.
    stride_length_calculation
        A valid instance of a stride length calculation algorithm.
        This will run on each "refined" gait sequence individually.
        This means the provided gait sequence will start and end at the first and last detected IC.
        The detected ICs (with L/R label) and all :class:`~mobgap.data.base.ParticipantMetadata` parameters are provided
        as keyword arguments.
        The concatenated raw stride length per second values are available via the ``raw_per_sec_parameters_``
        attribute.
    walking_speed_calculation
        A valid instance of a walking speed calculation algorithm.
        This will run on each "refined" gait sequence individually.
        This means the provided gait sequence will start and end at the first and last detected IC.
        The detected ICs (with L/R label), cadence per second, stride length per second values and all
        :class:`~mobgap.data.base.ParticipantMetadata` parameters are provided as keyword arguments.
        The concatenated raw walking speed per second values are available via the ``raw_per_sec_parameters_``
        attribute.

        .. note :: If either cadence or stride length is not provided, ``None`` will be passed to the algorithm.
                   Depending on the algorithm, this might raise an error, as the information is required.
    """,
        "turn_detection": """
    turn_detection
        A valid instance of a turn detection algorithm.
        This will run on each gait sequence individually.
        The concatenated raw turn detections are available via the ``raw_turn_list_`` attribute.
    """,
        "wba_parameters": """
    stride_selection
        A valid instance of a stride selection algorithm.
        This will be called with all interpolated stride parameters (``raw_per_stride_parameters_``) across all gait
        sequences.
    wba
        A valid instance of a walking bout assembly algorithm.
        This will be called with the filtered stride list from the stride selection algorithm.
        The final list of strides that are part of a valid WB are available via the ``per_stride_parameters_``
        attribute.
        The aggregated parameters for each WB are available via the ``per_wb_parameters_`` attribute.
    """,
        "aggregation_parameters": """
    dmo_thresholds
        A DataFrame with the thresholds for the individual DMOs.
        To learn more about the required structure and the filtering process, please refer to the documentation of the
        :func:`~mobgap.aggregation.get_mobilised_dmo_thresholds` and :func:`~mobgap.aggregation.apply_thresholds`.
    dmo_aggregation
        A valid instance of a DMO aggregation algorithm.
        This will be called with the aggregated parameters for each WB and the mask of the DMOs.
        The final aggregated parameters are available via the ``aggregated_parameters_`` attribute.
    """,
        "other_parameters": """
    datapoint
        The dataset instance passed to the run method.
    """,
        "primary_results": """
    per_stride_parameters_
        The final list of all strides including their parameters that are part of a valid WB.
        Note, that all per-stride parameters are interpolated based on the per-sec output of the other algorithms.
        Check out the pipeline examples to learn more about this.
    per_wb_parameters_
        Aggregated parameters for each WB.
        This contains "meta parameters" like the number of strides, duration of the WB and the average over all strides
        of cadence, stride length and walking speed (if calculated).
    per_wb_parameter_mask_
        A "valid" mask calculated using the :func:`~mobgap.aggregation.apply_thresholds` function.
        It indicates for each WB which DMOs are valid.
        NaN indicates that the value has not been checked
    aggregated_parameters_
        The final aggregated parameters.
        They are calculated based on the per WB parameters and the DMO mask.
        Invalid parameters are (depending on the implementation in the provided Aggregation algorithm) excluded.
        This output can either be a dataframe with a single row (all WBs were aggregated to a single value, default),
        or a dataframe with multiple rows, if the aggregation algorithm uses a different aggregation approach.
    """,
        "intermediate_results": """
    gs_list_
        The raw output of the gait sequence detection algorithm.
        This is a DataFrame with the start and end of each detected gait sequence.
    raw_ic_list_
        The raw output of the IC detection and the laterality classification.
        This is a DataFrame with the detected ICs and the corresponding L/R label.
    raw_turn_list_
        The raw output of the turn detection algorithm.
        This is a DataFrame with the detected turns (start, end, angle, ...).
    raw_per_sec_parameters_
        A concatenated dataframe with all calculated per-second parameters.
        The index represents the sample of the center of the second the parameter value belongs to.
    raw_per_stride_parameters_
        A concatenated dataframe with all calculated per-stride parameters and the general stride information (start,
        end, laterality).
    """,
        "debug_results": """
    gait_sequence_detection_
        The instance of the gait sequence detection algorithm that was run with all of its results.
    gs_iterator_
        The instance of the GS iterator that was run with all of its results.
        This contains the raw results for each GS, as well as the information about the constrained gs.
        These raw results (inputs and outputs per GS) can be used to test run individual algorithms exactly like they
        were run within the pipeline.
    stride_selection_
        The instance of the stride selection algorithm that was run with all of its results.
    wba_
        The instance of the WBA algorithm that was run with all of its results.
    dmo_aggregation_
        The instance of the DMO aggregation algorithm that was run with all of its results.
    """,
        "step_by_step": """
    The Mobilise-D pipeline consists of the following steps:

    1. Gait sequences are detected using the provided gait sequence detection algorithm.
    2. Within each gait sequence, initial contacts are detected using the provided IC detection algorithm.
       A "refined" version of the gait sequence is created, starting and ending at the first and last detected IC.
    3. Cadence, stride length and walking speed are calculated for each "refined" gait sequence.
       The output of these algorithms is provided per second.
    4. Using the L/R label for each IC calculated by the laterality classification algorithm, strides are defined.
    5. The per-second parameters are interpolated to per-stride parameters.
    6. The stride selection algorithm is used to filter out strides that don't fulfill certain criteria.
    7. The WBA algorithm is used to assemble the strides into walking bouts.
       This is done independent of the original gait sequences.
    8. Aggregated parameters for each WB are calculated.
    9. If DMO thresholds are provided, these WB-level parameters are filtered based on physiological valid thresholds.
    10. The DMO aggregation algorithm is used to aggregate the WB-level parameters to either a set of values
        per-recording or any other granularity (i.e. one value per hour), depending on the aggregation algorithm.

    For a step-by-step example of how these steps are executed, check out :ref:`mobilised_pipeline_step_by_step`.
    """,
    }
)

BaseGaitDatasetT = TypeVar("BaseGaitDatasetT", bound=BaseGaitDataset)


@mobilsed_pipeline_docfiller
class BaseMobilisedPipeline(Pipeline[BaseGaitDatasetT], Generic[BaseGaitDatasetT]):
    """Basic Pipeline structure of the Mobilise-D pipeline.

    .. warning:: While this class implements the basic structure of the Mobilise-D pipeline, we only consider it "The
             Mobilise-D pipeline" if it is used with the predefined parameters/algorithms for the cohorts these
             parameters are evaluated for.

    This pipeline class can either be used with a custom set of algorithms instances or the "official" predefined
    parameters for healthy or impaired walking (see Examples).
    However, when using the predefined parameters it is recommended to use the separate classes instead
    (:class:`MobilisedPipelineHealthy` and :class:`MobilisedPipelineImpaired`).

    For detailed steps on how this pipeline works, check the Notes section and the dedicated examples.


    Parameters
    ----------
    %(core_parameters)s
    %(turn_detection)s
    %(wba_parameters)s
    %(aggregation_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(primary_results)s
    %(intermediate_results)s
    %(debug_results)s

    Notes
    -----
    %(step_by_step)s

    See Also
    --------
    mobgap.pipeline.MobilisedPipelineHealthy : A predefined pipeline for healthy/mildly impaired walking.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

    gait_sequence_detection: BaseGsDetector
    initial_contact_detection: BaseIcDetector
    laterality_classification: BaseLRClassifier
    cadence_calculation: Optional[BaseCadCalculator]
    stride_length_calculation: Optional[BaseSlCalculator]
    walking_speed_calculation: Optional[BaseWsCalculator]
    turn_detection: Optional[BaseTurnDetector]
    stride_selection: StrideSelection
    wba: WbAssembly
    dmo_thresholds: Optional[pd.DataFrame]
    dmo_aggregation: BaseAggregator

    datapoint: BaseGaitDatasetT

    # Algos with results
    gait_sequence_detection_: BaseGsDetector
    gs_iterator_: GsIterator[FullPipelinePerGsResult]
    stride_selection_: StrideSelection
    wba_: WbAssembly
    dmo_aggregation_: BaseAggregator

    # Intermediate results
    gs_list_: pd.DataFrame
    raw_ic_list_: pd.DataFrame
    raw_turn_list_: pd.DataFrame
    raw_per_sec_parameters_: pd.DataFrame
    raw_per_stride_parameters_: pd.DataFrame

    # Final Results
    per_stride_parameters_: pd.DataFrame
    per_wb_parameters_: pd.DataFrame
    per_wb_parameter_mask_: Optional[pd.DataFrame]
    aggregated_parameters_: pd.DataFrame

    class PredefinedParameters:
        normal_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIluz(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(IcdShinImproved()),
                "stride_length_calculation": SlZijlstra(),
                "walking_speed_calculation": WsNaive(),
                "turn_detection": TdElGohary(),
                "stride_selection": StrideSelection(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_mobilised_dmo_thresholds(),
                "dmo_aggregation": MobilisedAggregator(groupby=None),
            }
        )

        impaired_walking: Final = MappingProxyType(
            {
                "gait_sequence_detection": GsdIonescu(),
                "initial_contact_detection": IcdIonescu(),
                "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
                "cadence_calculation": CadFromIcDetector(IcdHKLeeImproved()),
                "stride_length_calculation": SlZijlstra(),
                "walking_speed_calculation": WsNaive(),
                "turn_detection": TdElGohary(),
                "stride_selection": StrideSelection(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_mobilised_dmo_thresholds(),
                "dmo_aggregation": MobilisedAggregator(groupby=None),
            }
        )

    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsDetector,
        initial_contact_detection: BaseIcDetector,
        laterality_classification: BaseLRClassifier,
        cadence_calculation: Optional[BaseCadCalculator],
        stride_length_calculation: Optional[BaseSlCalculator],
        walking_speed_calculation: Optional[BaseWsCalculator],
        turn_detection: Optional[BaseTurnDetector],
        stride_selection: StrideSelection,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: BaseAggregator,
    ) -> None:
        self.gait_sequence_detection = gait_sequence_detection
        self.initial_contact_detection = initial_contact_detection
        self.laterality_classification = laterality_classification
        self.cadence_calculation = cadence_calculation
        self.stride_length_calculation = stride_length_calculation
        self.walking_speed_calculation = walking_speed_calculation
        self.turn_detection = turn_detection
        self.stride_selection = stride_selection
        self.wba = wba
        self.dmo_thresholds = dmo_thresholds
        self.dmo_aggregation = dmo_aggregation

    @mobilsed_pipeline_docfiller
    def run(self, datapoint: BaseGaitDatasetT) -> Self:
        """%(run_short)s.

        Parameters
        ----------
        %(run_para)s

        %(run_return)s
        """
        self.datapoint = datapoint

        imu_data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        self.gait_sequence_detection_ = self.gait_sequence_detection.clone().detect(
            imu_data, sampling_rate_hz=sampling_rate_hz
        )
        self.gs_list_ = self.gait_sequence_detection_.gs_list_
        self.gs_iterator_ = self._run_per_gs(self.gs_list_, imu_data, sampling_rate_hz, datapoint.participant_metadata)

        results = self.gs_iterator_.results_

        self.raw_per_sec_parameters_ = pd.concat(
            [
                results.cadence_per_sec,
                results.stride_length_per_sec,
                results.walking_speed_per_sec,
            ],
            axis=1,
        ).reset_index("r_gs_id", drop=True)
        self.raw_ic_list_ = results.ic_list
        self.raw_turn_list_ = results.turn_list
        self.raw_per_stride_parameters_ = self._sec_to_stride(
            self.raw_per_sec_parameters_, results.ic_list, sampling_rate_hz
        )

        flat_index = pd.Index(
            ["_".join(str(e) for e in s_id) for s_id in self.raw_per_stride_parameters_.index], name="s_id"
        )
        raw_per_stride_parameters = self.raw_per_stride_parameters_.reset_index("gs_id").rename(
            columns={"gs_id": "original_gs_id"}
        )
        raw_per_stride_parameters.index = flat_index

        self.stride_selection_ = self.stride_selection.clone().filter(
            raw_per_stride_parameters, sampling_rate_hz=sampling_rate_hz
        )
        self.wba_ = self.wba.clone().assemble(
            self.stride_selection_.filtered_stride_list_, sampling_rate_hz=sampling_rate_hz
        )

        self.per_stride_parameters_ = self.wba_.annotated_stride_list_
        self.per_wb_parameters_ = self._aggregate_per_wb(self.per_stride_parameters_, self.wba_.wb_meta_parameters_)
        if self.dmo_thresholds is None:
            self.per_wb_parameter_mask_ = None
        else:
            self.per_wb_parameter_mask_ = apply_thresholds(
                self.per_wb_parameters_,
                self.dmo_thresholds,
                cohort=datapoint.participant_metadata["cohort"],
                height_m=datapoint.participant_metadata["height_m"],
                measurement_condition=datapoint.recording_metadata["measurement_condition"],
            )

        self.dmo_aggregation_ = self.dmo_aggregation.clone().aggregate(
            self.per_wb_parameters_, wb_dmos_mask=self.per_wb_parameter_mask_
        )
        self.aggregated_parameters_ = self.dmo_aggregation_.aggregated_data_

        return self

    def _run_per_gs(
        self,
        gait_sequences: pd.DataFrame,
        imu_data: pd.DataFrame,
        sampling_rate_hz: float,
        participant_metadata: ParticipantMetadata,
    ) -> GsIterator:
        gs_iterator = GsIterator[FullPipelinePerGsResult]()
        # TODO: How to expose the individual algo instances of the algos that run in the loop?

        for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
            icd = self.initial_contact_detection.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
            lrc = self.laterality_classification.clone().predict(
                gs_data, icd.ic_list_, sampling_rate_hz=sampling_rate_hz
            )
            if self.turn_detection:
                r.ic_list = lrc.ic_lr_list_
                turn = self.turn_detection.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
                r.turn_list = turn.turn_list_

            refined_gs, refined_ic_list = refine_gs(r.ic_list)

            with gs_iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
                cad_r = None
                if self.cadence_calculation:
                    cad = self.cadence_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        sampling_rate_hz=sampling_rate_hz,
                        **participant_metadata,
                    )
                    cad_r = cad.cadence_per_sec_
                    rr.cadence_per_sec = cad_r
                sl_r = None
                if self.stride_length_calculation:
                    sl = self.stride_length_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        sampling_rate_hz=sampling_rate_hz,
                        **participant_metadata,
                    )
                    sl_r = sl.stride_length_per_sec_
                    rr.stride_length_per_sec = sl.stride_length_per_sec_
                if self.walking_speed_calculation:
                    ws = self.walking_speed_calculation.clone().calculate(
                        refined_gs_data,
                        initial_contacts=refined_ic_list,
                        cadence_per_sec=cad_r,
                        stride_length_per_sec=sl_r,
                        sampling_rate_hz=sampling_rate_hz,
                        **participant_metadata,
                    )
                    rr.walking_speed_per_sec = ws.walking_speed_per_sec_

        return gs_iterator

    def _sec_to_stride(
        self, sec_level_paras: pd.DataFrame, lr_ic_list: pd.DataFrame, sampling_rate_hz: float
    ) -> pd.DataFrame:
        stride_list = (
            lr_ic_list.groupby("gs_id", group_keys=False)
            .apply(strides_list_from_ic_lr_list)
            .assign(stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz)
        )

        stride_list = create_multi_groupby(
            stride_list,
            sec_level_paras,
            "gs_id",
            group_keys=False,
        ).apply(naive_sec_paras_to_regions, sampling_rate_hz=sampling_rate_hz)
        return stride_list

    def _aggregate_per_wb(self, per_stride_parameters: pd.DataFrame, wb_meta_parameters: pd.DataFrame) -> pd.DataFrame:
        # TODO: Make a class constant
        params_to_aggregate = [
            "stride_duration_s",
            "cadence_spm",
            "stride_length_m",
            "walking_speed_mps",
        ]
        return pd.concat(
            [
                wb_meta_parameters,
                per_stride_parameters.reindex(columns=params_to_aggregate)
                .groupby(["wb_id"])
                # TODO: Decide if we should use mean or trim_mean here!
                .mean(),
            ],
            axis=1,
        )


class MobilisedPipelineHealthy(BaseMobilisedPipeline):
    """Official Mobilise-D pipeline for healthy and mildly impaired gait (aka P1 pipeline).

    .. note:: When using this pipeline with its default parameters with healthy participants or participants with COPD
              or congestive heart failure, the use of the name "the Mobilise-D pipeline" is recommended.

    Based on the benchmarking performed in [1]_, the algorithms selected for this pipeline are the optimal choice for
    healthy and mildly impaired gait or more specifically for the cohorts "HA", "COPD", "CHF" within the Mobilise-D
    validation study.
    Performance metrics for the original implementation of this pipeline can be found in [2]_.
    This pipeline is referred to as the "P1" pipeline in the context of this and other publications.

    For detailed steps on how this pipeline works, check the Notes section and the dedicated examples.

    Parameters
    ----------
    %(core_parameters)s
    %(turn_detection)s
    %(wba_parameters)s
    %(aggregation_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(primary_results)s
    %(intermediate_results)s
    %(debug_results)s

    Notes
    -----
    %(step_by_step)s

    .. [1] Micó-Amigo, M., Bonci, T., Paraschiv-Ionescu, A. et al. Assessing real-world gait with digital technology?
       Validation, insights and recommendations from the Mobilise-D consortium. J NeuroEngineering Rehabil 20, 78
       (2023). https://doi.org/10.1186/s12984-023-01198-5
    .. [2] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
           multiple conditions with a wearable device. Sci Rep 14, 1754 (2024).
           https://doi.org/10.1038/s41598-024-51766-5

    See Also
    --------
    mobgap.pipeline.BaseMobilisedPipeline : A version of the pipeline without any default algorithms or parameters.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

    @set_defaults(**{k: cf(v) for k, v in BaseMobilisedPipeline.PredefinedParameters.normal_walking.items()})
    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsDetector,
        initial_contact_detection: BaseIcDetector,
        laterality_classification: BaseLRClassifier,
        cadence_calculation: Optional[BaseCadCalculator],
        stride_length_calculation: Optional[BaseSlCalculator],
        walking_speed_calculation: Optional[BaseWsCalculator],
        turn_detection: Optional[BaseTurnDetector],
        stride_selection: StrideSelection,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: BaseAggregator,
    ) -> None:
        super().__init__(
            gait_sequence_detection=gait_sequence_detection,
            initial_contact_detection=initial_contact_detection,
            laterality_classification=laterality_classification,
            cadence_calculation=cadence_calculation,
            stride_length_calculation=stride_length_calculation,
            walking_speed_calculation=walking_speed_calculation,
            turn_detection=turn_detection,
            stride_selection=stride_selection,
            wba=wba,
            dmo_thresholds=dmo_thresholds,
            dmo_aggregation=dmo_aggregation,
        )


class MobilisedPipelineImpaired(BaseMobilisedPipeline):
    """Official Mobilise-D pipeline for impaired gait (aka P2 pipeline).

    .. note:: When using this pipeline with its default parameters with participants with MS, PD, PFF, the use of the
              name "the Mobilise-D pipeline" is recommended.

    Based on the benchmarking performed in [1]_, the algorithms selected for this pipeline are the optimal choice for
    healthy and mildly impaired gait or more specifically for the cohorts "PD", "MS", "PFF" within the Mobilise-D
    validation study.
    Performance metrics for the original implementation of this pipeline can be found in [2]_.
    This pipeline is referred to as the "P1" pipeline in the context of this and other publications.

    For detailed steps on how this pipeline works, check the Notes section and the dedicated examples.

    Parameters
    ----------
    %(core_parameters)s
    %(turn_detection)s
    %(wba_parameters)s
    %(aggregation_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(primary_results)s
    %(intermediate_results)s
    %(debug_results)s

    Notes
    -----
    %(step_by_step)s

    .. [1] Micó-Amigo, M., Bonci, T., Paraschiv-Ionescu, A. et al. Assessing real-world gait with digital technology?
           Validation, insights and recommendations from the Mobilise-D consortium. J NeuroEngineering Rehabil 20, 78
           (2023). https://doi.org/10.1186/s12984-023-01198-5
    .. [2] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
           multiple conditions with a wearable device. Sci Rep 14, 1754 (2024).
           https://doi.org/10.1038/s41598-024-51766-5

    See Also
    --------
    mobgap.pipeline.BaseMobilisedPipeline : A version of the pipeline without any default algorithms or parameters.
    mobgap.pipeline.MobilisedPipelineImpaired : A predefined pipeline for impaired walking.

    """

    @set_defaults(**{k: cf(v) for k, v in BaseMobilisedPipeline.PredefinedParameters.impaired_walking.items()})
    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsDetector,
        initial_contact_detection: BaseIcDetector,
        laterality_classification: BaseLRClassifier,
        cadence_calculation: Optional[BaseCadCalculator],
        stride_length_calculation: Optional[BaseSlCalculator],
        walking_speed_calculation: Optional[BaseWsCalculator],
        turn_detection: Optional[BaseTurnDetector],
        stride_selection: StrideSelection,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: BaseAggregator,
    ) -> None:
        super().__init__(
            gait_sequence_detection=gait_sequence_detection,
            initial_contact_detection=initial_contact_detection,
            laterality_classification=laterality_classification,
            cadence_calculation=cadence_calculation,
            stride_length_calculation=stride_length_calculation,
            walking_speed_calculation=walking_speed_calculation,
            turn_detection=turn_detection,
            stride_selection=stride_selection,
            wba=wba,
            dmo_thresholds=dmo_thresholds,
            dmo_aggregation=dmo_aggregation,
        )
