# Import modules
from pprint import pprint
from mobgap.data import TVSLabDataset, TVSFreeLivingDataset
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired, MobilisedPipelineUniversal
import pandas as pd
from tqdm.auto import tqdm

# Define path to TVS dataset
dataset_path = "C:/Users/paolo/OneDrive - Politecnico di Torino/prova stefano/Projects/Mobilise-D/Data/tvs_dataset/tvs_dataset"
# Check data validity
# data_quality_fl = TVSFreeLivingDataset(dataset_path+"/").data_quality["INDIP"]
data_quality_lab = TVSLabDataset(dataset_path+"/").data_quality["INDIP"]
valid_subjects = data_quality_lab[data_quality_lab > 1]
multiIndexx = valid_subjects.index
valid_IDs = list(multiIndexx.get_level_values(1))
# Define laboratory dataset object with the following settings:
# - Use INDIP data as reference
# - Skip the trial entirely when the reference data is not available
labdata_all = TVSLabDataset(dataset_path, reference_system="INDIP", missing_reference_error_type="skip")
# A subset of the TVS dataset made of one HA subject (1091) and one MS subject (2022)
# labdata = labdata_all.get_subset(cohort = ["HA", "MS"], participant_id = ["1091", "2022"])
labdata = labdata_all.get_subset(participant_id = valid_IDs)

# Define a universal pipeline object including the two pipelines (healthy and impaired)
meta_pipeline = MobilisedPipelineUniversal(
    pipelines=[
        ("healthy", MobilisedPipelineHealthy()),
        ("impaired", MobilisedPipelineImpaired()),
    ]
)

per_wb_paras = {}
aggregated_paras = {}
ref_parameters = {}
for trial in tqdm(labdata): # tqdm -> adds progress bar
    ref_parameters[trial.group_label] = trial.reference_parameters_.wb_list # reference parameters of each wb in the current trial
    pipe = meta_pipeline.clone().safe_run(trial) # run pipeline on current trial "safely"
    if not (per_wb := pipe.per_wb_parameters_).empty: # if pipe.per_wb_parameters_ is not empty
        per_wb_paras[trial.group_label] = per_wb # pipe.per_wb_parameters_ is stored in dict per_wb_paras
    if not (agg := pipe.aggregated_parameters_).empty: # if pipe.aggregated_parameters_ is not empty
        aggregated_paras[trial.group_label] = agg # pipe.aggregated_parameters_ is stored in dict aggregated_paras

# concatenate into a single dataframe
per_wb_paras = pd.concat(per_wb_paras)
ref_parameters = pd.concat(ref_parameters)
# rename index of dataframes
per_wb_paras.index.names= ["cohort", "participant_id", "tm", "test_id", "trial_id", "wb_id"]
per_wb_paras.index.names
ref_parameters.index.names= ["cohort", "participant_id", "tm", "test_id", "trial_id", "wb_id"]
ref_parameters.index.names

# select only walking speed
detected_dmo = per_wb_paras[['start','end','walking_speed_mps']]
reference_dmo = ref_parameters[['start','end','avg_walking_speed_mps']]
# rename columns
detected_dmo.columns = ["start", "end", "speed_mps"]
reference_dmo.columns = ["start", "end", "speed_mps"]
# combine detected and reference DMOs into a single dataframe
combined_dmos = (
    pd.concat(
        [detected_dmo, reference_dmo], keys=["detected", "reference"], axis=1
    )
    .reorder_levels((1, 0), axis=1)
    .sort_index(axis=1)
)
combined_dmos.head()

# true positive evaluation
from mobgap.utils.df_operations import create_multi_groupby

# group detected and reference DMOs of each WB according to Time Measure
per_trial_participant_day_grouper = create_multi_groupby(
    detected_dmo,
    reference_dmo,
    groupby=["cohort", "participant_id", "tm", "test_id", "trial_id"],
)

from mobgap.pipeline.evaluation import categorize_intervals
# categorize intervals as false positives, false negatives, or true positives
wb_tp_fp_fn = per_trial_participant_day_grouper.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=0.8,
        multiindex_warning=False,
    )
)
wb_tp_fp_fn # this contains the matched indices of the WBs, plus the matching label (fp, tp, fn). nan are placed where fp or fn occur.
tp = len(wb_tp_fp_fn[wb_tp_fp_fn['match_type'] == 'tp'])
fp = len(wb_tp_fp_fn[wb_tp_fp_fn['match_type'] == 'fp'])
fn = len(wb_tp_fp_fn[wb_tp_fp_fn['match_type'] == 'fn'])
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1_score = 2*recall*precision/(precision+recall)

from mobgap.pipeline.evaluation import get_matching_intervals
# take only true positive WBs
wb_matches = get_matching_intervals(
    metrics_detected=detected_dmo,
    metrics_reference=reference_dmo,
    matches=wb_tp_fp_fn,
)
wb_matches.T
wb_matches = wb_matches.dropna()

from mobgap.pipeline.evaluation import ErrorTransformFuncs as E

custom_errors = [
    ("speed_mps", [E.error, E.rel_error,  E.abs_error, E.abs_rel_error])
]

from mobgap.utils.df_operations import apply_transformations

custom_wb_errors = apply_transformations(wb_matches, custom_errors)
custom_wb_errors.T

wb_matches_with_errors = pd.concat([wb_matches, custom_wb_errors], axis=1)
wb_matches_with_errors.T
# Save and read
wb_matches_with_errors.to_json('wb_matches_with_errors_TPE_InLab.json')
wb_matches_with_errors2 = pd.read_json('wb_matches_with_errors_TPE_InLab.json')

# Filter cohorts
import numpy as np
# HA
HA = wb_matches_with_errors[np.in1d(wb_matches_with_errors.index.get_level_values(0), ['HA'])]
# PFF
PFF = wb_matches_with_errors[np.in1d(wb_matches_with_errors.index.get_level_values(0), ['PFF'])]
# PD
PD = wb_matches_with_errors[np.in1d(wb_matches_with_errors.index.get_level_values(0), ['PD'])]
# MS
MS = wb_matches_with_errors[np.in1d(wb_matches_with_errors.index.get_level_values(0), ['MS'])]
# CHF
CHF = wb_matches_with_errors[np.in1d(wb_matches_with_errors.index.get_level_values(0), ['CHF'])]
# COPD
COPD = wb_matches_with_errors[np.in1d(wb_matches_with_errors.index.get_level_values(0), ['COPD'])]


metrics = [
    "speed_mps"
]
aggregations_simple = [
    ((m, o), ["mean"]) # instead of ["mean","std"]
    for m in metrics
    for o in ["detected", "reference", "error", "rel_error", "abs_error", "abs_rel_error"]
]
pprint(aggregations_simple)

from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import CustomOperation

aggregations_custom1 = [
    CustomOperation(identifier=m, function=A.icc, column_name=(m, "all"))
    for m in metrics
]
pprint(aggregations_custom1)

aggregations_custom2 = [
    CustomOperation(identifier=(m,o), function=A.loa, column_name=(m, o))
    for m in metrics
    for o in ["error", "rel_error"]
]
pprint(aggregations_custom2)

aggregations_custom3 = [
    CustomOperation(identifier=(m,o), function=A.quantiles, column_name=(m,o))
    for m in metrics
    for o in ["detected", "reference","abs_error", "abs_rel_error"]
]
pprint(aggregations_custom3)
# divide into cohorts
sub_df = wb_matches_with_errors.loc[:, "speed_mps"]

from mobgap.utils.df_operations import apply_aggregations

aggregations = aggregations_simple + aggregations_custom1 + aggregations_custom2+ aggregations_custom3
agg_results = (
    apply_aggregations(wb_matches_with_errors, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# HA
agg_results_HA = (
    apply_aggregations(HA, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# PD
agg_results_PD = (
    apply_aggregations(PD, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# CHF
agg_results_CHF = (
    apply_aggregations(CHF, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# MS
agg_results_MS = (
    apply_aggregations(MS, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# PFF
agg_results_PFF = (
    apply_aggregations(PFF, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# COPD
agg_results_COPD = (
    apply_aggregations(COPD, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
agg_results
# Combined evaluation
# group detected and reference DMOs of each WB according to Test and Trial

# Assuming the dataframe is already loaded and called df
# First, let's group by the trial levels and compute the median for the speed_mps columns

# Define the levels of the index related to the trial
test_matches = (
    combined_dmos.groupby(
        level=["cohort", "participant_id", "tm", "test_id", "trial_id"], axis=0
    )
    .median()
    .dropna()
)
test_matches.T

test_matches_errors = apply_transformations(test_matches, custom_errors)
test_matches_errors.T

test_matches_with_errors = pd.concat([test_matches, test_matches_errors], axis=1)
test_matches_with_errors.T
# HA
HA_ca = test_matches_with_errors[np.in1d(test_matches_with_errors.index.get_level_values(0), ['HA'])]
# PFF
PFF_ca = test_matches_with_errors[np.in1d(test_matches_with_errors.index.get_level_values(0), ['PFF'])]
# PD
PD_ca = test_matches_with_errors[np.in1d(test_matches_with_errors.index.get_level_values(0), ['PD'])]
# MS
MS_ca = test_matches_with_errors[np.in1d(test_matches_with_errors.index.get_level_values(0), ['MS'])]
# CHF
CHF_ca = test_matches_with_errors[np.in1d(test_matches_with_errors.index.get_level_values(0), ['CHF'])]
# COPD
COPD_ca = test_matches_with_errors[np.in1d(test_matches_with_errors.index.get_level_values(0), ['COPD'])]

agg_results_ca = (
    apply_aggregations(test_matches_with_errors, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# HA
agg_results_HA_ca = (
    apply_aggregations(HA_ca, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# PD
agg_results_PD_ca = (
    apply_aggregations(PD_ca, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# CHF
agg_results_CHF_ca = (
    apply_aggregations(CHF_ca, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# MS
agg_results_MS_ca = (
    apply_aggregations(MS_ca, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# PFF
agg_results_PFF_ca = (
    apply_aggregations(PFF_ca, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
# COPD
agg_results_COPD_ca = (
    apply_aggregations(COPD_ca, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
agg_results_ca
# Select 'start/detected' and 'end/detected' rows
walking_time = wb_matches_with_errors2.iloc[:,0] - wb_matches_with_errors2.iloc[:,4]
tot_walking_time = walking_time.sum()/360000
# number of strides
index_ = wb_matches.index
n_strides_tot = per_wb_paras["n_strides"].loc[index_].sum()