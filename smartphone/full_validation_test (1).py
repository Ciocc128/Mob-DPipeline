#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Full DMO Analysis

# In[2]:


# Import modules
from pprint import pprint
from mobgap.data import TVSLabDataset, TVSFreeLivingDataset
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired, MobilisedPipelineUniversal
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Memory

# Define path to TVS dataset
# dataset_path = "C:/Users/paolo/OneDrive - Politecnico di Torino/prova stefano/Projects/Mobilise-D/Data/tvs_dataset/tvs_dataset"
dataset_path = "/home/arne/Documents/repos/private/mobilised_tvs_data/tvs_dataset/"


# In[3]:


labdata_all = TVSLabDataset(dataset_path, reference_system="INDIP", missing_reference_error_type="skip", memory=Memory("./.cache"))
valid_participants = labdata_all.data_quality.query("INDIP > 1 & SU > 0").reset_index()["participant_id"].to_list()
labdata = labdata_all.get_subset(participant_id=valid_participants)
labdata


# In[4]:


print("n_participants: ", len(labdata.index["participant_id"].unique()))  # N-pariticpnats different then in Paolos case? why? -> Filter SU > 0


# In[5]:


# # For testing, only n participants per cohort:
# labdata = labdata.get_subset(participant_id = labdata.index[["cohort", "participant_id"]].drop_duplicates().groupby("cohort").head(2)["participant_id"].to_list())
# labdata


# In[29]:


import joblib

# Define a universal pipeline object including the two pipelines (healthy and impaired)
meta_pipeline = MobilisedPipelineUniversal(
    pipelines=[
        ("healthy", MobilisedPipelineHealthy()),
        ("impaired", MobilisedPipelineImpaired()),
    ]
)


def run(datapoint):
    return (datapoint.group_label, datapoint.reference_parameters_.wb_list, meta_pipeline.clone().safe_run(datapoint).pipeline_.per_wb_parameters_)


loop = joblib.Parallel(n_jobs=4, return_as="generator")(joblib.delayed(run)(d) for d in labdata)

index_names = [*labdata.index.columns, "wb_id"]

detected_dmo = {}
reference_dmo = {}

for label, ref, res in tqdm(loop, total=len(labdata)):
    detected_dmo[label] = res
    reference_dmo[label] = ref

detected_dmo = pd.concat(detected_dmo, names=index_names).drop(columns="rule_obj")
reference_dmo = pd.concat(reference_dmo, names=index_names)
reference_dmo.columns = reference_dmo.columns.str.lstrip("avg_") 


# ## True positive analysis

# In[30]:


from mobgap.utils.df_operations import create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals

reference_dmo = reference_dmo.dropna()
detected_dmo = detected_dmo.dropna()


per_trial_groupby = create_multi_groupby(
    detected_dmo,
    reference_dmo,
    groupby=labdata.index.columns.to_list(),
)
wb_tp_fp_fn = per_trial_groupby.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=0.8,
        multiindex_warning=False,
    )
)
wb_matches = get_matching_intervals(
    metrics_detected=detected_dmo,
    metrics_reference=reference_dmo,
    matches=wb_tp_fp_fn,
)
wb_matches


# In[31]:


from mobgap.pipeline.evaluation import get_default_error_transformations
from mobgap.utils.df_operations import apply_transformations

wb_errors = apply_transformations(wb_matches, get_default_error_transformations(), missing_columns="ignore")
wb_matches_with_errors = pd.concat([wb_matches, wb_errors], axis=1).sort_index(axis=1)
wb_matches_with_errors


# In[32]:


# Aggregate everything
from mobgap.pipeline.evaluation import get_default_error_aggregations
from mobgap.utils.df_operations import apply_aggregations


# We apply the aggregations per cohort
final_agg_errors = (
    wb_matches_with_errors
    .groupby("cohort")
    .apply(lambda g: apply_aggregations(g, get_default_error_aggregations(), missing_columns="skip"))
    .rename_axis(columns=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"], axis=1)
    .sort_index(axis=1)
)
final_agg_errors.filter(like="walking_speed_mps").loc[["HA", "CHF", "COPD", "MS", "PD", "PFF"]]


# ## Aggregated Analysis

# In[33]:


combined_dmos = (
    pd.concat(
        [detected_dmo, reference_dmo], keys=["detected", "reference"], axis=1
    )
    .reorder_levels((1, 0), axis=1)
    .sort_index(axis=0)
    .infer_objects()
)
# We filter for columns that exist in both dataframes.
combined_dmos = combined_dmos[list(set(detected_dmo.columns).intersection(reference_dmo))].sort_index(axis=1)

combined_dmos


# In[34]:


per_trial_averages = (
    combined_dmos.select_dtypes(['number']).groupby(
        level=labdata.index.columns.to_list()
    )
    .median()
    .dropna()
)
per_trial_averages


# In[35]:


per_trail_errors = apply_transformations(per_trial_averages, get_default_error_transformations(), missing_columns="ignore")
per_trail_errors_with_raw = pd.concat([per_trial_averages, per_trail_errors], axis=1).sort_index(axis=1)
per_trail_errors_with_raw


# In[36]:


# We apply the aggregations per cohort
final_agg_errors_agg_analysis = (
    per_trail_errors_with_raw
    .groupby("cohort")
    .apply(lambda g: apply_aggregations(g, get_default_error_aggregations(), missing_columns="ignore"))
    .rename_axis(columns=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"], axis=1)
    .sort_index(axis=1)
)
final_agg_errors_agg_analysis.filter(like="walking_speed_mps").loc[["HA", "CHF", "COPD", "MS", "PD", "PFF"]]


# ## Testing with the old results

# In[37]:


from mobgap.utils.conversions import as_samples

def unify(df):
    return (
        df
        .rename(columns={
            "participant": "participant_id", 
            "Start": "start",
            "End": "end",
            "AverageStrideSpeed": "walking_speed_mps",
            "AverageStrideLength": "stride_length_m",
            "AverageCadence": "cadence_spm",
            "Duration": "duration_s"
        })
        .set_index(["cohort", "participant_id", "test", "trial", "wb_id"])
        .drop(["wb_type", "index", "test_pretty"], axis=1)
        .assign(start=lambda df_: as_samples(df_.start, 100).to_list(), end=lambda df_: as_samples(df_.end, 100).to_list())
    )


# In[38]:


old_reference = pd.read_csv("/home/arne/Documents/repos/work/mobilised/wba_paper/wba_analysis/experiments/full_pipeline_analysis/results/all_reference_data.csv").query("test != 'FreeLiving_Wild'").pipe(unify)
old_detected = pd.read_csv("/home/arne/Documents/repos/work/mobilised/wba_paper/wba_analysis/experiments/full_pipeline_analysis/results/all_single_sensor_data.csv").query("test != 'FreeLiving_Wild'").pipe(unify)


# In[39]:


old_detected


# In[40]:


old_reference


# In[41]:


per_trial_groupby = create_multi_groupby(
    old_detected,
    old_reference,
    groupby=["cohort", "participant_id", "test", "trial"],
)
wb_tp_fp_fn = per_trial_groupby.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=0.8,
        multiindex_warning=False,
    )
)
wb_matches = get_matching_intervals(
    metrics_detected=old_detected,
    metrics_reference=old_reference,
    matches=wb_tp_fp_fn,
)
wb_matches


# In[42]:


wb_errors = apply_transformations(wb_matches, get_default_error_transformations(), missing_columns="ignore")
wb_matches_with_errors = pd.concat([wb_matches, wb_errors], axis=1).sort_index(axis=1)
wb_matches_with_errors


# In[44]:


# We apply the aggregations per cohort
final_agg_errors = (
    wb_matches_with_errors
    .groupby("cohort")
    .apply(lambda g: apply_aggregations(g, get_default_error_aggregations(),  missing_columns="ignore"))
    .rename_axis(columns=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"], axis=1)
    .sort_index(axis=1)
)
final_agg_errors.filter(like="walking_speed_mps").loc[["HA", "CHF", "COPD", "MS", "PD", "PFF"]]


# ## Error Analysis
# 
# MS 3035 seems to show super large difference in N WBs for free living

# In[ ]:


old_reference_fl = pd.read_csv("/home/arne/Documents/repos/work/mobilised/wba_paper/wba_analysis/experiments/full_pipeline_analysis/results/all_reference_data.csv").query("test == 'FreeLiving_Wild'").pipe(unify)
old_detected_fl = pd.read_csv("/home/arne/Documents/repos/work/mobilised/wba_paper/wba_analysis/experiments/full_pipeline_analysis/results/all_single_sensor_data.csv").query("test == 'FreeLiving_Wild'").pipe(unify)
old_detected_fl


# In[ ]:


from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIonescu

fldata_all = TVSFreeLivingDataset(dataset_path, reference_system="INDIP", missing_reference_error_type="skip", memory=Memory("./.cache"))

test_p = fldata_all.get_subset(participant_id="1095")


# In[ ]:


pipe = MobilisedPipelineHealthy().run(test_p)


# In[ ]:


pipe.gait_sequence_detection_.gs_list_


# In[ ]:


pipe.per_wb_parameters_


# In[ ]:


test_p.reference_parameters_.wb_list


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def plot_gsd_outputs(data, **kwargs):
    fig, ax = plt.subplots()

    ax.plot(data["acc_x"].to_numpy(), label="acc_x")

    color_cycle = iter(plt.rcParams["axes.prop_cycle"])

    y_max = 1.1
    plot_props = [
        {
            "data": v,
            "label": k,
            "alpha": 0.2,
            "ymax": (y_max := y_max - 0.1),
            "color": next(color_cycle)["color"],
        }
        for k, v in kwargs.items()
    ]

    for props in plot_props:
        for gsd in props.pop("data").itertuples(index=False):
            ax.axvspan(
                gsd.start, gsd.end, label=props.pop("label", None), **props
            )

    ax.legend()
    return fig, ax


# In[ ]:


plot_gsd_outputs(test_p.data_ss, detected=pipe.per_wb_parameters_, reference=test_p.reference_parameters_.wb_list, old_detected_wbs=old_detected_fl.query("participant_id==1008"))


# In[ ]:




