"""
Shin Algo
=========

This example shows how to use the improved Shin algorithm and some examples on how the results compare to the original
matlab implementation.

"""

import pandas as pd
from matplotlib import pyplot as plt

from mobgap.cad import CadFromIc
from mobgap.data import LabExampleDataset
from mobgap.icd import IcdShinImproved

# %%
# Loading data
# ------------
# .. note:: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP output for initial contacts ("ic") as ground truth.

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")

single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data_ss
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.ic_list

reference_wbs
# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
from mobgap.pipeline import GsIterator

iterator = GsIterator()

for gs_id, (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    icd = IcdShinImproved()
    icd.detect(data, sampling_rate_hz=sampling_rate_hz)
    # TODO: Need a version of IC-list that has the "refined GS ID in the index"
    result.ic_list = (
        icd.ic_list_.assign(refined_gs_id=0)
        .set_index("refined_gs_id", append=True)
        .reorder_levels(["refined_gs_id", "step_id"])
    )

    refined_gs_list = pd.DataFrame({"start": [5], "end": [gs.end - gs.start - 5]}).rename_axis(index="gs_id")

    for rgs_id, (refined_gs, refined_data), refined_result in iterator.iterate_subregions(refined_gs_list):
        refined_result.cad_per_sec = (
            CadFromIc()
            .calculate(refined_data, result.ic_list.loc[rgs_id], sampling_rate_hz=sampling_rate_hz)
            .cad_per_sec_
        )

    result.gait_speed = pd.DataFrame()


detected_ics = iterator.results_.ic_list
detected_ics
# %%
# Matlab Outputs
# --------------
# To check if the algorithm was implemented correctly, we compare the results to the matlab implementation.
import json

from mobgap import PACKAGE_ROOT


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PACKAGE_ROOT.parent
        / f"example_data/original_results/icd_shin_improved/lab/{p.cohort}/{p.participant_id}/SD_Output.json"
    ).open() as f:
        original_results = json.load(f)["SD_Output"][p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["SD"]

    if not isinstance(original_results, list):
        original_results = [original_results]

    ics = {}
    for i, gs in enumerate(original_results, start=1):
        ics[i] = pd.DataFrame({"ic": gs["IC"]}).rename_axis(index="step_id")

    return (pd.concat(ics, names=["wb_id", ics[1].index.name]) * datapoint.sampling_rate_hz).astype("int64")


detected_ics_matlab = load_matlab_output(single_test)
detected_ics_matlab
# %%
# Plotting the results
# --------------------
# With that we can compare the python, matlab and ground truth results.
# We zoom in into one of the gait sequences to better see the output.
#
# We can make a couple of main observations:
#
# 1. The python version finds the same ICs as the matlab version, but wil a small shift to the left (around 5-10
#    samples/50-100 ms).
#    This is likely due to some differences in the downsampling process.
# 2. Compared to the ground truth reference, both versions detect the IC too early most of the time.
# 3. Both algorithms can not detect the first IC of the gait sequence.
#    However, this is expected, as per definition, this first IC marks the start of the WB in the reference system.
#    Hence, there are no samples before that point the algorithm can use to detect the IC.

imu_data.reset_index(drop=True).plot(y="acc_x")

plt.plot(ref_ics["ic"], imu_data["acc_x"].iloc[ref_ics["ic"]], "o", label="ref")
plt.plot(detected_ics["ic"], imu_data["acc_x"].iloc[detected_ics["ic"]], "x", label="shin_algo_py")
plt.plot(detected_ics_matlab["ic"], imu_data["acc_x"].iloc[detected_ics_matlab["ic"]], "+", label="shin_algo_matlab")
plt.xlim(reference_wbs.iloc[2]["start"] - 50, reference_wbs.iloc[2]["end"] + 50)
plt.legend()
plt.show()

# %%
# Evaluation of the algorithm against a reference
# --------------------------------------------------
# To quantify how the Python output compares to the reference labels, we are providing a range of evaluation functions.
# See the :ref:`example on ICD evaluation <icd_evaluation>` for more details.
