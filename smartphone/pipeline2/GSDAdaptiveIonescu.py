# %%
import json
import os 
import numpy as np
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.gsd import GsdAdaptiveIonescu

# Define subject ID and data path
subject_id = "005"
data_path = f'C:/Users/giorg/OneDrive - Politecnico di Torino/Giorgio Trentadue/Acquisizioni/{subject_id}/In Lab/Results final/'

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=None
)

# Initialize the JSON structure
gsdb_output = {"GSDB_Output": {}}

# Helper function to plot results (can be ignored for the purpose of saving JSON)
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
            ax.axvspan(gsd.start, gsd.end, label=props.pop("label", None), **props)

    ax.legend()
    return fig, ax

# Helper function to convert numpy types to native Python types
def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(elem) for elem in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Process the trials from the 4th to the last one
for trial in mobDataset[3:]:
    short_trial = trial
    imu_data = short_trial.data_ss
    reference_data = short_trial.reference_parameters_.wb_list
    gs_output = GsdAdaptiveIonescu().detect(imu_data, sampling_rate_hz=short_trial.sampling_rate_hz)

    # Extract time_measure, test, and trial values
    params = trial.get_params()
    subset_index = params['subset_index']
    time_measure = subset_index.iloc[0]['time_measure']
    test = subset_index.iloc[0]['test']
    trial_name = subset_index.iloc[0]['trial']

    # Get the start and end time from gs_output
    gs_start = gs_output.gs_list_["start"].values[0]  # assuming a single row for simplicity
    gs_end = gs_output.gs_list_["end"].values[0]  # assuming a single row for simplicity
    fs = 100  # Replace with actual sampling rate if available

    # Nested dictionary creation
    if time_measure not in gsdb_output["GSDB_Output"]:
        gsdb_output["GSDB_Output"][time_measure] = {}

    if test not in gsdb_output["GSDB_Output"][time_measure]:
        gsdb_output["GSDB_Output"][time_measure][test] = {}

    if trial_name not in gsdb_output["GSDB_Output"][time_measure][test]:
        gsdb_output["GSDB_Output"][time_measure][test][trial_name] = {}

    gsdb_output["GSDB_Output"][time_measure][test][trial_name]["SU"] = {
        "LowerBack": {
            "GSD": {
                "Start": gs_start,
                "End": gs_end,
                "fs": fs
            }
        }
    }

    print("Reference Parameters:\n\n", reference_data)
    print("\nPython Output:\n\n", gs_output.gs_list_)

    fig, ax = plot_gsd_outputs(
        short_trial.data_ss,
        reference=reference_data,
        python=gs_output.gs_list_,
    )
    fig.show()

# Convert gsdb_output to native Python types
gsdb_output_native = convert_to_native_types(gsdb_output)

"""# Determine the current script directory
current_directory = os.path.dirname(os.path.abspath(__file__))

 # Save the results to a JSON file in the current directory
json_file_path = os.path.join(current_directory, "gsdb_output.json")
with open(json_file_path, "w") as json_file:
    json.dump(gsdb_output_native, json_file, indent=4)

print(f"Data saved to {json_file_path}") """


# Save the results to a JSON file
json_file_path = f"{data_path}gsdb_output.json"
with open(json_file_path, "w") as json_file:
    json.dump(gsdb_output_native, json_file, indent=4)

print(f"Data saved to {json_file_path}")

# %%
