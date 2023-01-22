# Simultaneous sampling of uncertainties and simulation using EMA Workbench

from ema_workbench import RealParameter, ScalarOutcome, Model, Policy
from ema_workbench import MultiprocessingEvaluator, ema_logging
from streamflow_generation import read_historical_series, monthly_generator
import sys
import os
import numpy as np
import pandas as pd
import random

module_path = os.path.abspath(os.path.join("../../model"))
if module_path not in sys.path:
    sys.path.append(module_path)
from model_nile import ModelNile

nile_model = ModelNile()


# Generating one realization and assigning it to the model:
def model_with_uncertainty(mean_factor=1, stdev_factor=1, **policy_input):

    catchments_1 = ["BlueNile", "WhiteNile", "Atbara"]
    catchments_2 = ["Rahad", "Dinder"]

    for realization in range(1):

        all_catchments_hydrology = dict()
        correlated_catchment_groups = [catchments_1, catchments_2]
        nr_of_years = 20
        random_int = np.random.randint(1000)
        random_seeds = [random_int, random_int + 100]

        for i, catchment_group in enumerate(correlated_catchment_groups):
            for catchment in catchment_group:
                historical_time_stamped = read_historical_series(catchment)
                # leap_cleaned_hist = clean_up_leap_years(historical_time_stamped)
                historical_vector = np.array(historical_time_stamped.iloc[:, -1])
                synthetic_streamflow = monthly_generator(
                    historical_vector,
                    nr_of_years,
                    random_seeds[i],
                    mean_adjustment=mean_factor,
                    sd_adjustment=stdev_factor,
                )
                all_catchments_hydrology[catchment] = synthetic_streamflow

        full_df = pd.DataFrame.from_dict(all_catchments_hydrology)
        nile_model.synthetic_hydrology = [full_df]
        egypt_def, min_HAD, sudan_def, ethiopia_hydro = nile_model.evaluate(
            list(policy_input.values())
        )

        return egypt_def, min_HAD, sudan_def, ethiopia_hydro

    if __name__ == "__main__":
        ema_logging.log_to_stderr(ema_logging.INFO)

        output_directory = "../outputs/"

        em_model = Model("NileProblem", function=model_with_uncertainty)
        em_model.uncertainties = [
            RealParameter("mean_factor", 0.7, 1.3),
            RealParameter("stdev_factor", 1, 1.5),
        ]

        total_parameter_count = (
            nile_model.overarching_policy.get_total_parameter_count()
        )
        release_parameter_count = nile_model.overarching_policy.functions[
            "release"
        ].get_free_parameter_number()
        n_inputs_release = nile_model.overarching_policy.functions["release"].n_inputs
        n_outputs_release = nile_model.overarching_policy.functions["release"].n_outputs
        RBF_count = nile_model.overarching_policy.functions["release"].RBF_count
        p_per_RBF = 2 * n_inputs_release + n_outputs_release

        # Since we first introduce the release policy to the model, first parameters
        # belong to the RBFs. Only then, the parameters of hedging functions.
        # Let's first put the levers for the release policy.

        lever_list = list()
        for i in range(release_parameter_count):
            modulus = (i - n_outputs_release) % p_per_RBF
            if (
                (i >= n_outputs_release)
                and (modulus < (p_per_RBF - n_outputs_release))
                and (modulus % 2 == 0)
            ):  # centers:
                lever_list.append(RealParameter(f"v{i}", -1, 1))
            else:  # linear parameters for each release, radii and weights of RBFs:
                lever_list.append(RealParameter(f"v{i}", 0, 1))

        for j in range(release_parameter_count, total_parameter_count):
            lever_list.append(RealParameter(f"v{i}", 0, 1))

        em_model.levers = lever_list
        # specify outcomes
        em_model.outcomes = [
            ScalarOutcome("egypt_def", ScalarOutcome.MINIMIZE),
            ScalarOutcome("min_HAD", ScalarOutcome.MAXIMIZE),
            ScalarOutcome("sudan_def", ScalarOutcome.MINIMIZE),
            ScalarOutcome("ethiopia_hydro", ScalarOutcome.MAXIMIZE),
        ]

        n_scenarios = 5000
        policy_df = pd.read_csv(f"{output_directory}policies_exploration.csv")
        my_policies = [
            Policy(policy_df.loc[i, "name"], **(policy_df.iloc[i, :-1].to_dict()))
            for i in policy_df.index
        ]

        random.seed(123)
        np.random.seed(123)

        with MultiprocessingEvaluator(em_model) as evaluator:
            experiments, outcomes = evaluator.perform_experiments(
                n_scenarios, my_policies
            )

        outcomes = pd.DataFrame.from_dict(outcomes)
        experiments.to_csv(f"{output_directory}experiments_exploration.csv")
        outcomes.to_csv(f"{output_directory}outcomes_exploration.csv")
