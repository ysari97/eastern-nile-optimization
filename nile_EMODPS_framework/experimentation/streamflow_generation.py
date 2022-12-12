import numpy as np
import pandas as pd
import os


def cholesky_extended(input_matrix):
    from scipy.linalg import cholesky, eig
    input_matrix = np.nan_to_num(input_matrix, nan=1e-6)

    try:
        output = cholesky(input_matrix)
    except np.linalg.LinAlgError:
        print("Just try except")
        # For now, ignore correlations and make sure that streamflows are generated
        # from historical data, properly!
        return np.identity(np.shape(input_matrix)[0])

    #     k = min(min(np.real(eig(input_matrix)[0])), -1 * np.spacing(1))
    #     input_matrix = input_matrix - k * np.identity(np.shape(input_matrix)[0])
    #     input_matrix = input_matrix / input_matrix[0, 0]
    #     output = cholesky_extended(input_matrix)

    return output


def monthly_generator(
    historical_monthly_series,
    synthetic_years,
    random_seed,
    mean_adjustment=1,
    sd_adjustment=1,
):
    # This function receives the historical monthly series as an iterable (list,
    # 1D np.array) and the number of years for which synthetic monthly streamflow
    # data is to be generated

    # Since a shifting of the synthetic flows is needed to not miss year-to-year
    # correlation, we should start by increasing the synthetic years by one to end
    # up with the desired number of years
    synthetic_years += 1

    # First convert the historical data into a 2D matrix where rows correspond to
    # years and each column is a month (0-January, 11-December):
    hist_years = int(len(historical_monthly_series) / 12)  # Nr of years in input
    historical_matrix = np.reshape(historical_monthly_series, (-1, 12))

    # Following the method described by Quinn, we sample integers for each cell in
    # our (prospective) synthetic data. This part is needed to preserve spatial
    # correlation i.e. to draw the same historical year's observation for each of
    # the spatially correlated sites:
    np.random.seed(random_seed)
    random_int_matrix = np.random.randint(
        low=0, high=hist_years, size=(synthetic_years, 12)
    )

    # Making a log transform due to the log-normal distribution assumption for
    # streamflows:
    log_historical = np.log(historical_matrix)

    # Mean and std parameters of each month's distribution:
    monthly_mean_vector = np.mean(log_historical, axis=0)
    monthly_std_vector = np.std(log_historical, axis=0)

    # Standardization of the historical logged matrix to standard normal
    # distribution (Z(0,1)):
    z_historical = (log_historical - monthly_mean_vector) / monthly_std_vector

    # We get another (shifted) version of this Z matrix for year-to-year
    # correlation. Get rid of first and last 6 months of the dataset:
    z_shifted = np.reshape(((np.ndarray.flatten(z_historical))[6:-6]), (-1, 12))

    # For temporal correlation adjustment, calculate upper cholesky matrices:
    u_cholesky = cholesky_extended(np.corrcoef(z_historical, rowvar=False))
    u_cholesky_shifted = cholesky_extended(np.corrcoef(z_shifted, rowvar=False))

    # Now we generate our synthetic Z by adhering to the random_int_matrix:
    uncorr_synthetic = np.zeros(shape=(synthetic_years, 12))
    for year in range(synthetic_years):
        for month in range(12):
            uncorr_synthetic[year, month] = z_historical[
                random_int_matrix[year, month], month
            ]
    # Get the shifted version as usual:
    uncorr_shifted = np.reshape(
        ((np.ndarray.flatten(uncorr_synthetic))[6:-6]), (-1, 12)
    )

    # Matrix multiplication with the upper triangular matrix from Cholesky
    # decomposition gives the correlated matrices:
    corr_z_synthetic = np.matmul(uncorr_synthetic, u_cholesky)
    corr_z_shifted = np.matmul(uncorr_shifted, u_cholesky_shifted)

    # Concatenating the last 6 columns of the shifted matrix (January to June)
    # starting from row 1 with the last 6 months of the original matrix (July to
    # December) starting from row 2 gives the fully correlated final version:
    final_log = np.concatenate(
        (corr_z_shifted[:, 6:], corr_z_synthetic[1:, 6:]), axis=1
    )

    # Before backtransforming the log-normal series into their true scale, do mean and sd
    # adjustments if there is any. If both are set to 1, below functions do not change their
    # values at all
    common_ln_term = np.log(
        (sd_adjustment / mean_adjustment) ** 2
        * (np.exp(np.power(monthly_std_vector, 2)) - 1)
        + 1
    )
    monthly_mean_vector = (
        np.log(mean_adjustment)
        + monthly_mean_vector
        + (np.power(monthly_std_vector, 2) * 0.5)
        - 0.5 * common_ln_term
    )
    monthly_std_vector = np.sqrt(common_ln_term)

    # Back transform the log-normal standardized flow values:
    generated_matrix = np.exp((final_log * monthly_std_vector) + monthly_mean_vector)
    flat_streamflows = np.ndarray.flatten(generated_matrix)
    np.place(flat_streamflows, flat_streamflows < 0.001, 0)

    return list(flat_streamflows)


def multi_site_nile_generator(
    correlated_catchment_groups, nr_of_years, random_seeds, realization_id, scenario
):

    all_catchments_hydrology = dict()

    for i, catchment_group in enumerate(correlated_catchment_groups):
        for catchment in catchment_group:
            historical_time_stamped = read_historical_series(catchment)
            # leap_cleaned_hist = clean_up_leap_years(historical_time_stamped)
            historical_vector = np.array(historical_time_stamped.iloc[:, -1])
            synthetic_streamflow = monthly_generator(
                historical_vector, nr_of_years, random_seeds[i]
            )
            all_catchments_hydrology[catchment] = synthetic_streamflow

    directory = "../synthetic_hydrology/"
    full_df = pd.DataFrame.from_dict(all_catchments_hydrology)
    full_df.to_csv(f"{directory}{scenario}/{realization_id}.csv", index=False)


def read_historical_series(catchment_name):
    directory = "../historical_hydrology/"
    df = pd.read_csv(f"{directory}{catchment_name}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    monthly_aggregated = df.groupby([pd.Grouper(key="Date", freq="1M")]).sum()
    monthly_aggregated = monthly_aggregated.reset_index()

    return monthly_aggregated


def clean_up_leap_years(monthly_df):
    idx_no_leap = [np.where(monthly_df["Date"].dt.is_leap_year == False)][0][0]

    return np.array(monthly_df.iloc[:, -1])


if __name__ == "__main__":

    catchments_1 = ["BlueNile", "WhiteNile", "Atbara"]
    catchments_2 = ["Rahad", "Dinder"]

    for realization in range(30):
        multi_site_nile_generator([catchments_1, catchments_2], 20, [realization, 100 + realization], realization,
                                  "baseline")
