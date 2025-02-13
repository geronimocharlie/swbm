from typing import List, Union
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import itertools
from tqdm import tqdm  # Import progress bar

class SimpleWaterBalanceModel:
    def __init__(
        self,
        cs: float = 420,  # Soil water holding capacity (mm)
        alpha: float = 4.0,  # Runoff function shape
        gamma: float = 0.5,  # ET function shape
        beta: float = 0.8,  # ET function maximum
        cs_init_coef: float = 0.9,  # Initial soil moisture coefficient
        countries: List[str] = ["Germany", "Sweden"],  # List of countries
        calibration_countries: List[str] = ["Germany_new", "Sweden_new", "Spain_new"], # List of callibration countries
        data_root_dir = 'data',
        verbose = False,
    ) -> None:
        self.cs = cs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.initial_soil_moisture = cs_init_coef * cs  # Initial soil moisture in mm
        self.initial_soil_moisture_volumetric = self.initial_soil_moisture / cs  # Volumetric for comparison
        self.rn_conversion_factor = 2.26  # Convert from W/m² to mm/day
        self.countries = countries
        self.calibration_countries = calibration_countries
        self.data_root_dir = data_root_dir
        self.data = [self.read_data(c) for c in countries]
        self.data_calibration = [self.read_data(c) for c in calibration_countries]
        self.verbose = verbose

    def read_data(self, country: str) -> pd.DataFrame:
        """
        Read input data for the given country.
        """
        try:
            data = pd.read_csv(f"{self.data_root_dir}/Data_swbm_{country}.csv")
            data["time"] = pd.to_datetime(data["time"])
            data["Year"] = data["time"].dt.year  # Extract year

            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for {country} not found.")


    # Runoff and ET functions
    def runoff_function(self, soil_moisture: float, precipitation: float) -> float:
        """
        Calculate runoff (in mm).
        """
        runoff = ((soil_moisture / self.cs) ** self.alpha) * precipitation
        return runoff

    def et_function(self, soil_moisture: float, net_radiation: float) -> float:
        """
        Calculate evapotranspiration (in mm/day).
        """
        net_radiation_mm = net_radiation / self.rn_conversion_factor  # Convert W/m² to mm/day
        et = self.beta * ((soil_moisture / self.cs) ** self.gamma) * net_radiation_mm
        return et

    def forward(self, soil_moisture: float, net_radiation: float, precipitation: float) -> tuple:
        """
        Update soil moisture after accounting for runoff and ET.
        """
        runoff = self.runoff_function(soil_moisture, precipitation)
        et = self.et_function(soil_moisture, net_radiation)
        soil_moisture_new = max(0, min(soil_moisture + precipitation - runoff - et, self.cs))
        return runoff, et, soil_moisture_new

    def run_model(self, country: Union[str, None] = None) -> dict:
        """
        Run the water balance model for all countries or a specific one.
        """
        results = {}
        countries_to_run = [country] if country and country in self.countries else self.countries

        for ci, country in enumerate(countries_to_run):
            data = self.data[ci]
            result = pd.DataFrame()
            result["Precipitation"] = data["tp_[mm]"]
            result["Net_Radiation"] = data["le_[W/m2]"] / self.rn_conversion_factor  # Convert W/m² to mm/day
            result["Soil_Moisture"] = [0.0] * len(data)
            result["Evapotranspiration"] = 0.0
            result["Runoff"] = 0.0

            soil_moisture = self.initial_soil_moisture  # Start with initial soil moisture

            for i in range(len(data)):
                precipitation = result.loc[i, "Precipitation"]
                net_radiation = result.loc[i, "Net_Radiation"]

                runoff, et, soil_moisture = self.forward(soil_moisture, net_radiation, precipitation)
                result.loc[i, "Soil_Moisture"] = soil_moisture
                result.loc[i, "Runoff"] = runoff
                result.loc[i, "Evapotranspiration"] = et

            results[country] = result

        return results
    
    def run_model_calibrate(self, data: pd.DataFrame) -> dict:
        """
        Run the water balance model for all countries or a specific one.
        """
      
        result = pd.DataFrame()
        result["Precipitation"] = data["tp_[mm]"]
        result["Net_Radiation"] = data["le_[W/m2]"] / self.rn_conversion_factor  # Convert W/m² to mm/day
        result["Soil_Moisture"] = [0.0] * len(data)
        result["Evapotranspiration"] = 0.0
        result["Runoff"] = 0.0
        result.reset_index(inplace=True)

        soil_moisture = self.initial_soil_moisture  # Start with initial soil moisture

        for i in range(len(data)):
            precipitation = result.loc[i, "Precipitation"]
            net_radiation = result.loc[i, "Net_Radiation"]

            runoff, et, soil_moisture = self.forward(soil_moisture, net_radiation, precipitation)
            result.loc[i, "Soil_Moisture"] = soil_moisture
            result.loc[i, "Runoff"] = runoff
            result.loc[i, "Evapotranspiration"] = et


        return result

    def save_results_to_csv(self, results: dict, output_dir: str = "output"):
        """
        Save the water balance model results to separate CSV files for each country.

        Args:
            results (dict): Dictionary containing results for each country as DataFrames.
            output_dir (str): Directory where the CSV files will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for country, data in results.items():
            # Save data as-is (units already consistent)
            file_path = os.path.join(output_dir, f"{country}_results.csv")
            data.to_csv(file_path, index=False)
            print(f"Results for {country} saved to {file_path}")

    def calibrate_model(self, param_grid: dict, years_train=(2008, 2013), years_val=(2014, 2018), output_dir="calibration_results"):
        """
        Calibrates the SimpleWaterBalanceModel by testing different parameter combinations.
        Ensures that untested parameter combinations are logged even if the evaluation file is missing.
        Reloads and evaluates existing best combinations from the log file.

        Args:
            param_grid (dict): Dictionary of hyperparameter values to test.
            years_train (tuple): Training period (e.g., 2008-2013).
            years_val (tuple): Validation period (e.g., 2014-2018).
            output_dir (str): Directory to save calibration results.

        Returns:
            dict: Best parameter combinations for each site.
        """

        if param_grid is None:
            raise ValueError("Parameter grid cannot be None. Provide a dictionary of parameter ranges.")

        param_combinations = list(itertools.product(*param_grid.values()))  # Generate all parameter combinations
        best_params = {}

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        log_file = os.path.join(output_dir, "calibration_log.csv")
        log_eval_file = os.path.join(output_dir, "eval_log.csv")

        score_keys = [
            "train_overall_score", "train_et_score", "train_runoff_score", "train_sm_score",
            "val_overall_score", "val_et_score", "val_runoff_score", "val_sm_score"
        ]

        # Load existing calibration results if available
        if os.path.exists(log_file):
            print("Found existing calibration runs. Reading in.")
            existing_results = pd.read_csv(log_file)
        else:
            existing_results = pd.DataFrame(columns=["site"] + list(param_grid.keys()) + score_keys[:4])

        if os.path.exists(log_eval_file):
            existing_evals = pd.read_csv(log_eval_file)
        else:
            existing_evals = pd.DataFrame(columns=["site"] + list(param_grid.keys()) + score_keys)

        # Loop over each site
        for sidx, site in enumerate(self.calibration_countries):
            print(f"\nCalibrating model for {site}...")

            os.makedirs(f"{output_dir}/{site}", exist_ok=True)  # Ensure output directory exists
            site_data = self.data_calibration[sidx]

            train_data = site_data[(site_data["Year"] >= years_train[0]) & (site_data["Year"] <= years_train[1])]
            val_data = site_data[(site_data["Year"] >= years_val[0]) & (site_data["Year"] <= years_val[1])]

            best_score = -np.inf
            best_combination = None

            with tqdm(total=len(param_combinations), desc=f"Calibrating {site}", unit="comb") as pbar:
                for param_set in param_combinations:
                    param_dict = {"site": site, **dict(zip(param_grid.keys(), param_set))}

                    if not existing_results.empty and ((existing_results["site"] == site) &
                                                    (existing_results[list(param_grid.keys())] == param_set).all(axis=1)).any():
                        train_score = existing_results.loc[
                            (existing_results["site"] == site) &
                            (existing_results[list(param_grid.keys())] == param_set).all(axis=1),
                            "train_overall_score"
                        ].values[0]
                    else:
                        for param, value in param_dict.items():
                            if param != "site":
                                setattr(self, param, value)

                        results = self.run_model_calibrate(train_data)
                        train_score, train_et_score, train_sm_score, train_runoff_score = self.evaluate_model(results, train_data)

                        param_dict.update({
                            "train_overall_score": train_score,
                            "train_et_score": train_et_score,
                            "train_sm_score": train_sm_score,
                            "train_runoff_score": train_runoff_score
                        })

                        new_entry = pd.DataFrame([param_dict])
                        new_entry.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

                        existing_results = pd.concat([existing_results, new_entry], ignore_index=True)

                        param_result_file = os.path.join(f"{output_dir}/{site}/", f"calibration_{site}_{'_'.join(map(str, param_set))}.csv")
                        pd.DataFrame(results).to_csv(param_result_file, index=False)

                    if train_score > best_score:
                        best_score = train_score
                        best_combination = param_dict
                        best_set = param_set

                    pbar.update(1)

            best_params[site] = best_combination
            best_params[site]["train_overall_score"] = best_score

            print(f"Best parameters for {site}: {best_params[site]}")

            for param, value in best_params[site].items():
                if param != "site":
                    setattr(self, param, value)

            if not os.path.exists(log_eval_file) or \
                    not ((existing_evals["site"] == site) &
                        (existing_evals[list(param_grid.keys())] == best_set).all(axis=1)).any():
                eval_results = self.run_model_calibrate(val_data)
                val_score, val_et_score, val_sm_score, val_runoff_score = self.evaluate_model(eval_results, val_data)

                best_combination.update({
                    "val_overall_score": val_score,
                    "val_et_score": val_et_score,
                    "val_sm_score": val_sm_score,
                    "val_runoff_score": val_runoff_score
                })

                new_eval_entry = pd.DataFrame([best_combination])
                new_eval_entry.to_csv(log_eval_file, mode='a', header=not os.path.exists(log_eval_file), index=False)

                existing_evals = pd.concat([existing_evals, new_eval_entry], ignore_index=True)

                eval_result_file = os.path.join(f"{output_dir}/{site}/", f"evaluation_{site}_{'_'.join(map(str, best_set))}.csv")
                pd.DataFrame(eval_results).to_csv(eval_result_file, index=False)

        best_params_df = pd.DataFrame.from_dict(best_params, orient="index")
        best_params_df.to_csv(os.path.join(output_dir, "best_parameters.csv"))

        return best_params


    def evaluate_model(self, model_results, observed_data):
        """
        Compute correlation score between modeled and observed data for Runoff, ET, and Soil Moisture.
        The final score is the sum of the three correlations.
        
        Args:
            model_results (pd.DataFrame): Model output data.
            observed_data (pd.DataFrame): Observed reference data.

        Returns:
            float: Sum of correlation scores for Soil Moisture, Runoff, and ET.
        """

        scores = []
        variables = {
            "Soil_Moisture": "sm_[m3/m3]",  # Observed soil moisture (convert to mm if necessary)
            "Evapotranspiration": "le_[W/m2]",  # Latent heat flux (convert if necessary)
            "Runoff": "ro_[m]"  # Runoff in meters (convert to mm if necessary)
        }
        et_score=0
        sm_score=0
        runoff_score=0

        for var, obs_column in variables.items():
            if obs_column in observed_data.columns and var in model_results.columns:
                observed_series = observed_data[obs_column].dropna()  # Remove NaN values
                modeled_series = model_results[var].dropna()  # Remove NaN values
                
                # Ensure both series are of the same length
                min_length = min(len(observed_series), len(modeled_series))
                observed_series = observed_series.iloc[:min_length]
                modeled_series = modeled_series.iloc[:min_length]

                if len(observed_series) > 1 and len(modeled_series) > 1:  # Avoid errors with single values
                    correlation = np.corrcoef(observed_series, modeled_series)[0, 1]
                    scores.append(correlation)

                    if self.verbose:
                        print(f"Correlation for {var}: {correlation:.4f}")

                    if var == "Evapotranspiration":
                        et_score = correlation
                    elif var == "Runoff":
                        runoff_score = correlation
                    elif var == "Soil_Moisture":
                        sm_score = correlation
        
        total_score = np.nansum(scores)  # Sum of correlations for the final score
        total_score = total_score if not np.isnan(total_score) else -np.inf  # Avoid NaN scores
        return total_score, et_score, sm_score, runoff_score

    def study_parameter_influence(
        self, parameter_name: str, parameter_values: list, countries: List[str], output_dir: str = "parameter_study"
    ) -> dict:
        """
        Study the influence of a parameter on model results.

        Args:
            parameter_name (str): Name of the parameter to vary.
            parameter_values (list): List of parameter values to test.
            countries (List[str]): List of countries to analyze.
            output_dir (str): Directory to save results.

        Returns:
            dict: Results for each country, where keys are country names and values are DataFrames.
        """
        os.makedirs(output_dir, exist_ok=True)
        country_results = {}

        for country in countries:
            print(f"Studying {parameter_name} for {country}...")
            results = []

            for value in parameter_values:
                # Temporarily update the parameter
                original_value = getattr(self, parameter_name)
                setattr(self, parameter_name, value)

                # Run the model
                result = self.run_model(country)[country]
                result["Parameter_Value"] = value
                results.append(result)

                # Restore the parameter
                setattr(self, parameter_name, original_value)

            # Combine results
            combined_results = pd.concat(results, keys=parameter_values, names=["Parameter_Value", "Row_Index"])
            country_results[country] = combined_results

            # Plot parameter influence
            print("Plotting pramater influence")
            self.plot_parameter_influence(combined_results, parameter_name, country, output_dir)

        return country_results



    def compare_to_observations(self, results: dict):
        """
        Compare model results to observations for soil moisture and other variables.
        """
        for country, result in results.items():
            observed_data = self.data[self.countries.index(country)]
            # Convert observed soil moisture to mm
            observed_data["Soil_Moisture_mm"] = observed_data["sm_[m3/m3]"] * self.cs
            # Print comparison for debugging
            print(f"Comparison for {country}:")
            print("Modeled Soil Moisture (mm):", result["Soil_Moisture"].head())
            print("Observed Soil Moisture (mm):", observed_data["Soil_Moisture_mm"].head())

    def plot_parameter_influence(
        self, combined_results: pd.DataFrame, parameter_name: str, country: str, output_dir: str
        ):
        """
        Plot the influence of a parameter on model results.

        Args:
            combined_results (pd.DataFrame): Results for all parameter values.
            parameter_name (str): Name of the parameter varied.
            country (str): Country name.
            output_dir (str): Directory to save plots.
        """
        observed_data = self.data[self.countries.index(country)]
        observed_data["Soil_Moisture_mm"] = observed_data["sm_[m3/m3]"] * self.cs

        variables = {
            "Soil_Moisture": "Soil_Moisture_mm",  # Observed soil moisture converted to mm
            "Evapotranspiration": "le_[W/m2]",
            "Runoff": "ro_[m]"
        }


        for variable, obs_column in variables.items():
            plt.figure(figsize=(14, 7))
            
            
            for param_value, group in combined_results.groupby(level="Parameter_Value"):

                new_group = group[variable].reset_index()
                values = new_group[variable]
                
                plt.plot(
                    #group.reset_index()["Row_Index"],
                    new_group['Row_Index'],
                    values,
                    label=f"{parameter_name} = {param_value}",
                    alpha=0.8
                )

            if obs_column in observed_data.columns:
                plt.plot(
                    observed_data.index,
                    observed_data[obs_column],
                    label="Observed",
                    color="black",
                    linewidth=2
                )
          

            plt.title(f"{variable} vs {parameter_name} - {country}")
            plt.xlabel("Time")
            plt.ylabel(variable)
            plt.legend()
            plt.grid(True)

            os.makedirs(output_dir, exist_ok=True)
            file_name = f"{country}_{parameter_name}_effect_on_{variable.lower()}.png"
            file_path = os.path.join(output_dir, file_name)
            plt.savefig(file_path, bbox_inches="tight")
            print(f"Plot saved to {file_path}")
            plt.close()



def plot_time_series(data: pd.DataFrame, title: str, variables: List[str], ylabel: str, country: str, output_dir: str):
    """
    Plot and save time series for given variables in the data.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data.
        title (str): Title of the plot.
        variables (List[str]): List of column names to plot.
        ylabel (str): Label for the y-axis.
        country (str): Country name for labeling.
        output_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(12, 6))
    for variable in variables:
        plt.plot(data.index, data[variable], label=variable)
    
    plt.title(f"{title} - {country}")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{country}_{title.replace(' ', '_').lower()}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    plt.close()



def visualize_results(results: dict, output_dir: str = "output"):
    """
    Visualize and save the time series results for all countries.

    Args:
        results (dict): Dictionary containing results for each country as DataFrames.
        output_dir (str): Directory where plots will be saved.
    """
    for country, data in results.items():
        # Plot soil moisture
        plot_time_series(data, "Soil Moisture Time Series", ["Soil_Moisture"], "Soil Moisture (mm)", country, output_dir)
        # Plot evapotranspiration
        plot_time_series(data, "Evapotranspiration Time Series", ["Evapotranspiration"], "Evapotranspiration (mm/day)", country, output_dir)
        # Plot runoff
        plot_time_series(data, "Runoff Time Series", ["Runoff"], "Runoff (mm/day)", country, output_dir)


def plot_time_series_with_seasonality(
    data: pd.DataFrame, title: str, variables: List[str], ylabel: str, country: str, output_dir: str, plot_season_lines: bool = True
):
    """
    Plot time series with seasonality markers for multi-year data.

    Args:
        data (pd.DataFrame): DataFrame with time series data.
        title (str): Plot title.
        variables (List[str]): Variables to plot.
        ylabel (str): Label for the y-axis.
        country (str): Country name for labeling.
        output_dir (str): Directory to save plots.
        plot_season_lines (bool): Whether to include season markers.
    """
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"])
    else:
        data["time"] = pd.date_range(start="2000-01-01", periods=len(data))  # Synthetic dates if missing

    plt.figure(figsize=(14, 7))
    for variable in variables:
        plt.plot(data["time"], data[variable], label=variable)

    if plot_season_lines:
        seasons = {"Spring": "-03-21", "Summer": "-06-21", "Fall": "-09-21", "Winter": "-12-21"}
        for year in range(data["time"].dt.year.min(), data["time"].dt.year.max() + 1):
            for season, date_suffix in seasons.items():
                try:
                    season_date = datetime.strptime(f"{year}{date_suffix}", "%Y-%m-%d")
                    if data["time"].min() <= season_date <= data["time"].max():
                        plt.axvline(season_date, color="gray", linestyle="--", alpha=0.7)
                except ValueError:
                    continue

    plt.title(f"{title} - {country}")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{country}_{title.replace(' ', '_').lower()}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    plt.close()






if __name__ == "__main__":
    swbm = SimpleWaterBalanceModel()
    #results = swbm.run_model()
    #swbm.save_results_to_csv(results, "output_v2")
    #visualize_results(results, "output_v2")

    parameter_studies = {
        "cs": [200, 420, 600],
        "alpha": [1, 4, 8],
        "gamma": [0.1, 0.5, 0.9],
        "beta": [0.2, 0.8, 1.0]
    }

    #for param_name, values in parameter_studies.items():
    #    swbm.study_parameter_influence(param_name, values, countries=["Germany", "Sweden"], output_dir="parameter_study_v2")
       # Define parameter grid for callibration
    param_grid = {
            "cs": [210,420,840],  # How much inital soil moisture (of soil moisture max which is data drivn)
            "alpha": [2, 4, 8],  # Runoff function shape
            "gamma": [0.2, 0.5, 0.8],  # ET function shape
            "beta": [0.4, 0.6, 0.8],  # Maximum of ET function
    }

    
    swbm.calibrate_model(param_grid, output_dir="calibration_results_v2")