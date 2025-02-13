from typing import List, Union, Dict
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import xarray as xr
import itertools
from tqdm import tqdm  # Import progress bar

class SimpleWaterBalanceModel:
    def __init__(
        self,
        alpha: float = 4.0,  # Runoff function shape
        gamma: float = 0.5,  # ET function shape
        beta: float = 0.8,  # ET function maximum
        init_cm_coef: float = 1.0,  # Initial soil moisture coefficient,
        drainage_factor_factor : float = 1.0, # factor to scale drainage factor mapping
        countries: List[str] = ["Germany", "Sweden"],  # List of countries
        calibration_countries: List[str] = ["Germany_new", "Sweden_new", "Spain_new"], # List of callibration countries
        soil_type_file_path = "meta_data/slt.nc",
        soil_type_var = "slt", 
        root_depth_file_path = "meta_data/root_profiles_660/data/root_profiles_D50D95.csv",
        root_depth_var = "D95",
        root_depth_unit = 'm',
        verbose = False,
        data_root_dir = 'data',
        delta = 0.5


        
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.init_cm_coef = init_cm_coef
        self.rn_conversion_factor = 2.26  # Convert from W/m² to mm/day
        self.countries = countries
        self.calibration_countries = calibration_countries
        self.data_root_dir = data_root_dir
        self.data = [self.read_data(c) for c in countries]
        self.data_calibration = [self.read_data(c) for c in calibration_countries]
        self.verbose = verbose
        self.delta = delta
        
     
        soil_type_data = xr.open_dataset(soil_type_file_path)

        root_depth_data = pd.read_csv(root_depth_file_path, encoding='latin1')

       
        if soil_type_var not in soil_type_data.variables:
            raise ValueError(f"Variable '{soil_type_var}' not found. Available variables: {list(self.soil_type_data.variables)}")
        
        self.root_depth_var = root_depth_var
        self.soil_type_data = soil_type_data[soil_type_var]
        self.root_depth_unit = root_depth_unit

        # Extract latitude and longitude coordinates
        self.soil_latitudes = self.soil_type_data['latitude'].values
        self.soil_longitudes = self.soil_type_data['longitude'].values


        if root_depth_var not in root_depth_data.columns:
            raise ValueError(f"Variable '{root_depth_var}' not found. Available variables: {list(root_depth_data.columns)}")
        
        self.root_depth_data = root_depth_data
    

        # Mapping of soil type index to PWP and FC values
        self.soil_properties = {
            1: {"Soil": "Coarse", "PWP": 0.059, "FC": 0.242},
            2: {"Soil": "Medium", "PWP": 0.151, "FC": 0.346},
            3: {"Soil": "Medium-fine", "PWP": 0.133, "FC": 0.382},
            4: {"Soil": "Fine", "PWP": 0.279, "FC": 0.448},
            5: {"Soil": "Very fine", "PWP": 0.335, "FC": 0.541},
            6: {"Soil": "Organic", "PWP": 0.267, "FC": 0.662}
        }

        # Mapping from soil type index to infiltration rate (mm/day)
        # Data from: https://stormwater.pca.state.mn.us/index.php/Design_infiltration_rates?utm_source=chatgpt.com
       
        self.soil_infiltration_mapping_old = {
            1: 41.4,  # Gravel, Sandy Gravel (Hydrologic Group A)
            2: 20.3,  # Silty Gravel, Sand (Hydrologic Group A)
            3: 11.4,  # Silty Sands (Hydrologic Group B)
            4: 7.6,  # Loam, Silt Loam (Hydrologic Group B)
            5: 5.1,  # Sandy Clay Loam, Silts (Hydrologic Group C)
            6: 1.5   # Clay, Silty Clay, etc. (Hydrologic Group D)
        }

        self.soil_infiltration_mapping = {
            1: 20.3,  # Gravel, Sandy Gravel (Hydrologic Group A) -> or 41.4 there are two values dependent on how coarse (silty vs sandy gravel)
            2: 11.4,  # Silty Gravel, Sand (Hydrologic Group A)
            3: 7.6,  # Silty Sands (Hydrologic Group B)
            4: 5.1,  # Loam, Silt Loam (Hydrologic Group B)
            5: 1.5,  # Sandy Clay Loam, Silts (Hydrologic Group C)
            6: 1.5   # Clay, Silty Clay, etc. (Hydrologic Group D)
        }

        # Define drainage factor per soil type (Lower factor = more water retention)
        # This is made up...
        self.soil_drainage_factors = {
            1: 0.6,  # Sand, Gravel (High drainage)
            2: 0.5,  # Loamy Sand
            3: 0.3,  # Loam
            4: 0.1,  # Clay Loam
            5: 0.05,  # Silty Clay
            6: 0.005  # Clay (Low drainage)
        }
        self.drainage_factors_factor = drainage_factor_factor

       

    # Example function to get infiltration rate based on soil type index
    def get_infiltration_rate(self, soil_type_index: int) -> float:
        return self.soil_infiltration_mapping.get(soil_type_index, 0)  # Default to 0 if unknown

    def get_drainage_factor(self, soil_type_index: int) -> float:
        return self.soil_drainage_factors.get(soil_type_index, 0.5) # Defaults to 0.5 if unknown


    def get_soil_properties(self, index):
        """
        Returns the PWP and FC for a given soil type index.
        
        Parameters:
        - index (int): Soil type index
        
        Returns:
        - tuple: (PWP, FC) for the soil type
        """
        if index in self.soil_properties:
            properties = self.soil_properties[index]
            return properties["PWP"], properties["FC"]
        else:
            raise ValueError(f"Invalid soil type index: {index}")


  
    def get_soil_type(self, lat, lon):
        """
        Get the soil type based on latitude and longitude.

        Parameters:
        - lat (float): Latitude in the range -90 to 90.
        - lon (float): Longitude in the range -180 to 180.

        Returns:
        - int: Soil type index at the nearest latitude and longitude.
        """
        # Normalize longitude to match the dataset (0 to 360)
        if lon < 0:
            lon += 360

        # Select the soil type variable
        soil_type = self.soil_type_data.sel(latitude=lat, longitude=lon, method="nearest")

        # Extract the value as an integer safely

        soil_value = int(np.round(soil_type.values[0]))

        print(f"Latitude: {lat}, Longitude: {lon}, Soil Type Index: {soil_value}")
        return soil_value
    
    def get_root_depth(self, lat, lon):
        """
        Get the root depth value based on latitude and longitude.

        Parameters:
        - lat (float): Latitude in the range -90 to 90.
        - lon (float): Longitude in the range -180 to 180.

        Returns:
        - int: Root depth value at the nearest latitude and longitude.
        """
        # Calculate distances
        self.root_depth_data['distance'] = self.root_depth_data.apply(
            lambda row: haversine(lon, lat, row['Longitude'], row['Latitude']), axis=1
        )

        # Find the nearest point
        nearest = self.root_depth_data.loc[self.root_depth_data['distance'].idxmin()]
        root_depth = nearest[self.root_depth_var]
        print(f"Nearest Root Depth: {root_depth}, Distance: {nearest['distance']:.2f} km")


        # convert to mm
        if self.root_depth_unit == 'm':
            root_depth = root_depth *1000
        elif self.root_depth_unit == 'cm':
            root_depth = root_depth * 100
        elif self.root_depth_unit == 'mm':
            root_depth = root_depth
        else:
            print("Unknown unit for root depth:", self.root_depth_unit)
            raise ValueError
        

        print(f"Latitude: {lat}, Longitude: {lon}, Rooth Depth: {root_depth} mm")

        return root_depth
    
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


   

    def runoff_function(self, soil_moisture: float, precipitation: float, soil_moisture_max: float, infiltration_rate: float) -> tuple:
        """
        Calculate runoff (in mm) and infiltration.
        """
        # Ensure inputs are valid and non-negative
        precipitation = max(0, precipitation)
        soil_moisture = max(0, soil_moisture)
        soil_moisture_max = max(1e-6, soil_moisture_max)  # Prevent division by zero or invalid capacity

        # Compute how much capacity remains in the soil
        remaining_cap = max(0, soil_moisture_max - soil_moisture)
        

        if precipitation >= remaining_cap:
            # Introduce an infiltration rate limit to prevent full percolation
            infiltration_max_possible = precipitation - remaining_cap
           
        elif precipitation < remaining_cap:
            infiltration_max_possible = precipitation

        infiltration = min(infiltration_rate, infiltration_max_possible)
        runoff = max(0, precipitation - infiltration)


        return runoff, infiltration


    def et_function(self, soil_moisture: float, net_radiation: float, soil_moisture_max: float, beta: float = 0.8, gamma: float = 0.5, rn_conversion_factor: float = 2.26) -> float:
        """
        Calculate evapotranspiration (in mm/day).
        """
        # Ensure inputs are valid and non-negative
        soil_moisture = max(0, soil_moisture)
        soil_moisture_max = max(1e-6, soil_moisture_max)  # Prevent division by zero

        # Convert net radiation from W/m² to mm/day
        net_radiation_mm = net_radiation / rn_conversion_factor

        # Calculate evapotranspiration
        et = beta * ((soil_moisture / soil_moisture_max) ** gamma) * net_radiation_mm

        return max(0, et)  # Ensure non-negative ET


    def drainage(self, drainage_factor: float, soil_moisture: float, soil_moisture_max: float) -> float:
        """
        Drainage is a function of infiltration, but reduced based on soil type.
        """
    
        # delta changes function behaviour 
        drainage = (soil_moisture * drainage_factor)**self.delta
        drainage = min(drainage, soil_moisture_max)

        return drainage


    def forward(self, soil_moisture: float, net_radiation: float, precipitation: float, soil_moisture_max: float, infiltration_rate: float, drainage_factor: float) -> tuple:
        """
        Update soil moisture after accounting for runoff, ET, and drainage.
        """
        # Runoff and infiltration
        runoff, infiltration = self.runoff_function(soil_moisture, precipitation, soil_moisture_max, infiltration_rate)

        # Evapotranspiration
        et = self.et_function(soil_moisture, net_radiation, soil_moisture_max)

        # Drainage is now limited by the drainage rate
        drainage_amount = self.drainage(drainage_factor, soil_moisture, soil_moisture_max)


        # Calculate change in soil moisture
        soil_moisture_delta = infiltration - et - drainage_amount


        # Update soil moisture, ensuring it stays within valid bound

        soil_moisture_new = soil_moisture + soil_moisture_delta

        soil_moisture_new = min(soil_moisture_new, soil_moisture_max)

        soil_moisture_new = max(0, soil_moisture_new)

        # Debugging information
        if self.verbose:
            print("Old soil moisture", soil_moisture)
            print("Soil moisture change:", soil_moisture_delta)
            print("Infiltration:", infiltration)
            print("Evapotranspiration:", et)
            print("Drainage:", drainage_amount)
            print("New soil moisture:", soil_moisture_new)
            print("Runoff:", runoff)
            print("Precipitation:", precipitation)

        # Check for NaN values explicitly
        if np.any(np.isnan([runoff, et, soil_moisture_new])):
            raise ValueError("NaN value encountered in computation.")

        return runoff, et, soil_moisture_new, drainage_amount, infiltration


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
            result["Time"] = data['time']

            # Get latitude and longitude
            latitude = data.iloc[0]["latitude"]
            longitude = data.iloc[0]["longitude"]

            if self.verbose:
                print("Data lat/lon:", latitude, longitude)

            # Map longitude and latitude to soil type (texture)
            soil_type_index = self.get_soil_type(latitude, longitude)

            # Map longitude and latitude to rooth depth
            root_zone_depth = self.get_root_depth(latitude, longitude)

            wilting_point, field_capacity = self.get_soil_properties(soil_type_index)
            awc = field_capacity - wilting_point
            soil_moisture_max = awc * root_zone_depth

            infiltration_rate = self.get_infiltration_rate(soil_type_index)
            drainge_factor = self.get_drainage_factor(soil_type_index)

          

            soil_moisture = self.init_cm_coef * soil_moisture_max  # Start with initial soil moisture

            if self.verbose:
                print("Starting soil moisture:", soil_moisture)
                print("Infiltration_rate", infiltration_rate)
            
            
           

            for i in range(len(data)):
                precipitation = result.loc[i, "Precipitation"]
                net_radiation = result.loc[i, "Net_Radiation"]

                runoff, et, soil_moisture, drainage, infiltration = self.forward(soil_moisture, net_radiation, precipitation, soil_moisture_max=soil_moisture_max, infiltration_rate=infiltration_rate, drainage_factor=drainge_factor)
                result.loc[i, "Soil_Moisture"] = soil_moisture
                result.loc[i, "Runoff"] = runoff
                result.loc[i, "Evapotranspiration"] = et
                result.loc[i, "Drainage"] = drainage
                result.loc[i, "Infiltration"] = infiltration

            results[country] = result

        return results
    
    def run_model_calibrate(self, data: Union[pd.DataFrame, None], infiltration_rate: float, soil_moisture_max : float, soil_moisture: float, drainage_factor: float) -> dict:
        """
        Run the water balance model specific data
        """
     
        result = pd.DataFrame()
        result["Precipitation"] = data["tp_[mm]"]
        result["Net_Radiation"] = data["le_[W/m2]"] / self.rn_conversion_factor  # Convert W/m² to mm/day
        result["Soil_Moisture"] = [0.0] * len(data)
        result["Evapotranspiration"] = 0.0
        result["Runoff"] = 0.0
        result["time"] = data["time"]
        result.reset_index(inplace=True)
        
        for i in range(len(data)):
            precipitation = result.loc[i, "Precipitation"]
            net_radiation = result.loc[i, "Net_Radiation"]
            runoff, et, soil_moisture, drainage, infiltration = self.forward(soil_moisture, net_radiation, precipitation, soil_moisture_max=soil_moisture_max, infiltration_rate=infiltration_rate, drainage_factor=drainage_factor)
            result.loc[i, "Soil_Moisture"] = soil_moisture
            result.loc[i, "Runoff"] = runoff
            result.loc[i, "Evapotranspiration"] = et
            result.loc[i, "Drainage"] = drainage
            result.loc[i, "Infiltration"] = infiltration

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

            # Get latitude and longitude
            latitude = site_data.iloc[0]["latitude"]
            longitude = site_data.iloc[0]["longitude"]

            if self.verbose:
                print("Data lat/lon:", latitude, longitude)

            # Map latitude and longitude to soil parameters
            soil_type_index = self.get_soil_type(latitude, longitude)
            root_zone_depth = self.get_root_depth(latitude, longitude)
            wilting_point, field_capacity = self.get_soil_properties(soil_type_index)
            awc = field_capacity - wilting_point
            soil_moisture_max = awc * root_zone_depth
            infiltration_rate = self.get_infiltration_rate(soil_type_index)
            drainage_factor = self.get_drainage_factor(soil_type_index)
            soil_moisture = self.init_cm_coef * soil_moisture_max  # Start with initial soil moisture

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

                        if self.verbose:
                            print(param_set, "exists")
                    else:
                        for param, value in param_dict.items():
                            if param != "site":
                                setattr(self, param, value)

                        results = self.run_model_calibrate(train_data, infiltration_rate, soil_moisture_max, soil_moisture, drainage_factor)
                        train_score, train_et_score, train_sm_score, train_runoff_score = self.evaluate_model(results, train_data)

                        param_dict.update({
                            "soil_moisture_max": soil_moisture_max,
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
                eval_results = self.run_model_calibrate(val_data, infiltration_rate, soil_moisture_max, soil_moisture, drainage_factor)
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
        The final score is the sum of the three correlations, ensuring that zero or missing values do not cause NaNs.

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
        et_score = 0
        sm_score = 0
        runoff_score = 0

        for var, obs_column in variables.items():
            if obs_column in observed_data.columns and var in model_results.columns:
                observed_series = observed_data[obs_column].fillna(0)  # Replace NaN with 0
                modeled_series = model_results[var].fillna(0)  # Replace NaN with 0

                # Ensure both series are of the same length
                min_length = min(len(observed_series), len(modeled_series))
                observed_series = observed_series.iloc[:min_length]
                modeled_series = modeled_series.iloc[:min_length]

                # Avoid issues with all zeros
                if np.all(observed_series == 0) and np.all(modeled_series == 0):
                    correlation = 0  # If both are zero, define correlation as 0 to avoid NaN
                elif np.var(observed_series) == 0 or np.var(modeled_series) == 0:
                    correlation = 0  # If variance is zero, correlation is undefined, set to 0
                else:
                    correlation = np.corrcoef(observed_series, modeled_series)[0, 1]
                    correlation = np.nan_to_num(correlation, nan=0)  # Replace NaN with 0

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
        total_score = np.nan_to_num(total_score, nan=0)  # Replace NaN with 0 to ensure valid scoring
        return total_score, et_score, sm_score, runoff_score



# Function to calculate Haversine distance
def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


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
        plt.plot(data['Time'], data[variable], label=variable)
    
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
        # Plot Drainage
        plot_time_series(data, "Drainage Time Series", ["Drainage"], "Drainage (mm)", country, output_dir)
         # Plot Infiltration
        plot_time_series(data, "Infiltration Time Series", ["Infiltration"], "Infiltration (mm)", country, output_dir)
         # Plot Precipitation
        plot_time_series(data, "Precipitation Time Series", ["Precipitation"], "Precipitation (mm)", country, output_dir)




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
    swbm = SimpleWaterBalanceModel(verbose=False, calibration_countries=["Sweden_new", "Germany_new", "Spain_new"])
    results = swbm.run_model()
    swbm.save_results_to_csv(results, "output_v3_new")
    visualize_results(results, "output_v3_new")
    results = pd.read_csv("output_v3_new/Germany_new_results.csv")
    swbm.evaluate_model(results, swbm.data[0])



    # Define parameter grid for callibration
    param_grid = {
            "init_cm_coef": [1.0, 0.9, 0.8, 0.7],  # How much inital soil moisture (of soil moisture max which is data drivn)
            "alpha": [2, 4, 8],  # Runoff function shape
            "gamma": [0.2, 0.5, 0.8],  # ET function shape
            "beta": [0.4, 0.6, 0.8],  # Maximum of ET function
            "drainage_factor_factor": [1.2, 1.0 ,0.8, 0.0, 0.5],  # scaling of drainage factors
            "delta": [2, 0.5, 0.7, 1.2, 1] # other values suck anyway[1, 2, 0.5, 0.7, 1.2],
    }


    swbm.calibrate_model(param_grid, output_dir="calibration_results_v3")

    #parameter_studies = {
    #    "cs": [200, 420, 600],
    #    "alpha": [1, 4, 8],
    #    "gamma": [0.1, 0.5, 0.9],
    #    "beta": [0.2, 0.8, 1.0]
    #}

    #for param_name, values in parameter_studies.items():
    #    swbm.study_parameter_influence(param_name, values, countries=["Germany", "Sweden"], output_dir="parameter_study_v2")
