# All down the drain? An update to the simple water balance model

## Overview
This repository hosts implementations of the **Simple Water Balance Model (SWBM)**, including both the original version (`swbm_v2.py`) and an updated version (`swbm_v3_soil.py`). The SWBM is designed to simulate key hydrological components, such as soil moisture, evapotranspiration, and runoff, using precipitation and net radiation as primary inputs.

The model is computationally efficient, adaptable to various spatial scales, and requires minimal input data, making it an excellent tool for hydrological research and applications.

## Features
- **SWBM v2** (`swbm_v2.py`): Implements the basic version of the water balance model with parameters for soil water holding capacity, runoff function, and evapotranspiration function.
- **SWBM v3** (`swbm_v3_soil.py`): Enhances the model with additional soil layers and improved process representation.
- **Parameter Sensitivity Analysis**: Supports studying the impact of key parameters on hydrological outputs.
- **Comparison with Observations**: Allows evaluation against measured soil moisture and runoff data.
- **Visualization**: Includes tools to generate time series plots of soil moisture, evapotranspiration, and runoff.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To run the model for a specific country (e.g., Germany):
```python
from swbm_v2 import SimpleWaterBalanceModel
swbm = SimpleWaterBalanceModel(countries=["Germany"])
results = swbm.run_model()
swbm.save_results_to_csv(results)
```
To perform parameter sensitivity analysis:
```python
swbm.study_parameter_influence("cs", [200, 420, 600], ["Germany"], "output/parameter_study")
```

## Data Requirements
The model requires CSV input files containing time series data for precipitation and net radiation:
- `tp_[mm]`: Precipitation in mm.
- `le_[W/m2]`: Net radiation in W/m² (converted internally to mm/day).
- `sm_[m3/m3]`: (Optional) Observed soil moisture for validation.

## Reference
The Simple Water Balance Model is based on the conceptual framework presented by **René Orth** in the _Seminar on Earth System Modelling.



