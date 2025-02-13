import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the NetCDF file
file_path = "slt.nc"
data = xr.open_dataset(file_path)

# Define soil properties with corresponding labels
soil_properties = {
    1: "Coarse",
    2: "Medium",
    3: "Medium-fine",
    4: "Fine",
    5: "Very fine",
    6: "Organic"
}

# Define site locations (longitude, latitude)
sites = {
    "Germany": (8.125, 48.125),
    "Spain": (-3.625, 38.625),
    "Sweden": (15.875, 63.625)
}

# Function to find the nearest soil type at a given site location
def get_soil_type(lon, lat):
        """
        Get the soil type based on latitude and longitude.

        Parameters:
        - lat (float): Latitude in the range -90 to 90.
        - lon (float): Longitude in the range -180 to 180.

        Returns:
        - int: Soil type index at the nearest latitude and longitude.
        """
        soil_type_data = data[soil_type_var]
        # Normalize longitude to match the dataset (0 to 360)
        if lon < 0:
            lon += 360

        # Select the soil type variable
        soil_type = soil_type_data.sel(latitude=lat, longitude=lon, method="nearest")

        # Extract the value as an integer safely

        soil_value = int(np.round(soil_type.values[0]))

        print(f"Latitude: {lat}, Longitude: {lon}, Soil Type Index: {soil_value}")
        return soil_properties.get(soil_value, "Unknown")

# Update with actual variable names from the dataset
soil_type_var = "slt"  # Replace with actual soil type variable name
lat_var = "latitude"  # Replace with actual latitude variable name
lon_var = "longitude"  # Replace with actual longitude variable name

if soil_type_var not in data.variables:
    raise ValueError(f"Variable '{soil_type_var}' not found. Available variables: {list(data.variables)}")

# Extract latitude and longitude coordinates
latitudes = data[lat_var]
longitudes = data[lon_var]

# Define the bounding box for Europe (handling 0-360 longitude)
lat_min, lat_max = 35, 71
lon_min_1, lon_max_1 = 335, 360  # Wrap-around part
lon_min_2, lon_max_2 = 0, 45  # Standard part

# Select latitudes within the range
data_europe = data.sel({lat_var: slice(lat_max, lat_min)})  # Flip order if needed

# Select longitudes correctly (handling wrap-around at 360)
data_europe1 = data_europe.sel({lon_var: slice(lon_min_1, lon_max_1)})
data_europe2 = data_europe.sel({lon_var: slice(lon_min_2, lon_max_2)})

# Merge the two longitude subsets
data_europe = xr.concat([data_europe1, data_europe2], dim=lon_var)

# Extract soil type data and remove singleton dimensions
soil_type_data = data_europe[soil_type_var].squeeze()

# Ensure data is 2D
if len(soil_type_data.shape) > 2:
    print("dealing with time")
    soil_type_data = soil_type_data.isel(time=0).squeeze()  # Select first time step if time exists

# Convert to numpy array
soil_type_array = soil_type_data.values



# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data using imshow
im = ax.imshow(soil_type_array, extent=[-25, 45, lat_min, lat_max], origin='upper', cmap="viridis")

# Create a colorbar with labels for soil types
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax, ticks=list(soil_properties.keys()))
cbar.ax.set_yticklabels([soil_properties[i] for i in soil_properties.keys()])

# Add pins for the three sites with soil type labels
for site_name, (lon, lat) in sites.items():
    soil_texture = get_soil_type(lon, lat)  # Get the soil type at this site
    label_text = f"{site_name}: {soil_texture}"

    ax.scatter(lon, lat, color="red", marker="o", edgecolors="black", s=60, zorder=3)  # Reduced marker size
    ax.text(lon + 1, lat, label_text, color="white", fontsize=9, weight="bold", ha="left", va="center",
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))

# Set titles and labels
ax.set_title("Soil Type Map - Europe")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Show the plot
plt.show()

