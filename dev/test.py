from windwhisper import windturbines
import xarray as xr

# we can preload the wind speed data, otherwise, the tool will do it every time
filepath_wind_speed = "/Users/romain/GitHub/windwhisper/dev/fixtures/era5_mean_2013-2022_month_by_hour.nc"
filepath_correction = "/Users/romain/GitHub/windwhisper/dev/fixtures/ratio_gwa2_era5.nc"


def wind_speed_data():
    wind_speed = xr.open_dataset(filepath_wind_speed).to_array().mean(dim="month")
    correction = xr.open_dataset(filepath_correction).to_array()
    correction = correction.sel(variable='ratio_gwa2_era5_mean_WS').interp(
        latitude=wind_speed.latitude,
       longitude=wind_speed.longitude,
       method="linear"
    )
    return wind_speed * correction


data = wind_speed_data()

wind_turbines = {
    'Turbine 0':
        {'diameter': 70.0,
         'hub height': 85.0,
         'position': (43.67402852737928, 7.2169801653160395),
         'power': 2500.0},
    # 'Turbine 1':
    #     {'diameter': 60.0,
    #      'hub height': 55.0,
    #      'position': (43.678986127617584, 7.222049102839316),
    #      'power': 1500.0},
    # 'Turbine 2':
    #     {'diameter': 60.0,
    #      'hub height': 55.0,
    #      'position': (43.67262338001209, 7.217452048839465),
    #      'power': 2500.0},
}

wt = windturbines.WindTurbines(
    wind_turbines=wind_turbines,
    wind_speed_data=data,
)