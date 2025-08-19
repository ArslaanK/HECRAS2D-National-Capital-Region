# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 01:27:21 2025

@author: Arslaan Khalid
"""


import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from shapely import wkt 
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import plotly.graph_objs as go
import warnings;warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from noaa_coops import Station, get_stations_from_bbox
import tqdm
import dataretrieval.nwis as nwis
from meteostat import Stations,Hourly
import itertools


def get_fcast_date(url):
    response = requests.get(url)
    if response.status_code == 200:
        fcst_date = response.text.strip()
        print(f"Recent forecast start: {fcst_date}")
    else:
        print(f"Failed to fetch recent.txt. Status code: {response.status_code}")
    return fcst_date

def list_files_from_directory(url, extension=".tsv"):
    """
    Parse the directory-style HTML and list files with the given extension.
    """
    response = requests.get(url)
    if not response.ok:
        raise RuntimeError(f"Failed to access {url}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    files = []

    for row in soup.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) >= 3:
            filename = cols[2].text.strip()
            if filename.endswith(extension):
                files.append(filename.replace(extension, ''))  # Just the gage ID

    return sorted(files)

variables = [
    "gage_height",
    "precipitation",
    "pressure",
    "wind_speed",
    "wind_direction",
]


variables_units = {
    "gage_height":'Water Levels in feet above NAVD88',
    "precipitation": 'inches',
    "pressure":'mbar',
    "wind_direction":'degrees',
    "wind_speed":'feet per second'
}

# Assign each variable a color: [R, G, B, A]
variable_colors = {
    'gage_height': [0, 0, 255, 160],        # blue
    'precipitation': [0, 128, 0, 160],      # green
    'pressure': [0, 0, 255, 160],           # blue
    'wind_direction': [255, 165, 0, 160],   # orange
    'wind_speed': [128, 0, 128, 160],       # purple
}


# unit conversions"

kmph_to_mps = 0.27778
knots_to_mps = 0.5144444444
mm_to_inches = 0.03937008
inches_to_mm = 25.4

# =============================================================================
# getting forecast date
# =============================================================================

# Set the remote URL for the CSV

recent_fcst_url = 'https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/recent.txt'

fcst_date=get_fcast_date(recent_fcst_url)

# fcst_date='2025081818'

fcst_minus_1 = (datetime.strptime(fcst_date, '%Y%m%d%H') - timedelta(hours=12)).strftime('%Y%m%d%H')
fcst_minus_2 =( datetime.strptime(fcst_date, '%Y%m%d%H') - timedelta(hours=24)).strftime('%Y%m%d%H')


# =============================================================================
# fixed URLS
# =============================================================================
Gages_URL = f"https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/{fcst_date}/gages_found.csv"
Perimeter_URL = f"https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/{fcst_date}/model_domain.csv"
Cells_URL = f"https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/{fcst_date}/mesh_cells_points.csv"

# =============================================================================
# finding gages data available online
# =============================================================================

# https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/2025081806/Timeseries/wind_speed/8594900.tsv

# Optionally check which files exist (same as in Option 1)
def check_gage_exists(fcst_date,gage_id, variable):  
    url = f"https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/{fcst_date}/Timeseries/{variable}/{gage_id}.tsv"
    #print(url)
    return requests.head(url).status_code == 200

# =============================================================================
# gages viewer
# =============================================================================
st.title("HECRAS2D Forecast System for National Capital Region")

# Load CSV directly from URL
@st.cache_data
def load_gages(url):
    df = pd.read_csv(url)

    # Convert WKT geometry column to actual shapely geometries
    if 'geometry' in df.columns:
        df['geometry'] = df['geometry'].apply(wkt.loads)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # Extract lat/lon
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y

    return gdf

# loading the gage
gages_df = load_gages(Gages_URL)
del gages_df['Type']

# get all unique gage ids
gage_ids = gages_df['site_no'].astype(str).unique()  # or 'name' or another column
site_name_map = gages_df.set_index('site_no')['site_name'].to_dict()  # fallback using site_no as name

gages_for_var = {}

for var in variables:
    #gid='8594900'
    existing_gages = [gid for gid in gage_ids if check_gage_exists(fcst_date,gid,var)]
    #print(var,existing_gages)
    gages_for_var[var]=existing_gages


# loading the mesh cells
# cells_df = load_gages(Cells_URL)

# loading the mesh perimeter
perimeter_df = pd.read_csv(Perimeter_URL)
perimeter_df['geometry'] = perimeter_df['geometry'].apply(wkt.loads)
perimeter_gdf = gpd.GeoDataFrame(perimeter_df, geometry='geometry', crs="EPSG:4326")

perimeter_gdf['polygon_coords'] = perimeter_gdf['geometry'].apply(lambda geom: geom.__geo_interface__['coordinates'])


# =============================================================================
# Text Info
# =============================================================================
readable = datetime.strptime(fcst_date, '%Y%m%d%H').strftime('%B %d, %Y at %H:%M UTC')

st.subheader(f"Latest Forecast Date: {readable}")


st.markdown("""
### Model Overview

The model domain covers the **National Capital Region (NCR)**, which is critical for hydrological and meteorological forecasting. The model is forced by **HRRR (High-Resolution Rapid Refresh)** 48-hour forecasts, initialized every **0600** and **1800 UTC**. These forecasts provide real-time atmospheric conditions to drive the model, ensuring accurate and timely predictions.

#### Atmospheric Forcing Data:
- **Wind**
- **Pressure**
- **Precipitation**

#### Coastal Boundary Conditions:
The model uses **coastal water level boundaries** provided by **AHPS (Advanced Hydrologic Prediction Service)**, part of **NOAA**.

#### River Inflow Conditions:
The model uses observed discharge from last 2 days as input conditions for the forecasted model run. In future it will be replaced by NWM or similar discharge forecast model.

#### Model Outputs:
- **Water Surface Elevation**
- **Cell Velocity**
- **Cell Volume**

""")


with st.expander("Model Development (click to expand)"):
    st.markdown("""

---               
#### 🗺️ Topography and Bathymetry
- **Topography data sourced from USGS**
- **Bathy data sourced from CUDEM** 

---

#### 🔧 Breaklines (Terrain Enhancement)
Breaklines used in this model include:
- **Road Centerlines** – to shape road embankments.
- **NHD Plus** – from the National Hydrography Dataset to define flow paths and riverbanks.
- **Shorelines** – to model coastal/riverine transitions precisely.
- **USACE Levees** – to simulate flood protection and water diversion.

---
            
### 🗺️ Internal and External Boundary Conditions
- **Internal streamflow boundaries** use data from **USGS stream gages** to simulate inflows with real-time accuracy.
- **External coastal boundary** is set downstream near the **Alexandria AHPS station**, capturing tidal influences from the Potomac River.

--- 

#### 🧮 Model Resolution
- **Global resolution**: 150 ft  
- **Shoreline resolution**: 50 ft  
- **Road network resolution**: 90 ft  
- **Total mesh cells**: 651,238
- **Prorojection:**: NAD83(2011) / Virginia South (ftUS) [CRS: EPSG:6595]

""")




# =============================================================================
# Gage spatial plot
# =============================================================================

# Check if lat/lon columns exist
lat_col = 'lat' if 'lat' in gages_df.columns else 'latitude'
lon_col = 'lon' if 'lon' in gages_df.columns else 'longitude'

if lat_col in gages_df.columns and lon_col in gages_df.columns:
    st.subheader("🗺️ Gage Locations Map with Mesh Perimeter")

    initial_view = pdk.ViewState(
        latitude=gages_df[lat_col].mean(),
        longitude=gages_df[lon_col].mean(),
        zoom=10,
        pitch=0,
    )

    # Make a list of layers: 1 Polygon layer + 1 scatter layer per variable
    layers = []

    # Add mesh perimeter layer
    layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=perimeter_gdf,
            get_polygon="polygon_coords",  # Use the new column with coordinates
            get_fill_color=[0, 0, 0, 0],  # transparent fill
            get_line_color='[255, 255, 0]',
            line_width_min_pixels=2,
            pickable=False,
        )
    )

    # Add one ScatterplotLayer per variable
    for var, gage_id in gages_for_var.items():
        var_gages = gages_df[gages_df['site_no'].astype(str).isin(gage_id)]
        if not var_gages.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=var_gages,
                    get_position=f"[{lon_col}, {lat_col}]",
                    get_radius=500,
                    get_color=variable_colors[var],
                    pickable=True,
                    tooltip=True,
                )
            )

    # Display combined map
    st.pydeck_chart(pdk.Deck(
        initial_view_state=initial_view,
        layers=layers,
        tooltip={"text": "{site_name}" if "site_name" in gages_df.columns else "{name}"}
    ))

else:
    st.warning("Latitude and longitude columns not found in the data.")


# =============================================================================
# load Timeseries data
# =============================================================================

# These are your forecasts to load
forecast_timestamps = [
     fcst_minus_2,
     fcst_minus_1,
     fcst_date,
]


def preload_all_data(forecast_timestamps):
    data = {}  # structure: data[var][gage][timestamp] = df

    for var in variables:
        data[var] = {}
        for gage in gage_ids:
            for ts in forecast_timestamps:
                url = f"https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/{ts}/Timeseries/{var}/{gage}.tsv"
                response = requests.get(url)
                if response.status_code == 200:
                    try:
                        df = pd.read_csv(io.StringIO(response.text), sep="\t")
                        if 'Datetime(GMT)' in df.columns:
                            df['Datetime(GMT)'] = pd.to_datetime(df['Datetime(GMT)'])
                            df.set_index('Datetime(GMT)', inplace=True)

                            # Init nested dict
                            if gage not in data[var]:
                                data[var][gage] = {}

                            # Store by full timestamp
                            #data[var][gage][ts] = df[['obs', 'model']].copy()
                            data[var][gage][ts] = df[['model']].copy()

                    except Exception as e:
                        pass
                        print(f"Error parsing {url}: {e}")
                else:
                    pass
                    #print(f"URL not found: {url} — status code {response.status_code}")
    return data

# Load all 3 forecasts
data_cache = preload_all_data(forecast_timestamps)


@st.cache_data(ttl=10800)  # Cache expires after 3 hours

def fetch_latest_observations():
    # # =============================================================================
    # # NOAA latest observations
    # # =============================================================================
    
    gages_df_copy = gages_df.copy().set_index('site_no')
    
    time_window = timedelta(days=2)
    
    start_dt = datetime.strptime(fcst_date, "%Y%m%d%H")  # Corrected format string

    end_dt   = datetime.now() + time_window
    
    # meto start date
    start_dt_metostat = start_dt - time_window
    end_dt_metostat = datetime.now()
    
    
    # time format for noaa
    start_fmt_noaa = (start_dt - time_window).strftime("%Y%m%d")
    end_fmt_noaa   = end_dt.strftime("%Y%m%d")
    
    
    # time format for usgs
    start_fmt_usgs = (start_dt - time_window).strftime('%Y-%m-%d')
    end_fmt_usgs   = end_dt.strftime('%Y-%m-%d')
    
    
    # Unified container for NOAA observations
    noaa_data = {
        "gage height": pd.DataFrame(),     # Water level
        "wspd": pd.DataFrame(),   # Wind speed
        "wdir": pd.DataFrame(),   # Wind direction
        "pres": pd.DataFrame()    # Pressure
    }
    
    usgs_data = {
        "gage height": pd.DataFrame(),     # Water level
        "wspd": pd.DataFrame(),   # Wind speed
        "wdir": pd.DataFrame(),   # Wind direction
        "pres": pd.DataFrame(),    # Pressure
        "prcp": pd.DataFrame()    # Precip
    }
    
    metostat_data = {
        #"gage height": pd.DataFrame(),     # Water level
        "wspd": pd.DataFrame(),   # Wind speed
        "wdir": pd.DataFrame(),   # Wind direction
        "pres": pd.DataFrame(),    # Pressure
        "prcp": pd.DataFrame()    # Precip
    }
    
    
    
    # =============================================================================
    # fetch 'gage_height'
    # =============================================================================
    for stn_id in tqdm.tqdm(gages_for_var['gage_height'], desc="Fetching gage_height"):
        if gages_df_copy.loc[stn_id]['agency'] == 'noaa':
            stn = Station(id=stn_id)
            # Water level
            df_wl = stn.get_data(
                begin_date=start_fmt_noaa,
                end_date=end_fmt_noaa,
                product="water_level",
                datum="NAVD",
                units="english",
                time_zone="gmt"
            )
            noaa_data["gage height"][stn_id] = df_wl["v"].asfreq('h') # convert to hourly
            
        elif gages_df_copy.loc[stn_id]['agency'] == 'usgs':
            # get stage height
            siteINFO = nwis.get_info(sites=stn_id)
            gage_height = float(siteINFO[0]['alt_va'])
            
            # pull record
            df = nwis.get_record(
                sites=stn_id, 
                service='iv', 
                start=start_fmt_usgs, 
                end=end_fmt_usgs, 
                parameterCd='00065'
            )
            df.index = df.index.tz_localize(None)
            df = df.select_dtypes(include=['number']).resample('h').max()
            
            # add gage corrections
            df = df+gage_height
            
            if len(df.columns)>1:
                # Find the column with "navd88" in its name
                navd88_col = next((col for col in df.columns if "navd88" in col.lower()), None)
                
                if navd88_col:
                    usgs_data["gage height"][stn_id] = df[navd88_col]
                else:
                    st.warning(f"No NAVD88 column found for station {stn_id}")
            else:
                usgs_data["gage height"][stn_id] = df  # convert to hourly
            
       
    # # =============================================================================
    # # fetch 'pressure', 'wind_speed', 'precipitation'
    # # =============================================================================
    
    for var in tqdm.tqdm(['pressure', 'wind_speed','precipitation'], desc="Fetching air pressure and winds"):
        if var == 'pressure':
            var1 = 'air_pressure'
            usgs_var = '00025'
        elif var == 'wind_speed':
            var1 = 'wind'
            usgs_var = ['00035','00036']
        elif var == 'precipitation':
            var1 = 'precipitation'
            usgs_var = '00045'   
            
        for stn_id in tqdm.tqdm(gages_for_var[var], desc=f"Checking {var} stations"): 
            #print(stn_id)
            #stn_id=['0204293125'
            if gages_df_copy.loc[stn_id]['agency'] == 'noaa': #=========================== NOAA
                stn = Station(id=stn_id)
                # Water level
                df_ = stn.get_data(
                    begin_date=start_fmt_noaa,
                    end_date=end_fmt_noaa,
                    product=var1,
                    #datum="NAVD",
                    units="english",
                    time_zone="gmt"
                )
                if var == 'pressure':
                    noaa_data['pres'][stn_id] = df_["v"].asfreq('h') # convert to hourly
                elif var == 'wind_speed':
                    noaa_data["wspd"][stn_id] = df_["s"].asfreq('h') # convert to hourly
                    noaa_data["wdir"][stn_id] = df_["d"].asfreq('h') # convert to hourly
             
            elif gages_df_copy.loc[stn_id]['agency'] == 'usgs': #=========================== USGS
                if var1=='wind':
                    for usgs_cd in usgs_var:
                        # wind
                        df = nwis.get_record(
                            sites=stn_id, 
                            service='iv', 
                            start=start_fmt_usgs, 
                            end=end_fmt_usgs, 
                            parameterCd=usgs_cd
                        )
                        #print(df[usgs_cd].head())
                        if usgs_cd == '00035':
                            usgs_data["wspd"][stn_id] = df[usgs_cd].asfreq('h') # convert to hourly
                        elif usgs_cd == '00036':
                             usgs_data["wdir"][stn_id] = df[usgs_cd].asfreq('h') # convert to hourly
                else:
                    # pressure and precip
                    df = nwis.get_record(
                        sites=stn_id, 
                        service='iv', 
                        start=start_fmt_usgs, 
                        end=end_fmt_usgs, 
                        parameterCd=usgs_var
                    )
                    
                    if var == 'pressure':
                        usgs_data['pres'][stn_id] = df[usgs_var].asfreq('h') # convert to hourly
                    elif var == 'precipitation':
                        usgs_data['prcp'][stn_id] = df[usgs_var].asfreq('h') # convert to hourly
                    
            elif gages_df_copy.loc[stn_id]['agency'] == 'metostat': #=========================== Metostat
                #print(gages_df_copy.loc[stn_id]['agency'],stn_id)
                data = Hourly(stn_id, start_dt_metostat, end_dt_metostat)
                df = data.fetch()
                
                if df.empty:
                    continue 
                # Store each variable by station
                metostat_data["prcp"][stn_id] = df["prcp"].astype(float)
                metostat_data["wspd"][stn_id] = df["wspd"]
                metostat_data["wdir"][stn_id] = df["wdir"]
                metostat_data["pres"][stn_id] = df["pres"]
    
    
    # # Unit conversion: wind speed from km/h to m/s
    # # Unit conversion: precipitation from mm to inches
    
    metostat_data["wspd"] = metostat_data["wspd"].astype(float) * kmph_to_mps
    metostat_data["prcp"] = metostat_data["prcp"].astype(float) * mm_to_inches
          

    # Return all gathered data for later use
    return noaa_data, usgs_data, metostat_data


# On first load, initialize and fetch data once
if "noaa_data" not in st.session_state:
    st.session_state.noaa_data = {}
    st.session_state.usgs_data = {}
    st.session_state.metostat_data = {}

    noaa_data, usgs_data, metostat_data = fetch_latest_observations()
    st.session_state.noaa_data = noaa_data
    st.session_state.usgs_data = usgs_data
    st.session_state.metostat_data = metostat_data

# Button to manually fetch newer observations
if st.button("Fetch New Observations"):
    with st.spinner("Fetching latest observations..."):
        noaa_data, usgs_data, metostat_data = fetch_latest_observations()
        st.session_state.noaa_data = noaa_data
        st.session_state.usgs_data = usgs_data
        st.session_state.metostat_data = metostat_data
    st.success("Observations updated!")

# Use session state stored data for plots
noaa_data = st.session_state.noaa_data
usgs_data = st.session_state.usgs_data
metostat_data = st.session_state.metostat_data


# =============================================================================
# Plots
# =============================================================================

colors = ['gray', 'yellow', 'red']
color_cycle = itertools.cycle(colors)


for selected_var in data_cache.keys():
    st.subheader(f"📈 Time Series at Gages — {selected_var.replace('_',' ').title()}")

    gages_with_data = data_cache[selected_var].keys()

    for gage in gages_with_data:
        gage_data = data_cache[selected_var][gage]

        if not gage_data:
            st.write(f"Skipping gage {gage}, no forecast data available.")
            continue

        fig = go.Figure()
        obs_plotted = False

        # Find max forecast time among all forecast datasets for this gage
        max_fcst_time = None
        for ts in gage_data.keys():
            df = gage_data[ts]
            if df is not None and not df.empty:
                max_fcst_time = max(max_fcst_time, df.index.max()) if max_fcst_time else df.index.max()

        # Plot all forecast model lines
        for ts in sorted(gage_data.keys()):
            df = gage_data[ts]
            ts_dt = datetime.strptime(ts, '%Y%m%d%H')
            ts_label = ts_dt.strftime('Forecast %HZ — %b %d')

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['model'],
                mode='lines',
                name=ts_label,
                line=dict(width=2, color=next(color_cycle))
            ))
            
            
            if ts == fcst_date: # only do it once
                # Select observation data sources depending on variable
                if selected_var == "gage_height":
                    if gage in noaa_data["gage height"]:
                        obs_ts = noaa_data["gage height"][gage]
                    if gage in usgs_data["gage height"]:
                        obs_ts = usgs_data["gage height"][gage]
                    # Add metostat if available for gage height (usually not)  
                # windspeed plots
                elif selected_var == "wind_speed":
                    if gage in metostat_data["wspd"]:
                        obs_ts = metostat_data["wspd"][gage]
                    if gage in noaa_data["wspd"]:
                        obs_ts = noaa_data["wspd"][gage]
                    if gage in usgs_data["wspd"]:
                        obs_ts = usgs_data["wspd"][gage]
                # wind direction plots
                elif selected_var == "wind_direction":
                    if gage in metostat_data["wdir"]:
                        obs_ts = metostat_data["wdir"][gage]
                    if gage in noaa_data["wdir"]:
                        obs_ts = noaa_data["wdir"][gage]
                    if gage in usgs_data["wdir"]:
                        obs_ts = usgs_data["wdir"][gage]
                        
                # pressure plots
                elif selected_var == "pressure":
                    if gage in metostat_data["pres"]:
                        obs_ts = metostat_data["pres"][gage]
                    if gage in noaa_data["pres"]:
                        obs_ts = noaa_data["pres"][gage]
                    if gage in usgs_data["pres"]:
                        obs_ts = usgs_data["pres"][gage]
                # precip plots
                elif selected_var == "precipitation":
                    if gage in metostat_data["prcp"]:
                        obs_ts = metostat_data["prcp"][gage]
                    elif gage in usgs_data["prcp"]:
                        obs_ts = usgs_data["prcp"][gage]
                           
                # now add the observations           
                fig.add_trace(go.Scatter(
                    x=obs_ts.index,
                    y=obs_ts,
                    mode='lines',
                    name='Observed',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                ))       
                    
        # Get site name if available
        site_name = gages_df.loc[gages_df['site_no'] == gage, 'site_name'].values[0] if gage in gages_df['site_no'].values else gage

        fig.update_layout(
            title=f"{selected_var.replace('_',' ').title()} — {site_name} ({gage})",
            xaxis_title="Datetime (GMT)",
            yaxis_title=variables_units[selected_var],
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    
    
    
    
    
