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




# Set the remote URL for the CSV

Gages_URL = "https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/2025081806/gages_found.csv"
Perimeter_URL = "https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/2025081806/model_domain.csv"
Cells_URL = "https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/2025081806/mesh_cells_points.csv"



# =============================================================================
# timeseries viewer
# =============================================================================
fcst_date='2025081806'

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
# Gage Data Preview
# =============================================================================
st.subheader(f"Latest Forecast Date: {fcst_date[:-2]}")
# st.subheader("📄 Gage Data Preview")
# st.dataframe(gages_df[['site_no','site_name','agency']].head(20))

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
# load timeseries data
# =============================================================================

forecast_hours = ["06", "18"]

#@st.cache_data(show_spinner="Loading all timeseries data...")
def preload_all_data(base_date):
    data = {}  # structure: data[var][gage][hour] = df
    for var in variables:
        data[var] = {}
        for gage in gage_ids:
            for hour in forecast_hours:
                fcst_path = f"{base_date}{hour}"
                url = f"https://data.iflood.vse.gmu.edu/Forecast/HECRAS2D_DC/{fcst_path}/Timeseries/{var}/{gage}.tsv"
                response = requests.get(url)
                if response.status_code == 200:
                    try:
                        df = pd.read_csv(io.StringIO(response.text), sep="\t")
                        if 'Datetime(GMT)' in df.columns:
                            df['Datetime(GMT)'] = pd.to_datetime(df['Datetime(GMT)'])
                            df.set_index('Datetime(GMT)', inplace=True)

                            # Initialize if not already
                            if gage not in data[var]:
                                data[var][gage] = {}

                            # Store separately by forecast hour
                            data[var][gage][hour] = df[['obs', 'model']].copy()
                    except Exception as e:
                        print(f"Error parsing {url}: {e}")
    return data

base_date = "20250818"  # YYYYMMDD
data_cache = preload_all_data(base_date)


# =============================================================================
# plots
# =============================================================================
# User selection

variables = list(gages_for_var.keys())
# selected_var = st.selectbox("Select variable", variables)


# # Check if there is any data for the selected variable
# gage_ids_with_data = gages_for_var.get(selected_var, [])


# selected_var = st.selectbox("Select variable", list(data_cache.keys()))


# Iterate over all available variables in data_cache
for selected_var in data_cache.keys():
    st.subheader(f"📈 Time Series at Gages — {selected_var.replace('_',' ').title()}")

    gages_with_data = data_cache[selected_var].keys()

    for gage in gages_with_data:
        gage_data = data_cache[selected_var][gage]

        # Skip if neither forecast exists
        if not any(k in gage_data for k in ['06', '18']):
            st.write(f"Skipping gage {gage}, no forecast data available.")
            continue

        fig = go.Figure()

        if '06' in gage_data:
            df_06 = gage_data['06']
            fig.add_trace(go.Scatter(
                x=df_06.index,
                y=df_06['model'],
                mode='lines',
                name='Forecast 06Z',
                line=dict(color='red')
            ))

        if '18' in gage_data:
            df_18 = gage_data['18']
            fig.add_trace(go.Scatter(
                x=df_18.index,
                y=df_18['model'],
                mode='lines',
                name='Forecast 18Z',
                line=dict(color='yellow')#, dash='dot
            ))

        # Add observed if available (same in both runs)
        if 'obs' in df_06.columns:
            fig.add_trace(go.Scatter(
                x=df_06.index,
                y=df_06['obs'],
                mode='lines',
                name='Observed',
                line=dict(color='blue')
            ))

        # Site name if available
        site_name = gages_df.loc[gages_df['site_no'] == gage, 'site_name'].values[0] if gage in gages_df['site_no'].values else gage

        fig.update_layout(
            title=f"{selected_var.replace('_',' ').title()} — {site_name} ({gage})",
            xaxis_title="Datetime (GMT)",
            yaxis_title=variables_units[selected_var],
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        # Create a new div-like section for each variable and gage
        with st.container():
            st.plotly_chart(fig, use_container_width=True)
