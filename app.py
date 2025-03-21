import os
import glob
import json
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.warp import transform as rio_transform
from rasterio.transform import xy
from rasterio.mask import mask
from shapely.geometry import mapping, Point
import geopandas as gpd
import pyproj
from scipy.stats import pearsonr

from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

###############################################################################
# 1. OPENAI CHATBOT SETUP
###############################################################################
def get_chatbot_response(user_message):
    """
    Calls the OpenAI Chat Completions API to get a chatbot response.
    """
    system_prompt = (
        "You are the friendly assistant of the Assaba Region EDA Satellite Data Visualization Dashboard (Beta)."
        "This is an early version, and you're here to help users explore satellite imagery, weather patterns, and other regional insights. You can answer general questions about satellite data and what’s visible on the dashboard."
        "Keep responses concise, clear, and upbeat — you’re excited to help, but always stay grounded in the dashboard data."
        "Avoid speculation or external advice. Stick to what the dashboard can show. Follow all standard safety guardrails."
    )
    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=150)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return "Sorry, I encountered an error processing your request."

###############################################################################
# 2. HELPER FUNCTIONS
###############################################################################
import markdown

def get_lat_lon_arrays_for_clipped_data(clipped_data, clipped_transform, src_crs):
    """
    Return 2D arrays (lon2d, lat2d) in EPSG:4326 for each pixel center.
    """
    _, height, width = clipped_data.shape
    rows, cols = np.indices((height, width))

    xs = clipped_transform.c + (cols * clipped_transform.a) + (rows * clipped_transform.b)
    ys = clipped_transform.f + (cols * clipped_transform.d) + (rows * clipped_transform.e)

    xs_flat = xs.flatten()
    ys_flat = ys.flatten()

    if src_crs and src_crs.to_string() != "EPSG:4326":
        lons, lats = rio_transform(src_crs, "EPSG:4326", xs_flat, ys_flat)
    else:
        lons, lats = xs_flat, ys_flat

    lon2d = np.array(lons).reshape(height, width)
    lat2d = np.array(lats).reshape(height, width)
    return lon2d, lat2d

def raster_to_points_latlon(raster, transform, src_crs, sample_step=10):
    rows, cols = raster.shape
    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    data = []
    for r in range(0, rows, sample_step):
        for c in range(0, cols, sample_step):
            val = raster[r, c]
            if np.isnan(val):
                continue
            x, y = xy(transform, r, c, offset='center')
            lon, lat = transformer.transform(x, y)
            data.append((lat, lon, val))
    df = pd.DataFrame(data, columns=['lat', 'lon', 'value'])
    return df

def load_raster(file_path, is_gpp=False):
    with rasterio.open(file_path) as src:
        arr = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs

        if is_gpp:
            arr[arr == 65533] = np.nan
            arr[arr <= 0] = np.nan
            arr[arr > 4000] = np.nan
            arr *= 0.0001
        else:
            arr[arr <= 0] = np.nan

    return arr, transform, crs

def resample_to_match(src_data, src_transform, src_crs,
                      dst_data, dst_transform, dst_crs,
                      method=Resampling.bilinear):
    resampled = np.zeros_like(dst_data, dtype=float)
    reproject(
        source=src_data,
        destination=resampled,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=method
    )
    return resampled

def detect_significant_change(gpp_start, gpp_end, threshold=200):
    gpp_diff = gpp_end - gpp_start
    invalid_mask = np.isnan(gpp_start) | np.isnan(gpp_end)
    change_mask = np.abs(gpp_diff) > threshold
    change_mask[invalid_mask] = False
    return change_mask

def detect_urbanization(pop_start, pop_end, gpp_change_mask, pop_threshold=50):
    pop_diff = pop_end - pop_start
    pop_increase_mask = pop_diff > pop_threshold
    min_rows = min(pop_increase_mask.shape[0], gpp_change_mask.shape[0])
    min_cols = min(pop_increase_mask.shape[1], gpp_change_mask.shape[1])
    pop_increase_cropped = pop_increase_mask[:min_rows, :min_cols]
    gpp_change_cropped = gpp_change_mask[:min_rows, :min_cols]
    hotspots = pop_increase_cropped & gpp_change_cropped
    return hotspots, pop_diff[:min_rows, :min_cols]

def raster_center(raster, transform):
    rows, cols = raster.shape
    x1, y1 = xy(transform, 0, 0)
    x2, y2 = xy(transform, rows - 1, cols - 1)
    return {'lat': (y1 + y2) / 2, 'lon': (x1 + x2) / 2}

###############################################################################
# 3. MAIN ANALYSIS
###############################################################################
def run_analysis():
    # Adjust these paths for your environment
    modis_folder = 'Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP/'
    pop_folder = 'Datasets_Hackathon/Gridded_Population_Density_Data/'

    gpp_files = sorted(glob.glob(os.path.join(modis_folder, '*GP.tif')))
    pop_files = sorted(glob.glob(os.path.join(pop_folder, '*Pop_*.tif')))

    gpp_2010_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2010'), None)
    gpp_2020_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2020'), None)

    pop_2010_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2010'), None)
    pop_2015_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2015'), None)
    pop_2020_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2020'), None)

    if not (gpp_2010_path and gpp_2020_path and pop_2010_path and pop_2015_path and pop_2020_path):
        raise ValueError("Missing one or more required raster files.")

    gpp_2010, gpp_2010_transform, gpp_2010_crs = load_raster(gpp_2010_path, is_gpp=True)
    gpp_2020, gpp_2020_transform, gpp_2020_crs = load_raster(gpp_2020_path, is_gpp=True)

    pop_2010, pop_2010_transform, pop_2010_crs = load_raster(pop_2010_path, is_gpp=False)
    pop_2015, pop_2015_transform, pop_2015_crs = load_raster(pop_2015_path, is_gpp=False)
    pop_2020, pop_2020_transform, pop_2020_crs = load_raster(pop_2020_path, is_gpp=False)

    # Resample population if needed
    if gpp_2010_crs != pop_2010_crs or gpp_2010.shape != pop_2010.shape:
        pop_2010 = resample_to_match(pop_2010, pop_2010_transform, pop_2010_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    if gpp_2010_crs != pop_2015_crs or gpp_2010.shape != pop_2015.shape:
        pop_2015 = resample_to_match(pop_2015, pop_2015_transform, pop_2015_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    if gpp_2010_crs != pop_2020_crs or gpp_2010.shape != pop_2020.shape:
        pop_2020 = resample_to_match(pop_2020, pop_2020_transform, pop_2020_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)

    GPP_THRESHOLD = 200
    gpp_change_mask = detect_significant_change(gpp_2010, gpp_2020, threshold=GPP_THRESHOLD)
    POP_THRESHOLD = 50
    hotspots, pop_diff = detect_urbanization(pop_2010, pop_2020, gpp_change_mask, pop_threshold=POP_THRESHOLD)
    gpp_diff = gpp_2020 - gpp_2010

    return {
        'pop_2010': pop_2010,
        'pop_2015': pop_2015,
        'pop_2020': pop_2020,
        'pop_diff': pop_diff,
        'gpp_diff': gpp_diff,
        'hotspots': hotspots,
        'transform': gpp_2010_transform,
        'crs': gpp_2010_crs
    }

# Load main results
results = run_analysis()
pop_2010 = results["pop_2010"]
pop_2015 = results["pop_2015"]
pop_2020 = results["pop_2020"]
pop_diff = results["pop_diff"]
gpp_diff = results["gpp_diff"]
hotspots = results["hotspots"]
transform_used = results["transform"]
crs_used = results["crs"]

###############################################################################
# 4. PREPARE STATIC FIGURES
###############################################################################
df_gpp_diff = raster_to_points_latlon(gpp_diff, transform_used, crs_used, sample_step=10)
df_gpp_diff = df_gpp_diff.dropna(subset=["value"])
df_gpp_diff = df_gpp_diff[df_gpp_diff["value"] > 0]

df_hotspots = raster_to_points_latlon(hotspots.astype(float), transform_used, crs_used, sample_step=10)
df_hotspots = df_hotspots.dropna(subset=["value"])
df_hotspots = df_hotspots[df_hotspots["value"] > 0]

fig_gpp_diff = px.density_map(
    df_gpp_diff, lat="lat", lon="lon", z="value",
    radius=10,
    center={"lat": 17, "lon": -11},
    zoom=6,
    map_style="carto-positron",
    title="GPP Difference (2010→2020)"
)
fig_gpp_diff.update_layout(width=1800, height=800)

fig_hotspots = px.density_map(
    df_hotspots, lat="lat", lon="lon", z="value",
    radius=10,
    center={"lat": 17, "lon": -11},
    zoom=6,
    map_style="carto-positron",
    title="Urbanization/Deforestation Hotspots"
)
fig_hotspots.update_layout(width=1800, height=800)

###############################################################################
# 5. BUILD DYNAMIC DICTIONARIES FOR POP & GPP
###############################################################################
df_pop_2010 = raster_to_points_latlon(pop_2010, transform_used, crs_used, sample_step=10)
df_pop_2010 = df_pop_2010.dropna(subset=["value"])
df_pop_2010 = df_pop_2010[df_pop_2010["value"] > 0]

df_pop_2015 = raster_to_points_latlon(pop_2015, transform_used, crs_used, sample_step=10)
df_pop_2015 = df_pop_2015.dropna(subset=["value"])
df_pop_2015 = df_pop_2015[df_pop_2015["value"] > 0]

df_pop_2020 = raster_to_points_latlon(pop_2020, transform_used, crs_used, sample_step=10)
df_pop_2020 = df_pop_2020.dropna(subset=["value"])
df_pop_2020 = df_pop_2020[df_pop_2020["value"] > 0]

pop_years_dict = {
    2010: df_pop_2010,
    2015: df_pop_2015,
    2020: df_pop_2020
}

# Administrative shapefile
muni_gdf = gpd.read_file("Datasets_Hackathon/Admin_layers/Assaba_Districts_layer.shp")
municipality_options = sorted(muni_gdf["ADM3_EN"].unique())

# GPP timeseries
gpp_timeseries_dict = {}
modis_folder = "Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP/"
gpp_years = range(2010, 2024)
for yr in gpp_years:
    file_path = os.path.join(modis_folder, f"{yr}_GP.tif")
    if os.path.exists(file_path):
        arr, transf, crs_ = load_raster(file_path, is_gpp=True)
        df = raster_to_points_latlon(arr, transf, crs_, sample_step=10)
        df = df.dropna(subset=["value"])
        df = df[df["value"] > 0]
        gpp_timeseries_dict[yr] = df

###############################################################################
# 6. PRECIPITATION BY DISTRICT
###############################################################################
def compute_precip_by_district(tif_path, shp_path):
    """
    For each district, mask the raster, compute mean precipitation.
    """
    gdf = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        means = []
        for _, row in gdf.iterrows():
            geom = [row.geometry]
            out_data, _ = mask(src, geom, crop=True)
            out_data = out_data.astype(float)
            if src.nodata is not None:
                out_data[out_data == src.nodata] = np.nan
            out_data[out_data == 0] = np.nan
            mean_val = np.nanmean(out_data)
            means.append(mean_val)
        gdf["precip"] = means
    return gdf

data_dir = "Datasets_Hackathon/Climate_Precipitation_Data/"
shp_file = "Datasets_Hackathon/Admin_layers/Assaba_Districts_layer.shp"

precip_choro_dict = {}
tif_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".tif")])
for tif in tif_files:
    try:
        year = int(tif[:4])
    except:
        continue
    file_path = os.path.join(data_dir, tif)
    gdf_year = compute_precip_by_district(file_path, shp_file)
    for col in gdf_year.columns:
        if ptypes.is_datetime64_any_dtype(gdf_year[col]):
            gdf_year[col] = gdf_year[col].astype(str)
    with rasterio.open(file_path) as ds:
        if gdf_year.crs != "EPSG:4326":
            gdf_year = gdf_year.to_crs(epsg=4326)
    if "ADM3_EN" not in gdf_year.columns:
        gdf_year["ADM3_EN"] = gdf_year.index.astype(str)
    precip_choro_dict[year] = gdf_year

if precip_choro_dict:
    precip_years = sorted(precip_choro_dict.keys())
else:
    precip_years = [2010]

###############################################################################
# 7. GPP BY DISTRICT
###############################################################################
def compute_gpp_by_district(tif_path, shp_path):
    """
    Mask the GPP raster for each district and compute mean GPP.
    """
    gdf = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        means = []
        for _, row in gdf.iterrows():
            geom = [row.geometry]
            out_data, _ = mask(src, geom, crop=True)
            out_data = out_data.astype(float)
            out_data[out_data <= 0] = np.nan
            out_data[out_data > 4000] = np.nan
            if src.nodata is not None:
                out_data[out_data == src.nodata] = np.nan
            out_data *= 0.0001
            mean_val = np.nanmean(out_data)
            means.append(mean_val)
        gdf["gpp_mean"] = means
    return gdf

gpp_choro_dict = {}
gpp_folder = "Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP/"
tif_files = sorted([f for f in os.listdir(gpp_folder) if f.endswith(".tif")])
for tif in tif_files:
    try:
        year = int(tif[:4])
    except:
        continue
    file_path = os.path.join(gpp_folder, tif)
    gdf_year = compute_gpp_by_district(file_path, shp_file)
    with rasterio.open(file_path) as ds:
        if gdf_year.crs != "EPSG:4326":
            gdf_year = gdf_year.to_crs(epsg=4326)
    if "ADM3_EN" not in gdf_year.columns:
        gdf_year["ADM3_EN"] = gdf_year.index.astype(str)
    gpp_choro_dict[year] = gdf_year

if gpp_choro_dict:
    gpp_years = sorted(gpp_choro_dict.keys())
else:
    gpp_years = [2010]

###############################################################################
# 8. LAND COVER HELPER & DATA
###############################################################################
landcover_data_folder = "Datasets_Hackathon/Modis_Land_Cover_Data/"

land_cover_classes = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Permanent Snow and Ice",
    16: "Barren"
}
valid_classes = set(land_cover_classes.keys())

def create_land_cover_geodataframe(tif_file, shp_file, year):
    """
    Creates a GeoDataFrame of land cover data for the whole map,
    reprojected to EPSG:4326.
    """
    assaba_map = gpd.read_file(shp_file)
    with rasterio.open(tif_file) as dataset:
        raster_crs = dataset.crs
        assaba_map = assaba_map.to_crs(raster_crs)
        try:
            clipped_data, clipped_transform = mask(dataset, assaba_map.geometry.map(mapping), crop=True)
        except ValueError:
            print(f"⚠️ Skipping {year} - No data for the raster.")
            return None

    clipped_data = clipped_data.astype(float)
    if dataset.nodata is not None:
        clipped_data[clipped_data == dataset.nodata] = np.nan
    clipped_data[~np.isin(clipped_data, list(valid_classes))] = np.nan

    lon2d, lat2d = get_lat_lon_arrays_for_clipped_data(clipped_data, clipped_transform, dataset.crs)
    _, height, width = clipped_data.shape

    points = []
    land_cover_codes = []
    for row in range(height):
        for col in range(width):
            val = clipped_data[0, row, col]
            if not np.isnan(val):
                lat = lat2d[row, col]
                lon = lon2d[row, col]
                points.append(Point(lon, lat))
                land_cover_codes.append(int(val))

    data = {
        "land_cover_code": land_cover_codes,
        "land_cover_class": [land_cover_classes.get(code, "Unknown") for code in land_cover_codes],
        "year": year,
        "geometry": points,
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf

landcover_tif_files = sorted([f for f in os.listdir(landcover_data_folder) if f.endswith(".tif")])
landcover_years = []
for tif in landcover_tif_files:
    try:
        year = int(tif[:4])
        landcover_years.append(year)
    except:
        pass
landcover_years = sorted(list(set(landcover_years)))

landcover_gdfs = {}
for tif in landcover_tif_files:
    try:
        year = int(tif[:4])
    except:
        continue
    file_path = os.path.join(landcover_data_folder, tif)
    gdf = create_land_cover_geodataframe(file_path, shp_file, year)
    if gdf is not None and not gdf.empty:
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        landcover_gdfs[year] = gdf

###############################################################################
# 9. MUNICIPALITY COMPARISON
###############################################################################
def extract_gpp_for_municipality(tif_file, shp_file, year, municipality_name):
    assaba_map = gpd.read_file(shp_file)
    with rasterio.open(tif_file) as dataset:
        raster_crs = dataset.crs  
        assaba_map = assaba_map.to_crs(raster_crs)  
        municipality = assaba_map[assaba_map["ADM3_EN"] == municipality_name]
        if municipality.empty:
            return None
        try:
            clipped_data, _ = mask(dataset, municipality.geometry.map(mapping), crop=True)
        except ValueError:
            return None
    clipped_data = clipped_data.astype(float)
    clipped_data[clipped_data == dataset.nodata] = np.nan
    clipped_data[clipped_data > 4000] = np.nan
    clipped_data *= 0.0001
    mean_gpp = np.nanmean(clipped_data)
    return {"Year": year, "Municipality": municipality_name, "Mean": mean_gpp}

def extract_precip_for_municipality(tif_file, shp_file, year, municipality_name):
    assaba_map = gpd.read_file(shp_file)
    with rasterio.open(tif_file) as dataset:
        raster_crs = dataset.crs  
        assaba_map = assaba_map.to_crs(raster_crs)  
        municipality = assaba_map[assaba_map["ADM3_EN"] == municipality_name]
        if municipality.empty:
            return None
        try:
            municipality_data, _ = mask(dataset, municipality.geometry.map(mapping), crop=True)
        except ValueError:
            return None
    municipality_data = municipality_data.astype(float)
    municipality_data[municipality_data == dataset.nodata] = np.nan
    mean_precip = np.nanmean(municipality_data)
    return {"Year": year, "Municipality": municipality_name, "Mean": mean_precip}

gpp_data_dir = "Datasets_Hackathon/MODIS_Gross_Primary_Production_GPP"
precip_data_dir = "Datasets_Hackathon/Climate_Precipitation_Data"
gpp_tif_files = sorted([f for f in os.listdir(gpp_data_dir) if f.endswith(".tif")])
precip_tif_files = sorted([f for f in os.listdir(precip_data_dir) if f.endswith(".tif")])

###############################################################################
# 10. MULTI-LANGUAGE TEXT LOADERS
###############################################################################
def load_text_markdown(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

language_dict = {
    "EN": {
        "header": "Addax",
        "population": "Population Over Time",
        "precipitation": "Precipitation Over Time (District-level)",
        "gpp": "GPP Over Time (District-level)",
        "land_cover": "Land Cover Map",
        "muni_comparison": "Municipality Comparison",
        "compare": "Compare",
        "additional_info": load_text_markdown("files_translation/english_report.md")
    },
    "FR": {
        "header": "Addax",
        "population": "Population au fil du temps",
        "precipitation": "Précipitations par district",
        "gpp": "Production Primaire Brute (GPP) par district",
        "land_cover": "Carte de Couverture du Sol",
        "muni_comparison": "Comparaison entre Municipalités",
        "compare": "Comparer",
        "additional_info": load_text_markdown("files_translation/french_report.md")
    },
    "AR": {
        "header": "أداكس",
        "population": "السكان مع مرور الوقت",
        "precipitation": "هطول الأمطار حسب المقاطعات",
        "gpp": "الإنتاج الأولي الإجمالي (GPP) حسب المقاطعات",
        "land_cover": "خريطة الغطاء الأرضي",
        "muni_comparison": "مقارنة بين البلديات",
        "compare": "مقارنة",
        "additional_info": load_text_markdown("files_translation/arabic_report.md")
    }
}

###############################################################################
# 11. DASH APP LAYOUT
###############################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# For consistent coloring of land cover classes:
color_discrete_map = {
    "Water": "#1f77b4",
    "Evergreen Needleleaf Forest": "#2ca02c",
    "Evergreen Broadleaf Forest": "#98df8a",
    "Deciduous Needleleaf Forest": "#8c564b",
    "Deciduous Broadleaf Forest": "#d62728",
    "Mixed Forests": "#9467bd",
    "Closed Shrublands": "#ff7f0e",
    "Open Shrublands": "#e377c2",
    "Woody Savannas": "#7f7f7f",
    "Savannas": "#bcbd22",
    "Grasslands": "#17becf",
    "Permanent Wetlands": "#aec7e8",
    "Croplands": "#ffbb78",
    "Urban and Built-up": "#ff9896",
    "Cropland/Natural Vegetation Mosaic": "#c5b0d5",
    "Permanent Snow and Ice": "#c49c94",
    "Barren": "#f7b6d2"
}

app.layout = dbc.Container([
    # Header row with language switcher
    dbc.Row([
        dbc.Col(
            html.H1(
                id="header-text",
                className="text-center mb-4",
                style={"color": "#2C3E50"}  # modern dark blue
            ),
            xs=12, sm=12, md=10
        ),
        dbc.Col(
            dcc.Dropdown(
                id="language-dropdown",
                options=[
                    {"label": "🇬🇧 EN", "value": "EN"},
                    {"label": "🇫🇷 FR", "value": "FR"},
                    {"label": "🇦🇪 AR", "value": "AR"}
                ],
                value="EN",
                clearable=False,
                searchable=False,
                style={"width": "100px"}
            ),
            xs=12, sm=12, md=2,
            style={"textAlign": "right", "marginTop": "20px"}
        )
    ], className="mb-4"),

    # Main content row: left side (tabs) and right side (additional info + chatbot)
    dbc.Row([
        # Left column (Tabs)
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label=language_dict["EN"]["population"], children=[
                    html.Div([
                        html.H3(id="population-title", style={"marginTop": "15px"}),
                        dcc.Slider(
                            id='pop-year-slider',
                            min=2010,
                            max=2020,
                            step=5,
                            value=2010,
                            marks={2010: '2010', 2015: '2015', 2020: '2020'}
                        ),
                        dcc.Graph(
                            id='pop-year-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '60vh'}
                        ),
                        dcc.Graph(
                            id='pop-timeseries-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '25vh', "marginTop": "20px"}
                        ),
                        html.Div(id="population-desc", className="text-muted", style={'marginTop': '10px'})
                    ])
                ]),
                dcc.Tab(label=language_dict["EN"]["precipitation"], children=[
                    html.Div([
                        html.H3(id="precipitation-title", style={"marginTop": "15px"}),
                        dcc.Slider(
                            id='precip-year-slider',
                            min=min(precip_years),
                            max=max(precip_years),
                            step=1,
                            value=min(precip_years),
                            marks={yr: str(yr) for yr in precip_years}
                        ),
                        dcc.Graph(
                            id='precip-year-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '60vh'}
                        ),
                        dcc.Graph(
                            id='precip-timeseries-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '25vh', "marginTop": "20px"}
                        ),
                        html.Div(id="precipitation-desc", className="text-muted", style={'marginTop': '10px'})
                    ])
                ]),
                dcc.Tab(label=language_dict["EN"]["gpp"], children=[
                    html.Div([
                        html.H3(id="gpp-title", style={"marginTop": "15px"}),
                        dcc.Slider(
                            id='gpp-year-slider',
                            min=min(gpp_years),
                            max=max(gpp_years),
                            step=1,
                            value=min(gpp_years),
                            marks={yr: str(yr) for yr in gpp_years}
                        ),
                        dcc.Graph(
                            id='gpp-year-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '60vh'}
                        ),
                        dcc.Graph(
                            id='gpp-timeseries-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '25vh', "marginTop": "20px"}
                        ),
                        html.Div(id="gpp-desc", className="text-muted", style={'marginTop': '10px'})
                    ])
                ]),
                dcc.Tab(label=language_dict["EN"]["land_cover"], children=[
                    html.Div([
                        html.H3(id="landcover-title", style={"marginTop": "15px"}),
                        dcc.Slider(
                            id='landcover-year-slider',
                            min=min(landcover_years),
                            max=max(landcover_years),
                            step=1,
                            value=min(landcover_years),
                            marks={yr: str(yr) for yr in landcover_years}
                        ),
                        dcc.Graph(
                            id='landcover-map-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '60vh'}
                        ),
                        dcc.Graph(
                            id='landcover-timeseries-graph',
                            className="graph-container",
                            style={'width': '100%', 'height': '25vh', "marginTop": "20px"}
                        ),
                        html.Div(id="landcover-desc", className="text-muted", style={'marginTop': '10px'})
                    ])
                ]),
                dcc.Tab(label=language_dict["EN"]["muni_comparison"], children=[
                    html.Div([
                        html.H3(id="muni-comparison-title", style={"marginTop": "15px"}),
                        dbc.Row([
                            dbc.Col([
                                html.Label(id="comparison-type-label"),
                                dcc.Dropdown(
                                    id='comparison-type-dropdown',
                                    options=[
                                        {'label': 'GPP', 'value': 'GPP'},
                                        {'label': 'Precipitation', 'value': 'Precipitation'}
                                    ],
                                    value='GPP',
                                    clearable=False,
                                    searchable=False
                                )
                            ], xs=12, sm=12, md=4),
                            dbc.Col([
                                html.Label(id="muni1-label"),
                                dcc.Dropdown(
                                    id='muni1-dropdown',
                                    options=[{'label': m, 'value': m} for m in municipality_options],
                                    value=municipality_options[0]
                                )
                            ], xs=12, sm=12, md=4),
                            dbc.Col([
                                html.Label(id="muni2-label"),
                                dcc.Dropdown(
                                    id='muni2-dropdown',
                                    options=[{'label': m, 'value': m} for m in municipality_options],
                                    value=municipality_options[1] if len(municipality_options) > 1 else municipality_options[0]
                                )
                            ], xs=12, sm=12, md=4)
                        ], style={"marginTop": "10px"}),
                        html.Button(
                            language_dict["EN"]["compare"],
                            id='compare-button',
                            n_clicks=0, 
                            className="btn btn-primary",
                            style={'margin': '10px'}
                        ),
                        dcc.Graph(
                            id='municipality-comparison-graph',
                            className="graph-container"
                        )
                    ])
                ])
            ], style={"backgroundColor": "#ffffff", "padding": "15px", "borderRadius": "5px"})
        ], xs=12, sm=12, md=6, style={"backgroundColor": "#f8f9fa", "padding": "15px", "borderRadius": "5px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"}),

        # Right column: Additional info + Chatbot
        dbc.Col([
            # Additional Info Card
            dbc.Card([
                dbc.CardHeader(
                    html.H4(id="additional-info-title", style={"color": "#2C3E50"})
                ),
                dbc.CardBody(
                    dcc.Markdown(
                        id="additional-info-text",
                        style={
                            "fontFamily": "Arial, sans-serif",
                            "maxHeight": "400px",
                            "overflowY": "auto",
                            "overflowX": "hidden",
                            "whiteSpace": "normal",
                            "wordBreak": "break-all"
                        }
                    )
                )
            ], className="mb-4", style={"backgroundColor": "#ffffff", "border": "none", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"}),

            # Chatbot Card
            dbc.Card([
                dbc.CardHeader("Dashboard Chatbot", style={"fontWeight": "bold", "color": "#2C3E50"}),
                dbc.CardBody([
                    html.Div(
                        id='chat-history',
                        children=[],
                        style={
                            'overflowY': 'auto',
                            'height': '200px',
                            'border': '1px solid #ccc',
                            'padding': '10px',
                            'backgroundColor': '#ffffff',
                            'borderRadius': '5px'
                        }
                    ),
                    html.Br(),
                    dbc.InputGroup([
                        dbc.Input(
                            id='chat-input',
                            placeholder='Enter your message...',
                            type='text'
                        ),
                        dbc.Button("Ask", id='chat-send', n_clicks=0, color="primary")
                    ])
                ], style={"marginBottom": "50px"})
            ], style={"backgroundColor": "#ffffff", "border": "none", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"})
        ], xs=12, sm=12, md=6, style={"backgroundColor": "#f5f5f5", "padding": "15px", "borderRadius": "5px"})
    ], className="mb-4")
],
fluid=True,
style={"backgroundColor": "#eaeaea", "minHeight": "100vh", "padding": "20px"}
)


###############################################################################
# 12. CALLBACKS
###############################################################################
@app.callback(
    Output('header-text', 'children'),
    Output('population-title', 'children'),
    Output('population-desc', 'children'),
    Output('precipitation-title', 'children'),
    Output('precipitation-desc', 'children'),
    Output('gpp-title', 'children'),
    Output('gpp-desc', 'children'),
    Output('landcover-title', 'children'),
    Output('landcover-desc', 'children'),
    Output('muni-comparison-title', 'children'),
    Output('comparison-type-label', 'children'),
    Output('muni1-label', 'children'),
    Output('muni2-label', 'children'),
    Output('compare-button', 'children'),
    Output('additional-info-title', 'children'),
    Output('additional-info-text', 'children'),
    Input('language-dropdown', 'value')
)
def update_language(lang):
    texts = language_dict.get(lang, language_dict["EN"])
    return (
        texts["header"],
        texts["population"],
        texts["population"],
        texts["precipitation"],
        texts["precipitation"],
        texts["gpp"],
        texts["gpp"],
        texts["land_cover"],
        texts["land_cover"],
        texts["muni_comparison"],
        "Comparison Type:",
        "Municipality 1:",
        "Municipality 2:",
        texts["compare"],
        "Additional Information",
        texts["additional_info"]
    )

@app.callback(
    Output('pop-year-graph', 'figure'),
    Input('pop-year-slider', 'value')
)
def update_population_map(selected_year):
    if selected_year not in pop_years_dict:
        return px.scatter_mapbox()
    df = pop_years_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value',
        radius=10,
        center={"lat": 17, "lon": -11},
        zoom=6,
        map_style="carto-positron",
        title=f"Population for {selected_year}",
        hover_data={'value': True, 'lat': False, 'lon': False}
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), autosize=True)
    return fig

@app.callback(
    Output('pop-timeseries-graph', 'figure'),
    Input('pop-year-slider', 'value')
)
def update_pop_timeseries(selected_year):
    years = sorted(pop_years_dict.keys())
    totals = []
    for yr in years:
        df = pop_years_dict[yr]
        totals.append(df["value"].sum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=totals, mode='lines+markers', name="Total Population"))
    fig.add_vline(x=selected_year, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Total Population Over Time",
        xaxis_title="Year",
        yaxis_title="Total Population",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

@app.callback(
    Output('gpp-year-graph', 'figure'),
    Input('gpp-year-slider', 'value')
)
def update_gpp_map(selected_year):
    if selected_year not in gpp_timeseries_dict:
        return px.scatter_mapbox()
    df = gpp_timeseries_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value',
        radius=10,
        center={"lat": 17, "lon": -11},
        zoom=6,
        map_style="carto-positron",
        title=f"GPP for {selected_year}"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), autosize=True)
    return fig

@app.callback(
    Output('gpp-timeseries-graph', 'figure'),
    Input('gpp-year-slider', 'value')
)
def update_gpp_timeseries(selected_year):
    years = sorted(gpp_timeseries_dict.keys())
    means = []
    for yr in years:
        df = gpp_timeseries_dict[yr]
        means.append(df["value"].mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=means, mode='lines+markers', name="Mean GPP"))
    fig.add_vline(x=selected_year, line_dash="dash", line_color="red")
    fig.update_layout(
        title="GPP Mean Over Time",
        xaxis_title="Year",
        yaxis_title="Mean GPP",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

@app.callback(
    Output('precip-year-graph', 'figure'),
    Input('precip-year-slider', 'value')
)
def update_precip_map(selected_year):
    if selected_year not in precip_choro_dict:
        return px.scatter_mapbox()
    gdf = precip_choro_dict[selected_year]
    gdf_json = json.loads(gdf.to_json())
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations="ADM3_EN",
        featureidkey="properties.ADM3_EN",
        color="precip",
        color_continuous_scale="Blues",
        range_color=(0, gdf["precip"].max()),
        mapbox_style="carto-positron",
        zoom=6,
        center={"lat": 17, "lon": -11},
        opacity=0.8,
        labels={"precip": "Precip (mm)"},
        title=f"Precipitation for {selected_year}",
        hover_data={"ADM3_EN": True, "precip": True}
    )
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), autosize=True)
    return fig

@app.callback(
    Output('precip-timeseries-graph', 'figure'),
    Input('precip-year-slider', 'value')
)
def update_precip_timeseries(selected_year):
    years = sorted(precip_choro_dict.keys())
    means = []
    for yr in years:
        gdf = precip_choro_dict[yr]
        means.append(gdf["precip"].mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=means, mode='lines+markers', name="Mean Precipitation"))
    fig.add_vline(x=selected_year, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Precipitation Mean Over Time",
        xaxis_title="Year",
        yaxis_title="Mean Precipitation (mm)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

@app.callback(
    Output('landcover-map-graph', 'figure'),
    Input('landcover-year-slider', 'value')
)
def update_landcover_map(selected_year):
    gdf = landcover_gdfs.get(selected_year)
    if gdf is None or gdf.empty:
        return px.scatter_map(title=f"No Land Cover data for {selected_year}")
    fig = px.scatter_map(
        gdf,
        lat="lat",
        lon="lon",
        color="land_cover_class",
        hover_data=["land_cover_code"],
        zoom=6,
        center={"lat": 17, "lon": -11},
        title=f"Land Cover for {selected_year}",
        map_style="carto-positron",
        color_discrete_map=color_discrete_map
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

@app.callback(
    Output('landcover-timeseries-graph', 'figure'),
    Input('landcover-year-slider', 'value')
)
def update_landcover_timeseries(selected_year):
    data_records = []
    years = sorted(landcover_gdfs.keys())
    for yr in years:
        gdf = landcover_gdfs.get(yr)
        if gdf is None or gdf.empty:
            continue
        class_counts = gdf["land_cover_class"].value_counts()
        for lc_class, cnt in class_counts.items():
            data_records.append({"Year": yr, "Class": lc_class, "Count": cnt})
    df = pd.DataFrame(data_records)
    if df.empty:
        return go.Figure()
    fig = px.line(
        df,
        x="Year",
        y="Count",
        color="Class",
        markers=True,
        title="Land Cover Classes Over Time",
        color_discrete_map=color_discrete_map
    )
    fig.update_traces(
        mode='lines+markers+text',
        text=df["Count"],
        texttemplate='%{text:.0f}',
        textposition='top center'
    )
    fig.add_vline(x=selected_year, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Pixel Count",
        legend_title="Land Cover Class",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

@app.callback(
    Output('municipality-comparison-graph', 'figure'),
    Input('compare-button', 'n_clicks'),
    State('muni1-dropdown', 'value'),
    State('muni2-dropdown', 'value'),
    State('comparison-type-dropdown', 'value')
)
def update_municipality_comparison(n_clicks, muni1, muni2, comp_type):
    if not n_clicks:
        raise PreventUpdate

    muni1_data = []
    muni2_data = []
    if comp_type == "GPP":
        for tif in gpp_tif_files:
            try:
                year = int(tif[:4])
            except:
                continue
            file_path = os.path.join(gpp_data_dir, tif)
            d1 = extract_gpp_for_municipality(file_path, shp_file, year, muni1)
            d2 = extract_gpp_for_municipality(file_path, shp_file, year, muni2)
            if d1 is not None:
                muni1_data.append(d1)
            if d2 is not None:
                muni2_data.append(d2)
        yaxis_title = "Mean GPP (kg_C/m²/year)"
        title = "GPP Comparison of Municipalities"
    else:
        for tif in precip_tif_files:
            try:
                year = int(tif[:4])
            except:
                continue
            file_path = os.path.join(precip_data_dir, tif)
            d1 = extract_precip_for_municipality(file_path, shp_file, year, muni1)
            d2 = extract_precip_for_municipality(file_path, shp_file, year, muni2)
            if d1 is not None:
                muni1_data.append(d1)
            if d2 is not None:
                muni2_data.append(d2)
        yaxis_title = "Mean Precipitation (mm)"
        title = "Precipitation Comparison of Municipalities"

    muni1_df = pd.DataFrame(muni1_data)
    muni2_df = pd.DataFrame(muni2_data)

    fig = go.Figure()
    if not muni1_df.empty:
        fig.add_trace(
            go.Scatter(x=muni1_df["Year"], y=muni1_df["Mean"], mode='lines+markers', name=f'{muni1}')
        )
    if not muni2_df.empty:
        fig.add_trace(
            go.Scatter(x=muni2_df["Year"], y=muni2_df["Mean"], mode='lines+markers', name=f'{muni2}')
        )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# Chatbot callback
@app.callback(
    Output('chat-history', 'children'),
    Input('chat-send', 'n_clicks'),
    State('chat-input', 'value'),
    State('chat-history', 'children')
)
def update_chat(n_clicks, user_message, chat_history):
    if n_clicks == 0 or not user_message:
        raise PreventUpdate

    user_div = html.Div(
        f"User: {user_message}",
        style={'margin': '5px 0', 'fontWeight': 'bold', 'color': '#2c3e50'}
    )
    bot_response = get_chatbot_response(user_message)
    bot_div = html.Div(
        f"Bot: {bot_response}",
        style={'margin': '5px 0', 'color': '#34495e'}
    )
    updated_history = chat_history + [user_div, bot_div]
    return updated_history


###############################################################################
# 13. RUN
###############################################################################
if __name__ == '__main__':
    app.run(
      debug=True,
      host='0.0.0.0',
      port='8050')
