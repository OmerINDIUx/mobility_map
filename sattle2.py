import cudf
import cuspatial
import geopandas as gpd

# Cargar el archivo GeoJSON con GeoPandas
gdf = gpd.read_file("seattle_sample_vehicle_density_2020_Q1.geojson")

# Convertir a cuDF para procesamiento en GPU
gpu_df = cuspatial.from_geopandas(gdf)

# Filtrar los datos (por ejemplo, con un índice de actividad >= 0.01)
gpu_df_filtered = gpu_df[gpu_df['activity_index'] >= 0.01]

# Agrupar los datos (por ejemplo, por 'road_id' y 'hour_code') y calcular el valor máximo de 'activity_index'
agg_gpu = gpu_df_filtered.groupby(['road_id', 'hour_code']).agg({'activity_index': 'max'}).reset_index()

# Convertir los resultados de nuevo a Pandas para su integración con la visualización
agg_df = agg_gpu.to_pandas()

import pydeck as pdk
import pandas as pd

# Crear un DataFrame de Pandas con los datos agregados
df = pd.DataFrame(agg_df)

# Asumiendo que tienes las coordenadas de latitud y longitud en el DataFrame
# Aquí se utilizarían las coordenadas correspondientes a cada "road_id" y "hour_code"
df['lat'] = 47.623901  # Latitud ejemplo
df['lon'] = -122.328491  # Longitud ejemplo

# Crear una capa de puntos en Deck.gl
scatter = pdk.Layer(
    "ScatterplotLayer",
    df,
    get_position=["lon", "lat"],
    get_radius=100,
    get_fill_color=[255, 0, 0],  # Rojo para mostrar actividad
    pickable=True,
)

# Crear el mapa con Deck.gl
view_state = pdk.ViewState(latitude=47.623901, longitude=-122.328491, zoom=12)

# Crear la visualización
deck = pdk.Deck(layers=[scatter], initial_view_state=view_state)

# Mostrar la visualización en un notebook (si usas Jupyter) o exportar como HTML
deck.to_html("seattle_trafico.html")

from keplergl import KeplerGl

# Crear un mapa en Kepler.gl
map_1 = KeplerGl(height=600)

# Puedes cargar un DataFrame de Pandas directamente para crear la visualización
map_1.add_data(data=df, name="traffic_data")

# Mostrar el mapa en un notebook (si usas Jupyter)
map_1
