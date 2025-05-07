import folium
import geopandas as gpd
from folium.plugins import TimeSliderChoropleth
import pandas as pd
import json

# Cargar archivo
gdf = gpd.read_file("seattle_sample_vehicle_density_2020_Q1.geojson")

# Timestamp por hour_code
gdf["timestamp"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(gdf["hour_code"], unit="h")

# Usar road_id como ID único
gdf["id"] = gdf["road_id"]

# Convertir geometrías a GeoJSON simple
gdf["geometry_json"] = gdf["geometry"].apply(lambda geom: json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"])

# Crear FeatureCollection
features = []
for _, row in gdf.iterrows():
    features.append({
        "type": "Feature",
        "geometry": row["geometry_json"],
        "properties": {"id": row["id"]}
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Crear styledict para TimeSliderChoropleth
styledict = {}

for _, row in gdf.iterrows():
    feature_id = row["id"]
    timestamp = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")
    opacity = min(row["activity_index"] * 10, 1.0)
    color = "red"  # puedes hacer esto dinámico si quieres
    styledict.setdefault(feature_id, {})[timestamp] = {
        "color": color,
        "opacity": opacity
    }

# Crear mapa
m = folium.Map(location=[47.6062, -122.3321], zoom_start=13)

# Añadir capa con slider
TimeSliderChoropleth(
    data=geojson,
    styledict=styledict
).add_to(m)

# Guardar
m.save("seattle_activity_slider.html")
print("Mapa guardado como seattle_activity_slider.html")
