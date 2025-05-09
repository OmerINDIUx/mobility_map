import pandas as pd
import geopandas as gpd
import torch
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

# Función para leer el archivo CSV y GeoJSON
def cargar_datos(csv_file, geojson_file):
    # Leer el archivo CSV
    df_csv = pd.read_csv(csv_file)
    
    # Leer el archivo GeoJSON con geopandas
    gdf_geojson = gpd.read_file(geojson_file)
    
    # Asegurarse de que ambos archivos tienen una columna común (taz_id)
    df_merged = pd.merge(df_csv, gdf_geojson, on='taz_id', how='inner')
    
    return df_merged, gdf_geojson

# Función para generar TAZs utilizando DBSCAN
def generar_tazs(df, eps=0.005, min_samples=5):
    coords = df[["xlat", "xlon"]].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df["taz_id"] = clustering.labels_
    return df

# Función para calcular la matriz OD
def calcular_od(df, device="cuda"):
    df = df[df["taz_id"] != -1]  # Eliminar ruido
    df["hour_code"] = df["hour_code"].astype(int)
    
    # Agrupar actividad por TAZ y hora
    grouped = df.groupby(["taz_id", "hour_code"])["activity_index_total"].sum().reset_index()
    pivot = grouped.pivot(index="taz_id", columns="hour_code", values="activity_index_total").fillna(0)
    
    # Cargar en VRAM
    activity_tensor = torch.tensor(pivot.values, device=device).float()
    
    # Calcular la distancia entre TAZs
    taz_coords = df.groupby("taz_id")[["xlat", "xlon"]].mean().loc[pivot.index]
    coords_tensor = torch.tensor(taz_coords.values, device=device)
    dists = torch.cdist(coords_tensor, coords_tensor, p=2)
    dists[dists == 0] = 1e-3  # Evitar división por cero

    flows = {}
    for h in range(activity_tensor.shape[1]):
        o = activity_tensor[:, h].unsqueeze(1)
        d = activity_tensor[:, h].unsqueeze(0)
        G = (o @ d) / (dists ** 2)  # Modelo de gravedad
        
        flows[h] = G.detach().cpu().numpy()

    return pivot.index.to_list(), flows  # Lista de TAZs, diccionario de flujos por hora

# Función para exportar la matriz OD a un archivo GeoJSON
def exportar_od(taz_ids, flows, taz_coords, output_file="od_flows.geojson"):
    features = []
    for h, mat in flows.items():
        for i, orig in enumerate(taz_ids):
            for j, dest in enumerate(taz_ids):
                if i != j and mat[i][j] > 0:
                    o_coord = taz_coords.loc[orig]
                    d_coord = taz_coords.loc[dest]
                    geom = {
                        "type": "LineString",
                        "coordinates": [
                            [o_coord["xlon"], o_coord["xlat"]],
                            [d_coord["xlon"], d_coord["xlat"]]
                        ]
                    }
                    features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "origin": int(orig),
                            "destination": int(dest),
                            "hour": int(h),
                            "flow": float(mat[i][j])
                        }
                    })

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"✅ OD exportado: {output_file}")

# Main: Cargar, generar TAZs, calcular OD y exportar resultados
if __name__ == "__main__":
    # Rutas de los archivos
    csv_file = "data/seattle_sample_daily.csv"
    geojson_file = "data/seattle_sample_vehicle_density_2020_Q1.geojson"

    print("📥 Cargando datos...")
    df_merged, gdf_geojson = cargar_datos(csv_file, geojson_file)

    print("🔍 Generando TAZs...")
    df_merged = generar_tazs(df_merged)  # Generar TAZs con DBSCAN
    
    print("🧠 Calculando matriz OD en GPU...")
    taz_ids, flows = calcular_od(df_merged, device="cuda")

    print("🗺️ Exportando a GeoJSON...")
    taz_coords = df_merged.groupby("taz_id")[["xlat", "xlon"]].mean()
    exportar_od(taz_ids, flows, taz_coords, output_file="od_map.geojson")

    print("✅ Proceso terminado.")
