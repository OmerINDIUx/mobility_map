# import geopandas as gpd
# import pandas as pd
# import numpy as np
# import torch
# import umap
# import hdbscan
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning)

# # ================================
# # ⚙️ 0. Verificar uso de GPU
# # ================================
# print("⚙️ 0. Verificar uso de GPU")
# cuda_available = torch.cuda.is_available()
# device = torch.device("cuda" if cuda_available else "cpu")
# print("CUDA disponible:", cuda_available)

# if cuda_available:
#     gpu_name = torch.cuda.get_device_name(0)
#     mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
#     print(f"🧠 Usando dispositivo: {gpu_name}")
#     print(f"💾 Memoria usada: {mem_alloc:.2f} MB")

# # ================================
# # 📥 1. Cargar CSV y GeoJSON
# # ================================
# print("📥 1. Cargar CSV y GeoJSON")
# print(f"🧠 Usando dispositivo: {gpu_name}")
# print(f"💾 Memoria usada: {mem_alloc:.2f} MB")
# csv_file = "data/seattle_sample_daily.csv"
# geojson_file = "data/seattle_sample_vehicle_density_2020_Q1.geojson"

# df_csv = pd.read_csv(csv_file)
# gdf_roads = gpd.read_file(geojson_file)

# # Imprimir las primeras filas del archivo CSV
# print("Primeras filas del CSV:")
# print(df_csv.head())

# # Imprimir las primeras filas del archivo GeoJSON
# print("Primeras filas del GeoJSON:")
# print(gdf_roads.head())

# # Ver tipos de columnas para cada archivo
# print("\nTipos de columnas del CSV:")
# print(df_csv.dtypes)

# print("\nTipos de columnas del GeoJSON:")
# print(gdf_roads.dtypes)

# # ================================
# # 🔑 2. Extraer clave de 'road_id' para empatar con 'geography'
# # ================================
# print("🔑 2. Extraer clave de 'road_id' para empatar con 'geography'")
# print(f"🧠 Usando dispositivo: {gpu_name}")
# print(f"💾 Memoria usada: {mem_alloc:.2f} MB")
# df_csv["geography"] = df_csv["geography"].astype(str)
# gdf_roads["road_id_suffix"] = gdf_roads["road_id"].str.extract(r";(\d+)$")

# # Verifica si geography termina en ese sufijo
# df_csv["geo_suffix"] = df_csv["geography"].str[-11:]
# print("Ejemplo comparación:")
# print(df_csv[["geography", "geo_suffix"]].head())
# print(gdf_roads[["road_id", "road_id_suffix"]].drop_duplicates().head())
# print(df_csv["geo_suffix"].unique())
# print(gdf_roads["road_id_suffix"].unique())
# gdf_roads["geography_key"] = gdf_roads["road_id"].str.extract(r";(\d+)$")
# print("geo_suffix:", df_csv["geo_suffix"].head())
# print("road_id_suffix:", gdf_roads["road_id_suffix"].head())

# df_csv["geo_suffix"] = df_csv["geo_suffix"].astype(str)
# gdf_roads["road_id_suffix"] = gdf_roads["road_id_suffix"].astype(str)

# # ================================
# # 📍 3. Reproyectar geometrías y calcular centroides
# # ================================
# print("📍 3. Reproyectar geometrías y calcular centroides"  )
# print(f"🧠 Usando dispositivo: {gpu_name}")
# print(f"💾 Memoria usada: {mem_alloc:.2f} MB")
# # Reproyectar a CRS proyectado (ejemplo: UTM)
# gdf_roads = gdf_roads.to_crs(epsg=3857)  # EPSG 3857 es un CRS proyectado en metros

# # Calcular centroides después de reproyectar
# gdf_roads["xlat"] = gdf_roads.geometry.centroid.y
# gdf_roads["xlon"] = gdf_roads.geometry.centroid.x

# # ================================
# # 🔗 4. Hacer merge entre CSV y GeoJSON
# # ================================
# print("🔗 4. Hacer merge entre CSV y GeoJSON")
# df_csv["geography"] = df_csv["geography"].astype(str).str.strip()
# gdf_roads["geography_key"] = gdf_roads["geography_key"].astype(str).str.strip()

# # Merge basado en la clave de geografía
# merged = pd.merge(
#     df_csv,
#     gdf_roads[["geography_key", "xlat", "xlon"]],
#     how="left",
#     left_on="geo_suffix",  # Cambiado de geography_key a geo_suffix
#     right_on="geography_key"
# )

# # Renombrar las columnas 'xlat_x' y 'xlon_x' a 'xlat' y 'xlon'
# merged = merged.rename(columns={"xlat_x": "xlat", "xlon_x": "xlon"})

# # Verificar las columnas del DataFrame mergeado
# print("Columnas después del merge:", merged.columns)

# # Asegurarse de que las columnas de latitud y longitud estén presentes
# if "xlat" in merged.columns and "xlon" in merged.columns:
#     print("Las columnas xlat y xlon están presentes.")
# else:
#     print("¡Error! Las columnas xlat y xlon no están presentes.")

# # ================================
# # 📊 5. Crear tabla pivote
# # ================================
# print("📊 5. Crear tabla pivote")
# # Crear tabla pivote correctamente usando las nuevas columnas 'xlat' y 'xlon'
# pivot_df = merged.pivot_table(
#     values="activity_index_total",
#     index=["xlat", "xlon"],
#     columns=["agg_day_period"],
#     aggfunc="sum",
#     fill_value=0
# )
# pivot_df.columns.name = None

# # Verifica si el pivote se creó correctamente
# print("pivot_df head:", pivot_df.head())
# print("pivot_df", pivot_df.isnull().sum())

# # ================================
# # 📏 6. Escalado
# # ================================
# print("📏 6. Escalado")
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(pivot_df)

# # ================================
# # 🔽 7. Reducción con UMAP
# # ================================
# print("🔽 7. Reducción con UMAP")
# reducer = umap.UMAP(
#     n_neighbors=15,
#     min_dist=0.1,
#     n_components=2,
#     metric='euclidean',
#     transform_seed=42,
#     random_state=42
# )
# embedding = reducer.fit_transform(X_scaled)

# # ================================
# # 🔍 8. Clustering con HDBSCAN
# # ================================
# print("🔍 8. Clustering con HDBSCAN")
# clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
# cluster_labels = clusterer.fit_predict(embedding)

# # ================================
# # 🧩 9. Añadir clusters al DataFrame
# # ================================
# print("🧩 9. Añadir clusters al DataFrame")
# pivot_df["cluster"] = cluster_labels
# pivot_df = pivot_df.reset_index()

# # ================================
# # 📈 10. Graficar resultados
# # ================================
# print("📈 10. Graficar resultados")
# plt.figure(figsize=(10, 6))
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=cluster_labels,
#     cmap="Spectral",
#     s=10,
#     alpha=0.8
# )
# plt.colorbar(label="Cluster")
# plt.title("Clusters de segmentos viales (HDBSCAN + UMAP)")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ================================
# # 🌍 11. Exportar GeoJSON
# # ================================
# print("🌍 11. Exportar GeoJSON")
# # Verificar que las coordenadas estén correctamente presentes
# if "xlat" not in merged.columns or "xlon" not in merged.columns:
#     print("¡Las coordenadas no están correctamente presentes!")
# else:
#     print("Coordenadas válidas:", merged[["xlat", "xlon"]].dropna().shape)

# # Exportar GeoJSON con los clusters
# cluster_gdf = gpd.GeoDataFrame(
#     pivot_df,
#     geometry=gpd.points_from_xy(pivot_df["xlon"], pivot_df["xlat"]),  # Usar "xlon" y "xlat" si son las correctas
#     crs="EPSG:4326"
# )
# cluster_gdf.to_file("seattle_clustered_segments.geojson", driver="GeoJSON")
# print("✅ GeoJSON con clusters guardado como 'seattle_clustered_segments.geojson'")


import os
import csv
import pandas as pd
import geopandas as gpd
import numpy as np
import cupy as cp
from tqdm import tqdm
from shapely.geometry import LineString
import geojson
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Parámetros
# -----------------------------
csv_file = "data/seattle_sample_daily.csv"
geojson_file = "data/seattle_sample_vehicle_density_2020_Q1.geojson"
output_dir = "output"
epsilon = 100.0
block_size = 50
flow_threshold = 0.001  # Mínimo para guardar línea OD en geojson

# -----------------------------
# Leer y procesar datos
# -----------------------------
df_csv = pd.read_csv(csv_file)
gdf = gpd.read_file(geojson_file)
gdf = gdf.to_crs(epsg=3857)
gdf["centroid"] = gdf.geometry.centroid
gdf["xlon"] = gdf["centroid"].x
gdf["xlat"] = gdf["centroid"].y

# Normalizar actividad
scaler = MinMaxScaler()
df_csv["activity_index_total"] = scaler.fit_transform(df_csv[["activity_index_total"]])

# Segmentos
segments = gdf["road_id"].unique()
segment_geoms = gdf.set_index("road_id")["centroid"].to_dict()
coords_all = [[segment_geoms[seg].x, segment_geoms[seg].y] for seg in segments]

# -----------------------------
# Inicialización
# -----------------------------
num_blocks = len(segments) // block_size + (1 if len(segments) % block_size else 0)
dist_matrix = cp.zeros((len(segments), len(segments)), dtype=cp.float32)

# -----------------------------
# Calcular matriz de distancias
# -----------------------------
with tqdm(total=num_blocks * num_blocks, desc="Calculando matriz de distancias") as pbar:
    for i in range(num_blocks):
        for j in range(num_blocks):
            start_i, end_i = i * block_size, min((i + 1) * block_size, len(segments))
            start_j, end_j = j * block_size, min((j + 1) * block_size, len(segments))

            coords_i = cp.array(coords_all[start_i:end_i])
            coords_j = cp.array(coords_all[start_j:end_j])

            dx = coords_i[:, cp.newaxis, 0] - coords_j[cp.newaxis, :, 0]
            dy = coords_i[:, cp.newaxis, 1] - coords_j[cp.newaxis, :, 1]
            dist_block = cp.sqrt(dx**2 + dy**2)

            dist_matrix[start_i:end_i, start_j:end_j] = dist_block
            pbar.update(1)

# -----------------------------
# Calcular matriz OD
# -----------------------------
activity = df_csv.groupby("geography")["activity_index_total"].mean()
activity = activity.reindex(segments).fillna(0).values
activity_gpu = cp.asarray(activity).reshape(-1, 1)

od_matrix = cp.zeros((len(segments), len(segments)), dtype=cp.float32)

with tqdm(total=num_blocks * num_blocks, desc="Calculando matriz OD") as pbar:
    for i in range(num_blocks):
        start_i, end_i = i * block_size, min((i + 1) * block_size, len(segments))
        ai = activity_gpu[start_i:end_i]

        for j in range(num_blocks):
            start_j, end_j = j * block_size, min((j + 1) * block_size, len(segments))
            aj = activity_gpu[start_j:end_j]

            d_sub = dist_matrix[start_i:end_i, start_j:end_j] + epsilon
            m_block = (ai @ aj.T) / d_sub

            od_matrix[start_i:end_i, start_j:end_j] = m_block
            pbar.update(1)

# -----------------------------
# Normalizar matriz OD
# -----------------------------
od_matrix /= cp.max(od_matrix)
od_matrix_np = cp.asnumpy(od_matrix)

# -----------------------------
# Crear carpeta de salida
# -----------------------------
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Guardar CSV con barra de progreso
# -----------------------------
# csv_path = os.path.join(output_dir, "od_matrix.csv")
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([""] + list(segments))  # Header

#     for i, row in enumerate(tqdm(od_matrix_np, desc="Guardando CSV")):
#         writer.writerow([segments[i]] + list(row))

# print(f"✅ CSV guardado en: {csv_path}")

# -----------------------------
# Guardar GeoJSON
# -----------------------------
features = []
for i in tqdm(range(len(segments)), desc="Generando GeoJSON"):
    for j in range(len(segments)):
        flow = od_matrix_np[i, j]
        if i != j:  # evitar líneas entre el mismo punto

            coord_i = coords_all[i]
            coord_j = coords_all[j]
            line = LineString([coord_i, coord_j])
            feature = geojson.Feature(
                geometry=line,
                properties={
                    "origin": segments[i],
                    "destination": segments[j],
                    "flow": float(flow)
                }
            )
            features.append(feature)

geojson_path = os.path.join(output_dir, "od_matrix.geojson")
with open(geojson_path, "w") as f:
    geojson.dump(geojson.FeatureCollection(features), f)

print(f"✅ GeoJSON guardado en: {geojson_path}")
