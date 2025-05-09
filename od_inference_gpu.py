# od_inference_gpu.py

import torch
import pandas as pd
from sklearn.cluster import DBSCAN

def generar_tazs(df, eps=0.005, min_samples=5):
    coords = df[["xlat", "xlon"]].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df["taz_id"] = clustering.labels_
    return df

def calcular_od(df, device="cuda"):
    df = df[df["taz_id"] != -1]  # Eliminar ruido
    df["hour_code"] = df["hour_code"].astype(int)
    
    # Agrupar la actividad por TAZ y hora
    grouped = df.groupby(["taz_id", "hour_code"])["activity_index_total"].sum().reset_index()
    pivot = grouped.pivot(index="taz_id", columns="hour_code", values="activity_index_total").fillna(0)
    
    # Cargar los datos en VRAM
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

    import geopandas as gpd
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"✅ OD exportado: {output_file}")
