from flask import jsonify, Flask, render_template, request, Response
import geopandas as gpd
import json
import csv

app = Flask(__name__)

# Carga el GeoDataFrame desde un archivo
gdf = gpd.read_file("data/seattle_sample_vehicle_density_2020_Q1.geojson")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/roads")
def get_roads():
    day_codes = request.args.getlist("day_codes[]", type=int)
    hour_start = request.args.get("hour_start", default=0, type=int)
    hour_end = request.args.get("hour_end", default=23, type=int)
    road_types = request.args.getlist("road_types[]")

    # Mostrar qué hay en el GeoDataFrame
    print("Todos los días disponibles:", gdf["day_code"].unique().tolist())
    print("Todas las horas disponibles:", gdf["hour_code"].unique().tolist())
    print("Todos los tipos de vía disponibles:", gdf["road_type"].unique().tolist())
    print("DAY CODES:", day_codes)
    print("HOURS:", hour_start, hour_end)
    print("ROAD TYPES:", road_types)

    # Filtros
    filtered = gdf.copy()
    filtered = filtered[
        (filtered["hour_code"] >= hour_start) & (filtered["hour_code"] <= hour_end)
    ]
    if day_codes:
        filtered = filtered[filtered["day_code"].isin(day_codes)]
    if road_types:
        filtered = filtered[filtered["road_type"].isin(road_types)]

    # Fallback para depurar
    if filtered.empty:
        print("⚠️ No se encontraron features con ese filtro. Devuelvo datos de prueba.")
        filtered = gdf.head(10)

    # Convertir a GeoJSON
    return Response(
        json.dumps(json.loads(filtered.to_crs(epsg=4326).to_json())),
        mimetype='application/json'
    )



@app.route('/api/daily_activity')
def daily_activity():
    features = []

    with open('data/seattle_sample_daily.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lon = float(row['xlon'])
            lat = float(row['xlat'])
            activity = float(row['activity_index_total'])

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "geography": row["geography"],
                    "activity_index": activity,
                    "date": row["agg_day_period"]
                }
            })

    return jsonify({
        "type": "FeatureCollection",
        "features": features
    })


if __name__ == "__main__":
    app.run(debug=True)
