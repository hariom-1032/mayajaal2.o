import shapefile
import json
import os

shp_path = "camlocations/CAMERA_POINTS.shp"
if not os.path.exists(shp_path):
    raise FileNotFoundError(f"Cannot find {shp_path}")

sf = shapefile.Reader(shp_path)
records = sf.shapeRecords()

marker_colors = {
    1: "#10b981",
    2: "#10b981",
    3: "#ef4444",
    4: "#ef4444",
    5: "#ef4444",
    6: "#ef4444",
    7: "#f59e0b",
    8: "#10b981",
    9: "#10b981",
    10: "#10b981",
}

features = []
for sr in records:
    d = sr.record.as_dict()
    ch = int(d["Channel"])
    name = str(d["Cam_name"]).strip()
    loc = str(d["Location"]).strip()
    ctype = str(d["Type"]).strip()
    lon, lat = sr.shape.points[0]

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "id": f"cam-ch-{ch}",
            "channel": ch,
            "name": name,
            "location": loc,
            "type": ctype,
            "title": f"Channel {ch}: {name}",
            "longitude": lon,
            "latitude": lat,
            "markerColor": marker_colors.get(ch, "#10b981")
        }
    })

features.sort(key=lambda x: x["properties"]["channel"])

geojson_data = {
    "type": "FeatureCollection",
    "name": "camlocations",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    },
    "features": features
}

os.makedirs("frontend/public/data", exist_ok=True)

with open("frontend/public/data/camlocations.geojson", "w") as f:
    json.dump(geojson_data, f, indent=2)

with open("frontend/public/data/recam_locations.geojson", "w") as f:
    json.dump(geojson_data, f, indent=2)

print("Generated camlocations.geojson and recam_locations.geojson successfully!")
for feat in features:
    p = feat["properties"]
    print(f"Channel {p['channel']}: {p['name']} ({p['location']}) [{p['type']}] -> lat: {p['latitude']:.6f}, lng: {p['longitude']:.6f}")
