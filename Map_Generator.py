import geopandas as gpd
import folium

path = r"C:\3rd Smester Project\Optimization Project\newzealandpaul-Shipping-Lanes-b0ad85c\data\Shipping_Lanes_v1.geojson"
save_path = r"C:\3rd Smester Project\Optimization Project\Routes.html"

routes = gpd.read_file(path)

# Ensure CRS is EPSG:4326
if routes.crs is None:
    print("Warning: input has no CRS — assuming EPSG:4326 (lon/lat).")
else:
    if routes.crs.to_string() != "EPSG:4326":
        routes = routes.to_crs("EPSG:4326")

minx, miny, maxx, maxy = 25, -10, 150, 80
asia_routes = routes.cx[minx:maxx, miny:maxy].copy()
asia_routes["geometry"] = asia_routes["geometry"].simplify(
    tolerance=0.2, preserve_topology=True
)

m = folium.Map(location=[20, 100], zoom_start=3, tiles="CartoDB positron")

# Add GeoJSON directly (folium handles coordinates)
folium.GeoJson(
    asia_routes.to_json(),
    name="asia_routes",
    style_function=lambda feat: {"color": "blue", "weight": 2, "opacity": 0.7},
).add_to(m)

m.save(save_path)
print("✔ Map saved as Routes.html")
