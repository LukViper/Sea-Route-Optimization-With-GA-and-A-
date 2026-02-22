#! python
import networkx as nx
import json
from math import radians, sin, cos, sqrt, atan2
import random
import time
import pickle
import os
import folium
from folium.plugins import AntPath
import webbrowser
import math
import csv
import xarray as xr
from functools import lru_cache


# -----------------------------
# Configuration / Constants
# -----------------------------
SEA_LANES_GEOJSON_PATH = r"/home/luk-viper/temp/Sea-Route-Optimization-With-GA-and-A-/Shipping_Lanes_v1.geojson"
OUTPUT_MAP_FILE = r"/home/luk-viper/temp/Sea-Route-Optimization-With-GA-and-A-/optimized_route_map.html"

CSV_Location = r"/home/luk-viper/temp/Sea-Route-Optimization-With-GA-and-A-/ports.csv"

NETCDF_WEATHER_PATH = r"/home/luk-viper/Downloads/weather_data.nc"
SELECTED_DATE = "2020-01-01"

AVERAGE_SPEED_KMH = 37.0
FUEL_CONSUMPTION_PER_KM = 0.04
GA_POPULATION_SIZE = 40
GA_GENERATIONS = 100
PORT_CONNECTION_THRESHOLD_KM = 500

# -----------------------------
# Helpers
# -----------------------------
def log_status(message, icon="ℹ️"):
    print(f"{icon} {message}")

def calculate_distance_km(coord1, coord2):
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# -----------------------------
# Load Ports
# -----------------------------
def load_ports_from_csv(path):
    ports = {}
    with open(path,'r',encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            ports[name] = (lat, lon)
    return ports

PORT_LOCATIONS = load_ports_from_csv(CSV_Location)

# -----------------------------
# Load Weather (.nc)
# -----------------------------
log_status("Loading weather dataset...", "🌦️")

ds = xr.open_dataset(NETCDF_WEATHER_PATH)
weather_slice = ds.sel(time=SELECTED_DATE, method="nearest")

log_status(f"Weather selected for {SELECTED_DATE}", "✅")

def get_weather_at(lat, lon):

    point = weather_slice.interp(latitude=lat, longitude=lon)

    if "depth" in point["uo"].dims:
        uo = float(point["uo"].isel(depth=0).values)
        vo = float(point["vo"].isel(depth=0).values)
    else:
        uo = float(point["uo"].values)
        vo = float(point["vo"].values)

    current_speed = (uo**2 + vo**2) ** 0.5

    return current_speed

    return current_speed
def weather_penalty(lat, lon):
    current = get_weather_at(lat, lon)

    penalty = 1 - current * 0.05

    # Clamp to stable physical bounds
    return max(0.7, min(1.3, penalty))

# -----------------------------
# Port Selection
# -----------------------------
print("Available Ports:")
available_ports = list(PORT_LOCATIONS.keys())
for i, port in enumerate(available_ports, 1):
    print(f"{i}. {port}")

def select_port_by_number(prompt):
    while True:
        try:
            num = int(input(prompt).strip())
            if 1 <= num <= len(available_ports):
                return available_ports[num - 1]
        except ValueError:
            pass

start_port = select_port_by_number("\nEnter START port number: ")
destination_port = select_port_by_number("Enter GOAL port number: ")

hub_ports = []

log_status(f"Selected Route: {start_port} → {destination_port}", "✅")

# -----------------------------
# Build Sea Graph
# -----------------------------
with open(SEA_LANES_GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

sea_graph = nx.Graph()

def make_sea_node_id(lat, lon):
    return f"sea_{lat:.6f}_{lon:.6f}"

for feature in geojson_data.get("features", []):
    geometry = feature.get("geometry", {})
    typ = geometry.get("type")
    coords = geometry.get("coordinates", [])

    lines = []
    if typ == "LineString":
        lines.append(coords)
    elif typ == "MultiLineString":
        lines.extend(coords)
    else:
        continue

    for line in lines:
        prev_node = None
        for lon, lat in line:
            node_id = make_sea_node_id(lat, lon)

            if node_id not in sea_graph:
                sea_graph.add_node(node_id, coord=(lat, lon))

            if prev_node:
                coord_a = sea_graph.nodes[prev_node]["coord"]
                coord_b = sea_graph.nodes[node_id]["coord"]

                mid_lat = (coord_a[0] + coord_b[0]) / 2
                mid_lon = (coord_a[1] + coord_b[1]) / 2

                penalty = weather_penalty(mid_lat, mid_lon)

                w = calculate_distance_km(coord_a, coord_b) * penalty

                sea_graph.add_edge(prev_node, node_id, weight=w)

            prev_node = node_id

log_status(f"Sea Nodes: {len(sea_graph.nodes)}", "🌊")

# Add ports
for pname, pcoord in PORT_LOCATIONS.items():
    sea_graph.add_node(pname, coord=pcoord)

# Connect ports to nearest sea node
for pname, pcoord in PORT_LOCATIONS.items():
    nearest_node = min(
        [n for n in sea_graph.nodes if str(n).startswith("sea_")],
        key=lambda n: calculate_distance_km(pcoord, sea_graph.nodes[n]["coord"])
    )

    dist = calculate_distance_km(pcoord, sea_graph.nodes[nearest_node]["coord"])
    sea_graph.add_edge(pname, nearest_node, weight=dist)

# -----------------------------
# A*
# -----------------------------
def heuristic_distance(u, v):
    return calculate_distance_km(sea_graph.nodes[u]["coord"], sea_graph.nodes[v]["coord"])

@lru_cache(maxsize=200000)
def weather_penalty(lat, lon):
    current = get_weather_at(lat, lon)

    penalty = 1 - current * 0.05

    return max(0.7, min(1.3, penalty))

path_nodes, distance_km = astar_between(start_port, destination_port)

# -----------------------------
# Weather Metrics
# -----------------------------
def compute_route_weather(path_nodes):
    currents = []

    for node in path_nodes:
        lat, lon = sea_graph.nodes[node]["coord"]
        currents.append(get_weather_at(lat, lon))

    return sum(currents) / len(currents)
avg_current = compute_route_weather(path_nodes)

print(f"🌊 Avg Current Speed: {avg_current:.2f} m/s")

# -----------------------------
# Estimates
# -----------------------------
travel_time = distance_km / AVERAGE_SPEED_KMH
fuel = distance_km * FUEL_CONSUMPTION_PER_KM

print("\n🧭 ROUTE SUMMARY")
print(f"Distance: {distance_km:.2f} km")
print(f"Travel Time: {travel_time:.2f} hrs")
print(f"Fuel: {fuel:.2f} tonnes")
print(f"🌊 Avg Wave Height: {avg_wave:.2f} m")
print(f"💨 Avg Wind Speed: {avg_wind:.2f} m/s")

# -----------------------------
# Visualization
# -----------------------------
coords = [sea_graph.nodes[n]["coord"] for n in path_nodes]

m = folium.Map(location=PORT_LOCATIONS[start_port], zoom_start=4)

AntPath(coords, color="orange", weight=5).add_to(m)

folium.Marker(PORT_LOCATIONS[start_port], tooltip="Start").add_to(m)
folium.Marker(PORT_LOCATIONS[destination_port], tooltip="Goal").add_to(m)

m.save(OUTPUT_MAP_FILE)
webbrowser.open(f"file://{os.path.realpath(OUTPUT_MAP_FILE)}")

print("\n🎉 Done!")