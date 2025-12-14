#!/usr/bin/env python3
# integrated_routing_with_open_meteo.py
# Requires: networkx, folium, requests
# pip install networkx folium requests

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
from functools import lru_cache
import requests

# -----------------------------
# Configuration / Constants
# -----------------------------
SEA_LANES_GEOJSON_PATH = r"C:\3rd Smester Project\Optimization Project\Sea-Route-Optimization-With-GA-and-A-\Shipping_Lanes_v1.geojson"
OUTPUT_MAP_FILE = r"C:\3rd Smester Project\Optimization Project\optimized_route_map.html"
CSV_Location = r"C:\3rd sem\OT project\New folder\ports.csv"

AVERAGE_SPEED_KMH = 37.0
FUEL_CONSUMPTION_PER_KM = 0.04
WEATHER_FACTOR = 1.25
GA_POPULATION_SIZE = 40
GA_GENERATIONS = 100
PORT_CONNECTION_THRESHOLD_KM = 300  # connect port to sea nodes within this radius

# Open-Meteo marine endpoint config
OPEN_METEO_MARINE_ENDPOINT = "https://api.open-meteo.com/v1/marine"
MARINE_CACHE_TTL_SEC = 15 * 60  # 15 minutes cache in-memory

# -----------------------------
# Utility helpers
# -----------------------------
def log_status(message, icon="ℹ️"):
    print(f"{icon} {message}")

def calculate_distance_km(coord1, coord2):
    """Haversine distance in km between two (lat, lon) coordinates."""
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def estimate_travel_time_hours(distance_km):
    return (distance_km / AVERAGE_SPEED_KMH) * WEATHER_FACTOR

def estimate_fuel_tonnes(distance_km):
    return distance_km * FUEL_CONSUMPTION_PER_KM * WEATHER_FACTOR

# -----------------------------
# Load ports CSV
# -----------------------------
def load_ports_from_csv(path):
    ports = {}
    with open(path, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("port") or row.get("Name")
            if not name:
                continue
            name = name.strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            ports[name] = (lat, lon)
    return ports

if not os.path.exists(CSV_Location):
    raise FileNotFoundError(f"Ports CSV not found: {CSV_Location}")

PORT_LOCATIONS = load_ports_from_csv(CSV_Location)

# -----------------------------
# Ask user for start/destination/hubs
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
            else:
                print(f"⚠️ Enter a number between 1 and {len(available_ports)}")
        except ValueError:
            print("⚠️ Invalid input! Enter a number corresponding to the port.")

start_port = select_port_by_number("\nEnter START port number: ")
destination_port = select_port_by_number("Enter GOAL port number: ")

hubs_input = input("Enter hub ports numbers (comma-separated, optional): ").strip()
hub_ports = []
if hubs_input:
    try:
        hub_numbers = [int(x.strip()) for x in hubs_input.split(",") if x.strip()]
        for n in hub_numbers:
            if 1 <= n <= len(available_ports):
                hub_name = available_ports[n - 1]
                if hub_name not in [start_port, destination_port]:
                    hub_ports.append(hub_name)
    except Exception:
        pass

log_status(f"Selected Route: {start_port} → {destination_port} (hubs allowed: {len(hub_ports)})", "✅")

# -----------------------------
# Load GeoJSON sea lanes -> build sea_graph
# -----------------------------
if not os.path.exists(SEA_LANES_GEOJSON_PATH):
    print(f"⚠️ Error: GeoJSON file '{SEA_LANES_GEOJSON_PATH}' not found.")
    exit(1)

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
        for line in coords:
            lines.append(line)
    else:
        continue

    for line in lines:
        prev_node = None
        for lon, lat in line:  # GeoJSON long,lat
            node_id = make_sea_node_id(lat, lon)
            if node_id not in sea_graph:
                sea_graph.add_node(node_id, coord=(lat, lon))
            if prev_node is not None and not sea_graph.has_edge(prev_node, node_id):
                coord_a = sea_graph.nodes[prev_node]["coord"]
                coord_b = sea_graph.nodes[node_id]["coord"]
                w = calculate_distance_km(coord_a, coord_b)
                sea_graph.add_edge(prev_node, node_id, weight=w)
            prev_node = node_id

log_status(f"Built sea-graph with {len([n for n in sea_graph.nodes if str(n).startswith('sea_')])} sea nodes and {len(list(sea_graph.edges))} edges", "✅")

# add ports into graph and connect to nearby sea nodes
for pname, pcoord in PORT_LOCATIONS.items():
    sea_graph.add_node(pname, coord=pcoord)

for pname, pcoord in PORT_LOCATIONS.items():
    connected = []
    for node, data in list(sea_graph.nodes(data=True)):
        if not str(node).startswith("sea_"):
            continue
        d = calculate_distance_km(pcoord, data["coord"])
        if d <= PORT_CONNECTION_THRESHOLD_KM:
            sea_graph.add_edge(pname, node, weight=d)
            connected.append(node)

    if len(connected) == 0:
        print(f"⚠️ No sea-lane near {pname}. Adding intermediate connection.")
        nearest_node = min(
            [n for n in sea_graph.nodes if str(n).startswith("sea_")],
            key=lambda n: calculate_distance_km(pcoord, sea_graph.nodes[n]["coord"])
        )
        dist = calculate_distance_km(pcoord, sea_graph.nodes[nearest_node]["coord"])
        sea_graph.add_edge(pname, nearest_node, weight=dist)
        print(f"🔧 Connected {pname} to nearest sea-lane node {nearest_node} at {dist:.2f} km")
    log_status(f"Port {pname} connected to sea nodes: {len(connected)}", "🔗")

# -----------------------------
# A* helper (cached)
# -----------------------------
def heuristic_distance(u, v):
    cu = sea_graph.nodes[u].get("coord")
    cv = sea_graph.nodes[v].get("coord")
    if cu is None or cv is None:
        return 0.0
    return calculate_distance_km(cu, cv)

@lru_cache(maxsize=None)
def astar_between(u, v):
    try:
        path = nx.astar_path(sea_graph, u, v, heuristic=heuristic_distance, weight="weight")
        length = nx.path_weight(sea_graph, path, weight="weight")
        return tuple(path), length
    except Exception:
        return None, float("inf")

# Precompute port-to-port raw paths (using A* on sea_graph)
log_status("Precomputing port-to-port paths...", "⚙️")
important_ports = [start_port] + hub_ports + [destination_port]
port_paths = {}
for i in range(len(important_ports)):
    for j in range(len(important_ports)):
        if i == j:
            continue
        port_paths[(important_ports[i], important_ports[j])] = astar_between(important_ports[i], important_ports[j])
log_status("Precompute done.", "✅")

# -----------------------------
# Open-Meteo marine fetch + cache
# -----------------------------
_marine_cache = {}  # key -> (timestamp, json)

def _cache_get(key):
    entry = _marine_cache.get(key)
    if not entry:
        return None
    ts, data = entry
    if (time.time() - ts) > MARINE_CACHE_TTL_SEC:
        del _marine_cache[key]
        return None
    return data

def _cache_set(key, data):
    _marine_cache[key] = (time.time(), data)

def fetch_marine_at(lat, lon, variables=None, forecast_days=1, timezone="auto"):
    """Fetch Open-Meteo marine forecast for lat,lon. Returns JSON or None on failure."""
    if variables is None:
        variables = ["wave_height", "wave_direction", "wave_period", "windspeed_10m"]

    # round coords for stable cache keys
    key = (round(lat, 3), round(lon, 3), ",".join(sorted(variables)))
    cached = _cache_get(key)
    if cached:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "forecast_days": forecast_days,
        "timezone": timezone
    }
    try:
        r = requests.get(OPEN_METEO_MARINE_ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        _cache_set(key, j)
        return j
    except Exception as e:
        print(f"⚠️ Marine fetch error for {lat},{lon}: {e}")
        return None

# -----------------------------
# Weather -> multipliers mapping
# -----------------------------
def weather_multipliers_from_snapshot(marine_json, ts_index=0):
    """
    Map marine variables to (speed_multiplier, fuel_multiplier).
    speed_multiplier <1 reduces speed (slower), >1 increases speed (rare).
    """
    if not marine_json or "hourly" not in marine_json:
        return 1.0, 1.0

    h = marine_json["hourly"]
    def _val(name):
        arr = h.get(name)
        if not arr:
            return None
        idx = min(ts_index, len(arr) - 1)
        return arr[idx]

    wave_h = _val("wave_height")          # meters
    wind_sp = _val("windspeed_10m")       # m/s typically
    wave_period = _val("wave_period")     # seconds (may be None)

    speed_mul = 1.0
    fuel_mul = 1.0

    # Example conservative rules (tune to vessel)
    if wave_h is not None:
        if wave_h >= 4.0:
            speed_mul *= 0.6
            fuel_mul *= 1.6
        elif wave_h >= 2.5:
            speed_mul *= 0.8
            fuel_mul *= 1.25
        elif wave_h >= 1.0:
            speed_mul *= 0.95
            fuel_mul *= 1.05

    if wind_sp is not None:
        if wind_sp >= 15:   # very strong
            speed_mul *= 0.85
            fuel_mul *= 1.2
        elif wind_sp >= 8:
            speed_mul *= 0.95
            fuel_mul *= 1.05

    # clamp
    speed_mul = max(0.35, min(1.2, speed_mul))
    fuel_mul = max(0.7, min(2.5, fuel_mul))
    return speed_mul, fuel_mul

# -----------------------------
# Precompute adjusted metrics per port-pair (Option B)
# -----------------------------
log_status("Precomputing adjusted port metrics (using marine snapshot)...", "⚙️")
adjusted_port_metrics = {}  # (a,b) -> {distance_km, time_hours, fuel_tonnes}

for (a, b), (path_nodes, dist_km) in list(port_paths.items()):
    if dist_km == float("inf"):
        adjusted_port_metrics[(a, b)] = {"distance_km": dist_km, "time_hours": float("inf"), "fuel_tonnes": float("inf")}
        continue

    # compute midpoint as average of port coordinates (simple)
    midlat = (PORT_LOCATIONS[a][0] + PORT_LOCATIONS[b][0]) / 2.0
    midlon = (PORT_LOCATIONS[a][1] + PORT_LOCATIONS[b][1]) / 2.0

    marine = fetch_marine_at(midlat, midlon, variables=["wave_height", "windspeed_10m"], forecast_days=1, timezone="auto")
    speed_mul, fuel_mul = weather_multipliers_from_snapshot(marine, ts_index=0)

    time_hours = estimate_travel_time_hours(dist_km) / speed_mul
    fuel = estimate_fuel_tonnes(dist_km) * fuel_mul

    adjusted_port_metrics[(a, b)] = {
        "distance_km": dist_km,
        "time_hours": time_hours,
        "fuel_tonnes": fuel,
        "speed_mul": speed_mul,
        "fuel_mul": fuel_mul
    }

log_status("Adjusted metrics precompute complete.", "✅")

# -----------------------------
# Genetic Algorithm (uses adjusted_port_metrics)
# -----------------------------
def total_distance_using_adjusted(seq):
    dist = 0.0
    for i in range(len(seq) - 1):
        metrics = adjusted_port_metrics.get((seq[i], seq[i+1]))
        if not metrics or metrics["distance_km"] == float("inf"):
            return float("inf")
        dist += metrics["distance_km"]
    return dist

def fitness(route_seq, goal):
    """Use precomputed adjusted time/fuel to compute fitness score."""
    total_time = 0.0
    total_fuel = 0.0
    for i in range(len(route_seq) - 1):
        m = adjusted_port_metrics.get((route_seq[i], route_seq[i+1]))
        if not m or m["time_hours"] == float("inf"):
            return 0.0
        total_time += m["time_hours"]
        total_fuel += m["fuel_tonnes"]

    stops = max(0, len(route_seq) - 2)
    if goal == "fastest":
        score = 1.0 / (total_time + stops * 8 + 1e-9)
    else:
        score = 1.0 / (total_fuel * (1 + 0.03 * stops) + 1e-9)
    return score

def mutate_route(route_seq):
    if len(route_seq) <= 3:
        return route_seq
    a, b = random.sample(range(1, len(route_seq) - 1), 2)
    route_seq[a], route_seq[b] = route_seq[b], route_seq[a]
    return route_seq

def run_genetic_algorithm(goal):
    if not hub_ports:
        seq = [start_port, destination_port]
        return seq, total_distance_using_adjusted(seq)

    population = []
    for _ in range(GA_POPULATION_SIZE):
        mid = hub_ports.copy()
        random.shuffle(mid)
        population.append([start_port] + mid + [destination_port])

    best_seq, best_fit = None, 0.0
    for gen in range(GA_GENERATIONS):
        scored = [(seq, fitness(seq, goal)) for seq in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > best_fit:
            best_seq, best_fit = scored[0][0][:], scored[0][1]

        # Elitism
        new_pop = [s[0] for s in scored[:6]]

        while len(new_pop) < GA_POPULATION_SIZE:
            top = [s[0] for s in scored[:12]] if len(scored) >= 12 else [s[0] for s in scored]
            if len(top) < 2:
                new_pop.append(random.choice(population))
                continue
            p1, p2 = random.sample(top, 2)
            merged = list(dict.fromkeys(p1[1:-1] + p2[1:-1]))
            random.shuffle(merged)
            child = [start_port] + merged + [destination_port]
            if random.random() < 0.35:
                child = mutate_route(child)
            new_pop.append(child)

        population = new_pop

    return best_seq, total_distance_using_adjusted(best_seq)

# Run GA both goals
log_status("Running GA for fastest and fuel-efficient goals...", "🏁")
t0 = time.time()
fastest_route, fastest_distance_km = run_genetic_algorithm("fastest")
fuel_efficient_route, fuel_distance_km = run_genetic_algorithm("fuel")
elapsed = time.time() - t0

# -----------------------------
# Reporting helpers
# -----------------------------
def display_summary(route, dist_km, label):
    print("\n" + "-" * 50)
    print(f"🧭 {label} — Route Summary")
    print("-" * 50)
    print(f"✅ Route: {' → '.join(route)}")
    print(f"📏 Distance: {dist_km:.2f} km")
    hrs = 0.0
    for i in range(len(route)-1):
        m = adjusted_port_metrics.get((route[i], route[i+1]))
        if m:
            hrs += m["time_hours"]
    print(f"⏱️ Estimated Travel Time: {hrs:.2f} hours ({hrs/24:.2f} days)")
    fuel = sum(adjusted_port_metrics.get((route[i], route[i+1]), {}).get("fuel_tonnes", 0.0) for i in range(len(route)-1))
    print(f"⛽ Estimated Fuel: {fuel:.2f} tonnes")
    emissions = fuel * 3.15
    print(f"🌍 Estimated CO2: {emissions:.2f} tonnes")

display_summary(fastest_route, fastest_distance_km, "Fastest (orange)")
display_summary(fuel_efficient_route, fuel_distance_km, "Fuel-efficient (green)")
print(f"\n⏱️ GA elapsed time: {elapsed:.2f}s")

# -----------------------------
# Helper to turn a node-path into lat/lon coordinates
# -----------------------------
def nodes_to_latlon(path_nodes):
    coords = []
    for n in path_nodes:
        coord = sea_graph.nodes[n].get("coord")
        if coord:
            coords.append(coord)
    return coords

def build_full_route_coordinates(seq):
    coords = []
    for i in range(len(seq) - 1):
        start = seq[i]; end = seq[i+1]
        path_nodes, _ = port_paths.get((start, end), (None, float("inf")))
        if path_nodes:
            seg = nodes_to_latlon(path_nodes)
            if coords and seg and coords[-1] == seg[0]:
                coords.extend(seg[1:])
            else:
                coords.extend(seg)
        else:
            # fallback: just append start and end coordinates if no sea path
            coords.append(PORT_LOCATIONS[start])
            coords.append(PORT_LOCATIONS[end])
    return coords

fast_coords = build_full_route_coordinates(fastest_route)
fuel_coords = build_full_route_coordinates(fuel_efficient_route)

# Great-circle interpolation (for direct line)
def gc_interpolate(p1, p2, n=60):
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    def to_vec(lat, lon):
        x = math.cos(lat) * math.cos(lon); y = math.cos(lat) * math.sin(lon); z = math.sin(lat); return (x, y, z)
    v1 = to_vec(lat1, lon1); v2 = to_vec(lat2, lon2)
    dot = max(min(v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2], 1.0), -1.0)
    omega = math.acos(dot)
    if abs(omega) < 1e-12:
        return [p1, p2]
    points = []
    for i in range(n):
        t = i / (n - 1)
        s1 = math.sin((1 - t) * omega) / math.sin(omega)
        s2 = math.sin(t * omega) / math.sin(omega)
        x = s1 * v1[0] + s2 * v2[0]
        y = s1 * v1[1] + s2 * v2[1]
        z = s1 * v1[2] + s2 * v2[2]
        lat = math.atan2(z, math.sqrt(x*x + y*y)); lon = math.atan2(y, x)
        points.append((math.degrees(lat), math.degrees(lon)))
    return points

direct_path_coords = gc_interpolate(PORT_LOCATIONS[start_port], PORT_LOCATIONS[destination_port], n=120)

# -----------------------------
# Map visualization using folium
# -----------------------------
m = folium.Map(location=PORT_LOCATIONS[start_port], zoom_start=4, tiles=None)
folium.TileLayer(tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png", attr="&copy; OpenSeaMap contributors", name="OpenSeaMap", overlay=False, control=True).add_to(m)
folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)

folium.GeoJson(SEA_LANES_GEOJSON_PATH, name="Sea Lanes (GeoJSON)", style_function=lambda x: {"color": "blue", "weight": 1, "opacity": 0.4}).add_to(m)

fg_fast = folium.FeatureGroup(name="Fastest Route (orange)", show=True)
fg_fuel = folium.FeatureGroup(name="Fuel-efficient Route (green)", show=False)
fg_direct = folium.FeatureGroup(name="Direct Great-Circle (dashed)", show=False)
fg_hubs = folium.FeatureGroup(name="Hubs & Ports", show=True)

if fast_coords:
    AntPath(fast_coords, color="orange", weight=5, delay=1000).add_to(fg_fast)
if fuel_coords:
    AntPath(fuel_coords, color="green", weight=5, delay=1000).add_to(fg_fuel)

folium.PolyLine(direct_path_coords, color="darkgreen", weight=3, opacity=0.6, dash_array="8,8", tooltip="Direct Great-Circle Path").add_to(fg_direct)

folium.Marker(PORT_LOCATIONS[start_port], tooltip="Start: " + start_port, icon=folium.Icon(color="darkgreen", icon="play")).add_to(fg_hubs)
folium.Marker(PORT_LOCATIONS[destination_port], tooltip="Goal: " + destination_port, icon=folium.Icon(color="darkred", icon="stop")).add_to(fg_hubs)

hub_colors = {}
for hub in fastest_route:
    if hub in hub_ports:
        hub_colors[hub] = "orange"
for hub in fuel_efficient_route:
    if hub in hub_ports:
        if hub in hub_colors:
            hub_colors[hub] = "blue"
        else:
            hub_colors[hub] = "green"

for hub, color in hub_colors.items():
    folium.CircleMarker(PORT_LOCATIONS[hub], radius=7, tooltip=f"Hub: {hub}", fill=True, color=color, fill_color=color).add_to(fg_hubs)

fg_fast.add_to(m); fg_fuel.add_to(m); fg_direct.add_to(m); fg_hubs.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

legend_html = """
<div style="
position: fixed;
bottom: 30px; left: 30px;
z-index:9999;
background-color:white;
padding: 10px;
border-radius:8px;
box-shadow: 0 0 5px rgba(0,0,0,0.3);
font-size:14px;">
<b>🗺️ Route Legend</b><br>
<span style="color:darkgreen;">⬤</span> Start/Goal<br>
<span style="color:orange;">⬤</span> Fastest Route<br>
<span style="color:green;">⬤</span> Fuel-efficient Route<br>
<span style="color:blue;">⬤</span> Sea Lanes (GeoJSON)<br>
<span style="color:darkgreen;">⬤</span> Direct Great-Circle (dashed)<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save(OUTPUT_MAP_FILE)
log_status(f"Map saved as '{OUTPUT_MAP_FILE}' (orange=fastest, green=fuel-efficient).", "🗺️")
try:
    webbrowser.open(f"file://{os.path.realpath(OUTPUT_MAP_FILE)}")
except Exception:
    pass

# Save route + metrics
save_fn = "saved_route_with_weather.pkl"
with open(save_fn, "wb") as f:
    pickle.dump({
        "fast_seq": fastest_route,
        "fast_dist": fastest_distance_km,
        "fuel_seq": fuel_efficient_route,
        "fuel_dist": fuel_distance_km,
        "elapsed": elapsed,
        "adjusted_metrics": adjusted_port_metrics
    }, f)

print("\n🎉 Done! Map should now open in your browser.")
