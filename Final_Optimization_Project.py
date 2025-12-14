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
from functools import lru_cache


# -----------------------------
# Configuration / Constants
# -----------------------------
SEA_LANES_GEOJSON_PATH = r"C:\3rd sem\OT project\New folder\newzealandpaul-Shipping-Lanes-b0ad85c\data\Shipping_Lanes_v1.geojson"
OUTPUT_MAP_FILE = r"C:\3rd sem\OT project\New folder\optimized_route_map.html"

PORT_LOCATIONS = {}
CSV_Location = r"C:\3rd sem\OT project\New folder\ports.csv"
def load_ports_from_csv(path):
    ports ={}
    with open(path,'r',encoding="utf-8") as f :
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            ports[name] = (lat, lon)
    return ports 
PORT_LOCATIONS = load_ports_from_csv(CSV_Location)

AVERAGE_SPEED_KMH = 37.0
FUEL_CONSUMPTION_PER_KM = 0.04
WEATHER_FACTOR = 1.25
GA_POPULATION_SIZE = 40
GA_GENERATIONS = 100
PORT_CONNECTION_THRESHOLD_KM = 300 # can change for sea-lane data coverage but 500 is realistic due to the dataset we are using

# Helpers

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


#  select ports 

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
        hub_numbers = [int(x.strip()) for x in hubs_input.split(",")]
        for n in hub_numbers:
            if 1 <= n <= len(available_ports):
                hub_name = available_ports[n - 1]
                if hub_name not in [start_port, destination_port]:
                    hub_ports.append(hub_name)
    except Exception:
        pass

log_status(f"Selected Route: {start_port} → {destination_port} (hubs allowed: {len(hub_ports)})", "✅")


# Load GeoJSON and build sea-graph
if not os.path.exists(SEA_LANES_GEOJSON_PATH):
    print(f"⚠️ Error: GeoJSON file '{SEA_LANES_GEOJSON_PATH}' not found.")
    exit(1)

with open(SEA_LANES_GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

sea_graph = nx.Graph()


def make_sea_node_id(lat, lon):
    """Create a stable id string for a sea node from lat/lon."""
    return f"sea_{lat:.6f}_{lon:.6f}"


# Iterate GeoJSON features and add sea nodes/edges
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
        # GeoJSON uses [lon, lat]
        for lon, lat in line:
            node_id = make_sea_node_id(lat, lon)
            if node_id not in sea_graph:
                sea_graph.add_node(node_id, coord=(lat, lon))
            if prev_node is not None and not sea_graph.has_edge(prev_node, node_id):
                coord_a = sea_graph.nodes[prev_node]["coord"]
                coord_b = sea_graph.nodes[node_id]["coord"]
                w = calculate_distance_km(coord_a, coord_b)
                sea_graph.add_edge(prev_node, node_id, weight=w)
            prev_node = node_id

log_status(
    f"Built sea-graph with {len([n for n in sea_graph.nodes if str(n).startswith('sea_')])} sea nodes and {len(list(sea_graph.edges))} edges",
    "✅",
)

# add ports as nodes (with coord )
for pname, pcoord in PORT_LOCATIONS.items():
    sea_graph.add_node(pname, coord=pcoord)


# connect each port to nearby sea nodes only
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
        print(f"⚠️ No sea-lane near {pname}. Adding intermediate node.")

        # find nearest sea lane node anywhere in the full dataset
        nearest_node = min(
            [n for n in sea_graph.nodes if str(n).startswith("sea_")],
            key=lambda n: calculate_distance_km(
                pcoord,
                sea_graph.nodes[n]["coord"]
            )
        )

        # connect the port to the best possible sea lane node
        dist = calculate_distance_km(pcoord, sea_graph.nodes[nearest_node]["coord"])
        sea_graph.add_edge(pname, nearest_node, weight=dist)
        print(f"🔧 Connected {pname} to nearest sea-lane node {nearest_node} at {dist:.2f} km")

    log_status(f"Port {pname} connected to {len(connected)} sea nodes.", "🔗")




# A* helpers

def heuristic_distance(u, v):
    cu = sea_graph.nodes[u].get("coord")
    cv = sea_graph.nodes[v].get("coord")
    if cu is None or cv is None:
        return 0.0
    return calculate_distance_km(cu, cv)


@lru_cache(maxsize=None)
def astar_between(u, v):
    """Run A* between two nodes in the sea graph and return (path_tuple, length)."""
    try:
        path = nx.astar_path(sea_graph, u, v, heuristic=heuristic_distance, weight="weight")
        length = nx.path_weight(sea_graph, path, weight="weight")
        return tuple(path), length
    except Exception:
        return None, float("inf")



# Port-to-port A* paths (only between start, destination and candidate hubs)

log_status("Precomputing port-to-port paths...", "⚙️")
important_ports = [start_port] + hub_ports + [destination_port]
port_paths = {}
for i in range(len(important_ports)):
    for j in range(len(important_ports)):
        if i == j:
            continue
        port_paths[(important_ports[i], important_ports[j])] = astar_between(important_ports[i], important_ports[j])
log_status("Precompute done.", "✅")



# Genetic Algorithm 
def total_distance(port_sequence):
    """Sum of distances along a port sequence using precomputed port-to-port paths."""
    dist = 0.0
    for i in range(len(port_sequence) - 1):
        _, d = port_paths.get((port_sequence[i], port_sequence[i + 1]), (None, float("inf")))
        dist += d
    return dist


def fitness(route_seq, goal):
    """
    Compute fitness for GA.
    Introduces stop penalties to differentiate 'fastest' vs 'fuel-efficient' routes.
    """
    d = total_distance(route_seq)
    if d == float("inf"):
        return 0.0

    stops = len(route_seq) - 2  # exclude start and destination

    if goal == "fastest":
        # Penalize each hub stop slightly (e.g., docking time)
        t = estimate_travel_time_hours(d) + stops * 8  # 8 hours penalty per hub
        return 1.0 / (t + 1e-6)
    else:
        # Slightly higher fuel consumption with more stops (small penalty)
        f = estimate_fuel_tonnes(d) * (1 + 0.03 * stops)  # 3% penalty per hub
        return 1.0 / (f + 1e-6)


def mutate_route(route_seq):
    """Swap two intermediate hubs randomly to introduce variation."""
    if len(route_seq) <= 3:
        return route_seq
    a, b = random.sample(range(1, len(route_seq) - 1), 2)
    route_seq[a], route_seq[b] = route_seq[b], route_seq[a]
    return route_seq


def run_genetic_algorithm(goal):
    """
    Run genetic algorithm to optimize route.
    Returns best sequence and its total distance.
    """
    if not hub_ports:
        return [start_port, destination_port], total_distance([start_port, destination_port])

    # Initialize population
    population = []
    for _ in range(GA_POPULATION_SIZE):
        mid = hub_ports.copy()
        random.shuffle(mid)
        population.append([start_port] + mid + [destination_port])

    best_seq, best_fit = None, 0.0

    for gen in range(GA_GENERATIONS):
        # Evaluate fitness
        scored = [(seq, fitness(seq, goal)) for seq in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Track best
        if scored and scored[0][1] > best_fit:
            best_seq, best_fit = scored[0][0][:], scored[0][1]

        # Elitism: keep top 6
        new_pop = [s[0] for s in scored[:6]]

        # Generate new population
        while len(new_pop) < GA_POPULATION_SIZE:
            top12 = [s[0] for s in scored[:12]] if len(scored) >= 12 else [s[0] for s in scored]
            if len(top12) < 2:
                new_pop.append(random.choice(population))
                continue
            # Crossover: merge two parents
            p1, p2 = random.sample(top12, 2)
            merged = list(dict.fromkeys(p1[1:-1] + p2[1:-1]))  # keep unique hubs
            random.shuffle(merged)
            child = [start_port] + merged + [destination_port]

            # Mutation
            if random.random() < 0.35:
                child = mutate_route(child)

            new_pop.append(child)

        population = new_pop

    return best_seq, total_distance(best_seq)



# Run GA

log_status("Running GA for both fastest and fuel-efficient routes...", "🏁")
t0 = time.time()
fastest_route, fastest_distance_km = run_genetic_algorithm("fastest")
fuel_efficient_route, fuel_distance_km = run_genetic_algorithm("fuel")
elapsed = time.time() - t0


# Summary display

def display_summary(route, dist_km, label):
    print("\n" + "-" * 50)
    print(f"🧭 {label} — Route Summary")
    print("-" * 50)
    print(f"✅ Route: {' → '.join(route)}")
    print(f"📏 Distance: {dist_km:.2f} km")
    hrs = estimate_travel_time_hours(dist_km)
    print(f"⏱️ Estimated Travel Time: {hrs:.2f} hours ({hrs / 24:.2f} days)")
    fuel = estimate_fuel_tonnes(dist_km)
    print(f"⛽ Estimated Fuel: {fuel:.2f} tonnes")
    emissions = fuel * 3.15 #emission factor 
    print(f"🌍 Estimated CO2: {emissions:.2f} tonnes")
  


display_summary(fastest_route, fastest_distance_km, "Fastest (orange)")
display_summary(fuel_efficient_route, fuel_distance_km, "Fuel-efficient (green)")



# Build path coords from A* node sequences
def nodes_to_latlon(path_nodes):
    coords = []
    for n in path_nodes:
        coord = sea_graph.nodes[n].get("coord")
        if coord:
            coords.append(coord)
    return coords


def nearest_sea_nodes_for_port(port_name, max_km=None, top_k=3):
    """
    Return a list of sea node ids near the port ordered by distance (ascending).
    max_km: optional maximum search radius (defaults to PORT_CONNECTION_THRESHOLD_KM * 1.5)
    top_k: return up to top_k nearest nodes
    """
    if max_km is None:
        max_km = PORT_CONNECTION_THRESHOLD_KM * 1.5
    pcoord = sea_graph.nodes[port_name].get("coord")
    if pcoord is None:
        return []
    dlist = []
    for node, data in sea_graph.nodes(data=True):
        if not str(node).startswith("sea_"):
            continue
        d = calculate_distance_km(pcoord, data["coord"])
        if d <= max_km:
            dlist.append((d, node))
    dlist.sort(key=lambda x: x[0])
    return [n for _, n in dlist[:top_k]]


# Build full coordinates strictly along sea lanes (no land crossings)

def build_full_route_coordinates(seq):
    """
    Convert a sequence of ports into lat/lon coordinates strictly along sea-lanes.
    Uses A* between nearest sea nodes if direct port-to-port path is not found.
    """
    coords = []

    for i in range(len(seq) - 1):
        start = seq[i]
        end = seq[i + 1]

        # Try direct precomputed port-to-port path first
        path_nodes, _ = port_paths.get((start, end), (None, float("inf")))

        if path_nodes:
            seg = nodes_to_latlon(path_nodes)
            if coords and seg and coords[-1] == seg[0]:
                coords.extend(seg[1:])
            else:
                coords.extend(seg)
        else:
            # Sea-lane A* fallback using nearest sea nodes
            log_status(f"Finding sea-lane path for {start} → {end}...", "⚠️")
            s_nodes_start = nearest_sea_nodes_for_port(start, top_k=5)
            s_nodes_end = nearest_sea_nodes_for_port(end, top_k=5)
            best_path = None
            best_length = float("inf")

            for a in s_nodes_start:
                for b in s_nodes_end:
                    path_nodes2, length2 = astar_between(a, b)
                    if path_nodes2 and length2 < best_length:
                        best_path = path_nodes2
                        best_length = length2

            if best_path:
                seg = nodes_to_latlon(best_path)
                if coords and seg and coords[-1] == seg[0]:
                    coords.extend(seg[1:])
                else:
                    coords.append(PORT_LOCATIONS[start])
                    coords.extend(seg)
            else:
                # No sea-lane path found; skip segment
                log_status(f"⚠️ No sea-lane path found between {start} → {end}. Segment skipped.", "❌")

    return coords


fast_coords = build_full_route_coordinates(fastest_route)
fuel_coords = build_full_route_coordinates(fuel_efficient_route)



# Correct great-circle interpolation (spherical linear interpolation)

def gc_interpolate(p1, p2, n=60):
    """Spherical linear interpolation between two lat/lon points (returns n points)."""
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)

    def to_vec(lat, lon):
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return (x, y, z)

    v1 = to_vec(lat1, lon1)
    v2 = to_vec(lat2, lon2)
    dot = max(min(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2], 1.0), -1.0)
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
        lat = math.atan2(z, math.sqrt(x * x + y * y))
        lon = math.atan2(y, x)
        points.append((math.degrees(lat), math.degrees(lon)))
    return points


direct_path_coords = gc_interpolate(PORT_LOCATIONS[start_port], PORT_LOCATIONS[destination_port], n=120)



# Map visualization: create two overlay feature groups

m = folium.Map(location=PORT_LOCATIONS[start_port], zoom_start=4, tiles=None)

# Base tiles
folium.TileLayer(
    tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
    attr="&copy; OpenSeaMap contributors",
    name="OpenSeaMap",
    overlay=False,
    control=True,
).add_to(m)
folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)

# Add GeoJSON sea lanes (toggleable)
folium.GeoJson(
    SEA_LANES_GEOJSON_PATH,
    name="Sea Lanes (GeoJSON)",
    style_function=lambda x: {"color": "blue", "weight": 1, "opacity": 0.4},
).add_to(m)

# Feature groups for overlays
fg_fast = folium.FeatureGroup(name="Fastest Route (orange)", show=True)
fg_fuel = folium.FeatureGroup(name="Fuel-efficient Route (green)", show=False)
fg_direct = folium.FeatureGroup(name="Direct Great-Circle (dashed)", show=False)
fg_hubs = folium.FeatureGroup(name="Hubs & Ports", show=True)

# Add AntPath for fastest and fuel-efficient into their groups
if fast_coords:
    AntPath(fast_coords, color="orange", weight=5, delay=1000).add_to(fg_fast)
if fuel_coords:
    AntPath(fuel_coords, color="green", weight=5, delay=1000).add_to(fg_fuel)

# Add direct great-circle polyline to its group (dashed)
folium.PolyLine(
    direct_path_coords,
    color="darkgreen",
    weight=3,
    opacity=0.6,
    dash_array="8,8",
    tooltip="Direct Great-Circle Path",
).add_to(fg_direct)

# Add markers for start/goal
folium.Marker(
    PORT_LOCATIONS[start_port],
    tooltip="Start: " + start_port,
    icon=folium.Icon(color="darkgreen", icon="play"),
).add_to(fg_hubs)
folium.Marker(
    PORT_LOCATIONS[destination_port],
    tooltip="Goal: " + destination_port,
    icon=folium.Icon(color="darkred", icon="stop"),
).add_to(fg_hubs)

# Add hub markers and color them by inclusion (orange if in fastest, green if in fuel, blue if in both)
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
    folium.CircleMarker(
        PORT_LOCATIONS[hub],
        radius=7,
        tooltip=f"Hub: {hub}",
        fill=True,
        color=color,
        fill_color=color,
    ).add_to(fg_hubs)

# Add all feature groups to map
fg_fast.add_to(m)
fg_fuel.add_to(m)
fg_direct.add_to(m)
fg_hubs.add_to(m)

# Layer control (lets user toggle overlays)
folium.LayerControl(collapsed=False).add_to(m)

# Add a simple legend (keeps existing colors)
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

# Save & open
m.save(OUTPUT_MAP_FILE)
log_status(f"Map saved as '{OUTPUT_MAP_FILE}' (orange=fastest, green=fuel-efficient, blue=both).", "🗺️")
webbrowser.open(f"file://{os.path.realpath(OUTPUT_MAP_FILE)}")

# Save route data for later
save_fn = "saved_route.pkl"
with open(save_fn, "wb") as f:
    pickle.dump(
        {
            "fast_seq": fastest_route,
            "fast_dist": fastest_distance_km,
            "fuel_seq": fuel_efficient_route,
            "fuel_dist": fuel_distance_km,
            "elapsed": elapsed,
        },
        f,
    )


print("\n🎉 Done! Map should now open in your browser.")

