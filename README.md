
# Sea-Route-Optimization-With-GA-and-A-
=======
# 🌊 Shipping Route Optimization using Genetic Algorithm (GA) & A* Search

This project focuses on optimizing maritime shipping routes by combining the **A\*** search algorithm with a **Genetic Algorithm (GA)** to achieve efficient, safe, and fuel-optimized paths across the ocean.  
It uses real-world **GeoJSON sea-lane data**, distance calculations (Haversine formula), and cost-based optimization.

## 🚀 Features
- ⚓ Shortest Path Routing using A*
- 🧬 Evolution-based Optimization using GA
- 🗺️ Visualization of shipping lanes using GeoPandas + Folium
- ⛽ Fuel-based cost optimization
- 🌀 Mutation, crossover & generation control
- 🌍 Haversine distance for accurate geospatial calculations
- 📊 Reproducible results using fixed random seeds

## 🧠 Core Algorithms

### 1. A* Algorithm
A* finds the shortest valid path using:
- g(n): Cost from start
- h(n): Haversine heuristic
- f(n) = g(n) + h(n)

### 2. Genetic Algorithm
GA evolves better routes using:
- Population
- Fitness Function
- Selection
- Crossover
- Mutation
- Generations

## 📁 Project Structure
```
📦 Shipping-Route-Optimization
│
├── data/
│   └── Shipping_Lanes.geojson
│
├── optimizer.py
├── utils.py
├── main.py
├── README.md
└── requirements.txt
```

## 🛠️ Installation
```sh
git clone https://github.com/your-username/Shipping-Route-Optimization.git
cd Shipping-Route-Optimization
pip install -r requirements.txt
python main.py
```

## 🔮 Future Scope
- Integrate weather data
- Add piracy-risk zones
- Multi-objective optimization
- Live AIS datasets
- Reinforcement Learning

## 🙌 Contributors
- Shasank Dahal



Optimization of Sea Routes Using Genetic Algorithm and A\* Algorithm for getting Multiport feaseable path.
<br>
Auhor : Shasank Dahal


