# CVRP Route Solver

A nearest-neighbor heuristic for the capacitated vehicle routing problem. Orders are points on a plane. Vehicles take the closest remaining stop, return to the depot when they are full, and start again. The run draws the routes to `delivery_routes.png`.

Distance is Euclidean and computed from the coordinates you pass in. Capacity is one unit per order.

## Requirements

- Python 3.10 or newer
- `matplotlib`

```bash
pip install matplotlib
```

## Run

```bash
python cvrp_routes.py
```
