"""
# Capacitated Vehicle Routing Problem (CVRP) Solver

## What it is

A programmatic heuristic for the Capacitated Vehicle Routing Problem that
handles capacity constraints and spatial distance dynamically:

- **Euclidean distance** - Travel cost between any two points (depot or order)
  is computed from coordinates at runtime, so depot location and order sets are
  fully parameterized.
- **Nearest-neighbor assignment** - Unserved orders are claimed round-robin
  across vehicles; each vehicle always extends to the geographically closest
  remaining order from its current position.
- **Capacity resets** - Each order consumes one unit of capacity. When a
  vehicle is full, its path appends a return to the configured depot, load
  resets to zero, and routing continues to the next closest order.
- **Multi-vehicle routes** - Final itineraries are polylines that start and end
  at the depot, including any mid-route reload trips.

## What it is used for

Real-world logistics optimization, dynamic fleet management, and operational
cost reduction:

- **Last-mile delivery** - Assign stops to drivers under van/payload limits.
- **Fleet planning** - Estimate total distance and reload trips before dispatch.
- **Cost control** - Reduce empty miles and avoid overloading vehicles.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

Point = Tuple[float, float]
Order = Dict[str, Any]


def euclidean(a: Point, b: Point) -> float:
    """Return Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_orders(orders: Sequence[Any]) -> List[Order]:
    """Accept dicts or (x, y) / (id, x, y) tuples; return canonical order dicts."""
    normalized: List[Order] = []
    for i, item in enumerate(orders):
        if isinstance(item, dict):
            oid = item.get("id", i + 1)
            normalized.append({"id": oid, "x": float(item["x"]), "y": float(item["y"])})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            normalized.append({"id": i + 1, "x": float(item[0]), "y": float(item[1])})
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            normalized.append(
                {"id": item[0], "x": float(item[1]), "y": float(item[2])}
            )
        else:
            raise TypeError(f"Unsupported order format: {item!r}")
    return normalized


def _path_distance(path: Sequence[Point]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        total += euclidean(path[i], path[i + 1])
    return total


def optimize_delivery_routes(
    depot_location: Point,
    orders: Sequence[Any],
    num_vehicles: int,
    vehicle_capacity: int,
) -> Dict[str, Any]:
    """
    Assign orders to vehicles with a greedy nearest-neighbor CVRP heuristic.

    Capacity is the maximum number of orders a vehicle may carry before it must
    return to ``depot_location`` to reload.
    """
    if num_vehicles < 1:
        raise ValueError("num_vehicles must be >= 1")
    if vehicle_capacity < 1:
        raise ValueError("vehicle_capacity must be >= 1")

    depot: Point = (float(depot_location[0]), float(depot_location[1]))
    catalog = _normalize_orders(orders)
    unserved = set(range(len(catalog)))

    routes: List[List[Point]] = [[] for _ in range(num_vehicles)]
    order_sequences: List[List[Any]] = [[] for _ in range(num_vehicles)]
    loads = [0] * num_vehicles
    positions: List[Point] = [depot] * num_vehicles
    started = [False] * num_vehicles

    while unserved:
        progressed = False
        for v in range(num_vehicles):
            if not unserved:
                break

            if loads[v] >= vehicle_capacity:
                if positions[v] != depot:
                    routes[v].append(depot)
                    positions[v] = depot
                loads[v] = 0

            current = positions[v]
            nearest_idx = min(
                unserved,
                key=lambda i: euclidean(current, (catalog[i]["x"], catalog[i]["y"])),
            )
            order = catalog[nearest_idx]
            point: Point = (order["x"], order["y"])

            if not started[v]:
                routes[v].append(depot)
                started[v] = True

            routes[v].append(point)
            order_sequences[v].append(order["id"])
            positions[v] = point
            loads[v] += 1
            unserved.remove(nearest_idx)
            progressed = True

        if not progressed:
            break

    for v in range(num_vehicles):
        if started[v] and (not routes[v] or routes[v][-1] != depot):
            routes[v].append(depot)

    per_vehicle = [_path_distance(r) for r in routes]
    return {
        "routes": routes,
        "order_sequences": order_sequences,
        "total_distance": sum(per_vehicle),
        "per_vehicle_distance": per_vehicle,
        "orders": catalog,
        "depot": depot,
    }


def plot_routes(
    depot_location: Point,
    orders: Sequence[Any],
    routes: Sequence[Sequence[Point]],
    output_path: str = "delivery_routes.png",
    title: str = "CVRP Delivery Routes",
) -> None:
    """Plot depot, orders, and per-vehicle paths with directional arrows."""
    catalog = _normalize_orders(orders)
    depot: Point = (float(depot_location[0]), float(depot_location[1]))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    xs = [o["x"] for o in catalog]
    ys = [o["y"] for o in catalog]
    ax.scatter(xs, ys, c="#4a5568", s=40, zorder=3, label="Orders")
    for o in catalog:
        ax.annotate(
            str(o["id"]),
            (o["x"], o["y"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.scatter(
        [depot[0]],
        [depot[1]],
        marker="*",
        s=350,
        c="#c53030",
        zorder=5,
        label="Depot",
        edgecolors="black",
        linewidths=0.5,
    )

    for v, route in enumerate(routes):
        if len(route) < 2:
            continue
        color = cmap(v % 10)
        rx = [p[0] for p in route]
        ry = [p[1] for p in route]
        ax.plot(rx, ry, color=color, linewidth=2, alpha=0.85, label=f"Vehicle {v + 1}")

        for i in range(len(route) - 1):
            x0, y0 = route[i]
            x1, y1 = route[i + 1]
            arrow = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
            )
            ax.add_patch(arrow)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved route map to {output_path}")
    plt.show()


def print_itinerary(result: Dict[str, Any]) -> None:
    """Print a human-readable per-vehicle itinerary to the console."""
    print("=" * 60)
    print("Delivery itinerary")
    print("=" * 60)
    for v, (seq, dist, route) in enumerate(
        zip(
            result["order_sequences"],
            result["per_vehicle_distance"],
            result["routes"],
        )
    ):
        if not seq:
            print(f"Vehicle {v + 1}: (idle)")
            continue
        stops = " → ".join(str(oid) for oid in seq)
        print(f"Vehicle {v + 1}: Depot → {stops} → Depot")
        print(f"  Orders served: {len(seq)} | Distance: {dist:.2f}")
        print(f"  Path points: {len(route)}")
    print("-" * 60)
    print(f"Total distance: {result['total_distance']:.2f}")
    print("=" * 60)


def _generate_sample_orders(n: int = 20, seed: int = 42) -> List[Order]:
    """Generate clustered random order coordinates for a demo dataset."""
    rng = random.Random(seed)
    clusters = [(-25, -20), (20, 25), (-15, 30), (30, -25)]
    orders: List[Order] = []
    for i in range(n):
        cx, cy = clusters[i % len(clusters)]
        orders.append(
            {
                "id": i + 1,
                "x": cx + rng.uniform(-12, 12),
                "y": cy + rng.uniform(-12, 12),
            }
        )
    return orders


if __name__ == "__main__":
    depot = (0.0, 0.0)
    sample_orders = _generate_sample_orders(20, seed=42)
    num_vehicles = 4
    vehicle_capacity = 5

    print(
        f"Solving CVRP: {len(sample_orders)} orders, "
        f"{num_vehicles} vehicles, capacity={vehicle_capacity}, depot={depot}"
    )

    result = optimize_delivery_routes(
        depot_location=depot,
        orders=sample_orders,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )
    print_itinerary(result)
    plot_routes(
        depot_location=depot,
        orders=sample_orders,
        routes=result["routes"],
        output_path="delivery_routes.png",
    )
