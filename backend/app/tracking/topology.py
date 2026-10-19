from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import threading
from typing import Any, Optional

logger = logging.getLogger("SIH26187.CameraTopology")

DEFAULT_TOPOLOGY_FILE = Path(__file__).resolve().parent.parent / "data" / "camera_topology.json"


class CameraTopology:
    """
    Graph-based Camera Adjacency and Transition Matrix for Multi-Camera Tracking.
    Models physical corridors between cameras with expected transit times,
    providing spatio-temporal affinity scores to boost or penalize Re-ID matching.
    """

    def __init__(self, topology_path: Optional[Path] = None) -> None:
        self.topology_path = topology_path or DEFAULT_TOPOLOGY_FILE
        self.lock = threading.Lock()
        # Key: (from_camera_id, to_camera_id) -> corridor_dict
        self._corridors: dict[tuple[str, str], dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load topology definitions from JSON file."""
        with self.lock:
            self._corridors.clear()
            if not self.topology_path.exists():
                logger.info("No topology file at %s, initializing empty graph.", self.topology_path)
                return

            try:
                with open(self.topology_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("corridors", []):
                        from_c = str(item.get("from_camera", "")).strip()
                        to_c = str(item.get("to_camera", "")).strip()
                        if not from_c or not to_c:
                            continue

                        corridor_info = {
                            "from_camera": from_c,
                            "to_camera": to_c,
                            "distance_m": float(item.get("distance_m", 50.0)),
                            "bidirectional": bool(item.get("bidirectional", True)),
                            "expected_transit_sec": item.get(
                                "expected_transit_sec",
                                {"min": 5.0, "optimal": 25.0, "max": 120.0},
                            ),
                            "transit_count": int(item.get("transit_count", 0)),
                            "confidence": float(item.get("confidence", 0.9)),
                        }

                        self._corridors[(from_c, to_c)] = corridor_info
                        if corridor_info["bidirectional"]:
                            rev_info = dict(corridor_info)
                            rev_info["from_camera"] = to_c
                            rev_info["to_camera"] = from_c
                            self._corridors[(to_c, from_c)] = rev_info

                logger.info("Loaded %d camera corridors from %s.", len(self._corridors), self.topology_path)
            except Exception as err:
                logger.warning("Failed to load camera topology from %s: %s", self.topology_path, err)

    def save(self) -> None:
        """Persist current topology with updated transition stats to disk."""
        with self.lock:
            unique_corridors = []
            seen = set()
            for (c1, c2), info in self._corridors.items():
                pair_key = tuple(sorted([c1, c2])) if info.get("bidirectional", True) else (c1, c2)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                unique_corridors.append(info)

            try:
                self.topology_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.topology_path, "w", encoding="utf-8") as f:
                    json.dump({"corridors": unique_corridors}, f, indent=2)
                logger.debug("Persisted %d unique corridors to %s.", len(unique_corridors), self.topology_path)
            except Exception as err:
                logger.warning("Failed to save camera topology to %s: %s", self.topology_path, err)

    def register_corridor(
        self,
        from_camera: str,
        to_camera: str,
        min_sec: float = 5.0,
        optimal_sec: float = 25.0,
        max_sec: float = 120.0,
        distance_m: float = 50.0,
        bidirectional: bool = True,
        confidence: float = 0.9,
    ) -> None:
        """Programmatically register or update a corridor between two cameras."""
        from_c = str(from_camera).strip()
        to_c = str(to_camera).strip()

        corridor_info = {
            "from_camera": from_c,
            "to_camera": to_c,
            "distance_m": float(distance_m),
            "bidirectional": bool(bidirectional),
            "expected_transit_sec": {
                "min": float(min_sec),
                "optimal": float(optimal_sec),
                "max": float(max_sec),
            },
            "transit_count": 0,
            "confidence": float(confidence),
        }

        with self.lock:
            self._corridors[(from_c, to_c)] = corridor_info
            if bidirectional:
                rev_info = dict(corridor_info)
                rev_info["from_camera"] = to_c
                rev_info["to_camera"] = from_c
                self._corridors[(to_c, from_c)] = rev_info

    def get_corridor(self, from_camera: str, to_camera: str) -> Optional[dict[str, Any]]:
        """Retrieve corridor parameters for a transition between two cameras."""
        key = (str(from_camera).strip(), str(to_camera).strip())
        with self.lock:
            return self._corridors.get(key)

    def is_adjacent(self, from_camera: str, to_camera: str) -> bool:
        """Check if two cameras share an explicit corridor."""
        return self.get_corridor(from_camera, to_camera) is not None

    def compute_spatial_affinity(
        self,
        from_camera: str,
        to_camera: str,
        dt_sec: float,
        dist_km: Optional[float] = None,
    ) -> float:
        """
        Compute normalized spatial affinity score in [0.0, 1.0] for a candidate transition.

        - If explicit corridor exists: evaluates Gaussian probability around optimal transit window.
        - If no corridor but GPS known: evaluates realistic walking speed physics.
        - If unmapped test cameras: returns neutral baseline (0.65).
        """
        if from_camera == to_camera:
            return 1.0

        corridor = self.get_corridor(from_camera, to_camera)
        dt = max(0.01, float(dt_sec))

        if corridor is not None:
            # Explicit Corridor Modeling
            exp_transit = corridor.get("expected_transit_sec", {})
            t_min = float(exp_transit.get("min", 5.0))
            t_opt = float(exp_transit.get("optimal", 25.0))
            t_max = float(exp_transit.get("max", 120.0))

            if dt < t_min:
                # Arrived impossibly fast for this specific path
                underflow_ratio = dt / max(0.1, t_min)
                return max(0.05, round(underflow_ratio * 0.35, 3))

            if t_min <= dt <= t_max:
                # Gaussian affinity centered on t_opt
                sigma = max(8.0, (t_max - t_min) / 3.0)
                score = math.exp(-((dt - t_opt) ** 2) / (2.0 * (sigma ** 2)))
                # Scale between 0.60 and 1.0 within normal window
                scaled = 0.60 + 0.40 * score
                return round(min(1.0, scaled), 3)

            # dt > t_max: Lingering / delayed transit (decay toward 0.30 over 5 minutes)
            decay = math.exp(-(dt - t_max) / 180.0)
            return round(max(0.20, 0.20 + 0.40 * decay), 3)

        # No explicit corridor defined:
        if dist_km is not None and dist_km > 0:
            # Estimate speed in meters per second
            dist_m = dist_km * 1000.0
            v_ms = dist_m / dt

            # Normal walking / light running is 1.0 m/s to 3.5 m/s
            if 0.8 <= v_ms <= 3.5:
                # High realistic walking affinity
                return 0.85
            elif 0.3 <= v_ms < 0.8:
                # Slow walk or brief stop
                return 0.70
            elif 3.5 < v_ms <= 6.0:
                # Fast sprint
                return 0.60
            elif v_ms > 6.0:
                # Rapid motion (vehicle speed or questionable jump)
                return max(0.15, round(6.0 / v_ms * 0.5, 3))
            else:
                # Extremely slow transit (> 10x normal time)
                return 0.45

        # Fallback for uncalibrated / test camera pairs
        return 0.65

    def record_transition(self, from_camera: str, to_camera: str, dt_sec: float) -> None:
        """
        Record a verified transition and adaptively refine optimal transit time (Online Learning).
        Uses Exponential Moving Average (alpha=0.15) to track real-world traffic flows.
        """
        if from_camera == to_camera:
            return

        key = (str(from_camera).strip(), str(to_camera).strip())
        with self.lock:
            corridor = self._corridors.get(key)
            if corridor is None:
                return

            exp = corridor.setdefault("expected_transit_sec", {})
            curr_opt = float(exp.get("optimal", 25.0))
            t_min = float(exp.get("min", 5.0))
            t_max = float(exp.get("max", 120.0))

            # Only adapt if the transit occurred within reasonable bounds
            if t_min * 0.5 <= dt_sec <= t_max * 1.5:
                updated_opt = round(0.85 * curr_opt + 0.15 * dt_sec, 1)
                exp["optimal"] = max(t_min, min(t_max, updated_opt))

            corridor["transit_count"] = int(corridor.get("transit_count", 0)) + 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology for API responses and map visualization."""
        with self.lock:
            unique_corridors = []
            seen = set()
            for (c1, c2), info in self._corridors.items():
                pair_key = tuple(sorted([c1, c2])) if info.get("bidirectional", True) else (c1, c2)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                unique_corridors.append(info)

            return {
                "total_corridors": len(unique_corridors),
                "corridors": unique_corridors,
            }


# Singleton topology engine instance
camera_topology = CameraTopology()
