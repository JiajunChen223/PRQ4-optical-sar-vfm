"""Build a metadata-only SEN12TS split excluding SSL4EO patch proximity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SSL = Path(r"C:\Users\Administrator\Desktop\ssl4eo-s12_center_coords.csv")
COORDS = Path(r"F:\PRQ4\02_experiment\cloud\commands\outputs\PRQ4-SEN12TS-ALL-LABEL-COORDS-AUDIT-R2-RETRY1-20260824T022834802596-32240.out")
OLD = Path(r"F:\PRQ4\02_experiment\cloud\commands\outputs\sen12ts-full-metadata-manifest-r9-20260822T114332808217-21176.out")
OUTPUT = Path(r"F:\PRQ4\02_experiment\reports\sen12ts_ssl4eo_excluded_split_metadata_20260824_r2.json")
THRESHOLD_KM = 5.0


def unit_vector(longitude: float, latitude: float) -> tuple[float, float, float]:
    lon, lat = math.radians(longitude), math.radians(latitude)
    return math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def main() -> None:
    with SSL.open(encoding="utf-8", newline="") as handle:
        ssl = [(int(row[0]), float(row[1]), float(row[2])) for row in csv.reader(handle)]
    tree = cKDTree(np.asarray([unit_vector(row[1], row[2]) for row in ssl], dtype=np.float64))
    regions = json.loads(COORDS.read_text(encoding="utf-8"))["regions"]
    old_active = {row["parent"] for row in json.loads(OLD.read_text(encoding="utf-8"))["object_rows"]}
    selected = []
    region_summary = {}
    for region, rows in regions.items():
        eligible = []
        excluded = 0
        for row in rows:
            _, index = tree.query(unit_vector(row["longitude"], row["latitude"]))
            location_id, longitude, latitude = ssl[int(index)]
            distance = haversine(row["longitude"], row["latitude"], longitude, latitude)
            if distance <= THRESHOLD_KM:
                excluded += 1
                continue
            eligible.append({
                "region": region,
                "coord": row["coord"],
                "parent": f"{region}/{row['coord']}",
                "nearest_ssl4eo_location_id": location_id,
                "nearest_ssl4eo_distance_km": distance,
            })
        chosen = eligible[:400]
        chosen.sort(key=lambda item: hashlib.sha256(item["parent"].encode()).hexdigest())
        for index, item in enumerate(chosen):
            item["split"] = "train" if index < 280 else "validation" if index < 340 else "sealed_test"
        selected.extend(chosen)
        region_summary[region] = {
            "listed": len(rows), "excluded_within_5km": excluded,
            "eligible": len(eligible), "selected": len(chosen),
            "selected_min_distance_km": min(item["nearest_ssl4eo_distance_km"] for item in chosen),
        }
    assignments = [{"parent": item["parent"], "split": item["split"]} for item in sorted(selected, key=lambda item: item["parent"])]
    split_sha = hashlib.sha256(json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    new_active = {item["parent"] for item in selected if item["split"] != "sealed_test"}
    artifact = {
        "artifact_type": "sen12ts_ssl4eo_excluded_split_metadata",
        "schema_version": "researchpilot.sen12ts.ssl4eo_excluded_split.v1",
        "status": "pass_metadata_only",
        "dataset_id": "sen12ts_worldcover_3region_1200",
        "exclusion_threshold_km": THRESHOLD_KM,
        "threshold_basis": "The 5 km centre threshold exceeds the 3.676955 km sum of half-diagonals for square 2640 m SSL4EO and 2560 m SEN12TS patches by 1.323045 km.",
        "ssl4eo_patch_extent_source": "https://github.com/zhu-xlab/SSL4EO-S12",
        "nearest_neighbour_method": "3D unit-sphere cKDTree followed by haversine distance",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            "ssl4eo_coords_sha256": hashlib.sha256(SSL.read_bytes()).hexdigest(),
            "all_label_coords_sha256": hashlib.sha256(COORDS.read_bytes()).hexdigest(),
        },
        "region_summary": region_summary,
        "split_counts": {name: sum(item["split"] == name for item in selected) for name in ("train", "validation", "sealed_test")},
        "split_assignment_sha256": split_sha,
        "active_reused_parent_count": len(new_active & old_active),
        "active_new_parent_count": len(new_active - old_active),
        "active_retired_parent_count": len(old_active - new_active),
        "new_active_parents": sorted(new_active - old_active),
        "retired_active_parents": sorted(old_active - new_active),
        "selected_parents": assignments,
        "scope": {"metadata_only": True, "pixel_read": False, "label_content_read": False, "weights_read": False, "gpu_used": False, "training": False, "evaluation": False, "test_payload_accessed": False},
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: artifact[key] for key in ("status", "region_summary", "split_counts", "split_assignment_sha256", "active_reused_parent_count", "active_new_parent_count", "active_retired_parent_count")}, indent=2))


if __name__ == "__main__":
    main()
