#!/usr/bin/env python
"""Verify published artifacts and tables without importing the simulation stack.

Read-only: hashes, schemas, experiment coverage, and Tables VII–IX are checked
against the versioned building exports. No controller or historical exporter runs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ce1_thesis_final_results"
RUN = "ce1_occ_p2s2_all_occ"
CONTROLLERS = ["Baseline", "Safe-PI/RBC", "MPC", "GNU-RL adaptive model", "SAC"]
ALL_CONTROLLERS = CONTROLLERS + ["GNU-RL fixed model"]
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]
MONTHS = ["February", "March"]
KEYS = ["month", "mode", "occupant", "controller", "building_id"]
METRICS = {
    "electricity_consumption_total": "Elec. score",
    "daily_peak_average": "Daily peak",
    "all_time_peak_average": "Max peak",
    "ramping_average": "Ramping",
    "daily_one_minus_load_factor_average": "1-LF",
    "grid_efficiency_cost": "Grid",
    "discomfort_cold_proportion": "Cold [%]",
    "discomfort_hot_proportion": "Hot [%]",
    "comfort_cost": "Comfort [%]",
    "kelvin_hours": "KH",
}
PROPORTIONS = ["discomfort_cold_proportion", "discomfort_hot_proportion", "comfort_cost"]
GRID = ["ramping_average", "daily_one_minus_load_factor_average", "daily_peak_average", "all_time_peak_average"]
TEXT = {"month", "mode", "occupant", "building", "building_id", "controller", "source", "case", "scenario"}
# These blanks already exist in the historical export; published table metrics
# are never nullable. Carbon/pricing were not exported; baseline has no rollout.
BASELINE_NULLS = {
    "monthly_one_minus_load_factor_average", "net_electricity_consumption_kwh",
    "electricity_import_kwh", "max_net_electricity_consumption_kwh",
    "ramping_up_kwh", "ramping_abs_kwh", "heating_demand_kwh",
    "heating_electricity_consumption_kwh", "reward_total",
}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def compare(actual, expected, keys, label):
    """Compare keys and every numeric cell, ignoring row order and CSV roundoff."""
    require(set(actual.columns) == set(expected.columns), f"{label}: columns differ")
    for frame in (actual, expected):
        require(not frame.duplicated(keys).any(), f"{label}: duplicate keys")
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    require(left.index.equals(right.index), f"{label}: row identifiers differ")
    left = left[right.columns]
    try:
        np.testing.assert_allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                   rtol=1e-10, atol=1e-10, equal_nan=False)
    except (AssertionError, ValueError) as exc:
        raise VerificationError(f"{label}: numerical values differ: {exc}") from exc


def verify_manifest(root):
    manifest = json.loads((root / "analysis/artifact_manifest.json").read_text())
    require(manifest["source_run"] == RUN, "Unexpected source run")
    require(manifest["publication_export"] == PREFIX, "Unexpected publication export")
    require(manifest["source_notebook"] == "experiments/analysis/12_simple_v4_comparison.ipynb",
            "Unexpected source notebook")
    require(manifest["thesis_tables"] == {
        "VII": "no_occupant_mean_building_kpis", "VIII": "no_occupant_selected_building_kpis",
        "IX": "occupant_controller_mean_kpis"}, "Unexpected thesis table mapping")
    paths = [item["published"] for item in manifest["artifacts"]]
    require(len(paths) == len(set(paths)), "Duplicate manifest paths")
    published = {str(p.relative_to(root)) for folder in (
        "data", "figures/ce1_thesis_final_results", "figures/ce1_thesis_clean_load_figures")
        for p in (root / "analysis" / folder).rglob("*") if p.is_file()}
    require(set(paths) == published, "Manifest does not cover exactly the published data and figures")
    for item in manifest["artifacts"]:
        path = (root / item["published"]).resolve()
        require(path.is_relative_to(root.resolve() / "analysis"), "Artifact path escapes analysis directory")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                f"SHA-256 mismatch: {item['published']}")
    return len(paths)


def read_exports(root):
    frames = {}
    expected = {f"{PREFIX}_building_kpis_augmented.csv", f"{PREFIX}_district_kpis_augmented.csv",
                f"{RUN}_override_deltas.csv", f"{RUN}_gnurl_adaptive_ppo_delta_summary.csv",
                f"{RUN}_supplementary_comfort_grid_costs_by_mode.csv"}
    expected.update(f"{PREFIX}_{table}_{month.lower()}.csv" for table, month in itertools.product(
        ["no_occupant_mean_building_kpis", "no_occupant_selected_building_kpis", "occupant_controller_mean_kpis"], MONTHS))
    require({p.name for p in (root / "analysis/data").glob("*.csv")} == expected,
            "Unexpected published CSV set")
    for name in sorted(expected):
        frame = pd.read_csv(root / "analysis/data" / name)
        require(not frame.empty, f"{name}: empty CSV")
        if name == f"{PREFIX}_district_kpis_augmented.csv":
            required = {"month", "controller", "mode", "occupant", "total_kwh", "peak_kwh",
                        "daily_profile_peak_kwh", "daily_profile_peak_hour", "ramping_abs_kwh", "load_factor"}
        elif name == f"{RUN}_gnurl_adaptive_ppo_delta_summary.csv":
            required = {"month", "mode", "cases", "better", "worse", "mean_improve_pp", "median_improve_pp", "p25", "p75"}
        elif name == f"{RUN}_supplementary_comfort_grid_costs_by_mode.csv":
            required = {"month", "controller", "mode", "comfort_cost", "grid_efficiency_cost", *GRID}
        else:
            required = set()
        require(required.issubset(frame.columns), f"{name}: missing columns {required - set(frame.columns)}")
        for column in frame.columns:
            if column in TEXT:
                if column != "scenario":
                    require(frame[column].notna().all(), f"{name}: missing {column}")
                continue
            numeric = pd.to_numeric(frame[column], errors="raise")
            nullable = pd.Series(False, index=frame.index)
            if name == f"{PREFIX}_building_kpis_augmented.csv":
                if column in {"carbon_emissions_total", "cost_total"}:
                    nullable[:] = True
                elif column in BASELINE_NULLS:
                    nullable = frame["controller"].eq("Baseline")
                elif column in {"tdyn_mean", "tdyn_final"}:
                    nullable = frame["mode"].ne("bspline_tdyn")
            require((np.isfinite(numeric) | (numeric.isna() & nullable)).all(),
                    f"{name}: missing or nonfinite numeric value in {column}")
        frames[name] = frame
    return frames


def verify_buildings(buildings):
    required = set(KEYS + list(METRICS) + ["source", "case", "building", "overrides", "discomfort_proportion"])
    require(required.issubset(buildings.columns), f"Building export missing columns: {required - set(buildings.columns)}")
    expected = set()
    for month, controller, building in itertools.product(MONTHS, ALL_CONTROLLERS, range(1, 26)):
        expected.add((month, "no_occupant", "none", controller, f"building_{building:02}"))
        if controller != "Baseline":
            for mode, occupant in itertools.product(["bspline_tdyn", "without_tdyn"], OCCUPANTS):
                expected.add((month, mode, occupant, controller, f"building_{building:02}"))
    require(not buildings.duplicated(KEYS).any(), "Building export: duplicate experiment/building keys")
    require(set(buildings[KEYS].itertuples(index=False, name=None)) == expected,
            "Building export: expected 25-building experiment coverage differs")
    sources = {"Baseline": "baseline_dataset", "Safe-PI/RBC": "rbc_safe_pi", "MPC": "mpc",
               "SAC": "sac_merlin", "GNU-RL fixed model": "rlmpc", "GNU-RL adaptive model": "rlmpc"}
    tags = {"MPC": "mpc", "GNU-RL fixed model": "rlmpc_fixed", "GNU-RL adaptive model": "rlmpc_adaptive"}
    for row in buildings.itertuples(index=False):
        require(row.source == sources[row.controller], f"Unexpected source: {row.source}")
        month = "feb" if row.month == "February" else "mar"
        suffix = "no_occupant" if row.mode == "no_occupant" else f"{row.occupant}_{row.mode}"
        if row.controller == "Baseline":
            cases = {"baseline" if month == "feb" else "march_baseline"}
        elif row.controller == "Safe-PI/RBC":
            cases = {f"ce1_thesis_rbc_safepi_{month}_{suffix}"}
        elif row.controller == "SAC":
            stage = "jan20_feb" if month == "feb" else "feb10_mar"
            cases = {f"{RUN}_sac_{stage}_{suffix}"}
        else:
            case = f"{RUN}_{tags[row.controller]}_{month}_{suffix}"
            cases = {case}
            # Historical continuation runner sometimes appended the suffix twice.
            if row.controller == "GNU-RL adaptive model" and month == "mar" and row.mode != "no_occupant":
                cases.add(f"{case}_{suffix}")
        require(row.case in cases, f"Unexpected experiment identifier: {row.case}")
        require(row.building == f"Building {int(row.building_id.split('_')[1])}", "Building label mismatch")
    for column in PROPORTIONS + ["discomfort_proportion"]:
        require(buildings[column].between(0, 1).all(), f"{column}: outside [0, 1]")
    compare(buildings[KEYS + ["grid_efficiency_cost"]],
            buildings[KEYS].assign(grid_efficiency_cost=buildings[GRID].mean(axis=1)), KEYS, "Grid formula")
    compare(buildings[KEYS + ["comfort_cost"]], buildings[KEYS].assign(
        comfort_cost=buildings["discomfort_cold_proportion"] + buildings["discomfort_hot_proportion"]), KEYS, "Comfort formula")


def verify_tables(frames):
    buildings = frames[f"{PREFIX}_building_kpis_augmented.csv"]
    for month in MONTHS:
        no_occ = buildings[(buildings.month == month) & (buildings['mode'] == "no_occupant")
                           & buildings.controller.isin(CONTROLLERS)].copy()
        no_occ[PROPORTIONS] *= 100
        means = no_occ.groupby("controller", as_index=False)[list(METRICS)].mean().rename(columns=METRICS)
        means.controller = means.controller.replace({"GNU-RL adaptive model": "GNU-RL adaptive"})
        name = f"{PREFIX}_no_occupant_mean_building_kpis_{month.lower()}.csv"
        compare(frames[name], means, ["controller"], name)
        selected = no_occ[no_occ.building_id.isin(["building_08", "building_12", "building_15"])]
        name = f"{PREFIX}_no_occupant_selected_building_kpis_{month.lower()}.csv"
        columns = ["month", "building_id", "building", "controller", *METRICS]
        compare(frames[name], selected[columns], ["month", "building_id", "building", "controller"], name)
        occ = buildings[(buildings.month == month) & (buildings['mode'] == "bspline_tdyn")
                        & buildings.controller.isin(CONTROLLERS)].copy()
        occ[PROPORTIONS] *= 100
        means = occ.groupby(["occupant", "controller"], as_index=False)[[*METRICS, "overrides"]].mean()
        means.occupant = means.occupant.replace(dict(zip(OCCUPANTS, ["occ1 tolerant", "occ2 sensitive", "occ3 cold", "occ4 hot"])))
        means.controller = means.controller.replace({"GNU-RL adaptive model": "GNU-RL", "Safe-PI/RBC": "Safe-PI"})
        means = means.rename(columns={**METRICS, "overrides": "Mean overrides/building"})
        name = f"{PREFIX}_occupant_controller_mean_kpis_{month.lower()}.csv"
        compare(frames[name], means, ["occupant", "controller"], name)
        # Independently recorded Table VII / README values, to displayed precision.
        reference = {"February": [41.71, 1.10, 3.94, 27.09], "March": [34.18, 1.17, 3.75, 5.70]}
        table = frames[f"{PREFIX}_no_occupant_mean_building_kpis_{month.lower()}.csv"].set_index("controller")
        values = table.loc[["Safe-PI/RBC", "MPC", "GNU-RL adaptive", "SAC"], "Comfort [%]"].round(2)
        require(np.array_equal(values.to_numpy(), reference[month]), f"{month}: reported Table VII discomfort differs")


def verify_overrides(frames):
    buildings = frames[f"{PREFIX}_building_kpis_augmented.csv"]
    keys = ["month", "controller", "building", "building_id", "occupant"]
    metrics = {"overrides": "overrides", "override_hours": "override_hours",
               "discomfort_proportion": "discomfort", "electricity_consumption_total": "electricity_consumption"}
    paired = None
    for mode, suffix in [("without_tdyn", "without"), ("bspline_tdyn", "bspline")]:
        part = buildings[buildings['mode'] == mode][keys + list(metrics)].rename(
            columns={column: f"{label}_{suffix}" for column, label in metrics.items()})
        paired = part if paired is None else paired.merge(part, on=keys, validate="one_to_one")
    for label, delta in [("overrides", "delta_overrides"), ("override_hours", "delta_override_hours"),
                         ("discomfort", "delta_discomfort_proportion"), ("electricity_consumption", "delta_electricity_consumption_total")]:
        paired[delta] = paired[f"{label}_bspline"] - paired[f"{label}_without"]
    compare(frames[f"{RUN}_override_deltas.csv"], paired, keys, "Paired override comparisons")


def verify_auxiliary(frames):
    buildings = frames[f"{PREFIX}_building_kpis_augmented.csv"]
    districts = frames[f"{PREFIX}_district_kpis_augmented.csv"]
    keys = ["month", "controller", "mode", "occupant"]
    expected = buildings[keys].drop_duplicates()
    require(not districts.duplicated(keys).any(), "District export: duplicate keys")
    require(set(districts[keys].itertuples(index=False, name=None)) ==
            set(expected.itertuples(index=False, name=None)), "District export: experiment identifiers differ")
    require(districts.daily_profile_peak_hour.between(0, 23).all(), "District export: invalid peak hour")
    summary = frames[f"{RUN}_gnurl_adaptive_ppo_delta_summary.csv"]
    require(not summary.duplicated(["month", "mode"]).any(), "Adaptation export: duplicate keys")
    require(set(summary[["month", "mode"]].itertuples(index=False, name=None)) ==
            set(itertools.product(MONTHS, ["bspline_tdyn", "without_tdyn", "no_occupant"])),
            "Adaptation export: experiment identifiers differ")
    expected_cases = np.where(summary['mode'].eq("no_occupant"), 25, 100)
    require(np.array_equal(summary.cases, expected_cases), "Adaptation export: invalid case counts")
    require(((summary.better >= 0) & (summary.worse >= 0) &
             (summary.better + summary.worse <= summary.cases)).all(), "Adaptation export: invalid better/worse counts")
    costs = frames[f"{RUN}_supplementary_comfort_grid_costs_by_mode.csv"]
    keys = ["month", "controller", "mode"]
    # This historical audit predates the final March baseline augmentation.
    expected = buildings[keys].drop_duplicates()
    expected = expected[~((expected.month == "March") & (expected.controller == "Baseline"))]
    require(not costs.duplicated(keys).any(), "Supplementary export: duplicate keys")
    require(set(costs[keys].itertuples(index=False, name=None)) ==
            set(expected.itertuples(index=False, name=None)), "Supplementary export: experiment identifiers differ")
    compare(costs[keys + ["grid_efficiency_cost"]], costs[keys].assign(
        grid_efficiency_cost=costs[GRID].mean(axis=1)), keys, "Supplementary Grid formula")


def verify_results(root=ROOT):
    count = verify_manifest(root)
    frames = read_exports(root)
    verify_buildings(frames[f"{PREFIX}_building_kpis_augmented.csv"])
    verify_tables(frames)
    verify_overrides(frames)
    verify_auxiliary(frames)
    print(f"PASS: {count} artifact hashes, {len(frames)} CSVs, 2,300 building rows, "
          "six thesis tables, Table VII discomfort, and 1,000 paired comparisons.")
    print("Scope: saved exports and analysis only; original training and raw-trajectory KPIs are not revalidated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        verify_results()
    except (VerificationError, OSError, KeyError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
