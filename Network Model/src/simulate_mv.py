import argparse, os, yaml
import numpy as np
import pandas as pd
import pandapower as pp
from src.caps import compute_station_caps

# Use absolute import
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_mv import build_network

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _group_load_profiles():
    """Group loads into industrial, commercial, and residential categories."""
    return {
        "industrial": [4, 9],
        "commercial": [5, 3, 7, 8, 10, 11, 12, 13, 14],
        "residential": [6, 15, 16, 17]  # Includes charging stations (CSs)
    }

def _assign_unique_profiles_to_buses(categories, profile_dir):
    """Assign unique datasets to each bus based on its category."""
    category_prefix_map = {
        "industrial": "Industrial",
        "commercial": "Commercial",
        "residential": "Reference"
    }
    bus_profiles = {}
    for category, buses in categories.items():
        prefix = category_prefix_map[category]
        available_files = [
            f for f in os.listdir(profile_dir) if f.startswith(prefix)
        ]
        if len(available_files) < len(buses):
            raise ValueError(f"Not enough datasets for category '{category}'.")
        for bus, file in zip(buses, available_files):
            path = os.path.join(profile_dir, file)
            df = pd.read_csv(path)
            raw_profile = df.iloc[:, 0].values
            bus_profiles[bus] = raw_profile
    return bus_profiles

def _distribute_and_scale_profiles(bus_profiles, bus_peak_map):
    """Scale and randomize profiles for individual buses."""
    scaled_profiles = {}
    for bus, raw_profile in bus_profiles.items():
        if bus in bus_peak_map:
            normalized = raw_profile / np.max(raw_profile)
            scaled = normalized * bus_peak_map[bus]
            noise = np.random.normal(1.0, 0.05, size=scaled.shape)
            scaled_profiles[bus] = scaled * noise
        else:
            scaled_profiles[bus] = raw_profile * 0.7  # Scale down other buses

        scaled_profiles[bus] = np.nan_to_num(scaled_profiles[bus], nan=0.0, posinf=0.0, neginf=0.0)
    return scaled_profiles

def _assign_loads_to_buses(net, bus_profiles, t_idx, cos_phi):
    """Assign active and reactive power to buses."""
    for load_idx, load_row in net.load.iterrows():
        bus_idx = load_row.bus
        if bus_idx in bus_profiles:
            p_mw = bus_profiles[bus_idx][t_idx] / 1000.0
            phi = np.arccos(cos_phi)
            q_mvar = p_mw * np.tan(phi)
            net.load.at[load_idx, "p_mw"] = p_mw
            net.load.at[load_idx, "q_mvar"] = q_mvar

def _collect_results(net, t_idx, bus_rows, line_rows, trafo_rows):
    """Collect simulation results for buses, lines, and transformers."""
    for bus_idx in net.bus.index:
        rb = net.res_bus.loc[bus_idx]
        bus_rows.append({
            "t_idx": t_idx,
            "bus": int(bus_idx),
            "name": str(net.bus.at[bus_idx, "name"]),
            "vn_kv": float(net.bus.at[bus_idx, "vn_kv"]),
            "vm_pu": float(rb["vm_pu"]),
            "va_degree": float(rb["va_degree"]),
            "p_mw": float(rb.get("p_mw", np.nan)),
            "q_mvar": float(rb.get("q_mvar", np.nan)),
        })
    for line_idx in net.line.index:
        rl = net.res_line.loc[line_idx]
        line_rows.append({
            "t_idx": t_idx,
            "line": int(line_idx),
            "from_bus": int(net.line.at[line_idx, "from_bus"]),
            "to_bus": int(net.line.at[line_idx, "to_bus"]),
            "loading_percent": float(rl.get("loading_percent", np.nan)),
            "p_from_mw": float(rl.get("p_from_mw", np.nan)),
            "p_to_mw": float(rl.get("p_to_mw", np.nan)),
            "q_from_mvar": float(rl.get("q_from_mvar", np.nan)),
            "q_to_mvar": float(rl.get("q_to_mvar", np.nan)),
        })
    for tr_idx in net.trafo.index:
        rt = net.res_trafo.loc[tr_idx]
        trafo_rows.append({
            "t_idx": t_idx,
            "trafo": int(tr_idx),
            "name": str(net.trafo.at[tr_idx, "name"]),
            "hv_bus": int(net.trafo.at[tr_idx, "hv_bus"]),
            "lv_bus": int(net.trafo.at[tr_idx, "lv_bus"]),
            "sn_mva": float(net.trafo.at[tr_idx, "sn_mva"]),
            "loading_percent": float(rt.get("loading_percent", np.nan)),
            "p_hv_mw": float(rt.get("p_hv_mw", np.nan)),
            "q_hv_mvar": float(rt.get("q_hv_mvar", np.nan)),
            "p_lv_mw": float(rt.get("p_lv_mw", np.nan)),
            "q_lv_mvar": float(rt.get("q_lv_mvar", np.nan)),
        })

# Calculate P_max_LV
def run_pf_and_caps(net, trafo_for_cs, cos_phi: float):
    """
    Execute PF without EV injections and return station caps (kW).
    """
    # pp.runpp(net, algorithm="nr", calculate_voltage_angles=False, init="results")
    caps = compute_station_caps(net, trafo_for_cs, cos_phi=cos_phi)
    return caps


def run(cfg):
    """Main simulation loop."""
    horizon_h = cfg["time"]["horizon_hours"]
    step_min = cfg["time"]["step_minutes"]
    steps = int(horizon_h * 60 / step_min)

    with_der = bool(cfg["network"]["with_der"])
    net, ids = build_network(with_der=with_der)

    profile_dir = "Dataset/Dataset on Hourly Load Profiles for 24 Facilities (8760 hours)"
    cos_phi = 0.95

    categories = _group_load_profiles()
    bus_profiles = _assign_unique_profiles_to_buses(categories, profile_dir)

    bus_peak_map = {
        15: 60, 
        16: 60, 
        17: 30
    }  # Charging stations and household

    scaled_profiles = _distribute_and_scale_profiles(bus_profiles, bus_peak_map)

    outdir = cfg["outputs_dir"]
    _ensure_dir(outdir)
    bus_rows, line_rows, trafo_rows = [], [], []
    cap_rows = []

    trafo_for_cs = ids["trafos"]

    for t_idx in range(steps):
        _assign_loads_to_buses(net, scaled_profiles, t_idx, cos_phi)

        if with_der and len(net.sgen):
            base_sgen_p = net.sgen["p_mw"].copy()
            sgen_scale = np.ones(steps, dtype=float)
            net.sgen["p_mw"] = base_sgen_p * sgen_scale[t_idx]

        try:
            pp.runpp(net)
        except Exception as e:
            print(f"Power flow did not converge at t_idx={t_idx}. Bus loads:")
            print(net.load[["bus", "p_mw", "q_mvar"]])
            raise e

        _collect_results(net, t_idx, bus_rows, line_rows, trafo_rows)

        # Calculate the P_max_LV
        caps_kw = run_pf_and_caps(net, trafo_for_cs, cos_phi=cos_phi)
        
        for cs_id, pmax_kw in caps_kw.items():
            cap_rows.append({"t_idx": t_idx, "cs_id": cs_id, "pmax_kw": pmax_kw})

    # Extract simulation result
    pd.DataFrame(bus_rows).to_csv(os.path.join(outdir, "bus_timeseries.csv"), index=False)
    pd.DataFrame(line_rows).to_csv(os.path.join(outdir, "line_timeseries.csv"), index=False)
    pd.DataFrame(trafo_rows).to_csv(os.path.join(outdir, "trafo_timeseries.csv"), index=False)
    print(f"Wrote {outdir}/bus_timeseries.csv, line_timeseries.csv, trafo_timeseries.csv")

    caps_df = pd.DataFrame(cap_rows)
    caps_df.to_csv(os.path.join(outdir, "cs_caps.csv"), index=False)
    print(f"Wrote {len(caps_df)} rows → {os.path.join(outdir,'cs_caps.csv')}")

    

