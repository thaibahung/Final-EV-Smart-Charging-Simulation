import argparse, os, yaml
import numpy as np
import pandas as pd
import pandapower as pp

# Use absolute import
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_mv import build_network

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _synth_profile(steps):
    # smooth 24h curve in [0.6, 1.1] typical weekday-ish load shape
    x = np.linspace(0, 2*np.pi, steps, endpoint=False)
    return np.clip(0.85 + 0.25*np.sin(x - 0.5) + 0.1*np.sin(2*x), 0.6, 1.1)

def _load_profile_csv(path, steps):
    if not path or not os.path.exists(path):
        return _synth_profile(steps)
    df = pd.read_csv(path)
    if "scale" not in df.columns:
        raise ValueError(f"{path} must have a 'scale' column")
    if len(df) != steps:
        raise ValueError(f"{path} must have {steps} rows, found {len(df)}")
    return df["scale"].to_numpy(dtype=float)

def run(cfg):
    # timing
    horizon_h = cfg["time"]["horizon_hours"]
    step_min  = cfg["time"]["step_minutes"]
    steps     = int(horizon_h * 60 / step_min)

    # build network
    with_der = bool(cfg["network"]["with_der"])
    net = build_network(with_der=with_der)

    # snapshot base P for loads & sgens (we scale from these each step)
    base_load_p = net.load["p_mw"].copy() if len(net.load) else pd.Series(dtype=float)
    base_sgen_p = net.sgen["p_mw"].copy() if with_der and len(net.sgen) else pd.Series(dtype=float)

    # profiles
    load_scale = _load_profile_csv(cfg["profiles"]["load_profile_file"], steps)
    if with_der and len(net.sgen):
        sgen_scale = _load_profile_csv(cfg["profiles"]["sgen_profile_file"], steps)
    else:
        sgen_scale = np.ones(steps, dtype=float)

    # outputs
    outdir = cfg["outputs_dir"]
    _ensure_dir(outdir)
    bus_rows, line_rows, trafo_rows = [], [], []

    for t_idx in range(steps):
        # scale loads
        if len(net.load):
            net.load["p_mw"] = base_load_p * load_scale[t_idx]

        # scale sgens (if any)
        if with_der and len(net.sgen):
            net.sgen["p_mw"] = base_sgen_p * sgen_scale[t_idx]

        # AC power flow
        pp.runpp(net)

        # collect bus results
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

        # collect line results
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

        # collect transformer results
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

    # write CSVs
    pd.DataFrame(bus_rows).to_csv(os.path.join(outdir, "bus_timeseries.csv"), index=False)
    pd.DataFrame(line_rows).to_csv(os.path.join(outdir, "line_timeseries.csv"), index=False)
    pd.DataFrame(trafo_rows).to_csv(os.path.join(outdir, "trafo_timeseries.csv"), index=False)
    print(f"Wrote {outdir}/bus_timeseries.csv, line_timeseries.csv, trafo_timeseries.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

if __name__ == "__main__":
    main()
