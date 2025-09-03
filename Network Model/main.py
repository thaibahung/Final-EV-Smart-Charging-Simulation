import argparse
import numpy as np
import pandas as pd
import pandapower as pp
from pathlib import Path

import os
import yaml
from src.build_mv import build_network
from src.simulate_mv import run, run_pf_and_caps, _collect_results
from rl.multi_env_ev_cs import MultiEVRuntimeEnv
from rl.sac_agent_multi import SACMultiAgent


def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)


# --- helper: synthetic price vector (replace with real prices)
# TODO: Change with the real dataset
def _price_vec(n_steps: int) -> np.ndarray:
    x = np.arange(n_steps)
    base = 0.2 + 0.1*np.sin(2*np.pi*(x-6)/24.0)
    peak = 0.25*np.exp(-0.5*((x-34)/4)**2)
    return base + peak

# --- helper: tiny synthetic arrivals per step (replace with your generator/data)
# TODO: Change with the real dataset
def _synthetic_arrivals(n_steps: int, seed=0):
    rng = np.random.default_rng(seed)
    arrivals = [[] for _ in range(n_steps)]
    for t in range(n_steps):
        # Poisson-ish arrivals
        k = rng.poisson(lam=0.5)  # avg 0.5 EV per step
        for _ in range(int(k)):
            arrivals[t].append({
                "soc0":    float(np.clip(rng.normal(0.4, 0.15), 0.05, 0.95)),
                "stay_h":  float(np.clip(rng.normal(2.0, 0.7), 0.5, 6.0)),
                "limit_kw": float(np.random.choice([7.4, 11.0, 22.0])),
                "target":  0.8
            })
    return arrivals


# -------------------- TRAIN (offline) --------------------
def train_sac_multi(outdir: str, n_plugs: int):
    caps = pd.read_csv(os.path.join(outdir, "cs_caps.csv"))
    caps_cs = caps[caps["cs_id"] == "CS15"].sort_values("t_idx")

    pmax_seq = caps_cs["pmax_kw"].to_numpy(float)
    price_seq = _price_vec(len(pmax_seq))
    arrivals_by_t = _synthetic_arrivals(len(pmax_seq), seed=1)  # or your real arrivals

    SACMultiAgent.train_offline(
        n_plugs=n_plugs,
        pmax_seq=pmax_seq,
        price_seq=price_seq,
        arrivals_by_t=arrivals_by_t,
        total_timesteps=1000,
        seed=0,
        save_dir="models/sac_multi_cs15"
    )

# -------------------- DEPLOY (PF-coupled) --------------------
def deploy_sac_multi(cfg_path: str, outdir: str, n_plugs: int):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    net, ids = build_network(cfg)
    lv_bus = ids["lv_buses"]["CS15"]

    trafo_for_cs = {"CS15": ids["trafos"]["CS15"]}
    # trafo_for_cs = ids["trafos"]
    
    horizon_h = int(cfg["time"]["horizon_hours"])
    step_min  = int(cfg["time"]["step_minutes"])
    n_steps   = horizon_h * 60 // step_min
    cos_phi   = float(cfg["network"].get("cos_phi", 0.95))

    price_seq = _price_vec(n_steps)
    env = MultiEVRuntimeEnv(n_plugs=n_plugs, step_minutes=step_min,
                            capacity_kwh=60.0, alpha_target=150.0,
                            allow_v2g=True, pmax_clip_kw=100.0,
                            price_max_for_norm=float(np.max(price_seq)))
    env.reset(t0=0)

    # use the same arrivals generator used in training OR your real arrivals
    arrivals_by_t = _synthetic_arrivals(n_steps, seed=7)
    agent = SACMultiAgent.load("models/sac_multi_cs15")

    # Check if the model file exists
    model_path = os.path.join("models", "sac_multi_cs15", "sac_multi_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    # create one load for the whole station (sum of EV powers)
    cs_load_idx = pp.create_load(net, bus=lv_bus, p_mw=0.0, q_mvar=0.0, name="CS15_MULTI")

    log_rows = []
    bus_rows, line_rows, trafo_rows = [], [], []

    for t in range(n_steps):
        # Apply arrivals of this step
        env.apply_arrivals(arrivals_by_t[t])

        # PF #1 → caps
        # print(net, trafo_for_cs)
        
        caps_kw = run_pf_and_caps(net, trafo_for_cs, cos_phi=cos_phi)
        pmax_kw = caps_kw["CS15"]

        # Build obs, act, advance env one step (runtime)
        price_t = float(price_seq[t])
        obs = env.obs(pmax_kw, price_t)
        action = agent.act(obs, deterministic=False)
        obs_next, reward, info = env.step(action, pmax_kw=pmax_kw, price_t=price_t)

        # Inject aggregate station power into PF
        P_sum_kw = float(np.sum(info["P_kw"]))
        net.load.at[cs_load_idx, "p_mw"] = P_sum_kw / 1000.0
        net.load.at[cs_load_idx, "q_mvar"] = 0.0

        # PF #2
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=False, init="results")

        # log
        log_rows.append({
            "t_idx": t,
            "pmax_kw": pmax_kw,
            "price": price_t,
            "P_sum_kw": P_sum_kw,
            "reward": reward,
            "departed_idx": ";".join(map(str, info["departed_idx"])),
            "departed_penalty": info["departed_penalty"]
        })

        _collect_results(net, t, bus_rows, line_rows, trafo_rows)

    pd.DataFrame(log_rows).to_csv(os.path.join(outdir, "cs15_multi_ev_log_sac.csv"), index=False)
    print("Multi-EV SAC deployed → cs15_multi_ev_log_sac.csv")

    pd.DataFrame(bus_rows).to_csv(os.path.join(outdir, "bus_timeseries_after.csv"), index=False)
    pd.DataFrame(line_rows).to_csv(os.path.join(outdir, "line_timeseries_after.csv"), index=False)
    pd.DataFrame(trafo_rows).to_csv(os.path.join(outdir, "trafo_timeseries_after.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="EV Smart Charging")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--outdir", default="logs", help="Output directory")
    parser.add_argument("--model-dir", default="models/sac_multi_cs15", help="SAC model dir")
    parser.add_argument("--cs-id", default="CS15", help="Station ID (e.g., CS15)")
    parser.add_argument("--n-plugs", type=int, default=6, help="Number of charging ports at the station")

    # modes
    parser.add_argument("--train-sac", action="store_true", help="Train multi-EV SAC offline")
    parser.add_argument("--deploy-sac", action="store_true", help="Deploy SAC in PF loop")
    parser.add_argument("--run-sim",  action="store_true", help="Run base simulator only")

    args = parser.parse_args()

    if args.train_sac:
        # Run simulation first to generate caps
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run(cfg)  # This will generate cs_caps.csv in the output directory
        train_sac_multi(args.outdir, args.n_plugs)
        return

    if args.deploy_sac:
        deploy_sac_multi(args.config, args.outdir, args.n_plugs)
        return

    if args.run_sim:
        # base simulator mode (if you still want it)
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run(cfg)
        return

    # default: print usage
    parser.print_help()

if __name__ == "__main__":
    main()
