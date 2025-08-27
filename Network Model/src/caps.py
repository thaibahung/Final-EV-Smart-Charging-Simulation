# grid/caps.py
from typing import Dict, List
import pandas as pd

def compute_station_caps(
    net,
    trafo_for_cs: Dict[str, int],
    cos_phi: float = 0.95,
    kw_per_mw: float = 1000.0
) -> Dict[str, float]:
    """
    Compute station Pmax (kW) from current trafo loading after a PF run.

    Args:
      net: pandapower net with res_trafo.loading_percent and trafo.sn_mva filled
      trafo_for_cs: mapping like {"CS15": trafo_idx_15, "CS16": trafo_idx_16}
      cos_phi: power factor used for active power headroom
      kw_per_mw: convert MVA≈MW to kW (assumes ~unity voltage regulation)

    Returns:
      dict {cs_id: Pmax_kW (>=0)}
    """
    caps = {}
    for cs_id, t_idx in trafo_for_cs.items():
        
        # if t_idx not in net.res_trafo.index or t_idx not in net.trafo.index:
        #   raise ValueError(f"Transformer index {t_idx} not found in net.res_trafo or net.trafo")
        
        # print(cs_id, t_idx)
        
        loading = float(net.res_trafo.loc[t_idx, "loading_percent"])  # %
        sn_mva  = float(net.trafo.loc[t_idx, "sn_mva"])               # MVA

        # Active-power headroom (Eq. like paper’s): Snom * (100-L)/100 * cosφ
        headroom_mw = sn_mva * (max(0.0, 100.0 - loading) / 100.0) * cos_phi
        pmax_kw = max(0.0, headroom_mw * kw_per_mw)
        caps[cs_id] = pmax_kw
    return caps


def caps_to_dataframe(step_idx: int, caps: Dict[str, float]) -> pd.DataFrame:
    """Convenience to log caps per step."""
    return pd.DataFrame(
        [{"t_idx": step_idx, "cs_id": cs, "pmax_kw": caps[cs]} for cs in caps]
    )
