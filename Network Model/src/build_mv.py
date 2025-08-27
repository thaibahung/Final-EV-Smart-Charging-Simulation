import pandapower as pp
import pandapower.networks as pn

def _add_mv_lv_trafo(net, mv_bus: int, sn_mva: float, name: str, vn_lv_kv: float = 0.4):
    """Create a new LV bus and a 20/0.4 kV transformer from the given MV bus."""
    lv_bus = pp.create_bus(net, vn_kv=vn_lv_kv, name=f"LV_{name}")
    tr_idx = pp.create_transformer_from_parameters(
        net,
        hv_bus=mv_bus,
        lv_bus=lv_bus,
        sn_mva=sn_mva,
        vn_hv_kv=20.0,
        vn_lv_kv=vn_lv_kv,
        vkr_percent=0.5,
        vk_percent=4.0,
        pfe_kw=0.5,
        i0_percent=0.1,
        shift_degree=0.0,
        name=name
    )
    return lv_bus, tr_idx

def _add_sgen_with_q_limits(net, bus: int, p_rated_kw: float, name: str,
                            q_cap_ratio: float = 0.33, set_at_max: bool = False):
    """
    Add an inverter-based DER (PV/wind) as sgen with symmetric Q capability.
    - p_rated_kw is the *active* power rating from your diagram
    - If set_at_max=True, we initialize p_mw at its rating; otherwise start at 0 (time-series can set it)
    """
    p_rated_mw = p_rated_kw / 1000.0
    q_lim = q_cap_ratio * p_rated_mw
    return pp.create_sgen(
        net, bus=bus, p_mw=(p_rated_mw if set_at_max else 0.0), q_mvar=0.0, name=name,
        min_p_mw=0.0, max_p_mw=p_rated_mw,
        min_q_mvar=-q_lim, max_q_mvar=+q_lim
    )

def build_network(with_der: bool = True):
    """
    Build the CIGRE MV benchmark network and then customize it to your diagram:

    DER (all active power ratings in kW):
      - PV@3  = 20
      - PV@4  = 20
      - PV@5  = 30
      - PV@10 = 40
      - PV@11 = 10
      - PV@9  = 30
      - PV@8  = 30
      - WIND@7 = 1500

    MV→LV transformers:
      - From MV bus 7  → LV bus (name: CS_Trafo_15),  sn = 0.25 MVA
      - From MV bus 14 → LV bus (name: CS_Trafo_16),  sn = 0.25 MVA
      - From MV bus 14 → LV bus (name: HH_Trafo_17),  sn = 1.00 MVA

    Notes:
      - LV buses are created but *no loads are attached here* (so you can plug
        EV chargers / household later). The MV simulation still runs fine.
      - If you pass with_der=False, we still create the sgen elements but keep them at 0 MW
        unless you set set_at_max=True below or provide a time-series CSV.
    """
    # Base CIGRE MV (no built-in DER — we add your units explicitly)
    net = pn.create_cigre_network_mv(with_der=False)

    # --- Add your LV transformers (structure only; no LV loads) ---
    _lv15, tr15 = _add_mv_lv_trafo(net, mv_bus=7,  sn_mva=0.25, name="CS_Trafo_15")
    _lv16, tr16 = _add_mv_lv_trafo(net, mv_bus=14, sn_mva=0.25, name="CS_Trafo_16")   
    _lv17, tr17 = _add_mv_lv_trafo(net, mv_bus=14, sn_mva=1.00, name="HH_Trafo_17")   

    ids = {
        "trafos": {"CS15": tr15, "CS16": tr16},
        "lv_buses": {"CS15": _lv15, "CS16": _lv16},
    }

    # --- Add your DER (as sgens) ---
    # If with_der=True, we let the time-series drive them (start at 0).
    # If you want them to sit at their nameplate when no CSV is provided, set set_at_max=True.
    set_at_max = True

    der_specs = [
        (3,   20.0,  "PV_3_20kW"),
        (4,   20.0,  "PV_4_20kW"),
        (5,   30.0,  "PV_5_30kW"),
        (10,  40.0,  "PV_10_40kW"),
        (11,  10.0,  "PV_11_10kW"),
        (9,   30.0,  "PV_9_30kW"),
        (8,   30.0,  "PV_8_30kW"),
        (7, 1500.0,  "WIND_7_1500kW")
    ]

    for bus, pkw, name in der_specs:
        _add_sgen_with_q_limits(net, bus, pkw, name, q_cap_ratio=0.33, set_at_max=(set_at_max and with_der))

    # Add loads to LV buses 15, 16, and 17 (15, 16 connects to charging stations, current not working)
    # pp.create_load(net, bus=_lv15, p_mw=0.06, q_mvar=0.02, name="Load_15")
    # pp.create_load(net, bus=_lv16, p_mw=0.06, q_mvar=0.02, name="Load_16")
    pp.create_load(net, bus=_lv17, p_mw=0.03, q_mvar=0.01, name="Load_17")

    return net, ids
