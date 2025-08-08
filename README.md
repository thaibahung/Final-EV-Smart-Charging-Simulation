# EV Grid Sim – Grid‑only (CIGRE MV + LV trafos)

Runs a 24h, 30‑min time‑series power‑flow on a CIGRE MV network, adds 3 LV buses via 0.25 MVA (20/0.4 kV) transformers at MV buses 15/16/17, and logs:
- Bus voltages/angles and P/Q injections
- Transformer loading%
- Line flows and loading%
- LV transformer headroom Pmax_LV (kW) using: Pmax_LV = S_nom * (100 − loading%) / 100 * cosφ
