# MV Distribution Network Simulation (CIGRE MV-Based)

This project simulates a **medium-voltage (MV)** distribution network based on the CIGRE MV benchmark model, adapted to match the topology shown in your diagram:

- **HV/MV subtransmission at 110 kV** (bus 0)
- Two **25 MVA 110/20 kV transformers** feeding buses 1 and 12
- MV distribution ring and branches interconnecting buses 3–4–5–6–7–8–3, with additional links 3–10–9–8 and 8–14–13–12
- No low-voltage (LV) loads or charging stations included (yet)
- Optional DER (distributed generation) can be included if `with_der: true` in the config

The simulation runs a **time-series AC power flow** over a 24-hour period with 30-minute steps, scaling existing network loads and (optionally) generators according to profiles.

---

## Features

- Adjustable **time horizon** and **step size**
- Synthetic or CSV-based **load/generation profiles**
- Results exported for every time step:
  - **Bus voltages, angles, P/Q injections**
  - **Line flows, loadings**
  - **Transformer loadings, HV/LV power flows**

---
