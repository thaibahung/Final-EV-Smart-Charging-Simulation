import pandapower.networks as pn

def build_network(with_der: bool = False):
    """
    Build the CIGRE MV network:
    - Slack/ext_grid at bus 0 (HV-MV subtransmission 110 kV)
    - Two 25 MVA 110/20 kV transformers from bus 0 to buses 1 and 12
    - MV ring/branches per CIGRE layout
    Currently No LV buses, no household, no charging station.
    """
    net = pn.create_cigre_network_mv(with_der=with_der)
    # That's it—pandapower's CIGRE MV already matches the topology you drew.
    # We are not adding any LV transformers or loads.
    return net
