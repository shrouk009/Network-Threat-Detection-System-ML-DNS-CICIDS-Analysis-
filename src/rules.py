from typing import Dict

import pandas as pd


def _col(df: pd.DataFrame, name: str):
    mapping = {str(c).strip().lower(): c for c in df.columns}
    return mapping.get(name.lower())


def simulate_detection_rules(df: pd.DataFrame) -> Dict:
    flow_duration_col = _col(df, "flow duration")
    fwd_pkt_col = _col(df, "total fwd packets")
    bwd_pkt_col = _col(df, "total backward packets")
    syn_flag_col = _col(df, "syn flag count")

    rule_hits = {}

    if flow_duration_col and fwd_pkt_col:
        duration = pd.to_numeric(df[flow_duration_col], errors="coerce").fillna(0)
        fwd_packets = pd.to_numeric(df[fwd_pkt_col], errors="coerce").fillna(0)
        rule_hits["possible_flood_short_duration"] = int(((duration < 20000) & (fwd_packets > 100)).sum())
    else:
        rule_hits["possible_flood_short_duration"] = 0

    if bwd_pkt_col and fwd_pkt_col:
        bwd_packets = pd.to_numeric(df[bwd_pkt_col], errors="coerce").fillna(0)
        fwd_packets = pd.to_numeric(df[fwd_pkt_col], errors="coerce").fillna(0)
        rule_hits["possible_asymmetric_ddos"] = int(((fwd_packets > 80) & (bwd_packets < 5)).sum())
    else:
        rule_hits["possible_asymmetric_ddos"] = 0

    if syn_flag_col:
        syn_flags = pd.to_numeric(df[syn_flag_col], errors="coerce").fillna(0)
        rule_hits["possible_syn_abuse"] = int((syn_flags > 2).sum())
    else:
        rule_hits["possible_syn_abuse"] = 0

    total_flagged = int(sum(rule_hits.values()))
    return {"rule_hits": rule_hits, "rule_flagged_total": total_flagged}
