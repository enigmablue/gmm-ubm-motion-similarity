"""Aggregate-baseline comparator, re-derived today.

The aggregate does not use the 12D/18D vectors at all -- it is seven summary
measures per speaker -- so "merging" it is not meaningful. What is needed is to
confirm the manuscript's aggregate row still reproduces from the data on disk,
and to record the one asymmetry it carries.

Asymmetry, recorded not corrected: the aggregate's side-view finger measures are
read from `output/{vid}_hands.csv`, which is sampled at stride 3 (10 Hz), while
the framework's hands are at stride 5 (6 Hz). Its head measures come from
`output/maincam/{vid}_features.csv` at stride 5. So the aggregate mixes sampling
rates internally and does not match the framework's grid. CurlMean, CurlStd and
CurlP25 are distributional and should be mildly sensitive; HeadGest and
YawPitchRatio are not affected because their source is already stride 5.

WRITES  output/paper_review/final/aggregate_reproduction.json
Usage:  python pipeline_aggregate.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from aggregate_seven import mouth_stats  # noqa: E402
from crossview_exact import agg_domains  # noqa: E402
from pipeline import bh, exact_rho  # noqa: E402
from review2_checks import consensus_from_scores  # noqa: E402

DEST = Path(r"D:\Soqqle\Unity_Animation\FacialCapture\output\paper_review\final")
LABELS = ["consensus", "Hand_Stella", "Hand_Aru", "Head_Stella", "Head_Aru"]


def main():
    m_mc, m_fb = mouth_stats()
    C, dom = {}, {}
    for view, mouths in [("maincam", m_mc), ("fullbody", m_fb)]:
        raw = {k: v for k, v in agg_domains(view, mouths).items()
               if not k.startswith("Mouth")}
        dom[view] = raw
        C[view], _ = consensus_from_scores(raw, False)

    print("aggregate baseline, four head/hand domains, mouth excluded")
    print("cross-view consistency")
    pv, rows = [], []
    for lab in LABELS:
        side = C["maincam"] if lab == "consensus" else dom["maincam"][lab]
        front = C["fullbody"] if lab == "consensus" else dom["fullbody"][lab]
        rho, p = exact_rho(side, front)
        pv.append(p)
        rows.append((lab, rho, p))
    adj = bh(pv)
    for (lab, rho, p), a in zip(rows, adj):
        print(f"  {lab:<14} rho = {rho:+.3f}  p = {p:.3f}  p_BH = {a:.3f}")

    payload = {"crossview": [{"test": l, "rho": r, "p": p, "p_bh": float(a)}
                             for (l, r, p), a in zip(rows, adj)],
               "side_C": C["maincam"]}
    (DEST / "aggregate_reproduction.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {DEST / 'aggregate_reproduction.json'}")


if __name__ == "__main__":
    main()
