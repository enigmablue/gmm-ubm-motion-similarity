"""Single manifest of every number the manuscript quotes, for one unit choice.

Run after `pipeline.py`, `pipeline_retrieval.py`, `pipeline_comparators.py`,
`pipeline_extras.py`, `pipeline_textstats.py` and the two ablation scripts.
Nothing is recomputed from scratch here beyond the handful of consensus ranks
that were never persisted; everything else is read back from the stored JSON.

Prints a manifest grouped by the place each number goes, so a table edit is a
transcription rather than a recall task.

Usage:  python report_manifest.py [--units fw|px]
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from aggregate_seven import mouth_stats  # noqa: E402
from crossview_exact import agg_domains  # noqa: E402
from pipeline import DEST  # noqa: E402
from pipeline_comparators import consensus as dtw_consensus, domain_dtw  # noqa: E402
from review2_checks import consensus_from_scores  # noqa: E402

vids = ["1", "2", "3", "4", "5", "6"]
PAIRS = list(combinations(vids, 2))
TARGET = "(4,5)"


def rank_of(C):
    vals = sorted(C.values(), reverse=True)
    t = C[TARGET]
    above = sum(1 for v in vals if v > t + 1e-12)
    tied = sum(1 for v in vals if abs(v - t) <= 1e-12)
    return 1 + above + (tied - 1) / 2


def second_gap(C):
    others = [v for k, v in C.items() if k != TARGET]
    return C[TARGET] - max(others)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=["fw", "px"], default="fw")
    args = ap.parse_args()
    units = args.units
    J = lambda n: json.loads((DEST / f"{n}_{units}.json").read_text())

    main_r = J("pipeline")
    det = J("pipeline_retrieval")
    dtw = J("pipeline_dtw")
    ext = J("pipeline_extras")
    hand = J("hand_ablation")
    head = J("head_ablation")
    texts = J("pipeline_textstats")
    agg = json.loads((DEST / "aggregate_reproduction.json").read_text())

    print(f"### manifest for units = {units}\n")

    print("## Table 2  per-domain, side view")
    for d, v in sorted(main_r["table2"].items()):
        rows = v["rows"] if v["rows"] is not None else "n/a"
        print(f"  {d:<13} dim {v['dim']:>2}  rows {rows:>6}  "
              f"K {v['K']}  r {v['r']:<4} llr {v['llr']:+.4f}  rank {v['rank']:.0f}/15")

    print("\n## Key sentence after Table 2 (side-view domain ranks in order)")
    order = ["Hand_Stella", "Hand_Aru", "Head_Stella", "Head_Aru"]
    ranks = [main_r["table2"][d]["rank"] for d in order]
    print(f"  ranks {ranks}  worst {max(ranks):.0f}  "
          f"all in top four: {max(ranks) <= 4}")
    Ks = {d: main_r["table2"][d]["K"] for d in order}
    rs = {d: main_r["table2"][d]["r"] for d in order}
    print(f"  K range {min(Ks.values())}-{max(Ks.values())}   "
          f"r range {min(rs.values())}-{max(rs.values())}")

    print("\n## Table 3  consensus, side view")
    side = main_r["consensus_side"]
    ordered = sorted(side.items(), key=lambda kv: -kv[1])
    for i, (k, v) in enumerate(ordered, 1):
        tgt = " <-- nominated" if k == TARGET else ""
        print(f"  {k:<7} {v:.3f}{tgt}")
    print(f"  rank of (4,5): {rank_of(side)}   gap to second: {second_gap(side):.3f}")

    print("\n## Table 5  cross-view correlation and retrieval")
    for row in main_r["crossview"]:
        print(f"  framework {row['test']:<12} rho {row['rho']:+.3f}  "
              f"p {row['p']:.3f}  BH {row['p_bh']:.3f}")
    print(f"  framework retrieval  speaker {det['speaker_hits']}/12 p "
          f"{det['speaker_p_exact']:.3f} MRR {det['speaker_mrr']:.3f}")
    print(f"                       domain  {det['domain_hits']}/24 p "
          f"{det['domain_p_exact']:.3f} MRR {det['domain_mrr']:.3f}")
    print(f"  DTW       consensus    rho {dtw['crossview'][0]['rho']:+.3f}  "
          f"p {dtw['crossview'][0]['p']:.3f}")
    d = ext["dtw_retrieval"]
    print(f"            retrieval  speaker {d['speaker_hits']}/12 p "
          f"{d['speaker_p']:.3f}  domain {d['domain_hits']}/24 p "
          f"{d['domain_p']:.3f}")
    a = ext["aggregate_retrieval"]
    print(f"  aggregate consensus    rho {agg['crossview'][0]['rho']:+.3f}  "
          f"p {agg['crossview'][0]['p']:.3f}")
    print(f"            retrieval  speaker {a['speaker_hits']}/12 p "
          f"{a['speaker_p']:.3f}  domain {a['domain_hits']}/24 p "
          f"{a['domain_p']:.3f}")

    print("\n## Table 6  consensus rank of (4,5) by method and view")
    print(f"  framework side   {rank_of(side):.1f}/15")
    print(f"  framework front  {rank_of(main_r['consensus_frontal']):.1f}/15")
    print(f"  DTW side         {rank_of(dtw['side_C']):.1f}/15")
    _, dtw_front = dtw_consensus(domain_dtw("fullbody", units))
    print(f"  DTW front        {rank_of(dtw_front):.1f}/15")
    print(f"  aggregate side   {rank_of(agg['side_C']):.1f}/15")
    m_mc, m_fb = mouth_stats()
    raw = {k: v for k, v in agg_domains("fullbody", m_fb).items()
           if not k.startswith("Mouth")}
    agg_front, _ = consensus_from_scores(raw, False)
    print(f"  aggregate front  {rank_of(agg_front):.1f}/15")
    print(f"  frontal first pair {main_r['frontal_first']}")

    print("\n## Frontal per-domain ranks of (4,5), Table 6 prose")
    for d, v in ext["frontal_ranks"].items():
        print(f"  {d:<13} rank {v['rank']:.1f}/15  K {v['K']} r {v['r']}")

    print("\n## Table 7  hand ablation")
    for k, v in hand.items():
        print(f"  {k:<22} dims {v['dims']:>2}  St {v['Hand_St']:.0f}  "
              f"Ar {v['Hand_Ar']:.0f}  C45 {v['C45']:.3f}  rank {v['cons_rank']}/15")

    print("\n## Table 8  head ablation")
    for k, v in head.items():
        print(f"  {k:<10} dims {v['dims']:>2}  St {v['Hd_St']:.0f}  "
              f"Ar {v['Hd_Ar']:.0f}  C45 {v['C45']:.3f}  z {v['z']:.2f}")

    print("\n## Prose statistics")
    o = texts["overlap"]
    print(f"  top-5 overlap {o['top5']}/5   top-10 {o['top10']}/10   "
          f"mean shift {o['mean_shift']:.2f}")
    ls = texts["loo_summary"]
    print(f"  leave-one-out rho {ls['lo']:+.3f} (without {ls['worst']}) to "
          f"{ls['hi']:+.3f} (without {ls['best']}); not significant in "
          f"{ls['below_p05']}/6")
    b = texts["boot_fw_minus_dtw"]
    print(f"  bootstrap framework-DTW  [{b['lo']:+.3f}, {b['hi']:+.3f}]")
    b2 = texts["boot_fw_minus_agg"]
    print(f"  bootstrap framework-agg  [{b2['lo']:+.3f}, {b2['hi']:+.3f}]")

    print("\n## Retrieval by modality (H2b)")
    sr = det["self_ranks"]
    ho = sum(1 for d in ("Head_Stella", "Head_Aru")
             for q in vids if sr[d][q] == 1)
    hd = sum(1 for d in ("Hand_Stella", "Hand_Aru")
             for q in vids if sr[d][q] == 1)
    print(f"  head domains {ho}/12   hand domains {hd}/12")


if __name__ == "__main__":
    main()
