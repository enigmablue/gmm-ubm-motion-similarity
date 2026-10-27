"""Build the review bundle: provenance.json and numbers.json.

`numbers.json` is the review contract. Each entry names one quantity that the
manuscript reports, states where it appears, and records how to recompute it
from a stored result file. `verify.py` re-derives every entry and fails loudly
on a mismatch, so the bundle can be checked rather than trusted.

Everything here is derived, not typed. If a script changes and the results move,
re-running this file changes numbers.json too, and `git diff` shows exactly
which reported values moved.

Usage:  python make_bundle.py
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
PR = HERE.parent                       # paper_review/
RES = PR.parent / "output" / "paper_review" / "final"
OUT = HERE / "provenance.json"
NUM = HERE / "numbers.json"

sys.path.insert(0, str(PR))

# Scripts that produce a reported number, in dependency order.
CORE = [
    "pipeline.py",
    "pipeline_retrieval.py",
    "pipeline_comparators.py",
    "pipeline_aggregate.py",
    "pipeline_extras.py",
    "pipeline_units_tiebreak.py",
    "pipeline_hand_ablation.py",
    "pipeline_head_ablation.py",
    "pipeline_dtw_ablation.py",
    "pipeline_comparator_a.py",
    "pipeline_bh_family.py",
    "pipeline_textstats.py",
    "report_manifest.py",
]
# Modules those scripts import.
DEPS = [
    "gmm_pure_fullbody.py",
    "review2_raw_scores.py",
    "review2_checks.py",
    "table_all_methods.py",
    "aggregate_seven.py",
    "crossview_exact.py",
    "session_retrieval.py",
    "canonical.py",
    "review2_method_tests.py",
]
# Extraction scripts that produce the corrected feature sources.
EXTRACT = [
    "extract_maincam_hands_std.py",
    "extract_solo_head_std.py",
    "extract_fullbody_mouth.py",
]

INPUTS = [
    "output/maincam_hands_std",
    "output/fullbody_std/landmarks",
    "output/fullbody_std/features",
    "output/maincam",
    "output/solo_head_std",
    "output/stella",
    "output/aru",
]
FILES_INPUTS = [
    "output/talking_pct.json",
    "output/fullbody_mouth.csv",
]

PAIRS = [f"({a},{b})" for a, b in combinations(["1", "2", "3", "4", "5", "6"], 2)]
TARGET = "(4,5)"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def inventory():
    out = {}
    for rel in INPUTS:
        d = PR.parent / rel
        if not d.exists():
            out[rel] = {"exists": False}
            continue
        files = sorted(d.glob("*.csv"))
        out[rel] = {
            "exists": True,
            "files": len(files),
            "bytes": sum(f.stat().st_size for f in files),
            "sha256_16": {f.name: sha(f) for f in files},
        }
    for rel in FILES_INPUTS:
        f = PR.parent / rel
        out[rel] = ({"exists": False} if not f.exists()
                    else {"exists": True, "bytes": f.stat().st_size,
                          "sha256_16": sha(f)})
    return out


def env():
    import numpy, scipy, sklearn
    try:
        import importlib.metadata as md
        fastdtw = md.version("fastdtw")
    except Exception:
        fastdtw = "unknown"
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "fastdtw": fastdtw, "platform": platform.platform(),
            "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ---------------------------------------------------------------- number set
def load(name, units):
    return json.loads((RES / f"{name}_{units}.json").read_text())


def rank_of(C, pair=TARGET):
    t = C[pair]
    above = sum(1 for v in C.values() if v > t + 1e-12)
    tied = sum(1 for v in C.values() if abs(v - t) <= 1e-12)
    return 1 + above + (tied - 1) / 2


def build_numbers(units="fw"):
    main = load("pipeline", units)
    ret = load("pipeline_retrieval", units)
    dtw = load("pipeline_dtw", units)
    ext = load("pipeline_extras", units)
    hand = load("hand_ablation", units)
    head = load("head_ablation", units)
    texts = load("pipeline_textstats", units)
    dtwab = load("dtw_ablation", units)
    compa = load("comparator_a", units)
    bhf = json.loads((RES / "bh_family.json").read_text())
    agg = json.loads((RES / "aggregate_reproduction.json").read_text())

    N = []

    def add(nid, where, value, source, note=""):
        N.append({"id": nid, "where": where, "value": value,
                  "source": f"output/paper_review/final/{source}",
                  "units": units, "note": note})

    # Table 2
    for dom in ("Hand_Stella", "Hand_Aru", "Head_Stella", "Head_Aru"):
        t = main["table2"][dom]
        add(f"t2.{dom}.rank", f"Table 2, {dom}, Rank column", t["rank"],
            f"pipeline_{units}.json")
        add(f"t2.{dom}.K", f"Table 2, {dom}, K column", t["K"],
            f"pipeline_{units}.json")
        add(f"t2.{dom}.r", f"Table 2, {dom}, r column", t["r"],
            f"pipeline_{units}.json")
        if t["rows"] is not None:
            add(f"t2.{dom}.rows", f"Table 2, {dom}, Frames column",
                t["rows"], f"pipeline_{units}.json")
        add(f"t2.{dom}.llr", f"Table 2, {dom}, (4,5) LLR column",
            round(t["llr"], 4), f"pipeline_{units}.json")

    # Table 3
    for p in PAIRS:
        add(f"t3.{p}.C", f"Table 3, pair {p}, C column",
            round(main["consensus_side"][p], 3), f"pipeline_{units}.json")
    add("t3.c45_rank", "Table 3, (4,5) consensus rank",
        rank_of(main["consensus_side"]), f"pipeline_{units}.json")
    second = max(v for k, v in main["consensus_side"].items() if k != TARGET)
    add("t3.c45_gap", "Results, gap from (4,5) to the next pair",
        round(main["consensus_side"][TARGET] - second, 3),
        f"pipeline_{units}.json")

    # Table 5
    cv = {r["test"]: r for r in main["crossview"]}
    add("t5.fw.rho", "Table 5, framework cross-view rho",
        round(cv["consensus"]["rho"], 3), f"pipeline_{units}.json")
    add("t5.fw.p", "Table 5, framework exact p", cv["consensus"]["p"],
        f"pipeline_{units}.json")
    add("t5.fw.handaru_rho", "Results H1a, strongest domain rho",
        round(cv["Hand_Aru"]["rho"], 3), f"pipeline_{units}.json")
    add("t5.fw.speaker_hits", "Table 5, framework speaker retrieval",
        ret["speaker_hits"], f"pipeline_retrieval_{units}.json")
    add("t5.fw.domain_hits", "Table 5, framework domain retrieval",
        ret["domain_hits"], f"pipeline_retrieval_{units}.json")
    add("t5.fw.speaker_p", "Table 5 caption, framework speaker p",
        round(ret["speaker_p_exact"], 3),
        f"pipeline_retrieval_{units}.json")
    add("t5.fw.domain_p", "Table 5 caption, framework domain p",
        round(ret["domain_p_exact"], 3), f"pipeline_retrieval_{units}.json")
    add("t5.fw.speaker_mrr", "Results H2a, speaker MRR",
        round(ret["speaker_mrr"], 3), f"pipeline_retrieval_{units}.json")
    add("t5.fw.domain_mrr", "Results H2a, domain MRR",
        round(ret["domain_mrr"], 3), f"pipeline_retrieval_{units}.json")
    add("t5.dtw.rho", "Table 5, sequence alignment rho",
        round(dtw["crossview"][0]["rho"], 3), f"pipeline_dtw_{units}.json")
    add("t5.dtw.p", "Table 5, sequence alignment p",
        round(dtw["crossview"][0]["p"], 3), f"pipeline_dtw_{units}.json")
    add("t5.dtw.speaker_hits", "Table 5, DTW speaker retrieval",
        ext["dtw_retrieval"]["speaker_hits"], f"pipeline_extras_{units}.json")
    add("t5.dtw.domain_hits", "Table 5, DTW domain retrieval",
        ext["dtw_retrieval"]["domain_hits"], f"pipeline_extras_{units}.json")
    add("t5.agg.rho", "Table 5, aggregate rho",
        round(agg["crossview"][0]["rho"], 3), "aggregate_reproduction.json")
    add("t5.agg.p", "Table 5, aggregate p",
        round(agg["crossview"][0]["p"], 3), "aggregate_reproduction.json")
    add("t5.agg.speaker_hits", "Table 5, aggregate speaker retrieval",
        ext["aggregate_retrieval"]["speaker_hits"],
        f"pipeline_extras_{units}.json")
    add("t5.agg.domain_hits", "Table 5, aggregate domain retrieval",
        ext["aggregate_retrieval"]["domain_hits"],
        f"pipeline_extras_{units}.json")

    # Table 6
    add("t6.fw.side", "Table 6, framework side view",
        rank_of(main["consensus_side"]), f"pipeline_{units}.json")
    add("t6.fw.front", "Table 6, framework frontal view",
        rank_of(main["consensus_frontal"]), f"pipeline_{units}.json")
    add("t6.dtw.side", "Table 6, sequence alignment side view",
        rank_of(dtw["side_C"]), f"pipeline_dtw_{units}.json")
    add("t6.agg.side", "Table 6, aggregate side view",
        rank_of(agg["side_C"]), "aggregate_reproduction.json")

    # Table 7
    for k, v in hand.items():
        add(f"t7.{k}.C45", f"Table 7, {k}, C45", round(v["C45"], 3),
            f"hand_ablation_{units}.json")
        add(f"t7.{k}.rank", f"Table 7, {k}, consensus rank", v["cons_rank"],
            f"hand_ablation_{units}.json")

    # Table 8
    for k, v in head.items():
        add(f"t8.{k}.C45", f"Table 8, {k}, C45", round(v["C45"], 3),
            f"head_ablation_{units}.json")
        add(f"t8.{k}.z", f"Table 8, {k}, z", round(v["z"], 2),
            f"head_ablation_{units}.json")

    # Comparator A, literature features
    for dom, rk in compa["side"]["ranks"].items():
        add(f"compa.{dom}", f"Comparator A, side view, {dom} rank of (4,5)",
            rk, f"comparator_a_{units}.json")
    add("compa.side.C", "Comparator A, side-view consensus score",
        round(compa["side"]["C"][TARGET], 3), f"comparator_a_{units}.json")
    add("compa.side.rank", "Comparator A, side-view consensus rank",
        compa["side"]["rank_of_target"], f"comparator_a_{units}.json")

    # H1a: the tested statistic and its descriptive decomposition
    for key, where in [("consensus", "Results H1a, tested consensus correlation"),
                       ("head", "Results H1a, head-only correlation (descriptive)"),
                       ("hand", "Results H1a, hand-only correlation (descriptive)"),
                       ("Hand_Aru", "Results H1a, Aru hands (descriptive)"),
                       ("Hand_Stella", "Results H1a, Stella hands (descriptive)")]:
        t = bhf["tests"][key]
        add(f"h1a.{key}.rho", f"{where}, rho", round(t["rho"], 3),
            "bh_family.json")
        add(f"h1a.{key}.p", f"{where}, exact p", t["p"], "bh_family.json")

    # plain-language concordance anchor, reported inside Results H1a
    for key, where in [("consensus", "Results H1a, comparisons agreeing"),
                       ("head only", "Results H1a, head-only comparisons agreeing"),
                       ("hand only", "Results H1a, hand-only comparisons agreeing")]:
        c = bhf["concordance"][key]
        tag = key.replace(" ", "_")
        add(f"h1a.conc.{tag}.agree", f"{where}, comparisons agreeing",
            c["agree"], "bh_family.json")
        add(f"h1a.conc.{tag}.total", f"{where}, comparisons usable",
            c["total"], "bh_family.json")

    # Prose statistics
    o = texts["overlap"]
    add("h1b.top5", "Results H1b, top-five overlap", o["top5"],
        f"pipeline_textstats_{units}.json")
    add("h1b.top10", "Results H1b, top-ten overlap", o["top10"],
        f"pipeline_textstats_{units}.json")
    add("h1b.mean_shift", "Results H1b, mean rank shift",
        round(o["mean_shift"], 2), f"pipeline_textstats_{units}.json")
    ls = texts["loo_summary"]
    add("h1a.loo_lo", "Results H1a, leave-one-out minimum",
        round(ls["lo"], 3), f"pipeline_textstats_{units}.json")
    add("h1a.loo_hi", "Results H1a, leave-one-out maximum",
        round(ls["hi"], 3), f"pipeline_textstats_{units}.json")
    add("h1a.loo_ns", "Results H1a, count not significant",
        ls["below_p05"], f"pipeline_textstats_{units}.json")

    # Sequence alignment with matched representations
    add("seq.9D.side", "Sequence alignment, 9D head, side consensus",
        dtwab["head"]["9D"]["side_cons"], f"dtw_ablation_{units}.json")
    add("seq.18D.side", "Sequence alignment, 18D head, side consensus",
        dtwab["head"]["18D"]["side_cons"], f"dtw_ablation_{units}.json")
    add("seq.4D.finger.aru", "Sequence alignment, 4D finger, Aru hand rank",
        dtwab["hand"]["4D finger"]["hand_ranks"]["Aru"],
        f"dtw_ablation_{units}.json")

    # Frontal per-domain ranks
    for d, v in ext["frontal_ranks"].items():
        add(f"front.{d}.rank", f"Results, frontal {d} rank of (4,5)",
            v["rank"], f"pipeline_extras_{units}.json")

    return N


def write_manifest(entries, units):
    """MANIFEST.md: one table per manuscript area, generated from the contract."""
    groups = [
        ("t2", "Table 2 — per-domain settings and ranks (side view)",
         "Each domain's BIC-selected component count, self-consistency relevance "
         "factor, frame count, the nominated pair's raw likelihood-ratio score, "
         "and its rank among 15 pairs."),
        ("t3", "Table 3 — consensus ranking of all 15 pairs (side view)",
         "C is the proportion of the 56 domain-to-domain comparisons a pair wins. "
         "Mean across the 15 pairs is exactly 0.5 by construction."),
        ("t5", "Table 5 — cross-view consistency and matched-session retrieval",
         "Spearman correlation across the 15 pairs with an exact session-label "
         "permutation p-value; retrieval is hits over trials, chance 2/12 and 4/24."),
        ("t6", "Table 6 — consensus rank of the nominated pair by method and view",
         "Lower is better; chance is 8.0 of 15."),
        ("t7", "Table 7 — hand representation ablation",
         "Head features held fixed, so rows differ only in the hand vector."),
        ("t8", "Table 8 — head temporal-context ablation",
         "Hand features held fixed at the corrected 12D vector. z measures how far "
         "the nominated pair's consensus sits from the other fourteen pairs."),
        ("h1a", "Results, H1a — the tested statistic and its decomposition",
         "One hypothesis test: the consensus correlation. The head-only, hand-only "
         "and per-speaker values are descriptive decomposition and are neither "
         "adjusted nor claimed as tests."),
        ("h1b", "Results, H1b — agreement between views",
         "Overlap of the top of each ordering, and mean rank movement per pair."),
        ("seq", "Sequence alignment with matched representations",
         "DTW fed the framework's own feature vectors."),
        ("front", "Frontal-view per-domain ranks",
         "Rank of the nominated pair within each frontal domain."),
        ("compa", "Comparator A — sequence alignment with literature features",
         "15-dimensional wrist-relative hands and 3-dimensional head angles. "
         "Reported only as a specification variation, not in the main comparison."),
    ]
    lines = [
        "# Manifest of reported numbers",
        "",
        f"Generated from `numbers.json` for unit choice `{units}`. "
        f"{len(entries)} entries.",
        "",
        "Every value below is recomputed by `verify.py`. The `source` column names "
        "the result file written by the script named in the section heading.",
        "",
    ]
    for prefix, title, blurb in groups:
        rows = [e for e in entries if e["id"].startswith(prefix + ".")]
        if not rows:
            continue
        lines += [f"## {title}", "", blurb, "",
                  "| id | reported in | value | source |",
                  "|---|---|---|---|"]
        for e in rows:
            v = e["value"]
            vs = f"{v:g}" if isinstance(v, (int, float)) else str(v)
            lines.append(f"| `{e['id']}` | {e['where']} | {vs} | "
                         f"`{Path(e['source']).name}` |")
        lines.append("")
    lines += [
        "## Not covered",
        "",
        "See `README.md` §6. In short: the literature-feature comparator still "
        "reports its original numbers, the comparison-method retrieval p-values "
        "were withdrawn rather than recomputed, the session bootstrap is not "
        "numerically stable at six sessions, and the pixel-unit sensitivity is "
        "reported separately rather than folded into the manuscript.",
        "",
    ]
    (HERE / "MANIFEST.md").write_text("\n".join(lines))


def main():
    prov = {
        "bundle_version": 1,
        "workspace": str(PR.parent),
        "environment": env(),
        "scripts_core": {n: sha(PR / n) for n in CORE},
        "scripts_dependencies": {n: sha(PR / n) for n in DEPS},
        "scripts_extraction": {n: sha(PR / n) for n in EXTRACT},
        "inputs": inventory(),
        "result_files": {p.name: {"sha256_16": sha(p),
                                  "bytes": p.stat().st_size}
                         for p in sorted(RES.glob("*.json"))},
    }
    OUT.write_text(json.dumps(prov, indent=1))

    entries = build_numbers("fw")
    nums = {"units_default": "fw", "generated": prov["environment"]["built"],
            "entries": entries}
    NUM.write_text(json.dumps(nums, indent=1))
    write_manifest(entries, "fw")

    print(f"wrote {OUT.name}: {len(prov['scripts_core'])} scripts, "
          f"{len(prov['inputs'])} input roots")
    print(f"wrote {NUM.name}: {len(nums['entries'])} reported numbers")
    print("wrote MANIFEST.md")
    missing = [f for f in prov["result_files"]
               if prov["result_files"][f]["sha256_16"] is None]
    if missing:
        print(f"WARNING missing result files: {missing}")


if __name__ == "__main__":
    main()
