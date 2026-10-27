"""The multiple-comparison family, made explicit.

The manuscript corrects across five cross-view tests: the four
speaker-by-body-part domains plus the consensus that aggregates them. That
choice is asserted, never justified, and it is the reason the consensus test
ends up at p_BH = .062.

This script computes the two aggregations that grouping by body part requires --
consensus over the two head domains, and consensus over the two hand domains --
and then reports the same observed correlations under four candidate family
definitions. It does not choose between them: it prints what each yields so the
choice can be argued on design grounds rather than on which one is significant.

Family definitions reported:
  m=1   consensus only. The pre-specified outcome; domains are decomposition.
  m=3   consensus, head-only, hand-only. Body part is the unit of the test.
  m=5   the current manuscript. Each speaker-by-body-part domain is its own test.
  m=2   head-only and hand-only, with the full consensus reported as the primary
        aggregate without adjustment.

WRITES  output/paper_review/final/bh_family.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from pipeline import DEST, PKEYS, bh, consensus, domain_scores, exact_rho  # noqa: E402


def restricted(dom, keys):
    sub = {k: v for k, v in dom.items() if k in keys}
    return consensus(sub)[1]


def concordance(C_side, C_front):
    """Of all pairs of session-pairs, in how many do the two views agree on order?

    A plain-language anchor for Spearman's rho that does not require the reader
    to know what a correlation is. There are C(15,2) = 105 ways to choose two of
    the 15 session pairs; for each, ask which of the two the side view ranks
    higher and which the frontal view ranks higher. Ties count as half.
    """
    keys = list(C_side)
    agree = total = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = C_side[keys[i]] - C_side[keys[j]]
            b = C_front[keys[i]] - C_front[keys[j]]
            if abs(a) < 1e-12 or abs(b) < 1e-12:
                continue
            total += 1
            if (a > 0) == (b > 0):
                agree += 1
    return agree, total


def report(label, C_side, C_front):
    rho, p = exact_rho(C_side, C_front)
    return {"label": label, "rho": rho, "p": p}


def main():
    side = domain_scores("maincam", "fw")
    front = domain_scores("fullbody", "fw")

    full_s = consensus(side)[1]
    full_f = consensus(front)[1]
    heads_s, heads_f = restricted(side, ("Head_SpeakerB", "Head_SpeakerA")), \
        restricted(front, ("Head_SpeakerB", "Head_SpeakerA"))
    hands_s, hands_f = restricted(side, ("Hand_SpeakerB", "Hand_SpeakerA")), \
        restricted(front, ("Hand_SpeakerB", "Hand_SpeakerA"))

    tests = {
        "consensus": report("full four-domain consensus", full_s, full_f),
        "head": report("head only, two domains", heads_s, heads_f),
        "hand": report("hand only, two domains", hands_s, hands_f),
        "Hand_SpeakerB": report("SpeakerB hands",
                              side["Hand_SpeakerB"][0], front["Hand_SpeakerB"][0]),
        "Hand_SpeakerA": report("SpeakerA hands",
                           side["Hand_SpeakerA"][0], front["Hand_SpeakerA"][0]),
        "Head_SpeakerB": report("SpeakerB head",
                              side["Head_SpeakerB"][0], front["Head_SpeakerB"][0]),
        "Head_SpeakerA": report("SpeakerA head",
                           side["Head_SpeakerA"][0], front["Head_SpeakerA"][0]),
    }

    print("observed cross-view correlations")
    for k in ["consensus", "head", "hand", "Hand_SpeakerB", "Hand_SpeakerA",
              "Head_SpeakerB", "Head_SpeakerA"]:
        t = tests[k]
        print(f"  {t['label']:<34} rho {t['rho']:+.3f}   exact p {t['p']:.4f}")

    print("\nplain-language anchor: in how many of the 105 pairs of session pairs "
          "do the two views agree on which is more alike?")
    out_conc = {}
    for label, s, f in [("consensus", full_s, full_f),
                        ("head only", heads_s, heads_f),
                        ("hand only", hands_s, hands_f)]:
        a, n = concordance(s, f)
        out_conc[label] = dict(agree=a, total=n)
        print(f"  {label:<12} {a} of {n}")
    tests_conf = out_conc

    families = {
        "m=1 consensus only": ["consensus"],
        "m=3 consensus + head + hand": ["consensus", "head", "hand"],
        "m=5 four domains + consensus":
            ["consensus", "Hand_SpeakerB", "Hand_SpeakerA", "Head_SpeakerB", "Head_SpeakerA"],
        "m=2 head + hand": ["head", "hand"],
    }

    print("\nadjusted p under each family definition")
    out = {"tests": tests, "families": {}, "concordance": tests_conf}
    for name, keys in families.items():
        ps = [tests[k]["p"] for k in keys]
        adj = bh(ps)
        row = {k: round(float(a), 4) for k, a in zip(keys, adj)}
        out["families"][name] = row
        shown = "  ".join(f"{k}={row[k]:.3f}" for k in keys)
        survivors = [k for k in keys if row[k] < .05]
        print(f"  {name:<32} {shown}")
        print(f"  {'':<32} survives at .05: "
              f"{', '.join(survivors) if survivors else 'none'}")

    (DEST / "bh_family.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {DEST / 'bh_family.json'}")


if __name__ == "__main__":
    main()
