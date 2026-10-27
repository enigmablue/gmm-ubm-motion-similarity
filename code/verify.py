"""Check that the reported numbers still reproduce.

Two modes:

  python verify.py            compare numbers.json against the result files
                              currently on disk. Fast (seconds). Detects drift
                              between the contract and the stored results.

  python verify.py --rerun    re-execute every producing script first, then
                              compare. Slow (about 8 minutes). This is the real
                              reproduction test.

Exit code 0 if every entry matches, 1 otherwise, so it can gate a review.

Tolerance is 0 for integer values and 5e-4 for decimals, which is tight enough
to catch a changed parse and loose enough to absorb last-digit rounding.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PR = HERE.parent
PY = PR.parent / ".venv" / "Scripts" / "python.exe"

sys.path.insert(0, str(HERE))
from make_bundle import CORE, build_numbers  # noqa: E402

TOL = 5e-4

# Order matters: later scripts read files written by earlier ones.
ORDER = [
    ("pipeline.py", []),
    ("pipeline_retrieval.py", []),
    ("pipeline_comparators.py", []),
    ("pipeline_aggregate.py", []),
    ("pipeline_extras.py", []),
    ("pipeline_units_tiebreak.py", []),
    ("pipeline_hand_ablation.py", []),
    ("pipeline_head_ablation.py", []),
    ("pipeline_dtw_ablation.py", []),
    ("pipeline_comparator_a.py", []),
    ("pipeline_bh_family.py", []),
    ("pipeline_textstats.py", []),
]


def rerun():
    for script, extra in ORDER:
        print(f"  running {script} ...", flush=True)
        r = subprocess.run([str(PY), str(PR / script), *extra],
                           cwd=str(PR.parent), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED {script}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            sys.exit(1)


def check(contract):
    want = {e["id"]: e for e in contract["entries"]}
    got = {e["id"]: e for e in build_numbers(contract["units_default"])}

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    fails, oks = [], 0
    for nid in sorted(set(want) & set(got)):
        a, b = want[nid]["value"], got[nid]["value"]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ok = abs(float(a) - float(b)) <= TOL
        else:
            ok = a == b
        if ok:
            oks += 1
        else:
            fails.append((nid, want[nid]["where"], a, b))

    print(f"\n{oks} of {len(want)} reported numbers reproduce")
    if missing:
        print(f"  MISSING from current results ({len(missing)}): {missing[:8]}")
    if extra:
        print(f"  NEW in current results, not in contract ({len(extra)}): "
              f"{extra[:8]}")
    if fails:
        print(f"\n  MISMATCHES ({len(fails)}):")
        for nid, where, a, b in fails:
            print(f"    {nid:<26} {where}")
            print(f"      reported {a!r}   recomputed {b!r}")
        return 1
    if missing:
        return 1
    print("\nAll reported numbers reproduce.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="re-execute the analysis scripts before comparing")
    args = ap.parse_args()

    contract = json.loads((HERE / "numbers.json").read_text())
    if args.rerun:
        print("re-running the analysis chain")
        rerun()
    sys.exit(check(contract))


if __name__ == "__main__":
    main()
