# Runbook

Ordered commands for reproducing this study's reported numbers. Paths are
relative to `d:\Soqqle\Unity_Animation\FacialCapture`.

Interpreter: `.\.venv\Scripts\python.exe` — note the leading dot.
Shell: Windows PowerShell. Chain commands with `;`, never `&&`.

Environment the bundle was built against:

| Component | Version |
|---|---|
| Python | 3.11.9 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| fastdtw | 0.3.4 |
| Platform | Windows 10.0.26200 |

Exact hashes of every script and input file are in `provenance.json`. If you want
to confirm you are running the reviewed code, compare
`Get-FileHash paper_review\pipeline.py -Algorithm SHA256` against the recorded
value. The recorded values are the first 16 hex characters.

---

## 1. Fast check — two commands, under a minute

```powershell
cd d:\Soqqle\Unity_Animation\FacialCapture
.\.venv\Scripts\python.exe paper_review\bundle\verify.py
```

Expected:

```
109 of 109 reported numbers reproduce

All reported numbers reproduce.
```

Exit code 0. If anything differs you get a per-number block naming the manuscript
location, the reported value and the recomputed value. A mismatch here means the
result files on disk no longer match the reviewed contract — investigate before
going further.

This mode reads only `output/paper_review/final/*.json`. It does not run any model.

---

## 2. Full check — re-run the analysis, about 7.5 minutes

```powershell
.\.venv\Scripts\python.exe paper_review\bundle\verify.py --rerun
```

This executes the eleven producing scripts in dependency order and then
compares again. Measured end to end: **4.9 minutes**, 101 of 101 passing. An
earlier run of the ten scripts before Comparator A was added took 7.3 minutes, so
treat anywhere from 5 to 8 minutes as normal depending on machine state.

Per-script timings observed on the build machine:

| Script | Purpose | Time |
|---|---|---|
| `pipeline.py` | tables 2 and 3, cross-view tests, frontal consensus | 0.4 min |
| `pipeline_retrieval.py` | matched-session retrieval, framework | 1.5 min |
| `pipeline_comparators.py` | DTW cross-view comparison | 0.2 min |
| `pipeline_aggregate.py` | seven-measure baseline | 0.1 min |
| `pipeline_extras.py` | frontal per-domain ranks, comparator retrieval | 1.0 min |
| `pipeline_units_tiebreak.py` | face-width vs pixel criterion | 0.3 min |
| `pipeline_hand_ablation.py` | table 7 | 0.5 min |
| `pipeline_head_ablation.py` | table 8 | 0.4 min |
| `pipeline_dtw_ablation.py` | DTW across feature configurations | 3.0 min |
| `pipeline_comparator_a.py` | DTW with the literature feature set | 0.6 min |
| `pipeline_textstats.py` | leave-one-out, view overlap, bootstrap | 0.6 min |

Everything is deterministic: `random_state=42` throughout, and the bootstrap uses
`numpy.random.default_rng(seed=0)`. Two runs should produce byte-identical JSON
apart from the `generated` timestamp.

---

## 3. Reading a single number

To see every reported number in one place, with the manuscript location beside it:

```powershell
.\.venv\Scripts\python.exe paper_review\report_manifest.py --units fw
```

To see the pixel-unit sensitivity variant:

```powershell
.\.venv\Scripts\python.exe paper_review\report_manifest.py --units px
```

Most scripts accept `--units fw|px`. `fw` is face-width normalisation and is what
the manuscript reports; `px` is the sensitivity analysis described in
`README.md` §6.

---

## 4. Level 3 — rebuild the features from landmarks

Only needed if you doubt the feature CSVs themselves rather than the analysis
above them. Roughly an hour.

```powershell
# Corrected side-view hands, stride 5, the source the analysis now uses
.\.venv\Scripts\python.exe paper_review\extract_maincam_hands_std.py

# Frontal head features
.\.venv\Scripts\python.exe paper_review\extract_solo_head_std.py

# Mouth activity for the aggregate baseline
.\.venv\Scripts\python.exe paper_review\extract_fullbody_mouth.py

# Then re-run everything
.\.venv\Scripts\python.exe paper_review\bundle\verify.py --rerun
```

Rebuilding landmarks from the source videos is not covered: the `.mp4` files are
gitignored and depict two identifiable people, so whether to share them is an
author decision rather than a technical one.

---

## 5. Common failures

**`ModuleNotFoundError` for `gmm_pure_fullbody` or similar.** The scripts insert
`paper_review/` into `sys.path` themselves, but only when run with the package
root as the working directory. Run from `FacialCapture/`.

**`verify.py` reports MISSING entries.** A result JSON is absent. Run
`verify.py --rerun` to regenerate.

**A single number differs in the third decimal.** Check `provenance.json` for the
hash of the script that produces it. If the script changed since the bundle was
built, run `make_bundle.py` to refresh the contract and `git diff numbers.json` to
see exactly which reported values move. That diff is the review artifact.

**`pipeline_dtw_ablation.py` takes much longer than 3 minutes.** Sequence
alignment cost scales with sequence length and the DTW radius. It is the slowest
step and is not parallelised.

---

## 6. If you change anything

`make_bundle.py` regenerates `numbers.json` and `provenance.json` from current
results. Regenerate them after any change and keep the diff:

```powershell
.\.venv\Scripts\python.exe paper_review\bundle\make_bundle.py
git diff paper_review/bundle/numbers.json
```

If a reported number moved, that diff is the thing a reviewer needs to see. Do not
edit `numbers.json` by hand to make `verify.py` pass — the whole point of the
bundle is that the contract is derived from the results rather than typed.
