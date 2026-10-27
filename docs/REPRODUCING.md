# Reproducing the reported numbers

## What this repository can and cannot establish

It **can** establish that every number printed in the manuscript follows from the published result files by way of the published code.

It **cannot** establish that the result files follow from the raw video, because the video is not distributed (see `DATA_AVAILABILITY.md`). This is the honest limit of an analysis-only release, and the paper says as much.

## The contract

`contract/numbers.json` holds one entry per statistic stated in the paper. Each entry records:

- `value` — the number as it appears in print;
- `key` / `where` — which result file it comes from and where it is stated;
- the quantity it measures.

`contract/provenance.json` holds a SHA-256 hash for each of the core scripts and the list of input roots. If a script is edited, its hash will no longer match, and the contract no longer describes the code you have.

## Running the check

```bash
python code/verify.py            # fast: recompute from results/ only
python code/verify.py --rerun    # slow (~5 min): re-run the pipelines, then compare
```

Expected output ends with:

```
115 of 115 reported numbers reproduce
```

The fast check reads `results/` and re-derives each reported number arithmetically. The slow check additionally re-runs the analysis scripts against the input data, which requires the recordings.

## Layout note

`verify.py` and `make_bundle.py` were written for the development layout, in which the contract files sat in the same directory as the result files, with the scripts one level up. In this repository the pieces are separated into `code/`, `results/` and `contract/`. Adjust the two module-level path constants at the top of each script before running them here. This is a known limitation, listed in the top-level README.

## Re-running the pipeline from raw video

See `RUNBOOK.md`. You will need the six side-view and twelve frontal recordings, `config.py` pointed at an extracted-feature root, and the absolute paths in each script replaced. Expect the full chain — feature extraction, four domains, two comparators, four ablations and the specification analyses — to take roughly half an hour on the hardware used in the study, most of it in MediaPipe and DTW.
