# GMM-UBM motion signatures for cross-recording similarity

Analysis code, derived results and reproduction contract for the manuscript:

> Lopez-Ozieblo, R., & Shen, D. J. *A GMM-UBM Framework for Cross-Recording Similarity of Conversational Head and Hand Motion* (under review).

Six conversations between one dyad, filmed from one side camera and two frontal cameras, are each summarised as a **motion signature**: a Gaussian Mixture Model adapted from a shared Universal Background Model. Recordings are compared by cross-fitted log-likelihood ratio, and four per-speaker, per-body-part rankings are merged into a consensus score.

> **Status: proof of concept, not a validated tool.**
> In the paper, the cross-view association is positive and nominally significant (rho = .547, p = .025) but does not survive removing a single session, and matched-session retrieval is exactly at chance (2 of 12 speaker-level trials). Read [`docs/DEFECTS.md`](docs/DEFECTS.md) before reusing anything here.

## What is in this repository

| Path | Contents |
|---|---|
| `code/` | The analysis scripts that produce every reported number (27 files) |
| `results/` | Derived result JSON files (39 files) — no video, no landmarks |
| `contract/` | `numbers.json` (every number stated in the paper) and `provenance.json` (SHA-256 of each script) |
| `docs/DEFECTS.md` | Three defects found in earlier versions of the pipeline, and what they changed |
| `docs/MANIFEST.md` | What each result file contains, grouped by research question |
| `docs/RUNBOOK.md` | How to re-run the pipeline end to end |
| `docs/REPRODUCING.md` | How to check the numbers in this repository |
| `docs/DATA_AVAILABILITY.md` | What data can and cannot be shared, and why |

## What is *not* here

**The video recordings.** They show two named, identifiable people who consented to research use, not to public distribution. No frames, landmarks, or per-frame features are included. See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

**A packaged library.** These are analysis scripts written against one dataset on one workstation. They are published so the analysis can be inspected and re-run, not so they can be imported.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python check_paths.py            # reports which scripts still hardcode a local path
```

Verified against Python 3.11.9 with numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0 and fastdtw 0.3.4.

## Known limitations

1. **Absolute paths.** 17 of the 27 shipped scripts hardcode `D:\Soqqle\...` paths from the development machine (23 occurrences; run `python check_paths.py` for the list). `config.py` is provided as the intended replacement, but the scripts have **not** been refactored to use it. To run on another machine you must edit the path constant at the top of each script you need.
2. **No input data.** With the recordings withheld, the scripts cannot be run from scratch by a third party. What *can* be checked independently is the arithmetic: `contract/numbers.json` maps every number in the paper to the result file and code path that produces it.
3. **`code/verify.py` and `code/make_bundle.py`** expect the original working-directory layout (`contract` files alongside `results`). Adjust the module-level path constants before running them here.
4. **Scope of the claim.** The dataset is one dyad, six sessions, 15 session pairs. Nothing in this repository establishes that the method generalises.

## The pipeline in one paragraph

Per domain (Stella-head, Aru-head, Stella-hand, Aru-hand) and per view: frames are subsampled every 5th frame (~6 Hz); head pose is 18-dimensional and hand kinematics 12-dimensional; features are scaled by median and interquartile range pooled over all six recordings; a Universal Background Model is fitted to the pooled frames, with the component count chosen by BIC and components under 1% occupancy rejected; each recording is adapted by MAP estimation, with the relevance factor chosen by a within-recording A/B self-consistency test that never compares recordings; recordings are scored by cross-fitted symmetric log-likelihood ratio; and the four domain rankings are converted to normalised ranks and averaged into a consensus score.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the code, please cite the manuscript; the version of record for the code will be minted at first release.

## Licence

MIT for the code and derived results — see [`LICENSE`](LICENSE). The licence does not extend to the original video recordings, which are not distributed here.
