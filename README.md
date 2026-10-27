# GMM-UBM motion signatures for cross-recording similarity

Analysis code for the manuscript:

> Lopez-Ozieblo, R., & Shen, D. J. *A GMM-UBM Framework for Cross-Recording Similarity of Conversational Head and Hand Motion* (under review).

Six conversations between one dyad, filmed from one side camera and two frontal cameras, are each summarised as a **motion signature**: a Gaussian Mixture Model adapted from a shared Universal Background Model. Recordings are compared by cross-fitted log-likelihood ratio, and four per-speaker, per-body-part rankings are merged into a consensus score.

> **Status: proof of concept, not a validated tool.**
> In the paper the cross-view association is positive and significant (rho = .547, exact p = .025) but does not survive removing a single session, and matched-session retrieval is exactly at chance (2 of 12 speaker-level trials). Reuse with that in mind.

## What is in this repository

`code/` only: the 24 analysis scripts that produce the numbers reported in the paper. Each script writes result JSON files to the output directory and prints the values it computed.

| Script group | What it produces |
|---|---|
| `pipeline.py`, `canonical.py`, `table_all_methods.py`, `review2_raw_scores.py` | The core run: frame features, UBM, MAP adaptation, cross-fitted log-likelihood ratio, per-domain and consensus rankings |
| `pipeline_retrieval.py`, `session_retrieval.py` | Matched-session retrieval trials |
| `pipeline_comparators.py`, `pipeline_comparator_a.py`, `pipeline_dtw_ablation.py` | Sequence-alignment comparators A and B |
| `aggregate_seven.py`, `pipeline_aggregate.py` | The seven-feature aggregate baseline |
| `pipeline_hand_ablation.py`, `pipeline_head_ablation.py`, `pipeline_units_tiebreak.py` | Representation and measurement-scale ablations |
| `pipeline_bh_family.py`, `pipeline_textstats.py`, `crossview_exact.py` | Exact permutation tests, leave-one-out range, overlap statistics |
| `pipeline_extras.py`, `review2_checks.py`, `review2_method_tests.py`, `gmm_pure_fullbody.py` | Frontal-view ranks, cross-checks, shared helpers |
| `extract_*.py` | Frame-level head-pose and hand-landmark extraction |

## What is *not* here

**The video recordings.** They show two named, identifiable people who consented to research use, not to public distribution. No frames, landmarks or per-frame features are included.

**The derived results.** The result JSON files that back each table are not distributed here. They are part of the data package, available **on request** under a data-sharing agreement.

**A packaged library.** These are analysis scripts written against one dataset on one workstation. They are published so the analysis can be read as implemented, not so they can be imported.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python check_paths.py            # reports which scripts still hardcode a local path
```

Verified against Python 3.11.9 with numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0 and fastdtw 0.3.4.

## Known limitations

1. **Absolute paths.** 17 of the 24 shipped scripts hardcode `D:\Soqqle\...` paths from the development machine (23 occurrences; run `python check_paths.py` for the list). `config.py` is provided as the intended replacement, but the scripts have **not** been refactored to use it. To run on another machine you must edit the path constant at the top of each script you need.
2. **No input data, and no result files.** With the recordings withheld, the scripts cannot be run from scratch by a third party, and nothing here can be recomputed. The code is published so the method can be read as implemented.
3. **Scripts, not a library.** Several are internal cross-checks written during review (`review2_*.py`, `pipeline_extras.py`). They import each other by module name, so they have to sit in one directory.
4. **Scope of the claim.** The dataset is one dyad, six sessions, 15 session pairs. Nothing in this repository establishes that the method generalises.

## The pipeline in one paragraph

Per domain (Stella-head, Aru-head, Stella-hand, Aru-hand) and per view: frames are subsampled every 5th frame (~6 Hz); head pose is 18-dimensional and hand kinematics 12-dimensional; features are scaled by median and interquartile range pooled over all six recordings; a Universal Background Model is fitted to the pooled frames, with the component count chosen by BIC and components under 1% occupancy rejected; each recording is adapted by MAP estimation, with the relevance factor chosen by a within-recording A/B self-consistency test that never compares recordings; recordings are scored by cross-fitted symmetric log-likelihood ratio; and the four domain rankings are converted to normalised ranks and averaged into a consensus score.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the code, please cite the manuscript.

## Licence

MIT — see [`LICENSE`](LICENSE). The licence does not extend to the original video recordings, which are not distributed here.
