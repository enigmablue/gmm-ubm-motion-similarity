# GMM-UBM motion signatures for cross-recording similarity

Analysis code for the manuscript:

> Lopez-Ozieblo, R., & Shen, D. J. *A GMM-UBM Framework for Cross-Recording Similarity of Conversational Head and Hand Motion* (under review).

Six conversations between one dyad, filmed from one side camera and two frontal cameras, are each summarised as a **motion signature**: a Gaussian Mixture Model adapted from a shared Universal Background Model. Recordings are compared by cross-fitted log-likelihood ratio, and four per-speaker, per-body-part rankings are merged into a consensus score.

> **Status: proof of concept, not a validated tool.**
> In the paper the cross-view association is positive (rho = .715, exact two-sided p = .0014), and the association varied but remained positive when single sessions were removed. Matched-session retrieval did not establish above-chance identification (3 of 12 speaker-level trials, 4 of 24 domain-level trials, against chance of 2/12 and 4/24). Reuse with that in mind.

## What is in this repository

`code/` only: the 27 analysis scripts that produce the numbers reported in the paper. Each script writes result JSON files to the output directory and prints the values it computed.

| Script group | What it produces |
|---|---|
| `pipeline.py`, `canonical.py`, `table_all_methods.py`, `review2_raw_scores.py` | The core run: frame features, UBM, MAP adaptation, cross-fitted log-likelihood ratio, per-domain and consensus rankings |
| `pipeline_retrieval.py`, `session_retrieval.py` | Matched-session retrieval trials |
| `pipeline_comparators.py`, `pipeline_comparator_a.py`, `pipeline_dtw_ablation.py` | Sequence-alignment comparators A and B |
| `aggregate_seven.py`, `pipeline_aggregate.py` | The seven-feature aggregate baseline |
| `pipeline_hand_ablation.py`, `pipeline_head_ablation.py`, `pipeline_units_tiebreak.py` | Representation and measurement-scale ablations |
| `pipeline_bh_family.py`, `pipeline_textstats.py`, `crossview_exact.py` | Exact permutation tests, leave-one-out range, overlap statistics |
| `pipeline_extras.py`, `review2_checks.py`, `review2_method_tests.py`, `gmm_pure_fullbody.py` | Frontal-view ranks, cross-checks, shared helpers |
| `cell_extractor.py`, `hand_two_channel.py`, `handview_2x2.py` | Shared helpers: domain fitting, per-hand channels, hand feature construction |
| `extract_*.py` | Frame-level head-pose and hand-landmark extraction |

## What is *not* here

**The video recordings.** They show two identifiable people who consented to research use, not to public distribution. No frames, landmarks or per-frame features are included.

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

## Citation

Archived at Zenodo: **https://doi.org/10.5281/zenodo.23688566**. That is the concept DOI, which always resolves to the newest version.

See [`CITATION.cff`](CITATION.cff). If you use the code, please cite the manuscript.

## Version history

**1.1.0** — added `cell_extractor.py`, `hand_two_channel.py` and `handview_2x2.py`, which are imported by `canonical.py`, `gmm_pure_fullbody.py`, `session_retrieval.py` and `table_all_methods.py` but were missing from 1.0.x. In 1.0.x the import chain failed at `canonical.py` and the analysis scripts could not run. All 27 modules now import cleanly. The README results were also corrected to the current analysis.

**1.0.1** — Zenodo DOI added.

**1.0.0** — first release.

## Licence

MIT — see [`LICENSE`](LICENSE). The licence does not extend to the original video recordings, which are not distributed here.
