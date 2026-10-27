# Data availability

## Summary

| Item | Available? | Where |
|---|---|---|
| Analysis code | Yes | `code/` (MIT licence) |
| Derived result files | Yes | `results/` — JSON, numeric only |
| Reproduction contract | Yes | `contract/numbers.json`, `contract/provenance.json` |
| Extracted landmarks and per-frame features | On request, subject to agreement | not included |
| Original video recordings | **No** | not included, not distributable |

## Why the recordings are not shared

The six recordings show two named individuals, filmed front-on and in profile, over roughly 2.5 minutes each. Faces are clearly visible and both speakers are identifiable from the footage. Consent covered participation in the study; it did not cover public release of video, and de-identification is not possible for video of this kind without destroying the kinematic signal the study exists to measure.

Landmark and feature files are therefore also withheld: they encode facial geometry and body position frame by frame, and for a small, known dyad they are close to re-identifying.

## What can be requested

Researchers who need the intermediate data to replicate or extend the analysis may request:

- MediaPipe Hand landmark files and YuNet face-detection outputs, per recording;
- the 18-dimensional head and 12-dimensional hand feature tables;
- the aggregate seven-feature tables per speaker.

Requests go to the corresponding author and are handled under an institutional data-sharing agreement. Please state the intended use.

## What can be checked without any data

`contract/numbers.json` lists every statistic stated in the manuscript together with the result file and code path that produce it, and `code/verify.py` recomputes each one. Someone with no access to the recordings can therefore confirm that the reported values follow from the published code and published result files — though not that the result files follow from the raw video, which is the limit of what an analysis-only release can offer.

## Statement for the manuscript

The following wording is suggested for the article's data availability statement; adjust before submission.

> The video recordings analysed in this study cannot be shared: they show two identifiable individuals who consented to participation, not to public release of footage. Landmark and kinematic feature files are available from the corresponding author on reasonable request under an institutional data-sharing agreement. All analysis code, derived result files and the mapping from each reported statistic to the code that produces it are openly available at \<repository URL\>.
