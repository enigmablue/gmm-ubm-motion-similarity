# Defects found in the pipeline that produced the earlier tables

Recorded so that a reviewer can see what changed, why the reported numbers moved,
and how the diagnosis was made. Dates are October 2026.

The published tables were produced by code in `src/gmm/gmm_ubm_v5.py` and its
descendants. Three independent defects were found in that code path. Each is
listed with the evidence that established it.

---

## Defect 1 — derivatives computed across hand switches

**Where.** `build_hand_frames` in `gmm_ubm_v5.py`, and the copies of that function
in `hand_ablations.py` and `head_ablations.py`.

**What.** The loop keeps one `prev_wrist` for the whole recording. The file
contains rows for both hands interleaved by frame, and the loop never checks the
`handedness` column. When consecutive rows belong to different hands, the
"velocity" is the distance between a left-hand landmark and a right-hand landmark
divided by their frame gap.

**Evidence.** A neutral test on one recording (video 1, Stella) examined every
row-to-row transition:

| Transition kind | Count | Effect |
|---|---|---|
| Same hand, valid | 272 | correct |
| Cross-hand, zero time gap | 39 | velocity forced to 0 |
| Cross-hand, non-zero gap | 44 | spurious velocity between two different hands |

83 of 355 transitions (23%) were invalid. All 39 zero-gap cases had a non-zero
true same-hand velocity, with a median of 2.4 px/s that was silently discarded.
The 44 non-zero-gap cases injected displacements that no hand ever made.

**Consequence.** The hand features contained movement that did not happen, in
both the wrist-trajectory channels and the acceleration channel that follows from
them. Finger-configuration channels were unaffected because they are within-frame.

---

## Defect 2 — the wrong sampling rate

**Where.** The same function.

**What.** It reads `output/landmarks/{vid}_{person}_hand_landmarks.csv`. That
directory is written at stride 3, so it already contains every third frame. The
function then applies `if fi % SUBSAMPLE != 0: continue` with `SUBSAMPLE = 5`.

Keeping frames whose index is divisible by 5 out of a stream containing indices
divisible by 3 retains every 15th frame of the original video: about 2 Hz at
30 fps, not the 6 Hz the Method section states.

**Evidence.** Direct inspection of the `frame` column in both directories, plus
the frame counts: 1,907 and 1,690 hand rows in the old tables against 5,363 and
5,233 from the corrected stride-5 source, a ratio of about 2.8, which is the
expected 15/5.

**Consequence.** Every velocity and acceleration value was computed over a gap
seven times longer than intended, and the head-to-hand frame alignment assumed by
the shared `SUBSAMPLE` constant did not hold.

---

## Defect 3 — two tracks that were never merged

**Where.** The repository layout, not a single function.

**What.** Two independent lineages existed:

- a *extraction* track that wrote corrected stride-5 feature files
  (`extract_maincam_hands_std.py`, `extract_solo_head_std.py`), and
- a *method* track (`gmm_ubm` → `gmm_ubm_v5` → `hand_two_channel` → `canonical`)

The method track never consumed the extraction track's output. The extraction
scripts had existed since 12 September, but `gmm_ubm_v5.py` still read the older
stride-3 directory.

**Consequence.** Different tables in the same manuscript were computed from
different feature lineages. An ablation whose "full" row is supposed to reproduce
the headline did not reproduce it, which is the signal that surfaced the problem.

`pipeline.py` now merges the two tracks: it is the single entry point, reads the
stride-5 sources, and applies the within-hand rule.

---

## What the fix changed, and what it did not

**Survived.** The nominated pair still ranks first by consensus in the side view.
The framework is still the only method of the three that reaches nominal
significance on cross-view consistency. The translation-invariant 23D hand
encoding is still the weakest configuration.

**Did not survive.** The sentence claiming the nominated pair was "the only pair
to appear in the top four of every domain" is false in every corrected
construction; the pair's domain ranks are 6, 5, 3 and 2. In the hand ablation,
"wrist trajectory carried more of the ordering than finger configuration" is
reversed: finger-only reaches the same consensus as the full representation,
while wrist-only is lower.

---

## A note on how this was missed

Two process failures are worth recording, because both are easy to repeat.

**A defect can survive its own description.** The Method section already stated
that derivatives were computed within a hand and that gaps were skipped. The text
was correct; the code was not. Reading the Method and assuming the code matched it
would have missed all three defects.

**Agreement between two tables proves nothing when both share a builder.** The
ablation's combined-12D row did reproduce the headline, at 3/4/.857, and that
agreement was reassuring at the time. It was also worthless: both were produced by
the same faulty hand builder, so they agreed on the wrong features. The check that
actually surfaced the problem was reading the code path against the Method text
and against the frame indices in the files it reads. Agreement is only evidence
when the two things agreeing are computed independently.
