# Manifest of reported numbers

Generated from `numbers.json` for unit choice `fw`. 115 entries.

Every value below is recomputed by `verify.py`. The `source` column names the result file written by the script named in the section heading.

## Table 2 — per-domain settings and ranks (side view)

Each domain's BIC-selected component count, self-consistency relevance factor, frame count, the nominated pair's raw likelihood-ratio score, and its rank among 15 pairs.

| id | reported in | value | source |
|---|---|---|---|
| `t2.Hand_Stella.rank` | Table 2, Hand_Stella, Rank column | 6 | `pipeline_fw.json` |
| `t2.Hand_Stella.K` | Table 2, Hand_Stella, K column | 4 | `pipeline_fw.json` |
| `t2.Hand_Stella.r` | Table 2, Hand_Stella, r column | 128 | `pipeline_fw.json` |
| `t2.Hand_Stella.rows` | Table 2, Hand_Stella, Frames column | 5363 | `pipeline_fw.json` |
| `t2.Hand_Stella.llr` | Table 2, Hand_Stella, (4,5) LLR column | -0.0802 | `pipeline_fw.json` |
| `t2.Hand_Aru.rank` | Table 2, Hand_Aru, Rank column | 5 | `pipeline_fw.json` |
| `t2.Hand_Aru.K` | Table 2, Hand_Aru, K column | 2 | `pipeline_fw.json` |
| `t2.Hand_Aru.r` | Table 2, Hand_Aru, r column | 32 | `pipeline_fw.json` |
| `t2.Hand_Aru.rows` | Table 2, Hand_Aru, Frames column | 5233 | `pipeline_fw.json` |
| `t2.Hand_Aru.llr` | Table 2, Hand_Aru, (4,5) LLR column | -0.207 | `pipeline_fw.json` |
| `t2.Head_Stella.rank` | Table 2, Head_Stella, Rank column | 3 | `pipeline_fw.json` |
| `t2.Head_Stella.K` | Table 2, Head_Stella, K column | 4 | `pipeline_fw.json` |
| `t2.Head_Stella.r` | Table 2, Head_Stella, r column | 64 | `pipeline_fw.json` |
| `t2.Head_Stella.llr` | Table 2, Head_Stella, (4,5) LLR column | -0.0341 | `pipeline_fw.json` |
| `t2.Head_Aru.rank` | Table 2, Head_Aru, Rank column | 2 | `pipeline_fw.json` |
| `t2.Head_Aru.K` | Table 2, Head_Aru, K column | 5 | `pipeline_fw.json` |
| `t2.Head_Aru.r` | Table 2, Head_Aru, r column | 128 | `pipeline_fw.json` |
| `t2.Head_Aru.llr` | Table 2, Head_Aru, (4,5) LLR column | 0.0873 | `pipeline_fw.json` |

## Table 3 — consensus ranking of all 15 pairs (side view)

C is the proportion of the 56 domain-to-domain comparisons a pair wins. Mean across the 15 pairs is exactly 0.5 by construction.

| id | reported in | value | source |
|---|---|---|---|
| `t3.(1,2).C` | Table 3, pair (1,2), C column | 0.554 | `pipeline_fw.json` |
| `t3.(1,3).C` | Table 3, pair (1,3), C column | 0.321 | `pipeline_fw.json` |
| `t3.(1,4).C` | Table 3, pair (1,4), C column | 0.232 | `pipeline_fw.json` |
| `t3.(1,5).C` | Table 3, pair (1,5), C column | 0.161 | `pipeline_fw.json` |
| `t3.(1,6).C` | Table 3, pair (1,6), C column | 0.286 | `pipeline_fw.json` |
| `t3.(2,3).C` | Table 3, pair (2,3), C column | 0.661 | `pipeline_fw.json` |
| `t3.(2,4).C` | Table 3, pair (2,4), C column | 0.5 | `pipeline_fw.json` |
| `t3.(2,5).C` | Table 3, pair (2,5), C column | 0.536 | `pipeline_fw.json` |
| `t3.(2,6).C` | Table 3, pair (2,6), C column | 0.482 | `pipeline_fw.json` |
| `t3.(3,4).C` | Table 3, pair (3,4), C column | 0.643 | `pipeline_fw.json` |
| `t3.(3,5).C` | Table 3, pair (3,5), C column | 0.589 | `pipeline_fw.json` |
| `t3.(3,6).C` | Table 3, pair (3,6), C column | 0.625 | `pipeline_fw.json` |
| `t3.(4,5).C` | Table 3, pair (4,5), C column | 0.786 | `pipeline_fw.json` |
| `t3.(4,6).C` | Table 3, pair (4,6), C column | 0.482 | `pipeline_fw.json` |
| `t3.(5,6).C` | Table 3, pair (5,6), C column | 0.643 | `pipeline_fw.json` |
| `t3.c45_rank` | Table 3, (4,5) consensus rank | 1 | `pipeline_fw.json` |
| `t3.c45_gap` | Results, gap from (4,5) to the next pair | 0.125 | `pipeline_fw.json` |

## Table 5 — cross-view consistency and matched-session retrieval

Spearman correlation across the 15 pairs with an exact session-label permutation p-value; retrieval is hits over trials, chance 2/12 and 4/24.

| id | reported in | value | source |
|---|---|---|---|
| `t5.fw.rho` | Table 5, framework cross-view rho | 0.547 | `pipeline_fw.json` |
| `t5.fw.p` | Table 5, framework exact p | 0.025 | `pipeline_fw.json` |
| `t5.fw.handaru_rho` | Results H1a, strongest domain rho | 0.836 | `pipeline_fw.json` |
| `t5.fw.speaker_hits` | Table 5, framework speaker retrieval | 2 | `pipeline_retrieval_fw.json` |
| `t5.fw.domain_hits` | Table 5, framework domain retrieval | 3 | `pipeline_retrieval_fw.json` |
| `t5.fw.speaker_p` | Table 5 caption, framework speaker p | 0.733 | `pipeline_retrieval_fw.json` |
| `t5.fw.domain_p` | Table 5 caption, framework domain p | 0.967 | `pipeline_retrieval_fw.json` |
| `t5.fw.speaker_mrr` | Results H2a, speaker MRR | 0.436 | `pipeline_retrieval_fw.json` |
| `t5.fw.domain_mrr` | Results H2a, domain MRR | 0.403 | `pipeline_retrieval_fw.json` |
| `t5.dtw.rho` | Table 5, sequence alignment rho | 0.342 | `pipeline_dtw_fw.json` |
| `t5.dtw.p` | Table 5, sequence alignment p | 0.433 | `pipeline_dtw_fw.json` |
| `t5.dtw.speaker_hits` | Table 5, DTW speaker retrieval | 3 | `pipeline_extras_fw.json` |
| `t5.dtw.domain_hits` | Table 5, DTW domain retrieval | 5 | `pipeline_extras_fw.json` |
| `t5.agg.rho` | Table 5, aggregate rho | -0.468 | `aggregate_reproduction.json` |
| `t5.agg.p` | Table 5, aggregate p | 0.164 | `aggregate_reproduction.json` |
| `t5.agg.speaker_hits` | Table 5, aggregate speaker retrieval | 3 | `pipeline_extras_fw.json` |
| `t5.agg.domain_hits` | Table 5, aggregate domain retrieval | 6 | `pipeline_extras_fw.json` |

## Table 6 — consensus rank of the nominated pair by method and view

Lower is better; chance is 8.0 of 15.

| id | reported in | value | source |
|---|---|---|---|
| `t6.fw.side` | Table 6, framework side view | 1 | `pipeline_fw.json` |
| `t6.fw.front` | Table 6, framework frontal view | 2 | `pipeline_fw.json` |
| `t6.dtw.side` | Table 6, sequence alignment side view | 13.5 | `pipeline_dtw_fw.json` |
| `t6.agg.side` | Table 6, aggregate side view | 1 | `aggregate_reproduction.json` |

## Table 7 — hand representation ablation

Head features held fixed, so rows differ only in the hand vector.

| id | reported in | value | source |
|---|---|---|---|
| `t7.23D wrist-relative.C45` | Table 7, 23D wrist-relative, C45 | 0.679 | `hand_ablation_fw.json` |
| `t7.23D wrist-relative.rank` | Table 7, 23D wrist-relative, consensus rank | 2 | `hand_ablation_fw.json` |
| `t7.8D wrist only.C45` | Table 7, 8D wrist only, C45 | 0.696 | `hand_ablation_fw.json` |
| `t7.8D wrist only.rank` | Table 7, 8D wrist only, consensus rank | 2 | `hand_ablation_fw.json` |
| `t7.4D finger only.C45` | Table 7, 4D finger only, C45 | 0.786 | `hand_ablation_fw.json` |
| `t7.4D finger only.rank` | Table 7, 4D finger only, consensus rank | 1 | `hand_ablation_fw.json` |
| `t7.12D combined.C45` | Table 7, 12D combined, C45 | 0.786 | `hand_ablation_fw.json` |
| `t7.12D combined.rank` | Table 7, 12D combined, consensus rank | 1 | `hand_ablation_fw.json` |

## Table 8 — head temporal-context ablation

Hand features held fixed at the corrected 12D vector. z measures how far the nominated pair's consensus sits from the other fourteen pairs.

| id | reported in | value | source |
|---|---|---|---|
| `t8.9D.C45` | Table 8, 9D, C45 | 0.679 | `head_ablation_fw.json` |
| `t8.9D.z` | Table 8, 9D, z | 1.31 | `head_ablation_fw.json` |
| `t8.9D+S.C45` | Table 8, 9D+S, C45 | 0.75 | `head_ablation_fw.json` |
| `t8.9D+S.z` | Table 8, 9D+S, z | 1.85 | `head_ablation_fw.json` |
| `t8.9D+E.C45` | Table 8, 9D+E, C45 | 0.75 | `head_ablation_fw.json` |
| `t8.9D+E.z` | Table 8, 9D+E, z | 1.83 | `head_ablation_fw.json` |
| `t8.9D+A.C45` | Table 8, 9D+A, C45 | 0.732 | `head_ablation_fw.json` |
| `t8.9D+A.z` | Table 8, 9D+A, z | 1.67 | `head_ablation_fw.json` |
| `t8.9D+E+S.C45` | Table 8, 9D+E+S, C45 | 0.804 | `head_ablation_fw.json` |
| `t8.9D+E+S.z` | Table 8, 9D+E+S, z | 2.3 | `head_ablation_fw.json` |
| `t8.9D+A+S.C45` | Table 8, 9D+A+S, C45 | 0.732 | `head_ablation_fw.json` |
| `t8.9D+A+S.z` | Table 8, 9D+A+S, z | 1.63 | `head_ablation_fw.json` |
| `t8.9D+A+E.C45` | Table 8, 9D+A+E, C45 | 0.768 | `head_ablation_fw.json` |
| `t8.9D+A+E.z` | Table 8, 9D+A+E, z | 2.03 | `head_ablation_fw.json` |
| `t8.18D.C45` | Table 8, 18D, C45 | 0.786 | `head_ablation_fw.json` |
| `t8.18D.z` | Table 8, 18D, z | 1.93 | `head_ablation_fw.json` |

## Results, H1a — the tested statistic and its decomposition

One hypothesis test: the consensus correlation. The head-only, hand-only and per-speaker values are descriptive decomposition and are neither adjusted nor claimed as tests.

| id | reported in | value | source |
|---|---|---|---|
| `h1a.consensus.rho` | Results H1a, tested consensus correlation, rho | 0.547 | `bh_family.json` |
| `h1a.consensus.p` | Results H1a, tested consensus correlation, exact p | 0.025 | `bh_family.json` |
| `h1a.head.rho` | Results H1a, head-only correlation (descriptive), rho | 0.189 | `bh_family.json` |
| `h1a.head.p` | Results H1a, head-only correlation (descriptive), exact p | 0.4875 | `bh_family.json` |
| `h1a.hand.rho` | Results H1a, hand-only correlation (descriptive), rho | 0.425 | `bh_family.json` |
| `h1a.hand.p` | Results H1a, hand-only correlation (descriptive), exact p | 0.113889 | `bh_family.json` |
| `h1a.Hand_Aru.rho` | Results H1a, Aru hands (descriptive), rho | 0.836 | `bh_family.json` |
| `h1a.Hand_Aru.p` | Results H1a, Aru hands (descriptive), exact p | 0.0111111 | `bh_family.json` |
| `h1a.Hand_Stella.rho` | Results H1a, Stella hands (descriptive), rho | -0.307 | `bh_family.json` |
| `h1a.Hand_Stella.p` | Results H1a, Stella hands (descriptive), exact p | 0.188889 | `bh_family.json` |
| `h1a.conc.consensus.agree` | Results H1a, comparisons agreeing, comparisons agreeing | 70 | `bh_family.json` |
| `h1a.conc.consensus.total` | Results H1a, comparisons agreeing, comparisons usable | 101 | `bh_family.json` |
| `h1a.conc.head_only.agree` | Results H1a, head-only comparisons agreeing, comparisons agreeing | 57 | `bh_family.json` |
| `h1a.conc.head_only.total` | Results H1a, head-only comparisons agreeing, comparisons usable | 98 | `bh_family.json` |
| `h1a.conc.hand_only.agree` | Results H1a, hand-only comparisons agreeing, comparisons agreeing | 65 | `bh_family.json` |
| `h1a.conc.hand_only.total` | Results H1a, hand-only comparisons agreeing, comparisons usable | 98 | `bh_family.json` |
| `h1a.loo_lo` | Results H1a, leave-one-out minimum | 0.347 | `pipeline_textstats_fw.json` |
| `h1a.loo_hi` | Results H1a, leave-one-out maximum | 0.697 | `pipeline_textstats_fw.json` |
| `h1a.loo_ns` | Results H1a, count not significant | 4 | `pipeline_textstats_fw.json` |

## Results, H1b — agreement between views

Overlap of the top of each ordering, and mean rank movement per pair.

| id | reported in | value | source |
|---|---|---|---|
| `h1b.top5` | Results H1b, top-five overlap | 2 | `pipeline_textstats_fw.json` |
| `h1b.top10` | Results H1b, top-ten overlap | 7 | `pipeline_textstats_fw.json` |
| `h1b.mean_shift` | Results H1b, mean rank shift | 3.47 | `pipeline_textstats_fw.json` |

## Sequence alignment with matched representations

DTW fed the framework's own feature vectors.

| id | reported in | value | source |
|---|---|---|---|
| `seq.9D.side` | Sequence alignment, 9D head, side consensus | 14 | `dtw_ablation_fw.json` |
| `seq.18D.side` | Sequence alignment, 18D head, side consensus | 13.5 | `dtw_ablation_fw.json` |
| `seq.4D.finger.aru` | Sequence alignment, 4D finger, Aru hand rank | 1 | `dtw_ablation_fw.json` |

## Frontal-view per-domain ranks

Rank of the nominated pair within each frontal domain.

| id | reported in | value | source |
|---|---|---|---|
| `front.Hand_Stella.rank` | Results, frontal Hand_Stella rank of (4,5) | 9 | `pipeline_extras_fw.json` |
| `front.Hand_Aru.rank` | Results, frontal Hand_Aru rank of (4,5) | 1 | `pipeline_extras_fw.json` |
| `front.Head_Stella.rank` | Results, frontal Head_Stella rank of (4,5) | 9 | `pipeline_extras_fw.json` |
| `front.Head_Aru.rank` | Results, frontal Head_Aru rank of (4,5) | 3 | `pipeline_extras_fw.json` |

## Comparator A — sequence alignment with literature features

15-dimensional wrist-relative hands and 3-dimensional head angles. Reported only as a specification variation, not in the main comparison.

| id | reported in | value | source |
|---|---|---|---|
| `compa.Hand_Stella` | Comparator A, side view, Hand_Stella rank of (4,5) | 10 | `comparator_a_fw.json` |
| `compa.Head_Stella` | Comparator A, side view, Head_Stella rank of (4,5) | 11 | `comparator_a_fw.json` |
| `compa.Hand_Aru` | Comparator A, side view, Hand_Aru rank of (4,5) | 6 | `comparator_a_fw.json` |
| `compa.Head_Aru` | Comparator A, side view, Head_Aru rank of (4,5) | 15 | `comparator_a_fw.json` |
| `compa.side.C` | Comparator A, side-view consensus score | 0.321 | `comparator_a_fw.json` |
| `compa.side.rank` | Comparator A, side-view consensus rank | 13 | `comparator_a_fw.json` |

## Not covered

See `README.md` §6. In short: the literature-feature comparator still reports its original numbers, the comparison-method retrieval p-values were withdrawn rather than recomputed, the session bootstrap is not numerically stable at six sessions, and the pixel-unit sensitivity is reported separately rather than folded into the manuscript.
