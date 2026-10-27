# What this archive is

A read-only snapshot of the review bundle for the FacialCapture manuscript, taken
2026-10-27.

```
review_bundle/
├── README.md            what is claimed, what changed, what is not covered
├── RUNBOOK.md           ordered commands, timings, failure modes
├── MANIFEST.md          every reported number, generated from numbers.json
├── DEFECTS.md           the three pipeline defects and how each was established
├── SNAPSHOT.md          this file
├── numbers.json         the review contract: 109 reported values
├── provenance.json      hashes of every script and input file
├── make_bundle.py       regenerates numbers.json, provenance.json, MANIFEST.md
├── verify.py            compares the contract against results; exit 0 or 1
├── code/                24 scripts, read-only copies at the reviewed hashes
└── results/             result JSON files, read-only copies
```

## Which parts run, and which are only for reading

`verify.py` resolves paths relative to the live workspace tree
(`FacialCapture/paper_review/bundle/`), so it will not run from inside this
archive. To reproduce, extract over the original workspace, or copy `code/` into
`FacialCapture/paper_review/` and `results/` into
`FacialCapture/output/paper_review/final/`, then:

```powershell
cd d:\Soqqle\Unity_Animation\FacialCapture
.\.venv\Scripts\python.exe paper_review\bundle\verify.py --rerun
```

Expected: `109 of 109 reported numbers reproduce`. Last measured run: 4.9 minutes.

The `code/` copy exists so a reviewer can read the analysis without repository
access, and so the hash in `provenance.json` can be checked against the file
actually present:

```powershell
Get-FileHash code\pipeline.py -Algorithm SHA256
```

`provenance.json` records the first 16 hex characters of each SHA-256.

## What is absent

The input feature CSVs are not included: about 54 MB across
`output/maincam_hands_std`, `output/fullbody_std`, `output/maincam`,
`output/solo_head_std`, `output/stella` and `output/aru`. Their per-file hashes
are recorded in `provenance.json`, so a reviewer can confirm a local copy matches.
Without those inputs, `verify.py` in fast mode still works, because it reads only
`results/`. `verify.py --rerun` needs the inputs.

The source videos are not included and are not expected to be shared: they show
two identifiable people. Obtaining consent to redistribute them is an author
decision.

## The one-line summary

The manuscript's headline survives a rebuild of the analysis: the nominated pair
ranks first of 15 by consensus. A supporting claim that it was "the only pair in
the top four of every domain" does not, and one ablation sentence reverses
direction. `DEFECTS.md` explains why the numbers moved; `README.md` §6 lists what
is still open.
