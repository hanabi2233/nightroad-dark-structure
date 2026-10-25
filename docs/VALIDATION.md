# Validation performed during local release preparation

- Matched all 22 critical training-source size/SHA records from the current manuscript metadata before copying.
- Matched all 21 final checkpoint SHA values and all three initializer/three post-constructor RNG SHA values. The model archive has 27 binary assets, with archive CRC and member hashes verified.
- Retained all reported seeds and both favourable and unfavourable results. The result package's checks compare existing image-level records to already reported summaries and displayed table values; they do not create a new scientific analysis.
- Parsed and compiled all 53 Python source files without executing their bodies. All eight publication CLI help commands passed.
- Imported both source-package families and checked their configuration contracts without model construction, training, inference or dataset access.
- Performed a static review of the new training adapter against the recorded initialization, RNG, data scheduling, gradient-update, 8000-step and fixed-last logic. No numerical-path change was identified by static inspection. This is not an executed training-equivalence test.
- Ran the source and model integrity CLI, checked that protected original manuscript/state files remained unchanged, and scanned the public text payload for a bounded set of credential patterns. No matching credentials were found; this is not a guarantee against every possible sensitive string.

No training, inference, dataset-image generation, metric recomputation, server connection, GitHub repository creation or upload was performed. Full end-to-end reproducibility, dependency installation on a clean machine, and cross-platform bitwise equivalence have not been established by these packaging checks.

The full payload is enumerated in `SHA256SUMS.csv`. This checksum manifest excludes itself. After downloading or extracting the source, `python -B tools/verify_release.py` checks it along with the original-source mapping.
