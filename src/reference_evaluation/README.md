# Archived evaluation implementations

These are byte-identical records of the original evaluation implementation and adapters. They retain historical directory, environment, authorization and deadline bindings. They are not the new portable command interface and should not be invoked blindly after moving the files.

- `run_fixed_iqa.py`: original fixed NIQE/MUSIQ computation and metric-loading configuration.
- `external_iqa_reference.py`: extension that reuses those metric definitions for external methods.
- `run_segment.py` and `component_night_reference.py`: retained orchestration for merged/component real-night results.
- `segmenter.py`, `segmenter_deeplab.py`, `nightroad_core.py`: fixed segmentation adapters and count/metric helper definitions.
- `road_fixed_infer.py`: original controlled ACDC PNG inference rule.

The two segmenter adapter SHA-256 values match the current supplementary metadata. MMSegmentation's complete inherited configuration tree and pretrained weights must be obtained at the specified revision from the upstream project; copying only a small top-level config would be incomplete.

Current numerical evidence is supplied in `results/`. Portability of the entire IQA/MMSegmentation pipeline and independent execution are not claimed by this packaging task. All copied paths and hashes are listed in `SOURCE_MANIFEST.csv`.
