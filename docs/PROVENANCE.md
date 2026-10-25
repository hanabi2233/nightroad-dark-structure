# Source and packaging provenance

The source manuscript package is `SIVP_DECLARATION_EDIT_20261025`. No manuscript, original training file, dataset, checkpoint, canonical project state or original result was edited to make this release.

`SOURCE_MANIFEST.csv` maps copied files to their project-relative source names, sizes and SHA-256 values. All 22 source identities listed in the manuscript's `metadata/TRAINING_SOURCES.csv` were matched before copying. Additional imports required by those files, including `img_util.py`, are recorded separately in the same source manifest.

The `src/frozen/confirmation` family corresponds to seed-20261004 A/SUM4/B. The `src/frozen/followup` family supplies the later primary conditions; `merged` and `components` retain their exact separate objective/update implementations. Historical source files may contain extra branches or stale execution guards from research development. Only the seven conditions listed in the release README are exposed by the new training command.

The `tools/` files were written for local release preparation on 25 October 2026. They provide relative-path discovery, explicit dataset bindings, original checkpoint checks, and fresh output directories. `train_reported.py` calls the existing scientific functions for data scheduling, augmentation, optimizer updates, objectives and PCGrad. It restores both the sealed initializer and post-constructor CPU RNG. It keeps 8000 updates, 500-step warmup, batch 8, crop 256, FP32, the recorded learning-rate formula and a fixed final checkpoint.

The new runner does not reuse expired server authorizations, fixed calendar deadlines, machine hostname restrictions, remote login tools or an original run's output directory. These publication entrypoints are derivative orchestration, not the original formal execution source and not evidence of a new successful experiment. A reproduction creates its own outputs; their hashes must not be relabelled as original paper identities.

Full-reference inference, scoring and diagnostics compile selected exact function definitions from the archived evaluation implementation. The LPIPS backbone path is supplied explicitly instead of using the original author's cache path. Original files stay byte-identical, including historical path strings needed to retain their source identity.

Metadata copied from Online Resource 2 remains an exact copy, including its original README. That README describes the scope of the metadata attachment, which does not itself contain model binaries. Model binaries for the proposed separate release are recorded in `MODEL_ASSETS_MANIFEST.csv`. The original manuscript and supplementary availability statements have not yet been rewritten to claim a public release.

Checks in this preparation establish file identity and packaging consistency. They do not establish cross-platform bitwise training equivalence or independently rerun all experiments.
