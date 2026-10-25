# Data and model access plan

This private companion repository is [https://github.com/hanabi2233/nightroad-dark-structure](https://github.com/hanabi2233/nightroad-dark-structure); model assets are attached to [v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0). Access requires repository permission. Public release and the authors' reuse terms remain pending; private upload does not establish public data or code availability.

| Material | Access route |
|---|---|
| LOLv2-real original images | Original provider: https://github.com/flyywh/CVPR-2020-Semi-Low-Light ; follow its access and use terms. |
| ACDC original images and labels | Original provider: https://acdc.vision.ee.ethz.ch/download ; follow its access and use terms. |
| LOLv2 membership and duplicate-reference grouping | `metadata/GROUP_AWARE_SPLIT_MANIFEST.csv` and `DATA_MANIFEST.csv`. |
| Synthetic ACDC generation | `metadata/minimal_degradation_v1.py`, YAML configuration, per-image parameters and original-image hashes. |
| Original synthetic PNG identities and real-night membership | `metadata/ACDC_PANEL_MEMBERS.csv`. |
| Recorded numerical results | `results/`, with coverage and provenance described in its README. |
| Original controlled model checkpoints and initializer states | [`SIVP_MODEL_ASSETS.zip`](https://github.com/hanabi2233/nightroad-dark-structure/releases/download/v0.1.0/SIVP_MODEL_ASSETS.zip) in private release v0.1.0; repository permission required. |
| External model weights/configuration identities | `metadata/EXTERNAL_ASSETS.csv`, with original-source URLs and verification scope. |

Original dataset images, synthetic-image caches and experimental image panels are not redistributed in this preparation. The generation command can reconstruct inputs from an authorized local copy; it checks the resulting encoded PNG hashes against the original cache and fails if they differ. This command was not run on dataset images as part of packaging.

Raw and derived image redistribution must follow the original datasets' terms. No new claim of permission to redistribute those images is made. The third-party weights are also not included in the model asset archive.

SHA-256 values identify files; they do not create an access route. Until the model archive and repository are actually accessible, the manuscript must not claim that those materials have already been publicly released.

The raw experiment's historical allocation program was not recovered. The provided frozen membership table defines the LOLv2 split; no reconstructed or newly generated split is substituted.
