# Data and model availability

The companion repository is publicly accessible at [https://github.com/hanabi2233/nightroad-dark-structure](https://github.com/hanabi2233/nightroad-dark-structure), with model assets attached to [v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0). The repository and release assets became public on 29 October 2026 and do not require a repository invitation. Access is separate from reuse licensing; see [third-party notices](../THIRD_PARTY_NOTICES.md).

| Material | Access route |
|---|---|
| LOLv2-real original images | Original provider: https://github.com/flyywh/CVPR-2020-Semi-Low-Light ; follow its access and use terms. |
| ACDC original images and labels | Original provider: https://acdc.vision.ee.ethz.ch/download ; follow its access and use terms. |
| LOLv2 membership and duplicate-reference grouping | `metadata/GROUP_AWARE_SPLIT_MANIFEST.csv` and `DATA_MANIFEST.csv`. |
| Synthetic ACDC generation | `metadata/minimal_degradation_v1.py`, YAML configuration, per-image parameters and original-image hashes. |
| Original synthetic PNG identities and real-night membership | `metadata/ACDC_PANEL_MEMBERS.csv`. |
| Recorded numerical results | `results/`, with coverage and provenance described in its README. |
| Original controlled model checkpoints and initializer states | [`SIVP_MODEL_ASSETS.zip`](https://github.com/hanabi2233/nightroad-dark-structure/releases/download/v0.1.0/SIVP_MODEL_ASSETS.zip) in public release v0.1.0; no repository invitation required. |
| External model weights/configuration identities | `metadata/EXTERNAL_ASSETS.csv`, with original-source URLs and verification scope. |

Original dataset images, synthetic-image caches and experimental image panels are not redistributed in this preparation. The generation command can reconstruct inputs from an authorized local copy; it checks the resulting encoded PNG hashes against the original cache and fails if they differ. This command was not run on dataset images as part of packaging.

Raw and derived image redistribution must follow the original datasets' terms. No new claim of permission to redistribute those images is made. The third-party weights are also not included in the model asset archive.

The links above provide access to the repository and model archive. SHA-256 values in the repository manifests and release checksum file identify the original files and can be used to verify downloaded copies.

The raw experiment's historical allocation program was not recovered. The provided frozen membership table defines the LOLv2 split; no reconstructed or newly generated split is substituted.
