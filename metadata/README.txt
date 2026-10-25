Online Resource 2: reproduction metadata

Article title: Dark-structure supervision and local-luminance gradient coordination for night-time road image enhancement
Journal: Signal, Image and Video Processing

This resource accompanies Online Resource 1, Sections S4, S6 and S8.
Read CSV identifiers and 64-bit seeds as strings or integers, preserving leading zeros.
The LOLv2 CSV files, degradation configuration/code/manifest, initializer identity JSON
and schedules are byte-identical copies of the preserved sources. Other CSVs are
documentation derivatives; their hashes are not the original experiment-manifest hashes.

GROUP_AWARE_SPLIT_MANIFEST.csv: authoritative sample and group membership (551/138).
DATA_MANIFEST.csv: ordered sample IDs, encoded image hashes, sizes and dimensions.
ACDC_SYNTHETIC_VAL_V1.csv: normal source path/hash, per-image seed and parameters.
ACDC_PANEL_MEMBERS.csv: normal reference, synthetic PNG and real-night correspondence.
The synthetic PNG hash is distinct from the normal file hash; synthetic images belong
to the frozen cache, not the original ACDC normal image at that source path.
minimal_degradation_v1.py/.yaml: unchanged generation implementation and configuration.
schedules/*.json: 8000 batches of eight ordered sample-ID strings for each seed.
initialization_identities/*.json: tensor identities, not tensor values or model weights.
INITIALIZERS.csv: serialized state, flattened parameter, CPU RNG and schedule identities.
TRAINING_INSTANCES.csv: all 21 reported final checkpoints and supporting record hashes.
TRAINING_SOURCES.csv: exact retained implementation paths and hashes.
EXTERNAL_ASSETS.csv: source/config/checkpoint identities, revisions and public source URLs.
GAMMA_BASELINE.json: fixed exponent, RGB/PNG conversion, source identities and
106 archived output hashes for the existing Gamma reference in Table 5.
SHA256SUMS.csv: hashes of all other metadata payloads; excludes itself.

Data images, synthetic derivatives, model weights and the complete training implementation
are not included. Hashes identify retained assets and are not download locations.
The full training-code and controlled-checkpoint release location remains an author decision.
Four classical external source commits were recovered from clean local repositories and
matched to execution-sealed source hashes. SCUF large-weight SHA values come from the final
archived verification; the present audit checked their presence and byte sizes. Individual
verification_basis fields distinguish this from a fresh direct SHA-256 calculation.
No new training, inference, metric evaluation, data split or degradation was executed here.
