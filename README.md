# Dark-structure supervision and local-luminance gradient coordination

Companion code and recorded results for **Dark-structure supervision and local-luminance gradient coordination for night-time road image enhancement**.

**Private manuscript-companion release v0.1.0, prepared 25 October 2026.** Repository: [https://github.com/hanabi2233/nightroad-dark-structure](https://github.com/hanabi2233/nightroad-dark-structure). Model assets: [release v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0). Access currently requires repository permission; these links do not establish public availability. Author metadata, publication citation and the authors' software/model reuse terms remain to be confirmed. The manuscript is not claimed to have been accepted.

The method retains the Retinexformer inference architecture and changes the training objectives and their gradient organization. The complete method coordinates RGB reconstruction (R), input-guided dark-structure supervision (S), and local-luminance supervision (M). The release covers seven reported conditions at three seeds, including the two component deletions.

The controlled results use LOLv2-real development validation and synthetic ACDC validation. Real-night segmentation averages remain below the unenhanced input; removing the luminance task improves several full-reference metrics. The complete results and limitations are retained. These panels are not an untouched independent formal test, and the study does not establish universal real-night improvement or state-of-the-art superiority.

## Contents

| Directory | Contents |
|---|---|
| `src/frozen/` | Byte-identical scientific source files and original training entrypoints retained for provenance. |
| `tools/` | New publication packaging commands for integrity checks, path binding, training, inference and full-reference scoring. |
| `metadata/` | Original Online Resource 2 membership, parameter, schedule and identity records. |
| `results/` | Recorded numerical results and their table/figure coverage map. |
| `environment/` | Recorded software versions; training and evaluation used different environments. |
| `docs/` | Reproduction instructions, source provenance, data and licensing notes. |
| `licenses/` | Third-party licence texts, with attribution in `THIRD_PARTY_NOTICES.md`. |

Original datasets and third-party pretrained models are obtained from their providers. The separate **SIVP_MODEL_ASSETS.zip** contains 21 original final checkpoints, three sealed initial states, and three post-constructor CPU RNG states. Authorized repository users can obtain it from [release v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0) ([model archive](https://github.com/hanabi2233/nightroad-dark-structure/releases/download/v0.1.0/SIVP_MODEL_ASSETS.zip)).

## First check

This standard-library-only command performs no training, inference, network download or dataset-image access:

```bash
python -B tools/verify_release.py
```

After extracting the model archive into an external folder, verify it with:

```bash
python -B tools/verify_release.py --release-assets /path/to/extracted/model-assets
```

See [reproduction instructions](docs/REPRODUCING.md) before running any experiment. The original frozen entrypoints include historical execution guards and paths; use the documented publication commands. The new adapters preserve the scientific function bodies but have **not been validated by repeating the 8000-step training or dataset inference** during this packaging task. Integrity, syntax, command interfaces and non-training checks are distinct from independent experimental reproduction.

## Conditions

| CLI identifier | Paper condition |
|---|---|
| `A` | RGB reconstruction |
| `SUM4` | Direct sum of the four signed objective terms |
| `B` | Four-task PCGrad |
| `GTMEAN` | Adapted GT-Mean reference |
| `MERGED_LUM_PCGRAD` | Complete three-task method [R,S,M] |
| `MERGED_DROP_S` | Without dark-structure task [R,M] |
| `MERGED_DROP_M` | Without local-luminance task [R,S] |

Seeds: `20261004`, `20261005`, `20261006`. All original controlled checkpoints are fixed at step 8000; no best-seed or intermediate-checkpoint selection is introduced here.

## Rights and citation

See [third-party notices](THIRD_PARTY_NOTICES.md) and [data availability](docs/DATA_AVAILABILITY.md). Third-party licence texts do not automatically license the authors' new code, model weights, or datasets. The authors' intended reuse terms and final citation are pending; there is no invented DOI, author list or accepted-paper citation in this local preparation.
