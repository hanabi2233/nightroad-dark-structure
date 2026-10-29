# Dark-structure supervision and local-luminance gradient coordination

Companion code and recorded results for **Dark-structure supervision and local-luminance gradient coordination for night-time road image enhancement**.

**Public manuscript-companion repository.** Release v0.1.0 was prepared on 25 October 2026; the repository and its release assets became publicly accessible on 29 October 2026. Repository: [https://github.com/hanabi2233/nightroad-dark-structure](https://github.com/hanabi2233/nightroad-dark-structure). Model assets: [release v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0). No repository invitation is required to access these materials.

**Authors:** Jie Xu, Lanxiang Wei, Zhifeng Wang.

**Affiliation:** School of Intelligent Manufacturing and Control Engineering, Shanghai Polytechnic University, Shanghai, China.

**Corresponding author:** Lanxiang Wei, [20251513219@sspu.edu.cn](mailto:20251513219@sspu.edu.cn).

**Manuscript status:** Submitted to *Signal, Image and Video Processing* on 28 October 2026; not yet accepted or published.

The v0.1.0 tag and downloadable archives retain the original 25 October snapshot. This README and the linked release notes provide the updated access and author information; see [provenance](docs/PROVENANCE.md).

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

Original datasets and third-party pretrained models are obtained from their providers. The separate **SIVP_MODEL_ASSETS.zip** contains 21 original final checkpoints, three sealed initial states, and three post-constructor CPU RNG states. It is publicly downloadable from [release v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0) ([model archive](https://github.com/hanabi2233/nightroad-dark-structure/releases/download/v0.1.0/SIVP_MODEL_ASSETS.zip)).

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

See [third-party notices](THIRD_PARTY_NOTICES.md) and [data availability](docs/DATA_AVAILABILITY.md). Third-party components retain their existing licences. No separate reuse licence is currently specified for the authors' original contributions and controlled model weights. For reuse permission, contact [Lanxiang Wei](mailto:20251513219@sspu.edu.cn). The manuscript title and author order are listed above; a publication citation and DOI are not yet available.
