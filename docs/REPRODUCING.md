# Reproduction guide

These are reader-facing commands for a future reproduction. None of the dataset generation, training, inference or scoring commands below was executed during release preparation. Use separate environments for training, full-reference metrics, IQA and MMSegmentation as recorded in `environment/README.md` and the companion supplement.

## 1. Verify and obtain assets

Run `python -B tools/verify_release.py` from the repository root. Authorized repository users can obtain the authors' model archive from [release v0.1.0](https://github.com/hanabi2233/nightroad-dark-structure/releases/tag/v0.1.0). The repository is currently private. Extract it to, for example, `/work/model-assets/`, leaving `checkpoints/`, `initialization/` and `MODEL_ASSETS_MANIFEST.csv` together. Run:

```bash
python -B tools/verify_release.py --release-assets /work/model-assets
```

The model archive has all seven conditions at all three seeds, not a selected best seed. Original checkpoint identities are in `metadata/TRAINING_INSTANCES.csv`. The initialized states are untrained; they are not external pretrained models. Keep the CPU RNG state alongside each initializer.

## 2. Obtain and bind the original datasets

Download LOLv2-real and ACDC from their original providers under their terms, as linked in `DATA_AVAILABILITY.md`. For LOLv2 use the 689 pairs in `Real_captured/Train/Low` and `Real_captured/Train/Normal`. The published grouping/membership table defines train551 and validation138; do not resplit or use the official test to choose a checkpoint.

```bash
python -B tools/prepare_lolv2_bindings.py \
  --low-root /data/LOL-v2/Real_captured/Train/Low \
  --normal-root /data/LOL-v2/Real_captured/Train/Normal \
  --output /work/lolv2-bindings
```

Every input/reference encoded file hash is checked before creating bindings. IDs retain leading zeros.

The ACDC root must contain the `rgb_anon/` tree. Rebuild the recorded synthetic validation panel from its normal correspondences:

```bash
python -B tools/generate_acdc_panel.py --acdc-root /data/ACDC --output /work/acdc-synthetic106
```

The generator uses the original function and recorded parameters/seeds, then compares each generated PNG hash with the paper cache. It reports failure rather than silently treating a different encoding/environment as the original input. This wrapper has not been exercised against the 106 original images in this packaging task.

Create inference lists:

```bash
python -B tools/make_input_lists.py \
  --lolv2-validation-binding /work/lolv2-bindings/validation.json \
  --acdc-root /data/ACDC --synthetic-root /work/acdc-synthetic106 \
  --output /work/input-lists
```

It creates distinct LOLv2 validation, ACDC synthetic, and ACDC real-night lists. Real-night inputs have no pixel-aligned restoration ground truth in this study. Do not use their normal correspondences to compute restoration PSNR/SSIM/LPIPS.

## 3. Inference using an original paper checkpoint

```bash
python -B tools/infer_reported.py \
  --seed 20261005 --variant MERGED_LUM_PCGRAD \
  --checkpoint /work/model-assets/checkpoints/20261005/MERGED_LUM_PCGRAD/final_iter8000.pth \
  --inputs /work/input-lists/lolv2_validation138.csv \
  --output /work/merged-20261005-lolv2
```

Use another explicit input list for synthetic roads or real-night inputs. The original checkpoint hash and all input hashes are checked before inference. Prediction uses native resolution, right/bottom reflect padding to multiples of four, FP32, cropping, clipping and `numpy.rint` conversion to RGB PNG. The output folder must not already exist.

The inference command accepts only the 21 registered original paper checkpoint hashes. A new reproduction checkpoint is a different asset and needs its own declared identity; this preparation does not substitute new outputs for original results.

## 4. Full-reference scoring

Obtain the original torchvision AlexNet backbone (`alexnet-owt-7be5be79.pth`) from its provider and the lpips 0.1.4 package. Required SHA-256 values are checked by the archived metric function:

- AlexNet backbone: `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
- LPIPS Alex linear weights, `weights/v0.1/alex.pth`: `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`.

The backbone filename must retain its original name in a `checkpoints/` directory, e.g. `/work/torch-hub/checkpoints/alexnet-owt-7be5be79.pth`, because the original metric loader binds its parent cache directory.

```bash
python -B tools/make_scoring_pairs.py \
  --inputs /work/input-lists/lolv2_validation138.csv \
  --prediction-root /work/merged-20261005-lolv2 \
  --output /work/lolv2-pairs.csv

python -B tools/score_full_reference.py --pairs /work/lolv2-pairs.csv \
  --alexnet-backbone /work/torch-hub/checkpoints/alexnet-owt-7be5be79.pth \
  --output /work/lolv2-scores
```

PSNR/SSIM/LPIPS are computed on redecoded saved RGB PNGs. SSIM is scikit-image, win_size 7, channel_axis 2, data_range 1, non-Gaussian, population covariance. LPIPS uses AlexNet v0.1 and inputs mapped to [-1,1]. The diagnostic function is also retained exactly. Do not mix these values with historical BasicSR SSIM contracts.

## 5. Reproduce training

Training uses only LOLv2 train551. The synthetic road normal correspondences are evaluation data for the reported controlled models. The saved state and RNG are loaded; changing only a seed in a fresh constructor is not equivalent.

```bash
python -B tools/train_reported.py \
  --seed 20261005 --variant MERGED_LUM_PCGRAD \
  --binding /work/lolv2-bindings/train.json \
  --initialization-root /work/model-assets/initialization \
  --output /work/reproduction-20261005-merged
```

This prints a plan only. Add `--execute` to deliberately start the complete 8000-step run. The command exposes no steps, loss-weight, crop, batch, LR, best-checkpoint or validation-selection override. Use the recorded training environment for the closest reproduction; another GPU/software stack is not asserted to be bitwise equivalent.

The new adapter calls the archived gradient/optimizer functions directly and preserves the shared per-step data RNG. It writes a fresh final checkpoint, trace steps 500/501/8000, configuration and run environment. It does not connect to the original server or resume any old run. Stop on a failure and retain the failed output rather than changing a parameter and calling it the same experiment.

## 6. Real-night quality and segmentation

The paper's numerical outcomes are in `results/`. The original real-night IQA and segmenter adapters are included as **reference source** under `src/reference_evaluation/`; their historical environment/path/authorization constraints are not a portable one-command benchmark. The real-night pipeline was not independently re-executed during packaging.

Use pyiqa 0.1.16 with the recorded NIQE statistics and MUSIQ-KonIQ10k weights. Both segmenters use the MMSegmentation revision and exact config/checkpoint identities in `metadata/EXTERNAL_ASSETS.csv`; follow original provider instructions to acquire third-party assets. The companion supplement describes image preprocessing, interpolation, crops/stride, channel order, normalization, valid-pixel pooling and the fixed Gamma baseline.

Do not run full-reference restoration metrics on real-night normal correspondences. Quality and segmentation scores must retain all 106 fixed night frames and the same method/seed coverage as the paper. Original external methods have different training sources and budgets; they are not equal-budget retrainings of the controlled method.
