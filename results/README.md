# Existing result data for the SIVP manuscript

Manuscript: **Dark-structure supervision and local-luminance gradient coordination for night-time road image enhancement**.

This folder contains previously generated numerical results and transparent exports for the manuscript and Online Resource 1. Packaging performed no training, inference, image scoring, new data splitting or new scientific analysis. It includes unfavourable as well as favourable results and all reported seeds. These panels are development validation and descriptive real-night evaluation, not an independent formal test.

## Contents and paper mapping

| Folder/file | Contents | Paper location |
|---|---|---|
| `full_reference/primary_per_image.csv` | 3,660 image-level records: 5 primary methods x 3 seeds x (138 LOLv2-real + 106 synthetic ACDC images) | Main Table 2; Supplementary Tables S1, S2, S10; quantitative supplementary figure |
| `full_reference/primary_seed_level.csv` | 30 existing panel/method/seed means | Main Table 2; Tables S1, S2, S10 |
| `full_reference/primary_seed_summary.csv` | Existing across-seed mean/minimum/maximum; this file does not contain standard deviations | Main Table 2 context; use the seed-level rows for the reported sample SD |
| `full_reference/component_per_image.csv` | 2,196 records for complete [R,S,M], without S [R,M], and without M [R,S], all 3 seeds and both panels | Main Table 3; Tables S3, S4 |
| `full_reference/component_seed_level.csv`, `component_mean_sd.csv` | Existing per-seed and across-seed component summaries | Main Table 3; Tables S3, S4 |
| `full_reference/component_*_differences.csv` | Existing matched removal-minus-complete differences at image and seed levels | Table S4 |
| `full_reference/external_per_image.csv` | 700 records for precisely the external method-panel pairs in the paper: SCI/EnlightenGAN on LOLv2-real; RetinexNet/SCI/EnlightenGAN/Zero-DCE on synthetic ACDC | Main Table 4 |
| `real_night/quality_per_image.csv` | 1,590 NIQE/MUSIQ records: 15 evaluated instances, each on the same 106 real-night frames | Main Table 5; Tables S5, S7 |
| `real_night/quality_*summary.csv`, `quality_by_sequence.csv`, `quality_paired_vs_input.csv` | Existing instance/condition/sequence summaries and paired quality differences | Main Table 5; Tables S5, S7 |
| `real_night/quality_and_segmentation.csv` | Existing integrated quality and segmentation table; Gamma is absent because its quality scores were not evaluated | Main Table 5 |
| `real_night/segmentation_instances.csv`, `segmentation_mean_sd.csv`, `segmentation_paired.csv` | Controlled instances, pooled segmentation results and existing paired differences | Main Table 5; Tables S5-S7 |
| `real_night/external_segmentation_summary.csv` | Existing current-paper method rows, including Input and Gamma | Main Table 5 |
| `real_night/segmentation_by_group_region.csv` | Existing per-sequence/region summaries | Supplementary real-night interpretation |
| `real_night/segmentation_details/` | Existing per-class counts and per-image scores, plus external class results; no pixel label maps | Supports the pooled segmentation values |
| `reported_tables/main_table_1.csv` ... `main_table_5.csv` | Exact displayed cells exported from the current English manuscript, including its rounding | Main Tables 1-5 |
| `reported_tables/supplement_table_1.csv` ... `supplement_table_10.csv` | Exact displayed cells from Online Resource 1 | Tables S1-S10; S9 gives all six fixed visual-example frame IDs and order |
| `SOURCE_MANIFEST.csv` | Source identities, transformations, destination hashes and sizes | Provenance of this package |

The complete [R,S,M] rows appear in both the primary and component collections because they are the shared comparator. Do not count these duplicate records as additional training instances.

## Identifiers and units

- Read `sample_id`, `group_id`, hashes and seeds as strings; LOLv2 identifiers include leading zeros.
- `LOLv2_val138` and `ACDC_synthetic_val106` identify the fixed 138-pair and 106-frame full-reference panels. `sequence`/`group` use the existing ACDC sequence or pooled-group labels.
- Seeds are `20261004`, `20261005`, `20261006`. Empty seed identifies the untrained Input or an external fixed checkpoint where applicable.
- Method codes: `A` = RGB reconstruction; `SUM4` = direct summation; `B` = four-task PCGrad; `GTMEAN` = adapted GT-Mean; `MERGED_LUM_PCGRAD` = complete three-task [R,S,M]; `MERGED_DROP_S` = [R,M]; `MERGED_DROP_M` = [R,S].
- PSNR is in dB (higher is better); SSIM is dimensionless (higher is better); LPIPS is dimensionless (lower is better). The per-instance values average image-level scores. Reported sample standard deviations are across three training seeds, not across images.
- NIQE and MUSIQ retain their original metric scales: lower NIQE and higher MUSIQ are better. They are not interchangeable with restoration or segmentation quality.
- Suffix `_pct` denotes a percentage; `_pp` denotes percentage points. Unsuffixed `mIoU`, `aAcc`, and `IoU` in class/group JSON and CSV are fractions, not percentages.
- Segmentation mIoU is calculated from pooled valid-pixel class counts and then equally averaged over the 19 classes; it is not the mean of per-image mIoU. `aAcc` is total correct valid pixels divided by total valid pixels. `TP`, `FP`, `FN`, `GT_pixels`, `union`, `confusion` and `valid_pixels` are integer pixel counts. Class arrays follow the original Cityscapes 19-class order; scalar class records also carry `class_name`.
- `region=all` means all valid pixels. Existing `inv0` and `inv1` strata use the source protocol's mask labels and are retained for audit; their scores must not be substituted for the all-pixel headline result.
- `delta_vs_input`, `delta_input`, and paired-difference column labels give the original subtraction direction. Component differences are removal variant minus complete model. The Table S10 direction is complete model minus SUM4. Blank values indicate not applicable or unavailable; the displayed Gamma IQA dash means not evaluated, not zero.

## Source preservation and transformations

Files labelled `byte-identical copy` in `SOURCE_MANIFEST.csv` retain the original bytes. The current manuscript tables are string-preserving Markdown-to-CSV exports, so their rounded display values are not replacements for the full-precision numerical sources.

`primary_per_image.csv` consolidates the 30 recorded panel/seed/method sources into a common schema. Original CSV/JSON decimal tokens are retained; fields outside the displayed PSNR/SSIM/LPIPS comparisons are omitted. Its 90 panel/seed/method mean values were checked against the existing `primary_seed_level.csv` to an absolute tolerance of 1e-10. The completed independent restart for three-task seed 20261006 is the included source; the earlier interrupted attempt is not substituted.

`external_per_image.csv` retains only the method-panel rows used by Main Table 4 and removes workstation paths and operational timing/status fields. `quality_per_image.csv` removes only the source image path column. The external segmentation summary and class JSON files retain the current paper's methods; old unrelated methods are excluded. JSON serialization changes are explicitly marked, and all retained numerical values and arrays compare equal to their source objects.

The manifest can list several source rows for one consolidated destination. `source_project_relative` records provenance in the development archive; it is not a public download address. Source hashes identify records but do not substitute for access to omitted assets. No absolute workstation paths or credentials are intentionally included.

## Coverage limits

This result-data folder does not contain raw dataset images, synthetic-image caches, enhanced-image plates, segmentation label maps, model weights, training logs or server configuration. Fixed frame identities and ordering are present in Table S9 and the repository reproduction metadata. Visual figures require the original providers' data and the corresponding implementation/checkpoints; no image redistribution is implied here.

The supplied numerical files permit inspection of the reported scores and their aggregation. Packaging checks do not independently rerun or validate the underlying model predictions. Real-night segmentation remains below Input on average for the complete method, and the removal-of-luminance trade-offs are retained.
