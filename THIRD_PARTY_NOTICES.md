# Third-party notices

This local preparation does not assign a blanket licence to the whole repository. The authors' intended licence for new contributions and controlled model weights remains to be confirmed before public release.

## Retinexformer

Retained architecture and related upstream source: https://github.com/caiyuanhao1998/Retinexformer . The retained project version is identified in the provenance records. Copyright (c) 2023 Yuanhao Cai. Its MIT licence is copied verbatim to `licenses/Retinexformer-MIT.txt` and remains with the archived vendor files.

## BasicSR-derived utilities

The Retinexformer fork includes utilities originating in BasicSR (https://github.com/xinntao/BasicSR), including transforms, image utilities and the restoration-model augmentation implementation. They are not wholly original work by the present study's authors.

The retained fork's `setup.py` identifies the BasicSR project and Apache License 2.0 (`url` and `license` metadata). The Apache 2.0 text is included separately as `licenses/Apache-2.0.txt`, obtained from https://www.apache.org/licenses/LICENSE-2.0.txt . Retain the source's own notices. The Retinexformer root MIT notice does not erase the provenance of inherited BasicSR components.

## Adapted GT-Mean

The archived `candidate_core.py` documents an independently expressed mathematical adaptation based on Liao et al., *GT-Mean loss: A simple yet effective solution for brightness mismatch in low-light image enhancement*, ICCV 2025. It records the consulted upstream commit. The upstream GPL-3.0 implementation is not copied into this file, according to the retained source record; the adaptation is identified separately from the original authors' complete method/training setup.

## Evaluation libraries and pretrained weights

PyTorch, torchvision, NumPy, SciPy, Pillow, OpenCV, einops, scikit-image, LPIPS, pyiqa and MMSegmentation retain their own licences. They are dependencies, not bundled here as the authors' original work. External enhancement methods and segmenters are obtained from their providers; source revisions, configurations and known weight identities are listed in `metadata/EXTERNAL_ASSETS.csv`.

The model archive contains only the study's 21 controlled final checkpoints and its initializer/RNG states. It does not redistribute third-party pretrained segmenters, IQA models, raw datasets, or original dataset labels.
