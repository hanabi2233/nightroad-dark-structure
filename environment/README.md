# Recorded environments

`confirmation_training_reference.json` and `followup_training_reference.json` retain the recorded training software version fields; hostnames, interpreter installation paths, GPU process listings and session timestamps are omitted. The original training used Python 3.12.3, PyTorch 2.8.0+cu128, NumPy 2.3.2, CUDA 12.8 and an RTX 4090.

The saved zero-step model and post-constructor CPU RNG were originally prepared under Python 3.11.9 and PyTorch 2.11.0.dev20260120+cu128. These files are provided rather than regenerated under the training environment.

The original full-reference evaluation recorded Python 3.11.9, PyTorch 2.11.0.dev20260120+cu128, torchvision 0.25.0.dev20260120+cu128, NumPy 2.3.4, Pillow 12.0.0, scikit-image 0.25.2 and lpips 0.1.4. Real-night quality scoring uses pyiqa 0.1.16. Fixed segmentation uses MMSegmentation 1.2.2 at commit c685fe6767c4cadf6b051983ca6208f1b9d1ccb8; native dependencies and exact model/config identities are documented separately.

`requirements-core.txt` lists distribution names, not a tested replacement for these recorded environments. It intentionally does not turn module-version strings into invented package/build identifiers. Exact historical wheels and CUDA compatibility may require an archived environment. No dependency installation or new environment reproduction was performed during packaging.

Do not combine metrics produced under changed implementations with the paper's original result tables without clearly identifying the difference.
