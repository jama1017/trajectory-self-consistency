# Self-Consistency for LLM-Based Motion Trajectory Generation and Verification

[Jiaju Ma](https://majiaju.io/), [R. Kenny Jones](https://rkjones4.github.io/), [Jiajun Wu](https://jiajunwu.com/), and [Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/)
<br />
Stanford University
<br />
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
<br />

![Trajectory Self-Consistency Teaser](https://github.com/jama1017/trajectory-self-consistency/blob/main/assets/tsc_teaser.png?raw=true)
<br />

[![arXiv](https://img.shields.io/badge/arXiv-2603.29301-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.29301)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)

This repository contains the official implementation of our self-consistency method for LLM-based motion-trajectory generation and verification. We model the family of shapes associated with a prompt as a *prototype trajectory* paired with a *group of geometric transformations* (e.g., rigid, similarity, affine): two trajectories are considered consistent if one can be warped into the other under the transformations allowed by the group. We sample diverse trajectories from an LLM, cluster the consistent ones using a hierarchy of transformation groups, and select the largest cluster as the most self-consistent set — improving generation accuracy without supervision and enabling verification.

Check out the [project page](https://majiaju.io/trajectory-self-consistency) for animation and benchmark results.


## Installation
1. Set up a virtual environment with your favorite tool (`python>=3.10`). We recommend `uv` for its speed.
    ```bash
    uv venv tsc_env --python 3.12 # or conda, venv, etc.
    source tsc_env/bin/activate
    ```

2. Install the dependencies (`numpy`, `scipy`, `opencv-python`, `numba`).
    ```bash
    uv pip install -r requirements.txt

    # or with pip
    pip install -r requirements.txt
    ```


## Quick Start
A trajectory is an `(N, 2)` array of ordered `(x, y)` points. Call `transformation_ransac(T, R, transformation_type)` and read the scalar distance from the returned optimization history.

```python
import numpy as np
from shape_distance_metrics import transformation_ransac

## A target trajectory and a 90°-rotated copy of it
T = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], dtype=float)
R = T @ np.array([[0, -1], [1, 0]], dtype=float)

## transformation_type is one of the six metric names (see the table above)
A, history, R_aligned = transformation_ransac(T, R, transformation_type="rigid")

distance = history[-1]["cost"]   # residual distance after the best alignment
print(f"rigid distance: {distance:.4f}")
```

`transformation_ransac` returns `(A, history, R_aligned)`:
- `A` — the best `3x3` transformation matrix found.
- `history` — the per-iteration optimization log; `history[-1]["cost"]` is the shape distance.
- `R_aligned` — the reference trajectory after applying `A`.

Closed paths (where the first and last points nearly coincide) are detected automatically and aligned with a coarse-to-fine search over cyclic shifts.


## Release Roadmap
- [x] **Core distance metrics** — 6 transformation-invariant trajectory distances (this release)
- [ ] Clustering & transformation-group hierarchy (self-consistency selection)
- [ ] LLM sampling pipeline & dataset generation, plus dataset release
- [ ] Evaluation, VLM baselines & full paper reproduction
- [ ] Notebooks, tutorial & figures

> Star/watch the repo for updates as later phases land.


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Contact
Jiaju Ma<br />
jiajuma@stanford.edu<br />
[majiaju.io](https://majiaju.io)<br />


## Citation
If you find this work useful in your project, please cite our paper:
```bibtex
@inproceedings{Ma2026selfconsistency,
    author    = {Ma, Jiaju and Jones, R. Kenny and Wu, Jiajun and Agrawala, Maneesh},
    title     = {Self-Consistency for LLM-Based Motion Trajectory Generation and Verification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {17357-17366}
}
```