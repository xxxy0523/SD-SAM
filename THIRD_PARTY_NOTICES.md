# Third-party code

This repository contains modified source from:

| Directory | Upstream project | License copy |
| --- | --- | --- |
| `segment_anything/` | [Meta Segment Anything](https://github.com/facebookresearch/segment-anything) | [Apache 2.0](licenses/SAM_LICENSE) |
| `dinov3/` | [Meta DINOv3](https://github.com/facebookresearch/dinov3) | [DINOv3 License](licenses/DINOV3_LICENSE.md) |

Upstream copyright notices are retained. The SD-SAM adaptations add intermediate
feature/QKV extraction, shallow adapters, deep feature fusion and training interfaces.
Local corrections also address explicit checkpoint loading, spatial alignment and
frozen-module behavior. These directories are modified copies, not unchanged upstream
packages. The exact upstream commit of the originally supplied copies was not provided.

License copies were retrieved from the official repositories on 2026-10-07:
[SAM license source](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
and [DINOv3 license source](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
Their licenses continue to govern the corresponding code and pretrained weights.
No pretrained weights or datasets are redistributed here.

The author has not yet selected a license for the original SD-SAM contributions.
No repository-wide MIT or Apache grant is inferred from the third-party licenses.
