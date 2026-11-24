# TransNetV2 provenance

- PyTorch implementation: https://github.com/YangTuanAnh/transnetv2_pytorch
- Copied revision: `b4927ebfa576791521c6fb02217b4a622c731c37`
- Original TransNetV2 project: https://github.com/soCzech/TransNetV2
- Checkpoint: `transnetv2-pytorch-weights.pth`
- Checkpoint location: this directory, beside `model.py` and `inference.py`
- Checkpoint SHA-256: `5e896c007888ae67761a2c673cfe0f2f3745b28e580631589696a7b665a442d0`

`model.py` preserves the upstream network architecture. `inference.py` adapts
the upstream 100-frame windows, 50-frame stride, central predictions, inclusive
shot conversion, and visualization. Local changes make checkpoint/device
selection explicit, validate inputs, use stable sigmoid probabilities, and log
progress. Decoding and filesystem orchestration live outside this package.

The copied checkout contains no LICENSE file. No new license is asserted for
the reused code or weights; retain this provenance when redistributing it.
