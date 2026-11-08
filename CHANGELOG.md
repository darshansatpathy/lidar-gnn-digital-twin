# Changelog

## 0.1.0

- Procedural mobile-LiDAR simulator: streets with buildings, trees, poles and parked vehicles; range-dependent density, occlusion, back-face culling, porous crowns, range noise and intensity; a second epoch with known changes.
- Features: height above a terrain model, eigenvalue shape descriptors.
- DGCNN on PyTorch Geometric EdgeConv with a chunked k-nearest-neighbour search in plain PyTorch; voting over random subsets at inference.
- Gradient-boosting baseline on handcrafted features.
- Twin: asset extraction (trees by trunk, poles and buildings by clustering), matching to a registry, change detection that ignores vehicles.
- Weights committed as a state dict with a SHA-256 manifest, loaded with `weights_only=True`.
- `ltw scan`, `train`, `evaluate` (a CI gate: mIoU at least 0.85 and better than the baseline) and `twin` (GeoJSON and PLY export).
