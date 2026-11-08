# 1. Train and evaluate on simulated mobile-LiDAR scans

**Status:** accepted

## Context

Public labelled urban point clouds (Toronto-3D, Paris-Lille-3D, SemanticKITTI) are large
downloads with licence terms that vary, and none of them comes with a matching asset registry or
a second epoch with known changes, which is what a digital twin has to be scored against.

## Decision

Streets are generated procedurally and scanned by a simulated mobile scanner that keeps the
properties that make real data hard: density falling with range, surfaces facing away from the
scanner missing, porous tree crowns, range noise, and intensity that depends on material, incidence
and range. Every point carries its class and its asset; every street has an asset registry and a
second scan with known changes.

## Consequences

- Segmentation, asset extraction and change detection can all be scored exactly, and the data
  regenerates bit-for-bit from seeds, so CI needs no dataset download.
- Distribution shift is testable on demand: sparser scans, noisier scans, taller buildings.
- Absolute numbers are for this simulator. The pipeline (terrain model, blocks, DGCNN, clustering,
  matching) is what transfers to real scans; the README says so.
