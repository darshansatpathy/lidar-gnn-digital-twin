# 2. A DGCNN on PyTorch Geometric's EdgeConv, with neighbours found in plain PyTorch

**Status:** accepted

## Context

Point clouds have no grid. Graph networks treat points as nodes and learn from their
neighbourhoods. Dynamic graph CNNs (Wang et al., 2019) rebuild the graph in feature space at every
layer, so points with similar learned descriptors exchange information even when they are apart.
PyTorch Geometric's `knn_graph` needs a compiled extension (`pyg-lib` or `torch-cluster`) that is
not available for every PyTorch build.

## Decision

Use `torch_geometric.nn.EdgeConv` for the layers, and an exact chunked k-nearest-neighbour search
written with `torch.cdist` and `topk`. Three EdgeConv layers (k = 16), a global context vector, and a
per-point classifier: 130,000 parameters.

## Consequences

- Installs from PyPI with PyTorch alone; CI uses CPU wheels.
- The exact search is O(n²) per block, bounded by working in 20 m blocks of 4,096 points; a
  production system on dense scans would add the compiled extension for speed.
- Inference splits each block into random subsets of the training density and votes over two
  passes, so the model sees the point density it was trained on.
