# Security policy

Report a vulnerability privately through [GitHub security advisories](https://github.com/darshansatpathy/lidar-gnn-digital-twin/security/advisories/new). Please do not open a public issue. You can expect a first reply within five working days.

## What is in scope

- Loading model weights. `ltw` loads a `state_dict` with `torch.load(..., weights_only=True)` and checks its SHA-256 against `models/dgcnn.json` before use. A way to run code or load unverified weights through `ltw` is a vulnerability.
- Reading PLY files. `ltw` reads only the binary little-endian layout it writes; a crafted file that causes more than a clean error is in scope.
- The CI workflows: actions are pinned to major versions and run with read-only permissions.

## Supported versions

The latest release receives fixes.
