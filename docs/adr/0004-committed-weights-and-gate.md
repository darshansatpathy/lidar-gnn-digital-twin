# 4. Commit the trained weights with a digest, and gate on them in CI

**Status:** accepted

## Context

Training took about an hour on two shared CPU cores (15 epochs). CI should check the model people will use, and
weights must not be a way to run code on load.

## Decision

The trained state dict (0.5 MB) is committed with a manifest recording its SHA-256, parameter count
and validation scores. It is loaded with `torch.load(weights_only=True)` after the digest is
checked. CI regenerates the test streets from their seeds and fails if the committed model's mIoU
falls below 0.85 or does not beat the gradient-boosting baseline; a short training run is tested
separately.

## Consequences

- A change to the simulator, the features or the model that degrades the committed model fails CI.
- Retraining is a deliberate step (`ltw train`) whose new weights and report are reviewed together.
