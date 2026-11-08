# 3. Build the twin from segmentation, and keep transient objects out of it

**Status:** accepted

## Context

A street-asset twin records persistent assets (poles, trees, buildings) with position and height,
and is useful when it reports what changed. Parked vehicles are in every scan and move between
scans; a twin that reported them would bury real changes.

## Decision

Assets are clustered from the predicted classes: poles in plan, trees by their trunks (crowns of
neighbouring trees overlap), buildings into facades. Vehicles are classified but never become
assets. Assets are matched to the registry and between epochs by position with a Hungarian
assignment and a distance limit per kind; buildings are flagged as modified when their observed
height moves by more than 2 m.

## Consequences

- Change detection is scored against the generator's known changes, and with ground-truth labels as
  an upper bound, so errors are attributable to the model or to the twin logic.
- A building's height is the highest point the scanner sees, which is below the roof for tall
  buildings close to the street. The report states this rather than hiding it.
