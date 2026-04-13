# Mocap Key Pose Extractor

Thank you .BVH dataset from Bandai-Namco and project.py is trial for me to understand this topic.
A Maya Python tool that automatically reduces dense motion capture keyframes down to only the essential poses, making mocap data editable for animators.

## Overview

Motion capture recordings typically produce a keyframe on every single frame — hundreds or thousands of keys that are impossible to edit by hand. This tool analyzes the motion data, identifies which frames are truly important, and deletes everything else, leaving the animator with a clean, workable set of key poses.

The extraction pipeline combines PCA-based dimensionality reduction, geometric curve simplification, and calculus-based critical point detection:

1. **PCA-Based Energy Curve Construction** — Instead of naively summing joint distances, the tool collects all joint positions per frame into a high-dimensional pose vector and applies Principal Component Analysis (PCA) to project the data onto its principal direction of variation. This produces a smarter 1D energy curve that captures the most significant axis of motion across the entire skeleton, rather than treating all joint movements equally. Joint weighting is also applied so core joints (hips, spine, chest) contribute more than extremities (fingers, toes).

2. **Smoothing** — Applies a moving average filter to remove sensor noise from the energy curve. The window size is adjustable and tuned for 60fps data.

3. **Subsampling** — Optionally analyzes every Nth frame to reduce noise density before detection, then maps results back to the original timeline.

4. **Ramer-Douglas-Peucker (RDP) Simplification** — Geometrically simplifies the energy curve by recursively removing points within an epsilon tolerance of a straight-line approximation. Epsilon controls how aggressively frames are reduced.

5. **Critical Point Detection** — Uses first and second derivatives to find local maxima (peak actions), local minima (holds/pauses), and inflection points (motion transitions/saddle points) that RDP alone may miss.

6. **Minimum Gap Filtering** — Enforces a minimum frame distance between detected key poses to prevent clusters of redundant frames.

7. **ROM Validation (Dimension 3)** — After keyframe deletion, checks interpolated frames against natural human joint angle limits (based on Winter 2009 biomechanics data). Re-inserts keyframes where Maya's interpolation would create physically impossible poses such as hyperextended knees or elbows.

The tool provides an interactive Maya UI with sliders for epsilon, subsample step, smoothing window, and minimum gap, plus a toggle for ROM validation. All operations are wrapped in a single undo chunk for safe iteration.

## Tools & Technologies

- **Autodesk Maya** — 3D animation and motion capture host environment
- **Python ** — Scripting language used via Maya


## Author

Kantinun Sucharitkul
