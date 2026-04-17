# PF-DEA-Net (Initial Development)

This directory contains the first implementation scaffold of:

**PF-DEA-Net: Physics-Guided and Frequency-Enhanced Dehazing Network**

## Implemented Modules

- `pf_dea_net/model.py`
  - End-to-end PF-DEA-Net forward path
  - Returns interpretable intermediate outputs:
    - transmission map
    - atmospheric light
    - physics reconstruction
- `pf_dea_net/physics.py`
  - `PhysicsHead` for `t(x)` and `A`
  - `AtmosphericReconstruction` for scattering inversion
- `pf_dea_net/frequency.py`
  - FFT-based frequency enhancement unit
- `pf_dea_net/edge.py`
  - Laplacian edge extractor
  - edge-guided refinement block
- `pf_dea_net/blocks.py`
  - DEA-style residual blocks
  - CGA fusion
  - contrast/histogram-aware CGA++ fusion

## Quick Check

From this folder:

```bash
python smoke_test.py
```

If successful, tensor shapes for all outputs are printed.
