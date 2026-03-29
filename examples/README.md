# Examples

This directory contains the notebooks for the SpectraScope quick start and case studies.

## Run Order

**1. Quick_Start.ipynb** — Start here. Walks through the core workflow end-to-end using a small grape NIR dataset: loading data, generating features, training models (RF and optionally LCEN), and visualizing results. Does not require LCEN/SPA to run fully.

**2. XANES_PDF.ipynb** — Case study: regressing mean Ti–O bond length from simulated XANES and PDF spectra of Ti-oxide structures. Run this before GRAPES.ipynb — it exports `Figures/XANES_rMAX_all.csv`, which the GRAPES notebook reads for the cross-dataset correlation figure.

**3. GRAPES.ipynb** — Case study: predicting grape sugar content (°Brix) and acidity from Vis-NIR and Raman spectra. Requires `Figures/XANES_rMAX_all.csv` produced by XANES_PDF.ipynb.

## Notes

- All notebooks should be launched from within the `examples/` directory so that relative paths to `../tests/` and `../ss.mplstyle` resolve correctly.
- LCEN model training requires SmartProcessAnalytics v1.5.1. See the [README](../README.md) for installation instructions. Quick_Start.ipynb will skip LCEN steps gracefully if SPA is not installed; the case study notebooks require it.
