# SpectraScope

SpectraScope is a Python package for feature generation, model training, and feature downselection for spectral data.
SpectraScope is primarily built for spectral data, but can be applied to any 1D array-like data currently.
Future development will expand to 2D data for more advanced experimentation techniques.

**Requires Python ≥ 3.9** | [Apache-2.0 License](LICENSE)

## Installation

### Without LCEN

```bash
git clone git@github.com:TRI-AMDD/spectrascope.git
cd spectrascope
pip install -e .
```

This installs all the required packages except for SmartProcessAnalytics, which holds the LCEN algorithm.
SpectraScope can run without LCEN, however you will lose feature selection and polynomial/interaction transformations.

### With LCEN

The publication notebooks assume LCEN is installed. You can install it via the optional extra:

```bash
pip install -e ".[lcen]"
```

Or manually install v1.5.1 of SmartProcessAnalytics ([releases](https://github.com/PedroSeber/SmartProcessAnalytics/releases)):

```bash
git clone https://github.com/PedroSeber/SmartProcessAnalytics.git
cd SmartProcessAnalytics
pip install -e .
# note : if there are errors with ace-cream installation, can comment out
```

## Getting Started

SpectraScope is most flexible when used directly with Python.
The [examples](examples) directory leads a user through the core workflow of spectrascope, and contains notebooks that reproduce the main text figures from the manuscript. The [README](examples/README.md) outlines further details on how to run the files. 

- Start with the [Quick Start notebook](examples/Quick_Start.ipynb) for an introduction to the basic functions of Spectra-Scope
- [GRAPES.ipynb](examples/GRAPES.ipynb) — grape spectral data analysis
- [XANES_PDF.ipynb](examples/XANES_PDF.ipynb) — multi-modal XANES + PDF analysis

## Streamlit App

Analysis can also be run through the Streamlit app. To run the app:

```
cd app
streamlit run home.py  
```

Or use the docker image: 
```
docker compose up --build
```

## Citation Guidelines
If you use Spectra-scope, please cite:
```
Amalya C. Johnson and Chris Fajardo and Leena Sansguiri and Weike Ye and Steven B. Torrisi,
Spectra-Scope : A toolkit for automated and interpretable characterization of material properties from spectral data, 2026, https://arxiv.org/abs/2603.06011
```

## External data used for publication

We thank the authors of the papers whose datasets were used in this manuscript.

Material PDF & XANES data comes from: 
``` Tina Na Narong, Zoe Zachko, Steven B. Torrisi, Simon Billinge, Interpretable multimodal machine learning analysis of X-ray absorption near-edge spectra and pair distribution functions. npj Computational Materials 11, 98 (2025). https://doi.org/10.1038/s41524-025-01589-3 ```

Grape data comes from the following manuscript:
``` Iman Ebrahimi, Ricardo de Castro, Reza Ehsani, Luca Brillante, Shuang Feng,
Advancing grape chemical analysis through machine learning and multi-sensor spectroscopy, Journal of Agriculture and Food Research, Volume 16, 2024, https://doi.org/10.1016/j.jafr.2024.101085 ```
And is hosted here. https://github.com/imeb/Grape-Chemical-Composition/tree/main

## AI Usage
The majority of Spectra-scope was written by hand with assistance from GitHub copilot 2024. Since then, LLMs were used to assist with cleaning up the codebase, primarily Claude Opus 4.6.