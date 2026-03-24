# Getting Started

SpectraScope is a Python package for feature generation, model training, and feature
downselection for spectral data. It is primarily built for spectral data but can be applied
to any 1D array-like data. Future development will expand to 2D data.

## Installation

### Without LCEN

```bash
git clone git@github.com:TRI-AMDD/spectrascope.git
cd spectrascope
pip install -r requirements.txt
pip install -e .
```

This installs all required packages except SmartProcessAnalytics, which holds the LCEN
algorithm. SpectraScope runs without LCEN, but you will lose feature selection and
polynomial/interaction transformations.

### With LCEN

To use LCEN, install v1.5.1 of SmartProcessAnalytics (must be v1.5.1).
Download the release [here](https://github.com/PedroSeber/SmartProcessAnalytics/releases).

```bash
git clone https://github.com/PedroSeber/SmartProcessAnalytics.git
cd SmartProcessAnalytics-1.5.1
pip install -e .
# note: if there are errors with ace-cream installation, you can comment it out

git clone git@github.com:TRI-AMDD/spectrascope.git
cd spectrascope
pip install -r requirements.txt
pip install -e .
```

## Running the Streamlit App

SpectraScope is most flexible when used directly in Python (see the Quick Start notebook
below). Analysis can also be run interactively through the Streamlit app:

```bash
cd app
streamlit run home.py
```

Or with Docker:

```bash
docker compose up --build
```

## Quick Start

The best way to get started is the **Quick Start example notebook**, which walks through
a full workflow: loading data, generating spectral features, training a model, and
visualizing the results.

```{toctree}
:maxdepth: 1

../notebooks/Quick_Start
```
