# SpectraScope 

SpectraScope is a python package for feature generation, model training, and feature downselection for spectral data. 
SpectraScope is primarily built for spectral data, but can be applied to any 1D array-like data currently. 
Future development will expand to 2D data for more advanced experimentation techniques. 

## Installation: 

### Without LCEN 

```bash 
git clone git@github.com:TRI-AMDD/spectrascope.git
cd spectrascope
pip install -r requirements.txt
pip install -e .
```

This installs all the required packages except for SmartProcessAnalytics, which holds the LCEN algorithm. 
SpectraScope can run without LCEN, however you will lose feature selection and polynomial/interaction transformations. 


### With LCEN 

To Use LCEN, install v1.5.1 of SmartProcessAnalytics. Currently must be v1.5.1.

Version 1.5.1 can be found [here](https://github.com/PedroSeber/SmartProcessAnalytics/releases). 

```bash 
git clone https://github.com/PedroSeber/SmartProcessAnalytics.git
cd SmartProcessAnalytics-1.5.0 
pip install -e .
# note : if there are errors with ace-cream installation, can comment out

git clone git@github.com:TRI-AMDD/spectrascope.git
cd spectrascope
pip install -r requirements.txt
pip install -e .
```

## Streamlit app usage 
SpectraScope is most flexible when used directly with python. Please reference the notebooks in [examples](examples) for a variety of examples. Analysis can also be run through the streamlit app. To run the app: 

```
cd app
streamlit run home.py  
```

Or use the docker image: 
```
docker compose up --build
```

