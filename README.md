# SpectraScope 

SpectraScope is a python package for feature generation, model training, and feature downselection for spectral data. 
SpectraScope is primarily built for spectral data, but can be applied to any 1D array-like data currently. 
Future development will expand to 2D data for more advanced experimentation techniques. 

## Installation: 
```
git clone git@github.com:TRI-AMDD/spectrascope.git
cd spectrascope
pip install -r requirements.txt
pip install -e .
```

This installs all the required packages except for SmartProcessAnalytics, which holds the LCEN algorithm. 
SpectraScope can run without LCEN, however you will lose feature selection and polynomial/interaction transformations. 

To Use LCEN, install v1.4.1 of SmartProcessAnalytics. Currently must be v1.4.1.

A working version can be found on the "main" branch of this [fork](https://github.com/AmalyaJohnson-TRI/SmartProcessAnalytics/tree/main). 
Clone this repo and run pip install -e .

## Streamlit app usage 
SpectraScope is most flexible when used directly with python. Please reference the notebooks in [examples](examples) for a variety of examples. Analysis can also be run through the streamlit app. To run the app: 

```
cd app
streamlit run home.py  
```

## Docs 
The [docs](/docs) contain html documentation on the details of spectrascope. Can be used to create github pages website.  
