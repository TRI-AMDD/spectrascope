from collections import defaultdict

import numpy as np
import streamlit as st
from attr import dataclass

from spectrascope.ss_classification import *
from spectrascope.plot_spectral_features import PlotSpectralFeatures as psf

st.set_page_config(page_title="SpectraScope", page_icon="👋", layout="wide")

st.write("# Welcome to SpectraScope! 👋")

st.markdown(
    """
    SpectraScope is an app and python package for machine learning on spectral data. 
    
"""
)


@dataclass
class SpecMetadata:
    name: str
    xaxis_label: str
    yaxis_label: str
    xaxis: np.ndarray


@dataclass
class SpectraScopeInput:
    X: np.ndarray | list[np.ndarray]
    y: np.ndarray
    specnames: list[str]
    yname: str
    spec_metadata: dict[str, SpecMetadata] = None


@st.cache_data
def load_data(X, y, specnames, yname):
    data = SpectraScope(
        X=X,
        y=y,
        specnames=specnames,
        yname=yname,
    )
    return data


st.markdown("Please Load Your Data")

data_option = st.radio(
    "Select Data Option", ("default (pdf,xanes)", "default (wine)", "custom")
)
spec_input = None
if data_option == "default (pdf,xanes)":
    fnX = "../tests/test_files/X_testfile.npy"
    fny = "../tests/test_files/y_testfile.npy"
    spec_metadata = {
        "pdf": SpecMetadata(
            name="pdf",
            xaxis_label="r",
            yaxis_label="G(r)",
            xaxis=np.linspace(0, 10, 200),
        ),
        "xanes": SpecMetadata(
            name="xanes",
            xaxis_label="Energy (eV)",
            yaxis_label="Intensity (arb)",
            xaxis=np.linspace(4969.0, 5021.0, 200),
        ),
    }
    spec_input = SpectraScopeInput(
        X=np.load(fnX),
        y=np.load(fny),
        specnames=["pdf", "xanes"],
        yname="nn",
        spec_metadata=spec_metadata,
    )
elif data_option == "default (wine)":
    XSF_br = np.load("../tests/test_files/grapes_example/XSF_br.npy")
    XRF_br = np.load("../tests/test_files/grapes_example/XRF_br.npy")
    br = np.load("../tests/test_files/grapes_example/br.npy")
    nir_wls = np.load("../tests/test_files/grapes_example/nir_xaxis.npy")
    raman_wls = np.load("../tests/test_files/grapes_example/rm_xaxis.npy")
    spec_metadata = {
        "nir": SpecMetadata(
            name="nir",
            xaxis_label="nir",
            yaxis_label="yaxis",
            xaxis=nir_wls,
        ),
        "rm": SpecMetadata(
            name="rm",
            xaxis_label="rm",
            yaxis_label="yaxis",
            xaxis=raman_wls,
        ),
    }
    spec_input = SpectraScopeInput(
        X=[XSF_br, XRF_br],
        y=br,
        specnames=["nir", "rm"],
        yname="br",
        spec_metadata=spec_metadata,
    )
else:
    try:
        specnames = st.text_input(
            "Enter the names of the spectra separated by commas", ""
        ).split(",")
        assert specnames != [""], "Please enter at least one spectrum name"

        custom_x_option = st.radio(
            "Select X upload option",
            ("all spectra in one .npy file", "one .npy file per spectrum"),
        )
        if custom_x_option == "all spectra in one .npy file":
            X = st.file_uploader("Upload X data", type=["npy"])
            X = np.load(X)
        else:
            X = []
            for spec in specnames:
                f = st.file_uploader(f"Upload X data for {spec}", type=["npy"])
                X.append(np.load(f))

        y = st.file_uploader("Upload y data", type=["npy"])
        yname = st.text_input("Enter the name of the prediction", "")
        assert y and specnames != [""] and yname != "", "Please fill in all the fields"

        spec_metadata = {}
        for spec in specnames:
            xaxis_label = st.text_input(f"Enter x-axis label for {spec}", "")
            yaxis_label = st.text_input(f"Enter y-axis label for {spec}", "")
            linspace = st.text_input(
                f"Enter linspace for {spec} as a comma separated tuple min,max,numvalues",
                "",
            ).split(",")
            assert xaxis_label != "" and yaxis_label != "" and len(linspace) == 3
            xaxis = np.linspace(
                float(linspace[0]), float(linspace[1]), int(linspace[2])
            )
            spec_metadata[spec] = SpecMetadata(
                name=spec,
                xaxis_label=xaxis_label,
                yaxis_label=yaxis_label,
                xaxis=xaxis,
            )
        spec_input = SpectraScopeInput(
            X=X,
            y=np.load(y),
            specnames=specnames,
            yname=yname,
            spec_metadata=spec_metadata,
        )
    except Exception:
        st.warning("Please fill in all the fields")
        st.stop()


data_load_state = st.text(f"Loading {data_option} data...")
data = load_data(spec_input.X, spec_input.y, spec_input.specnames, spec_input.yname)
data_load_state.text(f"{data_option} data loaded!")

st.session_state["data"] = data
st.session_state["ndatasets"] = len(spec_input.specnames)

st.write(f"Detected {st.session_state['ndatasets']} loaded datasets")

# add x axis
# add pointwise features
for name in data.specnames:
    data.add_x_axis(specname=name, value=spec_input.spec_metadata[name].xaxis)

# Look at dataframes
for i, name in enumerate(spec_input.specnames):
    if st.checkbox(f"Show {name} Data"):
        st.write(
            pd.DataFrame(
                spec_input.X[i],
                columns=[
                    str(round(i, 2)) for i in spec_input.spec_metadata[name].xaxis
                ],
            ).head()
        )

# Save name of yattr in session state
yattr = st.text_input("Name of prediction", spec_input.yname)
st.session_state["yattr"] = yattr

st.session_state["fitted"] = False

st.markdown("Raw Data")


# Plot raw data
fig1 = make_subplots(
    rows=1,
    cols=st.session_state["ndatasets"],
    subplot_titles=data.specnames,
    horizontal_spacing=0.15,
)

axis_dict = {}
for i, spec in enumerate(data.specnames):
    xaxis_label, yaxis_label = (
        spec_input.spec_metadata[spec].xaxis_label,
        spec_input.spec_metadata[spec].yaxis_label,
    )
    axis_dict[spec] = (xaxis_label, yaxis_label)
    x = getattr(data, f"{spec}_xaxis")
    y = getattr(data, f"{spec}")
    plotter = psf(
        xaxis=x,
        arr=y,
        xaxis_label=xaxis_label,
        yaxis_label=yaxis_label,
        name="",
        fig=fig1,
        row=1,
        col=i + 1,
    )
    plotter.plot_raw_spec()

st.plotly_chart(fig1)

st.session_state["axis_dict"] = axis_dict
