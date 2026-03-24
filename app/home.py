import pathlib

import streamlit as st
from attr import dataclass
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.google_analytics import inject_google_analytics
from spectrascope.SSClassification import *
from spectrascope.SSPlotting import PlotSpectralFeatures as psf

st.set_page_config(page_title="SpectraScope", page_icon="👋", layout="wide")

inject_google_analytics()

st.write("# Welcome to SpectraScope! 👋")
st.markdown("""
    SpectraScope is an app and python package for machine learning on spectral data.
    """)

TEST_DIR = os.path.join(os.path.dirname(__file__), "../tests/test_files")

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
    return SpectraScope(X=X, y=y, specnames=specnames, yname=yname)


# @st.cache_data  # Temporarily disabled to allow updates
def load_default_data(option):
    if option == "Default (PDF,XANES)":
        return load_pdf_xanes_data()
    elif option == "Default (Wine)":
        return load_wine_data()
    return None


def load_pdf_xanes_data():
    
    fnX = os.path.join(TEST_DIR, "X_testfile.npy")
    fny = os.path.join(TEST_DIR, "y_testfile.npy")
    spec_metadata = {
        "PDF": SpecMetadata(
            name="PDF",
            xaxis_label="r",
            yaxis_label="G(r)",
            xaxis=np.linspace(0, 10, 200),
        ),
        "XANES": SpecMetadata(
            name="XANES",
            xaxis_label="Energy (eV)",
            yaxis_label="Intensity (arb)",
            xaxis=np.linspace(4969.0, 5021.0, 200),
        ),
    }
    return SpectraScopeInput(
        X=np.load(fnX),
        y=np.load(fny),
        specnames=["PDF", "XANES"],
        yname="bondlength",
        spec_metadata=spec_metadata,
    )


def load_wine_data():
    XSF_br = np.load(os.path.join(TEST_DIR, "grapes_example/XSF_br.npy"))
    XRF_br = np.load(os.path.join(TEST_DIR, "grapes_example/XRF_br.npy"))
    br = np.load(os.path.join(TEST_DIR, "grapes_example/br.npy"))
    nir_wls = np.load(os.path.join(TEST_DIR, "grapes_example/nir_xaxis.npy"))
    raman_wls = np.load(os.path.join(TEST_DIR, "grapes_example/rm_xaxis.npy"))
    spec_metadata = {
        "NIR": SpecMetadata(
            name="NIR",
            xaxis_label="NIR",
            yaxis_label="Reflection [%]",
            xaxis=nir_wls,
        ),
        "Raman": SpecMetadata(
            name="Raman",
            xaxis_label="Raman",
            yaxis_label="Intensity (cts.)",
            xaxis=raman_wls,
        ),
    }
    return SpectraScopeInput(
        X=[XSF_br, XRF_br],
        y=br,
        specnames=["NIR", "Raman"],
        yname="sugarcontent",
        spec_metadata=spec_metadata,
    )


def load_custom_data(csv_uploads: list[UploadedFile]):
    """
    see toy_data.ipyb for explanation of the csv format
    """
    X = []
    y = None
    yname = None
    specnames = []
    spec_metadata = {}
    for csv_upload in csv_uploads:
        csv_df = pd.read_csv(csv_upload, index_col=0)
        curr_X = csv_df.iloc[:-1, :-1].to_numpy()
        curr_y = csv_df.iloc[:-1, -1].to_numpy()
        curr_y_name = csv_df.columns[-1]
        curr_xaxis = csv_df.iloc[-1, :-1].to_numpy()
        curr_xaxis_label = csv_df.index[-1]
        curr_specname = pathlib.Path(csv_upload.name).name.removesuffix(".csv")

        X.append(curr_X)
        if y is not None and not np.array_equal(y, curr_y):
            st.error("all datasets must have same y data")
            st.stop()
        y = curr_y
        yname = curr_y_name
        specnames.append(curr_specname)
        spec_metadata[curr_specname] = SpecMetadata(
            name=curr_specname,
            xaxis_label=curr_xaxis_label,
            yaxis_label=curr_y_name,
            xaxis=curr_xaxis,
        )
    return SpectraScopeInput(
        X=X, y=y, specnames=specnames, yname=yname, spec_metadata=spec_metadata
    )


def plot_raw_data(data, spec_input):
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


def main():
    st.markdown("Please Load Your Data")

    # Add a button to clear cache if needed
    if st.button("Clear Cache ( if data not updating)"):
        st.cache_data.clear()
        st.rerun()

    data_option = st.radio(
        "Select Data Option", ("Default (PDF,XANES)", "Default (Wine)", "custom")
    )

    if data_option == "custom":
        csv_uploads = st.file_uploader(
            "Upload CSV files", type=["csv"], accept_multiple_files=True
        )
        if csv_uploads:
            spec_input = load_custom_data(csv_uploads)
        else:
            st.warning("Upload CSV files to load custom data")
            st.stop()
    else:
        spec_input = load_default_data(data_option)
        if spec_input is None:
            st.error(f"Unknown data option: {data_option}")
            st.stop()

    data_load_state = st.text(f"Loading {data_option} data...")
    data = load_data(spec_input.X, spec_input.y, spec_input.specnames, spec_input.yname)
    data_load_state.text(f"{data_option} data loaded!")

    st.session_state["data"] = data
    st.session_state["ndatasets"] = len(spec_input.specnames)
    st.write(f"Detected {st.session_state['ndatasets']} loaded datasets")

    for name in data.specnames:
        data.add_x_axis(specname=name, value=spec_input.spec_metadata[name].xaxis)

    for i, name in enumerate(spec_input.specnames):
        if st.checkbox(f"Show {name} data"):
            st.write(
                pd.DataFrame(
                    spec_input.X[i],
                    columns=[
                        str(round(i, 2)) for i in spec_input.spec_metadata[name].xaxis
                    ],
                ).head()
            )

    yattr = st.text_input("Name of Target Variable", spec_input.yname)
    st.session_state["yattr"] = yattr
    st.session_state["fitted"] = False

    st.markdown("Raw Data")
    plot_raw_data(data, spec_input)


if __name__ == "__main__":
    main()
