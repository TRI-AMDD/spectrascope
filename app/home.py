import streamlit as st
from spectrascope.ss_classification import *
from spectrascope.plot_spectral_features import plot_spectral_features as psf
from spectrascope.plot_spectral_features import plot_model as pm

st.set_page_config(page_title="SpectraScope", page_icon="👋", layout="wide")

st.write("# Welcome to SpectraScope! 👋")

# st.sidebar.success("Select a prediction schema")

st.markdown(
    """
    SpectraScope is an app and python package for machine learning on spectral data. 
    
"""
)


fnX = "/Users/amalyajohnson/Desktop/Spectrascope/spectrascope_local/tests/test_files/X_testfile.npy"
fny = "/Users/amalyajohnson/Desktop/Spectrascope/spectrascope_local/tests/test_files/y_testfile.npy"
X = np.load(fnX)
y = np.load(fny)

fn = "/Users/amalyajohnson/Desktop/Spectrascope/Datasets/XAS_PDF/Ti/processed_billinge.json"
data = pd.read_json(fn)
structures = data["structure"].apply(lambda x: Structure.from_dict(x))
structures = structures.to_numpy()
mpids = data["mp_id"].to_list()

info = BinaryClassifications(
    structures,
    mpids,
    X=X,
    y=y,
    specnames=["pdf", "xanes"],
    yname="nn",
    random_seed=10,
    generate_dict=False,
)


@st.cache_data
def load_data(X, y, specnames=["pdf", "xanes"], yname="nn"):
    data = BinaryClassifications(
        structures=structures,
        mpids=mpids,
        X=X,
        y=y,
        specnames=specnames,
        yname=yname,
        generate_dict=False,
    )
    return data


st.markdown("Please Load Your Data")

data_load_state = st.text("Loading X data...")
data = load_data(X, y)
data_load_state.text("Done! (using st.cache_data)")

st.session_state["data"] = data
st.session_state["ndatasets"] = data.X.shape[0]

st.write(f"Detected {data.X.shape[0]} loaded datasets")

# add x axis
# add pointwise features
data.add_x_axis(specname="xanes", value=np.linspace(4969.0, 5021.0, 200))
data.add_x_axis(specname="pdf", value=np.linspace(0, 10, 200))


xaxis_pdf = "r"
xaxis_xanes = "Energy (eV)"
yaxis_pdf = "G(r)"
yaxis_xanes = "Intensity (arb)"

axis_dict = {"xanes": (xaxis_xanes, yaxis_xanes), "pdf": (xaxis_pdf, yaxis_pdf)}


for Xi, name in zip(data.X, data.specnames):
    df = pd.DataFrame(
        Xi, columns=[str(round(i, 2)) for i in getattr(data, f"{name}_xaxis")]
    )
    if st.checkbox(f"Show {name} Data"):
        st.write(df.head())

yattr = st.text_input("Name of prediction", "nn")
st.session_state["yattr"] = yattr

st.session_state["fitted"] = False

st.markdown("Raw Data")

fig1 = make_subplots(
    rows=1,
    cols=st.session_state["ndatasets"],
    subplot_titles=data.specnames,
    horizontal_spacing=0.15,
)

for i, spec in enumerate(data.specnames):
    xaxis_label, yaxis_label = axis_dict[spec]
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
