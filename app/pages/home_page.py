import pandas as pd
import streamlit as st
import numpy as np
import sys

from spectrascope.SpectraScope import *
from spectrascope.ss_classification import *
from spectrascope.plot_spectral_features import * 

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor()
# from sklearn.linear_model import Lasso, Ridge
# import SPA

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# st.session_state['fitted'] = False
st.title("SpectraScope")
st.sidebar.markdown("Spectrascope Home")

fnX = "/Users/amalyajohnson/Desktop/Spectrascope/spectrascope_local/tests/test_files/X_testfile.npy"
fny = "/Users/amalyajohnson/Desktop/Spectrascope/spectrascope_local/tests/test_files/y_testfile.npy"
X = np.load(fnX)
y = np.load(fny)

fn = '/Users/amalyajohnson/Desktop/Spectrascope/Datasets/XAS_PDF/Ti/processed_billinge.json'
data = pd.read_json(fn)
structures = data['structure'].apply(lambda x : Structure.from_dict(x))
structures = structures.to_numpy()
mpids = data['mp_id'].to_list()

info = BinaryClassifications(
    structures, mpids, X=X, y=y, specnames=["pdf", "xanes"], yname="nn", random_seed=10, generate_dict=False
)



@st.cache_data
def load_data(X, y, specnames=["pdf", "xanes"], yname="nn"):
    data = BinaryClassifications(structures=structures, mpids=mpids, X=X, y=y, specnames=specnames, yname=yname)
    return data


st.markdown("## 1. Load  data")

data_load_state = st.text("Loading X data...")
data = load_data(X, y)
data_load_state.text("Done! (using st.cache_data)")
st.session_state["data"] = data
st.session_state["ndatasets"] = data.X.shape[0]
st.write(f"Detected {data.X.shape[0]} loaded datasets")

data.add_x_axis(specname="xanes", value=np.linspace(4969.0, 5021.0, 200))
data.add_x_axis(specname="pdf", value=np.linspace(0, 10, 200))

for Xi, name in zip(data.X, data.specnames):
    df = pd.DataFrame(
        Xi, columns=[str(round(i, 2)) for i in getattr(data, f"{name}_xaxis")]
    )
    if st.checkbox(f"Show {name} Data"):
        st.write(df.head())

yattr = st.text_input("Input Column to predict", " ")
st.session_state['yattr'] = yattr



def textinput_to_list(string, delim):
    return string.replace(" ", "").split(delim)


st.markdown("## 2. Choose Spectra to Featurize")
cols_to_featurize = st.text_input("Input column(s) to featurize", "xanes,pdf")
cols_to_featurize = textinput_to_list(cols_to_featurize, ",")

st.session_state["cols_to_featurize"] = cols_to_featurize

ncols = len(cols_to_featurize)

fig = make_subplots(
    rows=1, cols=ncols, subplot_titles=cols_to_featurize, horizontal_spacing=0.15
)

st.session_state["fitted_models"] = []
from pages.visualizing import plot_raw_spec

for ic, col in enumerate(cols_to_featurize):
    plot_raw_spec(fig, col, 1, ic + 1)

st.plotly_chart(fig)


features = st.text_input(
    "Input feature(s) to generate", "polynomial_featuresall,cwt"
)  # , gaussian_peaks')
features = textinput_to_list(features, ",")

nfeats = len(features)


for col in cols_to_featurize:
    fig = make_subplots(
        rows=1, cols=nfeats, subplot_titles=features, horizontal_spacing=0.15
    )
    x = getattr(data, f"{col}_xaxis")
    if not hasattr(data, f"{col}_pointwise_features"):
        data.add_spectral_feature(col, "pointwise_features", xrange=x)
    spec = getattr(data, f"{col}_pointwise_features")[0]
    temp = spectral_features(x, spec)

    for ic, feat in enumerate(features):
        temp.__getattribute__(feat).__call__()
        if "gaussian_peaks" in feat:
            fig.add_trace(
                go.Scatter(x=temp.energies, y=temp.intensities, showlegend=False),
                row=1,
                col=ic + 1,
            )
            for i in range(0, len(temp.peak_features), 3):
                ctr = temp.peak_features[i]
                amp = temp.peak_features[i + 1]
                wid = temp.peak_features[i + 2]
                yfit = amp * np.exp(-((x - ctr) ** 2 / (2 * (wid / 2.355) ** 2)))
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=yfit,
                        mode="lines",
                        showlegend=False,
                        line=dict(color="#7f7f7f"),
                    ),
                    row=1,
                    col=ic + 1,
                )
        else:
            x, y = temp.__getattribute__("plot_" + feat).__call__()
            if "cwt" in feat:
                fig.add_trace(go.Heatmap(z=y), row=1, col=ic + 1)
            else:
                fig.add_trace(go.Scatter(x=x, y=y, showlegend=False), row=1, col=ic + 1)

    st.plotly_chart(fig)

features.append("pointwise_features")

st.markdown("## 3. Choose Features to Build Model On")

all_features = []
for col in cols_to_featurize:
    for feat in features:
        all_features.append(col + "_" + feat)

features_selected = st.multiselect("Select from Features Computed", all_features)

with st.spinner(text="featurizing....(say that 10 times fast)"):
    for col in cols_to_featurize:
        for feat in features_selected:
            x = getattr(data, f"{col}_xaxis")
            if not hasattr(data, feat):
                data.add_spectral_feature(x, feat.replace(f"{col}_", ""), convcol=col)

multi_modal_selected = st.text_input(
    "Would you like to combine any of these features? (format: feat1,feat2; feat1,feat3)",
    None,
)

if multi_modal_selected is not None:
    multi_modal_selected = textinput_to_list(multi_modal_selected, ";")
    for combo in multi_modal_selected:
        combo = combo.split(",")
        if len(combo) == 2:
            feat1, feat2 = combo
            attr_name = feat1.split("_")[0] + "_" + feat2
            if not hasattr(data, attr_name):
                data.combine_feature_json(feat1, feat2)

                features_selected.append(attr_name)


st.markdown("## 4. Build Models for Cross Validation")

st.markdown("#### Build Random Forest")
model = RandomForestRegressor()
n_estimators = st.text_input("n_estimators", "100")  # 50, 200, 300
criterion = st.text_input("criterion", "squared_error")  # absolute_error
max_depth = st.text_input("max_depth", "35")  # 10, None
rf_other_kwargs = st.text_input("Other RF kwargs")

n_estimators = textinput_to_list(n_estimators, ",")
n_estimators = [int(n) for n in n_estimators]
criterion = textinput_to_list(criterion, ",")
max_depth = textinput_to_list(max_depth, ",")
max_depth = [None if d == "None" else int(d) for d in max_depth]

rf_kwargs = {
    "estimator": model,
    "param_grid": {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
    },
    "refit": True,
    "metric": mean_squared_error,
}

sk_kwargs = [rf_kwargs]

st.markdown("#### Build Lasso-Clip-Elastic Net")

cutoff = st.number_input("LCEN cutoff", value=5e-2)
l1_ratio = st.text_input("l1_ratio", "0.01,0.1,0.5, 0.99")
l1_ratio = textinput_to_list(l1_ratio, ",")
l1_ratio = [float(l1) for l1 in l1_ratio]
Kfold = st.number_input("K_fold", value=3)
cv_method = st.text_input("cv_method", "Kfold")
alpha = st.number_input("alpha", value=5)
# alphas = np.concatenate(([0], np.logspace(-4.3, 0, alpha)))
# st.write('The values for alpha are: ', ','.join(['{a:.5f}'.format(a=a) for a in alphas]))

lcen_other_kwargs = st.text_input("Other LCEN kwargs")


degree = [1, 2]
trans_type = "poly"
lcen_interaction = False
n_jobs = -1
scale_X = True
scale_y = True

lcen_kwargs = {
    "l1_ratio": l1_ratio,
    "K_fold": Kfold,
    "cv_method": cv_method,
    "n_jobs": n_jobs,
    "alpha": int(alpha),
    "degree": degree,
    "trans_type": trans_type,
    "LCEN_cutoff": float(cutoff),
    "LCEN_interaction": lcen_interaction,
    "scale_X": scale_X,
    "scale_y": scale_y,
}


# st.markdown('#### Build Custom Sklearn Model')
# sk_kwargs.append([])

st.markdown("## 5. Run Cross Validation ! :D")

seed = st.number_input("Random Seed", 1, 100, 20, 1)
test_size = st.number_input("Test Set Size", 0.0, 1.0, 0.25)

data.random_seed = int(seed)

fit_sk = st.checkbox("Fit Random Forest")
fit_lcen = st.checkbox("Fit LCEN")

# model_names = []
# if fit_sk:
#     for sk in sk_kwargs:
#         model = sk['model']
#         model_names.append(model.__str__().split('()')[0])

# if fit_lcen:
#     model_names.append('LCEN')

# @st.cache_data


def fit(sk_kwargs, lcen_kwargs, yattr='nn'):
    Nm = 0
    if fit_sk:
        Nm = len(sk_kwargs)
    if fit_lcen:
        Nm += 1

    Nf = len(features_selected)

    progress_text = f"Fitting {Nm} model(s) to {Nf} feature(s)"
    my_bar = st.progress(0, text=progress_text)
    N = Nm * Nf
    completed = 0
    fitted_models = []
    for attr_name in features_selected:
        if fit_sk:
            for sk in sk_kwargs:
                data.do_sk(attr_name, yattr, test_size, None, **sk)
                completed += 1
                my_bar.progress(completed / N, text=progress_text)
                model_name = sk["estimator"].__str__().split("()")[0]
                fitted_models.append(attr_name + "_" + yattr + "_" + model_name)

        if fit_lcen:
            data.do_LCEN(attr_name, yattr, test_size, None, **lcen_kwargs)
            completed += 1
            my_bar.progress(completed / N, text=progress_text)
            model_name = "LCEN"
            fitted_models.append(attr_name + "_" + yattr + "_" + model_name)
    my_bar.empty()

    return pd.DataFrame(data.best_fits), fitted_models


@st.cache_resource
def get_models():
    fitted_models = {}
    for attr_name in features_selected:
        for model_name in model_names:
            full_name = attr_name + "_" + yattr + "_" + model_name
            model = getattr(data, full_name)
            fitted_models[full_name] = model

    return fitted_models


def fitted_callback():
    st.session_state["fitted"] = True


run_it = st.button("Run it!", False, on_click=fitted_callback())

if run_it:
    best_fit_df, fitted_models = fit(sk_kwargs, lcen_kwargs)
    st.success("Done!", icon="✅")

    fig = px.imshow(
        best_fit_df,
        text_auto=True,
        aspect="auto",
        title="Fitting Summary, Percent Error",
    )
    fig.update_traces(texttemplate="%{z:.5f}")
    st.plotly_chart(fig)

else:
    fitted_models = []

st.session_state["fitted_models"] = fitted_models
st.session_state["data"] = data
st.session_state["cols_to_featurize"] = cols_to_featurize


# Feature Importances

# Features Selected

# Plot it by attribute
