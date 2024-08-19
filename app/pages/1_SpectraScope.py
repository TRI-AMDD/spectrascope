import streamlit as st

from spectrascope.ss_classification import *
from spectrascope.plot_spectral_features import plot_model as pm 
from spectrascope.plot_spectral_features import plot_spectral_features as psf 

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import product


st.set_page_config(
    page_title="Feature Generation and Model Training",
    # page_icon="👋",
    layout="wide"
)


pretty_names = {'Raw Data':'pointwise_features', 
                'Principle Component Analysis':'pca', 
                'Fourier Transform':'fft_features', 
                'Cumulative Distribution Function':'cdf', 
                'Agglomerative Clustering':'featureagg', 
                'Continuous Wavelet Transform':'cwt', 
                'Polynomial Fitting':'polynomial_featuresall', 
                'T-distributed Stochastic Neighbor Embedding':'tsne'}


def textinput_to_list(string, delim):
    return string.replace(" ", "").split(delim)

data  = st.session_state["data"]
yattr = st.session_state["yattr"]
y = getattr(data, yattr)

col1, col2 = st.columns(2)

st.markdown("## 2. Choose Spectra to Featurize")
spec_to_featurize = st.multiselect("Input column(s) to featurize", data.specnames)

st.session_state["fitted_models"] = []

kwargs_dict = { 'pca' : {'mode':'markers', 'marker':dict(color=y, colorscale='Viridis')}, #, showscale=True, colorbar=dict(len=0.3, x=0.3, y=0.5))}
    'cwt' : {'colorscale':'Viridis','showscale':False}, #,'colorbar':dict(len=0.3, x=1, y=0.85)}
    'cdf' : {},
    'polynomial_featuresall' : {},
    'featureagg' : {'mode':'markers', 'marker':dict(color=y, colorscale='Viridis')}, #, showscale=True, colorbar=dict(len=0.3, x=0.6, y=0.5))}
    'fft_features' : {},
    'pointwise_features':{},
    'tsne':{'mode':'markers', 'marker':dict(color=y, colorscale='Viridis')}
    }

xaxis_pdf = 'r'
xaxis_xanes = 'Energy (eV)'
yaxis_pdf = 'G(r)'
yaxis_xanes = 'Intensity (arb)'

axis_dict = {'xanes':(xaxis_xanes, yaxis_xanes), 'pdf':(xaxis_pdf, yaxis_pdf)}

features_available = list(kwargs_dict.keys())

pretty_features = st.multiselect("Visualize Features", list(pretty_names.keys()))
features = [pretty_names[f] for f in pretty_features]


nfeats = len(features)

if nfeats > 0 : 
    for spec in spec_to_featurize: 
        fig = make_subplots(
                rows=1, cols=nfeats, subplot_titles=pretty_features, horizontal_spacing=0.15
            )
        for i, feat in enumerate(features): 
            kwargs = kwargs_dict[feat]
            if 'fft' in feat or 'pointwise' in feat: 
                data.add_spectral_feature(spec, feat, xrange=getattr(data, f'{spec}_xaxis'))
                if 'pointwise' in feat: 
                    xaxis_label, yaxis_label = axis_dict[spec] 
                else: 
                    xaxis_label = 'Frequency'
                    yaxis_label = 'FFT'
            else: 
                data.add_spectral_feature(spec, feat)
                if 'cdf' in feat: 
                    xaxis_label, yaxis_label = axis_dict[spec] 
                    yaxis_label = 'Cumulative Intensity'
                elif 'polynomial' in feat: 
                    xaxis_label = 'Feature Index'
                    yaxis_label = 'Polynomial Coefficientr'
                
                else: 
                    xaxis_label = ''
                    yaxis_label = ''
                
            arr = data.__getattribute__(spec + '_'+ feat)
            if feat == 'pca' or feat == 'featureagg' or feat =='tsne': 
                xaxis_label = 'Component 1'
                yaxis_label = 'Component 2'
            else: 
                arr = arr[0].ravel()
                
            xaxis = getattr(data, f'{spec}_xaxis')
            temp = psf(arr, xaxis=xaxis, fig=fig, row=1, col=i + 1, xaxis_label=xaxis_label, yaxis_label=yaxis_label)    
            getattr(temp, f"plot_{feat}").__call__(showlegend=False, **kwargs)

        st.plotly_chart(fig)

features.append("pointwise_features")

st.markdown("## 3. Choose Features to Build Model On")

all_features = []
for spec in spec_to_featurize:
    for feat in features:
        all_features.append(spec + "_" + feat)

features_selected = st.multiselect("Select from Features Computed", all_features)

with st.spinner(text="featurizing....(say that 10 times fast)"):
    for spec in spec_to_featurize:
        for feat in features_selected:
            x = getattr(data, f"{spec}_xaxis")
            if not hasattr(data, feat):
                data.add_spectral_feature(spec, feat.replace(f"{spec}_", ""))

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
                data.combine_feature(feat1, feat2)
                
            features_selected.append(attr_name)

st.write("#### Would you like to apply further transformations to any of these features?")
do_transform = st.toggle("yes", False)

if do_transform: 
    trans_type = st.radio("Transformation Type", ["all", "poly", "simple_interaction"])
    degree = st.radio(r"Degree of polynomial transformation ($x^2, x^3$)", [1,2,3])
    interaction = st.toggle(r'Feature Interaction ($x_1 * x_2 ..$)', False)
    
    if trans_type == "simple_interaction": 
        interaction = True

    transform_selected = st.multiselect(
        "Choose features to transform",
        features_selected,
    )
    for transform in transform_selected: 
        additional_naming = f"_{degree}_{interaction}_{trans_type}"
        data.transform_feature(transform, degree=degree, interaction=interaction, trans_type=trans_type, additional_naming=additional_naming)
        features_selected.append(transform + "_trans" + additional_naming)
    
st.write(f"#### Final Features selected for model building:") 
for feat in features_selected: 
    st.write(feat)

st.markdown("## 4. Train Models")

cross_validate = st.radio("Choose Training Method", ["Default Models", "Tune Hyperparameters"])
cross_validate = {"Default Models":False, "Tune Hyperparameters":True}[cross_validate]

if not cross_validate:
    rf_kwargs = {"estimator":RandomForestRegressor()}
    lcen_kwargs = lcen_kwargs = {'l1_ratio':[0.1, 0.5, 0.9],
               'K_fold':3, 
               'cv_method':'Kfold', 
               'alpha':5, 
               'degree':[1],
               'trans_type':'poly', 
               'LCEN_cutoff':4e-3, 
               'LCEN_interaction':False, 
               'scale_X':True, 
               'scale_y':True}
    
elif cross_validate: 

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

sk_kwargs = [rf_kwargs]
st.markdown("## 5. Run!")

seed = st.number_input("Random Seed", 1, 100, 20, 1)
test_size = st.number_input("Test Set Size", 0.0, 1.0, 0.25)

data.random_seed = int(seed)

scale_X = st.checkbox("Scale X")
fit_sk = st.checkbox("Fit Random Forest")
fit_lcen = st.checkbox("Fit LCEN")

model_names = []


def fit(sk_kwargs, lcen_kwargs, test_size, scale_X, yattr):
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
                data.do_sk(attr_name, yattr, test_size, use_pipeline=cross_validate, scale_X=scale_X, **sk)
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


@st.cache_data
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
    print(test_size)
    best_fit_df, fitted_models = fit(sk_kwargs, lcen_kwargs, test_size=test_size, scale_X=scale_X, yattr=yattr)
    st.success("Done!", icon="✅")

    fig = px.imshow(
        best_fit_df,
        text_auto=True,
        aspect="auto",
        title="Fitting Summary, MSE",
    )
    fig.update_traces(texttemplate="%{z:.5f}")
    st.plotly_chart(fig)

else:
    fitted_models = []

st.session_state["fitted_models"] = fitted_models
st.session_state["data"] = data
st.session_state["spec_to_featurize"] = spec_to_featurize



