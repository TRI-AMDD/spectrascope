import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from spectrascope.plot_spectral_features import plot_model as pm 
import numpy as np 
from PIL import Image

st.set_page_config(
    page_title="Model Visualization",
    # page_icon="👋",
    layout="wide"
)


pretty_names = {'pointwise_features':'Raw Data', 
                'pca':'Principle Component Analysis', 
                'fft_features': 'Fourier Transform', 
                'cdf':'Cumulative Distribution Function', 
                'featureagg':'Agglomerative Clustering', 
                'cwt':'Continuous Wavelet Transform', 
                'polynomial_featuresall':'Polynomial Fitting', 
                'tsne':'T-distributed Stochastic Neighbor Embedding'}


data = st.session_state["data"]
cols_to_featurize = st.session_state["spec_to_featurize"]
fitted_models = st.session_state["fitted_models"]
yattr = st.session_state['yattr']
y = getattr(data, yattr)


st.markdown("### Visualizing Feature Selection and Importance")
st.markdown("LCEN : vertical lines on selected features")
st.markdown("Random Forests : Normalized feature importance for every feature")

model_name = st.selectbox("Select Model to Visualize", fitted_models)

if model_name == None: 
    st.stop()
    
elif 'trans' in model_name: 
    model = getattr(data, model_name)

    st.markdown('Visualization not implemented yet')
    import pandas as pd 
    selected_features = model['selected_features']
    selected_vals = model['selected_vals']
    temp = pd.DataFrame({'Selected Feature IDs':selected_features, 'Values': selected_vals})
    st.write(temp)
    st.stop()
     
fig = make_subplots(1,1, specs=[[{"secondary_y": True}]], subplot_titles=[model_name])

split_name = model_name.split('_')
spec = split_name[0]
raw_name = '_'.join(split_name[:-2])
feat = raw_name.replace(f'{spec}_', '')

xaxis = getattr(data, f'{spec}_xaxis')
raw = getattr(data, raw_name)
if len(raw[0].ravel()) > len(xaxis): 
    # multi modal 
    spec = '_'.join(split_name[:2])
    spec1, spec2 = spec.split('_')
    point1 = getattr(data, f"{spec1}_pointwise_features")
    point2 = getattr(data, f"{spec2}_pointwise_features")
    point_raw = np.concatenate([point1, point2], axis=1)
    xaxis = np.arange(point_raw.shape[1])
    feat = 'pointwise_features'
else: 
    point_raw = getattr(data, f'{spec}_pointwise_features')

model = getattr(data, model_name)

if 'RandomForest' in model_name: 
    try:
        feature_importances = model['P'].feature_importances_ 
    except AttributeError: 
        feature_importances = model['P'][0].best_estimator_.feature_importances_
    selected_vals = feature_importances / feature_importances.max()
else: 
    if st.session_state['scale_X']:
        selected_vals = model['selected_vals']
    else: 
        selected_vals = None 
    
selected_features = model['selected_features']

if 'xanes_pdf' in model_name: 
    print('hi')
    xaxis_label = 'Feature Index'
    yaxis_label = ''
elif 'pdf' in model_name: 
    xaxis_label = 'r'
    yaxis_label = 'G(r)'
elif 'xanes' in model_name: 
    xaxis_label = 'Energy (eV)'
    yaxis_label = 'Intensity'
else: 
    xaxis_label = ''
    yaxis_label = ''
    
# selected_features_arr = np.zeros_like(xaxis)
# selected_features_arr[selected_features] = 1 

kwargs_pca = {"feat":raw, 'mode':'markers', 'marker':dict(color=y, colorscale='Viridis', showscale=True)} #, colorbar=dict(len=0.3, x=0.3, y=0.5))}
model_kwargs = {
    'pca':kwargs_pca, 
    'featureagg':kwargs_pca,
    'tsne': kwargs_pca,
    'pointwise_features':{},
    'polynomial_featuresall':{},
    'cdf':{},
    'fft_features':{}
}
if 'pca' in model_name or 'featureagg' in model_name or 'tsne' in model_name or "polynomial" in model_name or 'fft' in model_name:
    if "polynomial" in model_name or 'fft' in model_name: 
        st.markdown('Visualization not implemented yet')
        import pandas as pd 
        try: 
            if 'LCEN' in model_name: 
                temp = pd.DataFrame({'Selected Feature IDs':selected_features, 'Values': selected_vals})
                st.write(temp)
            elif 'RandomForest' in model_name: 
                temp = pd.DataFrame({'Feature IDs':np.argsort(feature_importances)[::-1], 'Feature Importances': feature_importances[np.argsort(feature_importances)[::-1]]})
                st.write(temp)
        except: 
            pass 
        st.stop()
    else: 
        i = selected_features[0]
        j = selected_features[1]
        xaxis_label = f'Component {i}'
        yaxis_label = f'Component {j}'
        plotter = pm(selected_features=selected_features, selected_vals=selected_vals, fig=fig, xaxis=xaxis, raw=raw, xaxis_label=xaxis_label, yaxis_label=yaxis_label)
        plotter.show_scale = st.session_state['scale_X']
        plotter.__getattribute__(f'plot_{feat}').__call__(**model_kwargs[feat])
        fig.layout.annotations[0].update(text="Top 2 Components/Clusters")

else: 
    raw = point_raw.mean(axis=0)
    if "LCEN" in model_name: 
        plotter = pm(selected_features=selected_features, selected_vals=selected_vals, fig=fig, xaxis=xaxis, raw=raw, xaxis_label=xaxis_label, yaxis_label=yaxis_label)
        plotter.show_scale = st.session_state['scale_X']
        plotter.__getattribute__(f'plot_{feat}').__call__(**model_kwargs[feat])
    else: 
        plotter = pm(selected_features=feature_importances, selected_vals=selected_vals, fig=fig, xaxis=xaxis, raw=raw, xaxis_label=xaxis_label, yaxis_label=yaxis_label, plot_importances=True)
        plotter.show_scale = st.session_state['scale_X']
        plotter.__getattribute__(f'plot_{feat}').__call__(**model_kwargs[feat])


st.plotly_chart(fig)

st.stop()















def plot_model(full_name, fig, ir, ic, color="#d62728", plot_raw=True):
    raw_to_plot = []
    for col in cols_to_featurize:
        if col in full_name.split("_"):
            raw_to_plot.append(col)

    if len(raw_to_plot) > 1:
        multi_modal = True
        spec = f"{raw_to_plot[0]}_{raw_to_plot[1]}"
    else:
        multi_modal = False
        spec = raw_to_plot[0]

    feat = full_name.split(f"{spec}_")[1].split("_")[0]

    # ncols = len(raw_to_plot)
    # fig = make_subplots(rows=1, cols=ncols, subplot_titles=raw_to_plot, specs=[[{"secondary_y": True}]])

    if plot_raw:
        plot_raw_spec(fig, spec, ir, ic)
    if not multi_modal:
        x = getattr(data, f"{spec}_xaxis")
    else:
        x = np.arange(400)

    if "RandomForest" in full_name:
        model_dict = getattr(data, full_name)
        P = model_dict["P"]
        score = model_dict["score"]
        ypred = model_dict["ypred"]
        
        # ytest = getattr(data, f'{yattr}_test')
        ytest = data.nn_test
        perc_err = np.mean(np.abs((ytest - ypred) / ytest)) * 100
        importances = P[0].best_estimator_.feature_importances_

        fig.add_trace(
            go.Scatter(
                x=x,
                y=importances / np.max(importances),
                name=f"RF, {feat}, mean err:{perc_err:.5f} % ",
                line=dict(color=color),
            ),
            secondary_y=True,
            row=ir,
            col=ic,
        )
        # fig.update_yaxes(title='Feature Importance', secondary_y=True)

    elif "LCEN" in full_name:
        model_dict = getattr(data, full_name)
        params = model_dict["model_params"]
        num_selected = len(params)
        par_names = np.array(params)[:, 0]
        par_locs = np.array(
            [
                p.split("^")[0].replace("x", "") if p != "intercept" else "intercept"
                for p in par_names
            ]
        )
        par_vals = np.array(params)[:, 1].astype("float64")
        par_vals_abs = np.abs(par_vals)
        importance = np.argsort(par_vals_abs)[::-1]

        selected_legend = 0
        score = model_dict["MSE_test_nontrans"]
        ypred = model_dict["yhat_test_nontrans"]

        ytest = getattr(data, f'{yattr}_test')
        perc_err = np.mean(np.abs((ytest - ypred) / ytest)) * 100

        for loc, val in zip(par_locs[importance][:10], par_vals[importance][:10]):
            if loc == "intercept":
                pass
            else:
                loc = int(loc)
                if selected_legend == 0:

                    fig.add_trace(
                        go.Scatter(
                            x=[x[loc], x[loc]],
                            y=[0, 1],
                            mode="lines",
                            line=dict(color=color),
                            name=f"LCEN, {feat}, mean err:{perc_err:.5f} %",
                            legendgroup=f"LCEN_{full_name}",
                        ),
                        row=ir,
                        col=ic,
                        secondary_y=True,
                    )
                    selected_legend += 1
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[x[loc], x[loc]],
                            y=[0, 1],
                            mode="lines",
                            line=dict(color=color),
                            showlegend=False,
                            legendgroup=f"LCEN_{full_name}",
                        ),
                        row=ir,
                        col=ic,
                        secondary_y=True,
                    )

            fig.update_yaxes(title="", secondary_y=True)

if len(cols_to_featurize) > 1:
    col1, col2 = cols_to_featurize
else:
    col1 = cols_to_featurize[0]
    col2 = "nope"

if any([col2 in f for f in fitted_models]) and any([col1 in f for f in fitted_models]):
    # used two modalities; might not have combined them but need two subplots
    multi_modality = True
    single_modality = False
    fig1 = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=(col1.upper(), col2.upper()),
    )

    fig2 = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=(col1.upper() + "+" + col2.upper(),),
    )
else:
    # only used one modality, only plot 1 subplot
    single_modality = True
    if any([col2 in f for f in fitted_models]):
        title = col2
    else:
        title = col1
    fig1 = make_subplots(
        rows=1, cols=1, specs=[[{"secondary_y": True}]], subplot_titles=(title,)
    )


colors = px.colors.qualitative.Dark24
plot_xanes = True
plot_xanes_pdf = True
plot_pdf = True
for i, full_name in enumerate(fitted_models):
    print(full_name)
    if f"{col1}_{col2}" in full_name:
        ir = 1
        ic = 1
        plot_model(full_name, fig2, ir, ic, color=colors[i], plot_raw=False)
        if plot_xanes_pdf:
            y1 = getattr(data, f"{col1}_pointwise_features")
            y2 = getattr(data, f"{col2}_pointwise_features")
            fig2.add_trace(
                go.Scatter(
                    x=np.arange(200),
                    y=y1.mean(axis=0),
                    mode="lines",
                    opacity=0.5,
                    name="average xanes",
                    showlegend=False,
                    line=dict(color="#7f7f7f"),
                ),
                row=ir,
                col=ic,
            )
            fig2.add_trace(
                go.Scatter(
                    x=np.arange(201, 400),
                    y=y2.mean(axis=0),
                    mode="lines",
                    opacity=0.5,
                    name="average pdf",
                    showlegend=False,
                    line=dict(color="#7f7f7f"),
                ),
                row=ir,
                col=ic,
            )
        plot_xanes_pdf = False
    else:
        if col1 in full_name and col2 not in full_name:
            ir = 1
            ic = 1
            plot_model(full_name, fig1, ir, ic, color=colors[i], plot_raw=plot_xanes)
            plot_xanes = False

        elif col2 in full_name and col1 not in full_name:
            ir = 1
            ic = 2
            if single_modality:
                ic = 1
            plot_model(full_name, fig1, ir, ic, color=colors[i], plot_raw=plot_pdf)
            plot_pdf = False

#     st.plotly_chart(fig)

# fig = make_subplots(rows=1, cols=ncols, subplot_titles=raw_to_plot, specs=[[{"secondary_y": True}]])


fig1.update_layout(legend=dict(font=dict(size=20)))
st.plotly_chart(fig1)

if not single_modality:
    fig2.update_layout(legend=dict(font=dict(size=20)))
    st.plotly_chart(fig2)
