import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


data = st.session_state["data"]
cols_to_featurize = st.session_state["cols_to_featurize"]
fitted_models = st.session_state["fitted_models"]


def plot_raw_spec(fig, col, ir, ic):
    x = getattr(data, f"x_{col}")

    if not hasattr(data, f"{col}_pointwise_features"):
        data.add_feature_json(x, "pointwise_features", convcol=col, xrange=x)
    y = getattr(data, f"{col}_pointwise_features")
    for yi in y:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=yi,
                mode="lines",
                opacity=0.1,
                showlegend=False,
                line=dict(color="#7f7f7f"),
            ),
            row=ir,
            col=ic,
        )

    xlabel = getattr(data, f"{col}_labelx")
    ylabel = getattr(data, f"{col}_labely")
    fig.add_trace(
        go.Scatter(x=x, y=y.mean(axis=0), mode="lines", name="average"), row=ir, col=ic
    )

    fig.update_xaxes(title_text=xlabel, row=ir, col=ic)
    fig.update_yaxes(title_text=ylabel, row=ir, col=ic)


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
        x = getattr(data, f"x_{spec}")
    else:
        x = np.arange(400)

    if "RandomForest" in full_name:
        model_dict = getattr(data, full_name)
        P = model_dict["P"]
        score = model_dict["score"]
        ypred = model_dict["ypred"]

        ytest = data.BL_mean_test
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

        ytest = data.BL_mean_test
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


st.markdown("### Visualizing Feature Selection and Importance")
st.markdown("LCEN : vertical lines on selected features")
st.markdown("RF : Normalized feature importance for every feature")

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
