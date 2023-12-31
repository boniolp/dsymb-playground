# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import gc
import streamlit as st

from constant import *

from streamlit_plotly_events import plotly_events

from dsymb import *


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def preprocess_data(uploaded_ts):
    with st.spinner("Preprocessing your dataset..."):
        all_ts = []
        for ts in uploaded_ts:
            all_ts.append(np.genfromtxt(ts, delimiter=","))
    return all_ts
    gc.collect()


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def plot_matrix(distance_arr, distance_name=""):
    fig = px.imshow(
        distance_arr,
        aspect="auto",
        title=f"Pairwise distance matrix between time series using {distance_name}",
    )
    fig.update_xaxes(title_text="Times series index")
    fig.update_yaxes(title_text="Times series index")
    return fig


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def plot_symbolization(df_temp, mode, n_symbols):
    tmp_df = df_temp
    if n_symbols <= 10:
        plotly_colors = px.colors.qualitative.G10
    else:
        plotly_colors = px.colors.qualitative.Alphabet
    n_signals = tmp_df["signal_index"].nunique()
    tmp_df = tmp_df.rename(
        columns={
            "segment_start": "Start",
            "segment_end": "Finish",
            "signal_index": "Task",
        }
    )
    tmp_df["segment_symbol"] = tmp_df["segment_symbol"].apply(str)
    tmp_df["Task"] = tmp_df["Task"].apply(str)

    if mode == "Normalized":
        all_max_length = []
        for i in range(len(tmp_df)):
            sig_index = tmp_df["Task"].values[i]
            max_length = max(
                tmp_df.loc[tmp_df["Task"] == sig_index]["Finish"].values
            )
            all_max_length.append(max_length)
        tmp_df["max"] = all_max_length
        tmp_df["Start"] = tmp_df["Start"] / tmp_df["max"]
        tmp_df["Finish"] = tmp_df["Finish"] / tmp_df["max"]
    dict_symbol_color = {
        key: plotly_colors[int(key)]
        for key in sorted(tmp_df["segment_symbol"].unique().tolist())
    }
    fig = ff.create_gantt(
        tmp_df,
        index_col="segment_symbol",
        bar_width=0.45,
        show_colorbar=True,
        showgrid_x=True,
        group_tasks=True,
        colors=dict_symbol_color,
    )
    # Adding the contour to each symbol
    fig.update_traces(
        mode="lines", line_color="black", selector=dict(fill="toself")
    )
    fig.update_xaxes(type="linear")
    for trace in fig.data:
        print(trace)
    # Update the layout
    fig.update_layout(
        xaxis_type="linear",
        height=max(400, n_signals * 20),
        xaxis_title="Time stamp",
        yaxis_title="Symbolic sequence index",
        title_text="Colorbars of all symbolic sequences in the dataset",
        legend=dict(title=dict(text="Symbol")),
    )
    return fig


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def plot_symbol_distr(df_temp, mode, n_symbols):
    """Plot the distribution of each symbol over time."""
    tmp_df = df_temp.sort_values(by=["segment_symbol"])
    tmp_df = tmp_df.rename(
        columns={
            "segment_start": "Start",
            "segment_end": "Finish",
            "signal_index": "Task",
        }
    )
    tmp_df["segment_symbol"] = tmp_df["segment_symbol"].apply(str)
    tmp_df["Task"] = tmp_df["Task"].apply(str)

    all_max_length = []
    for i in range(len(tmp_df)):
        sig_index = tmp_df["Task"].values[i]
        max_length = max(
            tmp_df.loc[tmp_df["Task"] == sig_index]["Finish"].values
        )
        all_max_length.append(max_length)

    bin_size = 0.01 * max(all_max_length)

    if mode == "Normalized":
        bin_size = 0.01
        tmp_df["max"] = all_max_length
        tmp_df["Start"] = tmp_df["Start"] / tmp_df["max"]
        tmp_df["Finish"] = tmp_df["Finish"] / tmp_df["max"]

    if n_symbols <= 10:
        plotly_colors = px.colors.qualitative.G10
    else:
        plotly_colors = px.colors.qualitative.Alphabet

    list_of_actual_symbols_str = tmp_df["segment_symbol"].unique().tolist()
    list_of_actual_symbols_int = sorted(
        [int(elem) for elem in list_of_actual_symbols_str]
    )
    list_of_actual_symbols_str_sorted = [
        str(elem) for elem in list_of_actual_symbols_int
    ]
    fig = make_subplots(
        rows=len(list_of_actual_symbols_str_sorted),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"Symbol {i}" for i in list_of_actual_symbols_str_sorted
        ],
    )

    for i, symbol in enumerate(list_of_actual_symbols_str_sorted):
        pos_symb = 0.5 * np.array(
            tmp_df.loc[tmp_df["segment_symbol"] == symbol]["Finish"].values
            + tmp_df.loc[tmp_df["segment_symbol"] == symbol]["Start"].values
        )
        fig_symb = ff.create_distplot(
            [pos_symb],
            group_labels=["{}".format(symbol)],
            show_hist=True,
            colors=[plotly_colors[int(symbol)]],
            bin_size=bin_size,
            show_curve=False,
            show_rug=False,
        )
        for trace in fig_symb.data:
            fig.add_trace(trace, row=1 + i, col=1)

    fig.update_xaxes(title_text="Time stamp", row=n_symbols, col=1)
    for i in range(0, len(list_of_actual_symbols_str_sorted)):
        fig.update_yaxes(
            title_text="Occurrence", showticklabels=False, row=i + 1, col=1
        )
    fig.update_layout(
        xaxis_type="linear",
        height=max(300, n_symbols * 120),
        title_text="Symbols' occurrence over time",
        legend=dict(title=dict(text="Symbol")),
    )
    return fig


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def plot_dendrogram(centroids):
    fig = ff.create_dendrogram(centroids)
    fig.update_layout(
        xaxis_title="Symbol",
        yaxis_title="Distance between symbols",
        title="Dendrogram",
        margin=dict(r=10),
    )
    return fig


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def plot_symb_hist(df_temp, n_symbols):
    if n_symbols <= 10:
        plotly_colors = px.colors.qualitative.G10
    else:
        plotly_colors = px.colors.qualitative.Alphabet
    fig = px.histogram(
        df_temp,
        x="segment_symbol",
        color="segment_symbol",
        category_orders=dict(segment_symbol=list(np.arange(0, n_symbols))),
        labels={"segment_symbol": "Symbol"},
        color_discrete_sequence=plotly_colors,
    )
    fig.update_layout(
        xaxis_title="Symbol",
        yaxis_title="Count",
        title="Histogram",
        bargap=0.2,
        height=300,
    )
    fig.update_xaxes(type="category")
    return fig


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def plot_symb_len(df_temp, n_symbols):
    if n_symbols <= 10:
        plotly_colors = px.colors.qualitative.G10
    else:
        plotly_colors = px.colors.qualitative.Alphabet
    fig = px.histogram(
        df_temp,
        x="segment_length",
        color="segment_symbol",
        category_orders=dict(segment_symbol=list(np.arange(0, n_symbols))),
        labels={"segment_symbol": "Symbol"},
        color_discrete_sequence=plotly_colors,
    )
    fig.update_layout(
        xaxis_title="Segment length",
        yaxis_title="Count",
        title="Histogram",
        bargap=0.2,
        height=400,
    )
    return fig


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def plot_cluster_centers(centroids, n_symbols):
    if n_symbols <= 10:
        plotly_colors = px.colors.qualitative.G10
    else:
        plotly_colors = px.colors.qualitative.Alphabet
    fig = go.Figure()
    for dim in list(np.arange(0, centroids.shape[0])):
        series_plot = centroids[dim, :]
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, len(series_plot)),
                y=series_plot,
                line=dict(color=plotly_colors[dim]),
                mode="lines+markers",
                name=f"{dim}",
            )
        )
    fig.update_layout(
        title="Plotting each centroid (corresponding to a symbol) along the dimensions",
        legend=dict(title=dict(text="Symbol")),
        xaxis_title="Dimension",
        yaxis_title="Amplitude",
    )
    return fig


def plot_time_series(ts, tmp_df, n_symbols, dims=[0, 20]):
    """Plot the symbolic sequence as well as the multivariate
    time series.
    """

    # Get the number of dimensions
    n_dim = dims[1] - dims[0]

    # Define the colors for the symbols
    if n_symbols <= 10:
        plotly_colors = px.colors.qualitative.G10
    else:
        plotly_colors = px.colors.qualitative.Alphabet

    # Transform the data frame to make it compatible with Gantt charts
    tmp_df = tmp_df.rename(
        columns={
            "segment_start": "Start",
            "segment_end": "Finish",
            "signal_index": "Task",
        }
    )
    tmp_df["segment_symbol"] = tmp_df["segment_symbol"].apply(str)
    tmp_df["Task"] = tmp_df["Task"].apply(str)
    dict_symbol_color = {
        key: plotly_colors[int(key)]
        for key in sorted(tmp_df["segment_symbol"].unique().tolist())
    }

    # Create the Gantt chart for the symbolic sequence
    fig_symb = ff.create_gantt(
        tmp_df,
        index_col="segment_symbol",
        bar_width=0.4,
        showgrid_x=True,
        show_colorbar=True,
        group_tasks=True,
        colors=dict_symbol_color,
    )
    # Add the contour
    fig_symb.update_traces(
        mode="lines", line_color="black", selector=dict(fill="toself")
    )
    fig_symb.update_xaxes(type="linear")
    for trace in fig_symb.data:
        print(trace)

    # Create the subplots
    fig = make_subplots(
        rows=n_dim + 2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Symbolic sequence (for all dimensions)",
            "",
            "Multivariate time series, dimension by dimension",
        ),
    )

    # Add the symbolic sequence
    for trace in fig_symb.data:
        fig.add_trace(trace, row=1, col=1)

    # Add each dimension
    for i_row, i in enumerate(range(dims[0], dims[1])):
        fig.add_trace(
            go.Scattergl(
                x=list(range(len(ts))),
                y=ts[:, i],
                mode="lines",
                line=dict(color="white", width=1),  # white or black
            ),
            row=i_row + 3,
            col=1,
        )

    # Layout
    fig.update_xaxes(title_text="Time stamp", row=n_dim + 2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_layout(
        xaxis_type="linear",
        height=max(800, n_dim * 52),
        showlegend=False,
        title="Chosen multivariate time series along with its symbolic sequence.",
    )

    st.plotly_chart(fig, use_container_width=True)
    del fig, fig_symb
    gc.collect()


def get_data_step():
    uploaded_ts = st.file_uploader(
        "Upload your time series:", accept_multiple_files=True
    )
    if len(uploaded_ts) >= 1:
        try:
            st.session_state.ALL_TS = preprocess_data(uploaded_ts)
        except Exception as e:
            st.error(
                "An error occured while processing the files. Please check if the time series have the correct format (n_timestamps, n_dims). The exception is the following: {}",
                icon="🚨",
            )


def Visualize_step():
    if len(st.session_state.ALL_TS) >= 1:
        n_symbols = st.slider(
            "Select the number of symbols to represent your time series using $d_{symb}$:",
            2,
            25,
            5,
        )

        st.markdown(
            """
        	Then, use the `Individual analysis` tab to visualize your chosen raw
            multivariate time series along with its symbolic
            representation.
        	Use the `Dataset analysis` tab to explore and interpret
            your dataset with only one glance using the $d_{symb}$ colorbars.
        	It also provides some insights on your symbolization to help you
        	interpret a symbol as a real-world event in your data.
        	"""
        )

        D1, df_temp, lookup_table, centroids = dsym(
            st.session_state.ALL_TS, n_symbols
        )
        tab_indiv, tab_all = st.tabs(
            ["Individual analysis", "Dataset analysis"]
        )
        with tab_indiv:
            col1, col2 = st.columns(2)
            with col1:
                time_series_selected = st.selectbox(
                    "Choose the index of a single time series :",
                    list(range(len(st.session_state.ALL_TS))),
                )
                range_dims = [
                    [20 * dim_s, 20 * (dim_s + 1)]
                    for dim_s in range(
                        len(st.session_state.ALL_TS[time_series_selected][0])
                        // 20
                    )
                ]
                if len(range_dims) > 0:
                    if range_dims[-1][1] < len(
                        st.session_state.ALL_TS[time_series_selected][0]
                    ):
                        range_dims += [
                            [
                                range_dims[-1][1],
                                len(
                                    st.session_state.ALL_TS[
                                        time_series_selected
                                    ][0]
                                ),
                            ]
                        ]
                range_dims += [
                    [0, len(st.session_state.ALL_TS[time_series_selected][0])]
                ]
            with col2:
                dims = st.selectbox(
                    "Choose the dimensions' range (for conciseness purposes):",
                    range_dims,
                )
            plot_time_series(
                ts=st.session_state.ALL_TS[time_series_selected],
                tmp_df=df_temp.loc[
                    df_temp["signal_index"] == time_series_selected
                ],
                dims=dims,
                n_symbols=n_symbols,
            )

        with tab_all:
            mode = st.radio(
                "Mode",
                ["Colorbars", "Distance matrix"],
                captions=[
                    "Visualize all the symbolized time series using colobars.",
                    "Visualize the distance matrix based on $d_{symb}$.",
                ],
                horizontal=True,
            )
            if mode == "Colorbars":
                mode_length = st.radio(
                    "Length",
                    ["True", "Normalized"],
                    captions=[
                        "Use the true time series' lengths.",
                        "Normalize the lengths between 0 and 1.",
                    ],
                    horizontal=True,
                )

                st.markdown("### Overview of your symbolized dataset")

                fig = plot_symbolization(
                    df_temp, mode=mode_length, n_symbols=n_symbols
                )
                st.markdown(
                    """
                    First of all, let us visualize the whole dataset of
                    multivariate time series at once.
                    In the following color bars, each row is a color bar
                    corresponding to the symbolic sequence of a
                    multivariate time series.
                    """
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    """
                    ### Insights on the symbolization of your dataset
                    
                    Now, let us provide some insights on your symbolization to
                    help you interpret a symbol as a real-world event in your data.
                    """
                )
                st.markdown(
                    """
                    Which symbols occur more often (out of the whole dataset)?
                    Which symbols represent more recurring events?
                    """
                )
                fig_symb_hist = plot_symb_hist(
                    df_temp=df_temp, n_symbols=n_symbols
                )
                st.plotly_chart(fig_symb_hist, use_container_width=True)

                st.markdown(
                    """
                    Which symbols have the shortest / longest duration?
                    Which symbols encode short / long events?
                    """
                )
                fig_symb_len = plot_symb_len(
                    df_temp=df_temp, n_symbols=n_symbols
                )
                st.plotly_chart(fig_symb_len, use_container_width=True)

                fig_dist = plot_symbol_distr(
                    df_temp, mode=mode_length, n_symbols=n_symbols
                )
                st.markdown(
                    """
                    Let us investigate around which time stamp each symbol
                    occurs most.
                    For each occurrence of a symbol, the position taken is the
                    middle of each segment.
                    For example, one could notice that a specific symbol occurs
                    mostly at the beginning of a time series.
                    """
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                st.markdown(
                    """
                    Let us visualize the centroids that are characteristic of
                	each symbol.
                    Are there centroids that look alike / afar?
                    What about the range of their amplitudes?
                    """
                )
                fig_centroids = plot_cluster_centers(
                    centroids=centroids, n_symbols=n_symbols
                )
                st.plotly_chart(fig_centroids, use_container_width=True)

                fig_dendrogam = plot_dendrogram(centroids)
                st.markdown(
                    """
                    Now, let us quantify our previous intuitions about the centroids.
                	We investigate the distance between the individual symbols
                    and how they are grouped according to hierarchical clustering
                    thanks to a dendrogram.
                    Each symbol is represented by its corresponding centroid
                    obtained using K-Means.
                    The distance between the individual symbols is the Euclidean
                    distance between their centroids.
                	Note: it is not the distance between time series, but between
                    individual symbols.
                    Do symbols with a "small distance", thus considered "close",
                    effectively look alike in in the above centroids plot?
                    """
                )
                st.plotly_chart(fig_dendrogam, use_container_width=True)
            elif mode == "Distance matrix":
                fig = plot_matrix(D1, distance_name="d_symb")
                st.plotly_chart(fig, use_container_width=True)


def run_explore_frame():
    st.markdown("## Explore your dataset with the $d_{symb}$ symbolization")
    st.markdown(
        """
        Upload your dataset of (multivariate) time series: each time series
        must be in a `.csv` file with the shape `(n_timestamps, n_dim)`.
        For example, a preprocessed version of the JIGSAWS dataset
        (described in the `Benchmark` tab) can be found
        [here](https://kiwi.cmla.ens-cachan.fr/index.php/s/ctEdTsz6sxPBxxX).
        """
    )

    get_data_step()

    Visualize_step()

    gc.collect()


def run_benchmark_frame():
    st.markdown(compare_text_1)

    tab_data_desc, tab_baseline_desc = st.tabs(
        ["Dataset description", "Baselines description"]
    )

    with tab_data_desc:
        st.markdown(data_JIGSAW)

    with tab_baseline_desc:
        st.markdown(Baseline_desc)

    st.markdown(compare_text_2)

    df_exectime = pd.read_csv("data/eval/exectime.csv", index_col=0)
    df_acc = pd.read_csv("data/eval/accuracy.csv", index_col=0)
    d_replace_eval = {
        "rand_score": "Rand score",
        "Adjusted_rand_score": "Ajusted rand score",
        "Adjusted_Mutual_Information": "Adjusted mutual information",
        "Normalized_Multual_Information": "Normalized mutual information",
        "homogeneity": "Homogeneity",
        "Completeness": "Completeness",
        "V_measure": "V-measure",
        "Fowlkes_Mallows": "Fowlkes-Mallows index",
    }
    d_replace_distance = {
        "dtw_dep": "DTW dependent",
        "ddtw_dep": "DDTW dependent",
        "wdtw_dep": "WDTW dependent",
        "wddtw_dep": "WDDTW dependent",
        "lcss": "LCSS",
        "edr": "EDR",
        "erp": "ERP",
        "msm": "MSM",
        "twe": "TWE",
        "dsymb": "d_symb",
    }
    d_replace_distance_inv = dict()
    for key, value in d_replace_distance.items():
        d_replace_distance_inv[value] = key
    list_distances = list(d_replace_distance_inv.keys())

    dist_name = st.selectbox(
        "Choose a distance measure to investigate:", list_distances
    )
    st.markdown("*Note*: For $d_{symb}$, the number of symbols is fixed to 5.")
    dist_abb = d_replace_distance_inv[dist_name]

    fig_mat = plot_matrix(
        pd.read_csv(
            "data/simMatrix/{}_DistanceMatrix.csv".format(dist_abb), index_col=0
        ).to_numpy(),
        distance_name=dist_name,
    )

    st.plotly_chart(fig_mat, use_container_width=True)

    st.markdown("#### Explore the clustering performances")

    df_exectime_plot = df_exectime.T.rename(columns=d_replace_distance).T
    fig_time = px.bar(
        df_exectime_plot,
        labels={"distance": "distance measure", "value": "execution time"},
    )
    fig_time.update_xaxes(
        categoryorder="array",
        categoryarray=list_distances,
    )
    fig_time.update_yaxes(
        type="log",
    )
    fig_time.update_layout(
        xaxis_title="Distance measure",
        yaxis_title="Clustering execution time (in seconds)",
        showlegend=False,
    )
    fig_time.update_xaxes(
        categoryorder="array",
        categoryarray=list(d_replace_distance_inv.keys()),
    )

    st.plotly_chart(fig_time, use_container_width=True)

    fig_acc = px.bar(
        df_acc.rename(columns=d_replace_distance).T.rename(
            columns=d_replace_eval
        ),
        barmode="group",
    )
    fig_acc.update_layout(
        xaxis_title="Distance measure",
        yaxis_title="Clustering performance evaluation",
        legend=dict(title=dict(text="Evaluation metric")),
    )
    fig_acc.update_xaxes(
        categoryorder="array",
        categoryarray=list_distances,
    )
    st.plotly_chart(fig_acc, use_container_width=True)


def run_about_frame():
    st.markdown(about_text)
    st.image(
        "figures/cebo_logos.png", caption="Centre Borelli and its affiliations."
    )
