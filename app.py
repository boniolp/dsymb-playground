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

import streamlit as st
from utils import *


def run():
    st.set_page_config(
        page_title="dsymb-playground",
    )

    if "ALL_TS" not in st.session_state:
        st.session_state["ALL_TS"] = []

    st.write("# Welcome to $d_{symb}$ playground!")
    st.markdown(
        """
        :technologist: :zap: Swiftly interpret and compare your multivariate time series dataset
        using $d_{symb}$.
        $d_{symb}$ transforms a multivariate time series into an interpretable
        symbolic sequence, and comes with an efficient distance
        measure defined on the obtained symbolic sequences.
        1. Use the `Explore` tab to interpret the $d_{symb}$ symbolization.
        Visualize your raw time series along with their $d_{symb}$
        symbolization using the colorbars.
        With a single glance at the color bars, the symbolization provides an
        immediate and comprehensive understanding of your data.
        You can also visualize the $d_{symb}$ pairwise distance matrix
        between the symbolic sequences.
        2. Use the `Benchmark` tab to assess the relevance of the $d_{symb}$
        distance measure, with regards to 9 other distance measures, on the
        JIGSAWS dataset.
        In particular, $d_{symb}$ is much faster than existing methods.
        For computational reasons of the benchmark, all results are precomputed.
        """
    )

    tab_explore, tab_benchmark, tab_about = st.tabs(
        ["Explore", "Benchmark", "About"]
    )

    with tab_explore:
        run_explore_frame()

    with tab_benchmark:
        run_benchmark_frame()

    with tab_about:
        run_about_frame()


if __name__ == "__main__":
    run()
