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
        page_title="d_symb",
    )

    if "ALL_TS" not in st.session_state:
        st.session_state["ALL_TS"] = []

    st.write("# Welcome to Symbol!")
    st.markdown(
        "Explore and interpret your multivariate time series data set using"
        " the $d_{symb}$ symbolic representation."
        " Use the `Explore` tab your visualize your raw time series along"
        " with its computed $d_{symb}$ symbolization: the colorbars' list"
        " of the symbolic sequences and the distance matrix between the"
        " symbolic sequences."
        " Use the `Compare` tab to assess the relevance of $d_{symb}$ compared"
        " to other distance measures on the JIGSAWS data set (all results are"
        " pre-computed)."
    )

    tab_explore, tab_compare, tab_about = st.tabs(
        ["Explore", "Compare", "About"]
    )

    with tab_explore:
        run_explore_frame()

    with tab_compare:
        run_compare_frame()

    with tab_about:
        run_about_frame()


if __name__ == "__main__":
    run()