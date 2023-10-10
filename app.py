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
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="dSymb",
    )

    st.write("# Welcome to Streamlit! 👋")

    tab_explore, tab_compare, tab_about = st.tabs(["Explore", "Compare", "About"])  

    with tab_explore:
        run_explore_frame()

    with tab_compare:
        run_compare_frame()

    with tab_about:
        run_about_frame()
    
    

    


if __name__ == "__main__":
    run()
