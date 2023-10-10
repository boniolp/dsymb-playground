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

import streamlit as st



def run_explore_frame():
	st.markdown('# Explore')
	uploaded_ts = st.file_uploader("Upload your time series",accept_multiple_files=True)
	if uploaded_ts is not None:
		try:
			all_ts = []
			for ts in uploaded_ts:
				all_ts.append(np.genfromtxt(ts, delimiter=','))
            
			time_series_selected = st.selectbox('Pick a time series', list(range(len(all_ts))))
			fig = plt.figure(10,50)
			for i in range(len(all_ts)):
				plt.subplot(len(all_ts),1,i+1)
				plt.plot(all_ts[i])
			st.pyplot(fig)

            
            
		except Exception as e:
			st.markdown('file format not supported yet, please upload a time series in the format described in the about tab: {}'.format(e))
    

def run_compare_frame():
	st.markdown('# Compare')

def run_about_frame():
	st.markdown('# About')
