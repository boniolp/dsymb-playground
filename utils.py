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

import streamlit as st


def plot_time_series(ts):
	fig = make_subplots(rows=len(ts[0]), cols=1,shared_xaxes=True)
	for i in range(len(ts[0])):
		fig.add_trace(
			go.Scattergl(x=list(range(len(ts))), y=ts[:,i]),
			row=i+1, col=1
		)
	fig.update_layout(height=2000, title_text="Time Series")
	st.plotly_chart(fig, use_container_width=True)

def run_explore_frame():
	st.markdown('# Explore')
	uploaded_ts = st.file_uploader("Upload your time series",accept_multiple_files=True)
	if len(uploaded_ts) == 1:
		st.markdown("Multiple time series should be provided")
	elif len(uploaded_ts) >= 2:
		#try:
		all_ts = []
		for ts in uploaded_ts:
			all_ts.append(np.genfromtxt(ts, delimiter=','))
    
		time_series_selected = st.selectbox('Pick a time series', list(range(len(all_ts))))
		plot_time_series(all_ts[time_series_selected])
		

            
            
		#except Exception as e:
		#	st.markdown('file format not supported yet, please upload a time series in the format described in the about tab: {}'.format(e))
    

def run_compare_frame():
	st.markdown('# Compare')

def run_about_frame():
	st.markdown('# About')
