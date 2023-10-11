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

import streamlit as st

from dsymb import *

@st.cache_data(ttl=3600,max_entries=2)
def preprocess_data(uploaded_ts):
	with st.spinner('Preprocessing data...'):
		all_ts = []
		for ts in uploaded_ts:
			all_ts.append(np.genfromtxt(ts, delimiter=','))	
	return all_ts


def plot_symbolization(df_temp):
	fig = px.timeline(df_temp, x_start='segment_start', x_end='segment_end', y='segment_symbol')
	fig.update_layout(xaxis_type='linear')
	st.plotly_chart(fig, use_container_width=True)
	

def plot_time_series(ts):
	fig = make_subplots(rows=len(ts[0]), cols=1,shared_xaxes=True)
	for i in range(len(ts[0])):
		fig.add_trace(
			go.Scattergl(x=list(range(len(ts))), y=ts[:,i],mode = 'lines', line = dict(color = 'white', width=1)),
			row=i+1, col=1
		)
	fig.update_layout(height=2000, title_text="Time Series",showlegend=False)
	st.plotly_chart(fig, use_container_width=True)

def run_explore_frame():
	st.markdown('## Explore Your dataset')
	st.markdown('Select the number of symbols to represent your time series. You can then drop your dataset (each time series in one .csv file with the shape (n_timestamp,n_dim).')
	N_symbol = st.slider('Number of symbols', 0, 25, 5)
	uploaded_ts = st.file_uploader("Upload your time series",accept_multiple_files=True)
	if len(uploaded_ts) == 1:
		st.markdown("Multiple time series should be provided")
	elif len(uploaded_ts) >= 2:
		#try:
		all_ts = preprocess_data(uploaded_ts)

		
		D1,df_temp,lookup_table = dsym(all_ts,N_symbol)
		
		time_series_selected = st.selectbox('Pick a time series', list(range(len(all_ts))))
		st.dataframe(df_temp.loc[df_temp['signal_index']==time_series_selected])
		plot_symbolization(df_temp.loc[df_temp['signal_index']==time_series_selected])
		plot_time_series(all_ts[time_series_selected])
		
		

            
            
		#except Exception as e:
		#	st.markdown('file format not supported yet, please upload a time series in the format described in the about tab: {}'.format(e))
    

def run_compare_frame():
	st.markdown('# Compare')

def run_about_frame():
	st.markdown('# About')
