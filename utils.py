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


from streamlit_plotly_events import plotly_events

from dsymb import *

r = lambda: random.randint(50,255)
DEFAULT_PLOTLY_COLORS={str(i):'#%02X%02X%02X' % (r(),r(),r()) for i in range(25)}




@st.cache_data(ttl=3600,max_entries=1,show_spinner=False)
def preprocess_data(uploaded_ts):
	with st.spinner('Preprocessing data...'):
		all_ts = []
		for ts in uploaded_ts:
			all_ts.append(np.genfromtxt(ts, delimiter=','))	
	return all_ts
	gc.collect()


@st.cache_data(ttl=3600,max_entries=1,show_spinner=False)
def plot_matrix(D1):
	return px.imshow(D1,aspect="auto")


@st.cache_data(ttl=3600,max_entries=3,show_spinner=False)
def plot_symbolization(df_temp,mode):
	tmp_df = df_temp
	tmp_df = tmp_df.rename(columns={'segment_start': 'Start', 'segment_end': 'Finish', 'signal_index': 'Task'})
	tmp_df['segment_symbol'] = tmp_df['segment_symbol'].apply(str)
	tmp_df['Task'] = tmp_df['Task'].apply(str)

	if mode == 'Normalized':
		all_max_length = []
		for i in range(len(tmp_df)):
			sig_index = tmp_df['Task'].values[i]
			max_length = max(tmp_df.loc[tmp_df['Task'] == sig_index]['Finish'].values)
			all_max_length.append(max_length)
		tmp_df['max'] = all_max_length
		tmp_df['Start'] = tmp_df['Start'] / tmp_df['max']
		tmp_df['Finish'] = tmp_df['Finish'] / tmp_df['max']

	fig = ff.create_gantt(tmp_df, index_col = 'segment_symbol',  bar_width = 0.4, show_colorbar=True,group_tasks=True,colors= {key:DEFAULT_PLOTLY_COLORS[key] for key in set(tmp_df['segment_symbol'].values)})
	fig.update_layout(xaxis_type='linear', height=1000,title_text="All symbolized Time Series")
	return fig

	

def plot_time_series(ts,tmp_df,dims=[0,20]):
	
	#tmp_df = df_temp.copy()
	tmp_df = tmp_df.rename(columns={'segment_start': 'Start', 'segment_end': 'Finish', 'signal_index': 'Task'})
	tmp_df['segment_symbol'] = tmp_df['segment_symbol'].apply(str)
	tmp_df['Task'] = tmp_df['Task'].apply(str)
	fig_symb = ff.create_gantt(tmp_df, index_col = 'segment_symbol',  bar_width = 0.4, show_colorbar=True,group_tasks=True,colors={key:DEFAULT_PLOTLY_COLORS[key] for key in set(tmp_df['segment_symbol'].values)})
	
	fig = make_subplots(rows=(dims[1]-dims[0])+1, cols=1,shared_xaxes=True)

	for trace in fig_symb.data:
    		fig.add_trace(trace, row=1, col=1)
	
	for i_row,i in enumerate(range(dims[0],dims[1])):
		fig.add_trace(
			go.Scattergl(x=list(range(len(ts))), y=ts[:,i],mode = 'lines', line = dict(color = 'white', width=1)),
			row=i_row+2, col=1
		)
	fig.update_layout(xaxis_type='linear',height=min(2000,(dims[1]-dims[0])*50), title_text="Time Series",showlegend=False)
	st.plotly_chart(fig, use_container_width=True)
	del fig,fig_symb 
	gc.collect()


def get_data_step():
	uploaded_ts = st.file_uploader("Upload your time series",accept_multiple_files=True)
	if len(uploaded_ts) == 1:
		st.markdown("Multiple time series should be provided")
	elif len(uploaded_ts) >= 2:
		#try:
		st.session_state.ALL_TS = preprocess_data(uploaded_ts)

def Visualize_step():
	if len(st.session_state.ALL_TS) > 1:
		N_symbol = st.slider('Number of symbols', 0, 25, 5)
		D1,df_temp,lookup_table = dsym(st.session_state.ALL_TS,N_symbol)
		tab_indiv, tab_all = st.tabs(["Each time series", "Dataset"])  
		with tab_indiv:
			time_series_selected = st.selectbox('Pick a time series', list(range(len(st.session_state.ALL_TS))))
			range_dims = [[20*dim_s,20*(dim_s+1)] for dim_s in range(len(st.session_state.ALL_TS[time_series_selected][0])//20)]
			if range_dims[-1][1] < len(st.session_state.ALL_TS[time_series_selected][0]):
				range_dims += [[range_dims[-1][1],len(st.session_state.ALL_TS[time_series_selected][0])]]
			range_dims += [[0,len(st.session_state.ALL_TS[time_series_selected][0])]]
			dims = st.selectbox('choose dimensions range', range_dims)
			plot_time_series(st.session_state.ALL_TS[time_series_selected],df_temp.loc[df_temp['signal_index']==time_series_selected],dims)
			
		with tab_all:
			mode = st.radio(
				"Mode",
				["Colorbar list", "Similarity Matrix"],
				captions = ["Visualize all symbolized time series", "Visualize the similarity matrix based on dsymb"],horizontal=True)
			if mode == "Colorbar list":
				mode_length = st.radio(
					"Length",
					["Real", "Normalized"],
					captions = ["Real time series length", "normalized between 0 and 1"],horizontal=True)
				
				fig = plot_symbolization(df_temp,mode=mode_length)
				st.plotly_chart(fig, use_container_width=True)
			elif mode == "Similarity Matrix":
				fig = plot_matrix(D1)
				st.plotly_chart(fig, use_container_width=True)
				





def run_explore_frame():
	st.markdown('## Explore Your dataset')
	st.markdown('Select the number of symbols to represent your time series. You can then drop your dataset (each time series in one .csv file with the shape (n_timestamp,n_dim).')
	
	
	get_data_step()

	Visualize_step()

	gc.collect()
			
		

            
            
		#except Exception as e:
		#	st.markdown('file format not supported yet, please upload a time series in the format described in the about tab: {}'.format(e))
    

def run_compare_frame():
	st.markdown('## Compare')

def run_about_frame():
	st.markdown(f""" ## A fast interactive exploration of multivariate time series datasets
		Symbol is a Python-based web interactive tool to visualize, navigate, and explore 
		large multivariate time series datasets. It is based on a new symbolic representation, 
		**dsymb**, for multivariate time series. With our tool, exploring a dataset of 80 time 
		series (with 80 dimensions and 5000 timestamps) requires 20 seconds instead of 2000 
		seconds for DTW-based analysis.

		### Reference

		> "dsymb: "<br/>
		> Authors<br/>
		> Proceedings of XXX, pages XXX-XXX, 2023<br/>

		```bibtex
		@article{symbol,
		  title={},
		  author={},
		  journal={},
		  volume={},
		  number={},
		  pages={},
		  year={},
		  publisher={}
		}
		```

		## Contributors

		* Paul Boniol (ENS Paris Saclay)
		* Sylvain Combettes (ENS Paris Saclay)
		* Charles Truong (ENS Paris Saclay)
		* Laurent Oudre (ENS Paris Saclay)
		""")
