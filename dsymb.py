import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
from sklearn.utils import Bunch
import random
import gc

from segmentation import Segmentation
from segment_feature import SegmentFeature

import ruptures as rpt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.ticker as ticker

import time

from symbolization import Symbolization
from symbolic_signal_distance import SymbolicSignalDistance
from sklearn.preprocessing import StandardScaler

from weighted_levenshtein import lev

from aeon.distances import (
    ddtw_distance,
    dtw_distance,
    edr_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance,
)

import streamlit as st

def compute_weighted_lev(
    n_symbols,
    symb_signal_1,
    symb_signal_2,
    insert_costs,
    delete_costs,
    substitute_costs,):
    """Compute the general edit distance (a.k.a weighted Levenshtein
    distance) between two symbolic signals.

    The distance is not normalized by the lengths of the symbolic signals.
    symb_signal_1 and symb_signal_2 are signals of integers (the labels of
    the segment classes).
    """

    # Avoid weird ASCII characters
    assert n_symbols <= 26, "`n_symbols` should be inferior to 26!"
    alphabet_signal_1 = [chr(i + ord("A")) for i in symb_signal_1]
    alphabet_signal_2 = [chr(i + ord("A")) for i in symb_signal_2]

    # Convert the list of strings / characters into long strings:
    str_alphabet_signal_1 = "".join(alphabet_signal_1)
    str_alphabet_signal_2 = "".join(alphabet_signal_2)

    # Compute the weighted Levenshtein distance:
    symb_signals_dist = lev(
        str_alphabet_signal_1,
        str_alphabet_signal_2,
        insert_costs=insert_costs,
        delete_costs=delete_costs,
        substitute_costs=substitute_costs,
    )
    return symb_signals_dist

def get_feat_df(segment_features_df: pd.DataFrame) -> pd.DataFrame:
    """Return the same df with only the feature columns."""
    feat_columns = [
        col for col in segment_features_df.columns if col.endswith("_feat")
    ]
    return segment_features_df[feat_columns]

def transform_costs(lookup_table):
    """Transform the substitute, insertion and deletion costs.

    Computed from the look-up table and used for the weighted Levenshtein
    distance.

    Our symbols are the A, B, C, ... ASCII characters.
    """

    # Integrate the lookup table into the substitute costs:
    substitute_costs = np.ones((128, 128), dtype=np.float64)
    n_symbols = lookup_table.shape[0]
    substitute_costs[
        ord("A"): ord("A") + n_symbols, ord("A"): ord("A") + n_symbols
    ] = lookup_table.astype(np.float64)

    # Scale up the insert and delete costs:
    lookup_table_max = lookup_table.max()
    insert_costs = np.ones(128, dtype=np.float64) * lookup_table_max
    delete_costs = np.ones(128, dtype=np.float64) * lookup_table_max

    b_transform_costs = Bunch(
        insert_costs=insert_costs,
        delete_costs=delete_costs,
        substitute_costs=substitute_costs,
    )
    return b_transform_costs



def compute_symbolisation(df_temp,Nsig):
    l_min=np.min(df_temp["segment_length"])
    symboli=[]
    for i in range(Nsig):
        k=np.where(df_temp["signal_index"]==i)
        k=k[0]
        sym_x=[]
        for j in range(len(k)):
            new_l=df_temp["segment_length"][k[j]]/l_min
            new_sym=[df_temp['segment_symbol'][k[j]]] * new_l.astype(int)
            sym_x=sym_x+new_sym
        symboli.append(sym_x)
    return symboli

def compute_matrix_distance(symboli,lookup_table,Nsig,n_clusters):
    b_transform_costs=transform_costs(lookup_table)
    D=np.zeros((Nsig,Nsig))
    for i in range(Nsig):
        for j in range(i,Nsig):
            D[i,j]=compute_weighted_lev(n_clusters,symboli[i],symboli[j],b_transform_costs.insert_costs,b_transform_costs.delete_costs,b_transform_costs.substitute_costs)
            D[j,i]=D[i,j]
    return D



def my_clustering(n_clusters,X):
    kmeans = KMeans(n_clusters=n_clusters,n_init=10).fit(X)
    lookup_table=np.zeros((n_clusters,n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            lookup_table[i,j]=np.sqrt(np.sum(np.abs(kmeans.cluster_centers_[i,:]-kmeans.cluster_centers_[j,:])**2))
    return kmeans.labels_,lookup_table,kmeans.cluster_centers_



def reconstruct_signal(id_signal,X,df_temp):
    k=np.where(df_temp["signal_index"]==id_signal)
    k=k[0]
    x_recons=np.tile(X[k[0],:],(df_temp["segment_length"][k[0]],1))
    for i in range(1,len(k)):
        x_recons=np.concatenate((x_recons,np.tile(X[k[i],:],(df_temp["segment_length"][k[i]],1))))
    return x_recons

def reconstruct_signal_quant(id_signal,df_temp,centroids):
    k=np.where(df_temp["signal_index"]==id_signal)
    k=k[0]
    x_recons=np.tile(centroids[df_temp["segment_symbol"][k[0]],:],(df_temp["segment_length"][k[0]],1))
    for i in range(1,len(k)):
        x_recons=np.concatenate((x_recons,np.tile(centroids[df_temp["segment_symbol"][k[i]],:],(df_temp["segment_length"][k[i]],1))))
    return x_recons


def get_multiscale_seg(X,n_clusters):
    labels,lookup_table,centroids=my_clustering(n_clusters,X)
    lookup_table = lookup_table/np.max(lookup_table)

    return labels,lookup_table

@st.cache_data(ttl=3600,max_entries=1,show_spinner=False)
def dsym(list_of_multivariate_signals,N_symbol):
    with st.spinner('Computing dsymb...'):
        pen_factor=1000000
        Nsig=len(list_of_multivariate_signals)

        # Define the segmentation
        seg = Segmentation(
            uniform_or_adaptive="adaptive",
            mean_or_slope="mean",
            n_segments=None,
            pen_factor=pen_factor,
        )
      
        echelle=np.zeros((Nsig,))
        for i in range(Nsig):
            echelle[i]=np.mean(np.var(list_of_multivariate_signals[i],axis=0))
        
        nb_rupt=np.zeros((Nsig,))
        big_list_of_multivariate_signals=[]
        big_list_of_bkps=[]
        for sig in range(Nsig):
            x=list_of_multivariate_signals[sig]
            big_list_of_multivariate_signals.append(x)
            n1,n2=np.shape(x)
            pen=n1*echelle[sig]
            algo = rpt.KernelCPD(kernel="linear", jump=1).fit(list_of_multivariate_signals[sig])
            result = algo.predict(pen=pen)
            big_list_of_bkps.append(result)
            nb_rupt[sig]=len(result)

        b_segmentation=Bunch(list_of_multivariate_signals=big_list_of_multivariate_signals,list_of_bkps=big_list_of_bkps)
        seg_feat = SegmentFeature(
            features_names=[
                "mean",
            ]
        )
        df_temp = seg_feat.fit(b_segmentation).transform(b_segmentation)
        X=df_temp.to_numpy()[:,:len(list_of_multivariate_signals[0][0])] 

        labels,lookup_table = get_multiscale_seg(X,N_symbol)
        df_temp["segment_symbol"]=labels

        symboli=compute_symbolisation(df_temp,Nsig)
        D1=compute_matrix_distance(symboli,lookup_table,Nsig,len(lookup_table))

    gc.collect()
    return D1,df_temp,lookup_table


