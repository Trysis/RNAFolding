#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import plotly.express as px

# Load the CSV file 
# RNA bases are coding by 1,2,3,4
# 1 row = 1 seq 

def list_rna_seq(csv_name):
    # Chargez le fichier CSV en tant que tableau NumPy
    rna_seqs = np.genfromtxt(csv_name, delimiter=',', usecols=range(1, 208))
    print("Données chargées")
    print(rna_seqs)
    return rna_seqs[1:]

# load csv and return np.array of lists with encoding rna sequences 
def list_rna_seq(csv_name): 
    rna_seqs = pd.read_csv(csv_name,usecols=range(1,208))
    print("data load")
    # initialization empty list 
    liste_all = []
    # list of lists where each list represents a sequence
    for index,row in rna_seqs.iterrows():
        liste_ligne = row.tolist()
        liste_all.append(liste_ligne)
    listearray = np.array(liste_all)
    print("all seq add in list")
    return(listearray)


def UMAP_build(array_seq,i_neigh,i_dist,la_coloration,name):
    # make TSNE
    um = UMAP(n_components=2, n_neighbors = i_neigh, min_dist = i_dist,random_state=42)
    X_emb = um.fit_transform(array_seq)
    print("TSNE calc")
    # representation
    df = pd.DataFrame()
    df["comp-1"] = X_emb[:,0]
    df["comp-2"] = X_emb[:,1]
    df["coloration"] = la_coloration  # Assuming la_coloration is the column with 0 and 1 values

    fig = px.scatter(df, x="comp-1", y="comp-2", color="coloration", color_continuous_scale=px.colors.diverging.Temps)
    fig.update_layout(title=f"UMAP Representation {name}")
    fig.show()

    #graph_name = f"UMAP_neighbors{i_neigh}_dist{i_dist}.png"
    #fig.write_image(f"representation/{graph_name}")
    print("representation save")



if __name__ == "__main__":
    csv_file_seq = "TEST_CONDING_RNA.csv"
    csv_file_pred = "data_prediction.csv"
    
    rna_array = list_rna_seq(csv_file_seq) # seq to npy array 

    df_pred = pd.read_csv(csv_file_pred)
    
    data_simple_2A3 = df_pred[['simple_LSTM_2A3']]
    UMAP_build(rna_array,100,0.4,data_simple_2A3,"simple LSTM 2A3")

    data_simple_DMS = df_pred[['simple_LSTM_DMS']]
    UMAP_build(rna_array,100,0.4,data_simple_DMS,"simple LSTM DMS")

    ## bi LSTM 
    data_bi_2A3 = df_pred[['bi_LSTM_2A3']]
    UMAP_build(rna_array,100,0.4,data_bi_2A3,"Bi-LSTM 2A3")

    data_bi_DMS = df_pred[['BI_LSTM_DMS']]
    UMAP_build(rna_array,100,0.4,data_bi_DMS,"Bi-LSTM DMS")


    ### spot RNA 
    data_spot_2A3 = df_pred[['SPOT_RNA_2A3']]
    UMAP_build(rna_array,100,0.4,data_spot_2A3,"SPOT RNA 2A3")

    data_spot_DMS = df_pred[['SPOT_RNA_DMS']]
    UMAP_build(rna_array,100,0.4,data_spot_DMS,"SPOT RNA DMS")

    ######### DMS 
    
