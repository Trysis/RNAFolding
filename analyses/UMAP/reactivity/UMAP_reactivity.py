#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP




def UMAP_build(array_seq,i_neigh,i_dist,table_number):
    # make TSNE
    um = UMAP(n_components=2, n_neighbors = i_neigh, min_dist = i_dist)
    X_emb = um.fit_transform(array_seq)
    print("TSNE calc")
    # representation

    df = pd.DataFrame()
    df["comp-1"] = X_emb[:,0]
    df["comp-2"] = X_emb[:,1]

    sns.scatterplot(x="comp-1", y="comp-2",data=df).set(title="")

    graph_name = f"UMAP_{table_number}neighbors{i_neigh}_dist{i_dist}.png"
    plt.savefig(f"representation/{graph_name}")
    plt.clf()
    print("representation save")



if __name__ == "__main__":
    csv_file = "cleared_train_data.csv"

    cleaned_train_data = pd.read_csv(csv_file,usecols=range(1,208))
    
    cleaned_train_data = cleaned_train_data.fillna(0)
    df_2A3_MaP = cleaned_train_data[cleaned_train_data['experiment_type'].str.contains('2A3_MaP', na=False)].copy()
    df_DMS_MaP = cleaned_train_data[cleaned_train_data['experiment_type'].str.contains('DMS_MaP', na=False)].copy()

    list_p_nghbs = [1000]
    list_p_dist = [0.5]
    
    df_2A3_MaP_0 = (df_2A3_MaP.iloc[:, 1:209])
    df_DMS_MaP_0 = (df_DMS_MaP.iloc[:, 1:209])

    np_2A3_MaP = df_2A3_MaP_0.values
    np_DMS_MaP = df_DMS_MaP_0.values

    liste_data = [np_DMS_MaP]

    list_p_nghbs = [1000]
    list_p_dist = [0.4]
    datas = 0
    for table in liste_data :
        datas += 1
        for j in range(0,len(list_p_nghbs)):
            for k in range(0,len(list_p_dist)):
                tSNEx = UMAP_build(table,list_p_nghbs[j],list_p_dist[k],datas)
    
