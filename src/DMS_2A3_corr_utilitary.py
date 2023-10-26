
import numpy as np
import plots
import preprocessing

filepath = "./data/cleared_data.csv"
values = preprocessing.getReactivities(filepath)
val_2A3 = values["2A3"]
val_DMS = values["DMS"]
idx = np.arange(val_2A3.shape[0])

plots.plot(idx,
           val_2A3,
           val_DMS,
           mode="scatter",
           title="DMS and 2A3 reactivities correlation",
           xlabel="2A3 reactivity",
           ylabel="DMS reactivity",
           save_to="./",
           filename="2A3_DMS_corr",
           alphas=(0.3),
           lab_1="", lab_2=""
           )

