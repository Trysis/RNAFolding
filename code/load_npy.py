import numpy as np 

loaded_data = np.load("../id_test.npy")
print(loaded_data)


# Spécifiez le chemin du fichier texte dans lequel vous souhaitez enregistrer les éléments
file_path = 'id_test_seq.txt'

# Enregistrez les éléments du tableau dans le fichier texte, un élément par ligne
with open(file_path, 'w') as file:
    for item in loaded_data:
        file.write("%s\n" % item)
