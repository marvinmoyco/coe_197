from sklearn.decomposition import PCA
import numpy as np
def pca_function(input_dataset,dimension = 32):
    pca_obj = PCA(n_components=dimension)
    data_matrix  = pca_obj.fit_transform(input_dataset)
    print(data_matrix.shape)
    return pca_obj, data_matrix

