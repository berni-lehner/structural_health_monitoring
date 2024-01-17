"""The manifold_utils module provides interfaces for manifold learning and dimensionality reduction.
 
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")


from sklearn.manifold import TSNE
import umap


def tsne_embedding(X, n_dim=2, perplexity=3):
    tsne = TSNE(n_components=n_dim,
                init='random',
                perplexity=perplexity,
                learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    return X_embedded


def umap_embedding(X, y=None, n_dim=2, n_neighbors=15):
    um = umap.UMAP(n_components=n_dim,
                   init='random',
                   n_neighbors=n_neighbors,
                   learning_rate=1.0)
    X_embedded = um.fit_transform(X, y)
    
    return um, X_embedded