#!/usr/bin/env python
# coding: utf-8

"""The manifold_utils module provides interfaces for manifold learning and dimensionality reduction.
 
"""

__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")


from sklearn.manifold import TSNE


def tsne_embedding(X, n_dim=2, perplexity=3):
    tsne = TSNE(n_components=n_dim,
                init='random',
                perplexity=perplexity,
                learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    return X_embedded