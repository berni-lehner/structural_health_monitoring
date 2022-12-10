""".
 
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")


from scipy.fft import dct

def extract_dctc(X, n_coeffs=30, skip_0th=True):
    y = dct(X, norm='ortho')

    start_idx = int(skip_0th)
    
    return y[:, start_idx:start_idx+n_coeffs]