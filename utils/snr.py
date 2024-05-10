"""
Utilities functions for SNR
"""
import numpy as np

# -------------------------------------------------------------------------------------------------

def create_pseudo_replica(im, added_noise_sd=0.1, N=100):
    """Perform the pseudo replica data
    """
    if np.iscomplex(im.dtype):
        n = (np.random.randn(*im.shape, N) + 1j * np.random.randn(*im.shape, N)) * added_noise_sd
    else:
        n = np.random.randn(*im.shape, N) * added_noise_sd

    im_noised = im[..., np.newaxis] + n.astype(im.dtype)

    return im_noised

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":
    pass
