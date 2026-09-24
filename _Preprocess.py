import numpy as np

import pyclesperanto_prototype as cle

from _HelperFunctions import img_gradient

def background_correction(img: np.ndarray, sigma: int):
    """
    Returns the original image minus the blurred image with sigma.
    """

    img_gaussian = np.array(cle.gaussian_blur(img, sigma_x=sigma, sigma_y=sigma, sigma_z=0))
    img_gaussian = img - img_gaussian

    return img_gaussian.astype("uint8")  # This type wraps up between 0-255. This means negative values get mapped to 255


def blur_gradient(img: np.ndarray, sigma: int):
    """
    Computes the gradient of the image, surrounded by gaussian blurs so the noise is not present in the result.
    """
    result = cle.gaussian_blur(img, sigma_x = sigma, sigma_y = sigma)
    result = img_gradient(result)
    result = np.array(cle.gaussian_blur(result, sigma_x = 10, sigma_y = 10))

    return result

def preprocess(img: np.ndarray, sigma: int = 2, sigma2: int = 2, thresh = 230, midpoint = None):
    """
    Chains blur_gradient and background corrections, and combines them. It eliminates the 
    background, based on certain parameters.

    sigma: tuple of the values used in blur_gradient.
    thresh: indicates the value for the neuron bodies, these areas are not touched in preprocess.
    midpoint: If not null, blur gradient is thresholded at that value.
              If null, blur gradient is thresholded at the midpoint between the maximum and 
                minimum value reached.
    """
    
    img_gradient = blur_gradient(img, sigma)

    if type(midpoint) == type(None):
        midpoint = (img_gradient.max() - img_gradient.min()) / 2.

    img_gradient[img_gradient <= 0.8 * midpoint] = 0; img_gradient[img_gradient > 0.8 * midpoint] = 1
    img_gradient_2 = np.array(cle.gaussian_blur(img_gradient, sigma_x = 3, sigma_y = 3))
    

    # The best sigma for the process so far has been found around 2 
    img_corrected = background_correction(img, sigma2)
    img_corrected[img_gradient_2 == 0] = 255  # This avoids overwritting the background
    img_corrected[img >= thresh] = 0  # This avoids overwriting neuron bodies


    img_treated = np.minimum(255 - img_corrected, img)
    img_treated = np.array(cle.gaussian_blur(img_treated, sigma_x = 2, sigma_y = 2)).astype("uint8")
    
    return img_treated


