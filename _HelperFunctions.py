# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:07:16 2024

@author: gerdy
"""

"""
This script contains general utility functions. They are ordered alphabetically.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pyclesperanto_prototype as cle

import cv2 as cv
### ___________________________________________________________________________
def img_gradient(img: np.ndarray) -> np.ndarray:
    """
    Computes the color gradient of the image. Returns an image with the length 
    of the grad at each point.
    Good to detect borders coded in changing color.
    """
    gradx, grady = np.gradient(img)

    length=np.zeros(gradx.shape)
    for i in range(gradx.shape[0]):
        for j in range(gradx.shape[1]):
            length[i,j] = np.linalg.norm([gradx[i,j],grady[i,j]])
    return length


def convolve_derivative(x: np.ndarray, data: np.ndarray, s = 3, maximum_width = 0.1, *, DERIVATIVE = ["first"], show = False) -> tuple:
    """ 
    Computes the derivatives of noisy data by applying a gaussian filter, of sigma = s.
    
    x : x values for data, meaning data(x) as the plot representation.
    data : Input data to be filtered and derivated.
    s : parameter for the gaussian.
    
    DERIVATIVE: [zero, first, second] -> computes and returns the derivatives indicated
                The derivatives no computed will still be returned as np.NaN
    
    show : If True, plot the results.
    """
    X = linspace(- 3 * s, 3 * s, 1).astype(int)
    kernel_size = len(X)
    superposed_x = x[ kernel_size//2 : -(kernel_size)//2 + 1]
    
    # Init all return values
    f_conv = np.NAN; df_conv = np.NaN; d2f_conv = np.NaN; stable = np.NaN
    
    norm = 1 / np.sqrt(2 * np.pi) / s
            
    if "first" in DERIVATIVE:    
        # First order convolution
        dG_s = - X / s**2 * norm * np.exp(-X**2 / 2 / s**2)
        df_conv = np.convolve(data, dG_s, mode="valid")
        
        prev_max = 0 
        for n, i in enumerate(df_conv):
            # if i == 0 or i == len(df_conv): continue
            # centered_mean = (abs(df_conv[n-1]) + abs(df_conv[n+1])) / 2
            # print(centered_mean, abs(df_conv[n]))
            if (abs(i)  < maximum_width * abs(df_conv.max())) and (n > len(data) / 10):
                if prev_max == n - 1:
                    stable = superposed_x[n] # Finds the x value for the first relevant 0 in the first derivative.
                    break
                prev_max = n 
                
        if show:
            # This variable st is used to manually change the range of the plot.
            # st = int(stable)
            st = 10
            # plt.title(f"Filtered data's first order derivative | {stable}")
            plt.plot(superposed_x[st-10:], df_conv[st-10:])
            plt.plot(superposed_x[st-10:], np.zeros(len(superposed_x))[st-10:], linestyle = (0, (5,5)) )
            plt.show()

    if "zero" in DERIVATIVE:    
                # Zero-order convolution
        G_s = norm * np.exp(-X**2 / 2 / s**2)
        f_conv = np.convolve(data, G_s, mode="valid")

        if show:

            # plt.title("Number of pixels that change from FG to BG")
            plt.xlabel("Tolerance")
            plt.ylabel(r"$\Delta$ pixels")

            plt.plot(x[st-10:], data[st-10:], label="Raw data")
            plt.plot(superposed_x[st-10:], f_conv[st-10:], label = "Filtered data")  # The limits sign the places where they overlap
            plt.legend()

            plt.savefig(r"C:\Users\gerdy\Documentos\3 Documentación\Papers\AutomaticMaskGeneration\Pixels_delta_full_4-8.png", dpi = 400)
            plt.show()

    if "second" in DERIVATIVE:
        # Second order convolution
        d2G_s = ((X / s)**2 - 1 ) / s**2 * norm * np.exp(-X**2 / 2 / s**2)
        d2f_conv = np.convolve(data, d2G_s, mode="valid")
    
        if show:
            plt.title("Filtered data's seconds order derivative")
            plt.plot(superposed_x, d2f_conv)
            plt.plot(superposed_x, np.zeros(len(superposed_x)), linestyle = (0, (5,5)) )
            plt.show()
    
    
    return f_conv, df_conv, d2f_conv, stable # np.ndarray x3, float
    
def detect_maxima(img: np.ndarray, sigma = 3) -> tuple:
    """ 
    Detects the relevant locations of maxima in img, by using a parameter for
    the gaussian blur defined by sigma.
    """
              
    # Gaussian blur (sigma =3) to detect maxima locations.
    img_gaussian = cle.gaussian_blur(img, sigma_x= sigma, sigma_y=sigma, sigma_z=3)
    
    box = 0
    img_maxima_locations = cle.detect_maxima_box(img_gaussian, radius_x=box, radius_y=box, radius_z=box)
    
    # Detects relevant maxima locations (aka: possible neuron locations).
    img_gaussian2 = cle.gaussian_blur(img, sigma_x=1, sigma_y=1, sigma_z=1)
    np.seterr(all='raise')
    
    try: 
        img_thresh = cle.threshold_otsu(img_gaussian2)
    except FloatingPointError as e:
        print(f"Warning: {e} | function: _HelperFunctions.detect_maxima")
        return (np.array([[0,0],[0,0]]),0)
    
    img_relevant_maxima = cle.binary_and(img_thresh, img_maxima_locations)
    number_of_relevant_maxima_locations = cle.sum_of_all_pixels(img_relevant_maxima)
    
    return (np.array(img_relevant_maxima), number_of_relevant_maxima_locations) # np.ndarray, float

def morph(img: np.ndarray, MODE = None, *, count_lim = 4) -> np.ndarray:
    """
    MODE = ["ERODE", "DILATE"]

    count_lim: if MODE is ERODE, pixels surrounded by less or equal to (count_lim + 1) are set to 0.
               if MODE is DILATE, pixels surrounded by more or equal to than count_lim are set to 1. 
    WARNING: This implementation generates a lot of copies of the original object, maybe they are not needed.
    """
    
    if img.max() == 0: return None
    
    R = img.copy()
    RCheck = img.copy()
    if R.max() != 1:
        R = (-1 * (img // (-1 * img.max()))).astype("uint8")
        RCheck = (-1 * (img // (-1 * img.max()))).astype("uint8")
    X, Y = img.shape
    
    Rindices = cv.dilate(RCheck, np.ones((3,3)))
    Indices = extract_non_zero(Rindices)
    
    if MODE == "ERODE":
        for index in Indices:
            i, j = index   
            if i == 0 or i == X or j == 0 or j == Y: continue

            img_c = RCheck[i-1:i+2,j-1:j+2]
            Count = np.sum(img_c)
            if Count <= count_lim + 1: 
                R[i,j] = 0        
            else: 
                R[i,j] = 1
    elif MODE == "DILATE":
        for index in Indices:
            i, j = index   
            if i == 0 or i == X or j == 0 or j == Y: continue

            img_c = RCheck[i-1:i+2,j-1:j+2]
            Count = np.sum(img_c)
            if Count >= count_lim: 
                R[i,j] = 1        
            else: 
                R[i,j] = 0
    else:
        print("Warning:")
        print("Please, input one of both modes as in Morph(image, MODE = 'ERODE' or 'DILATE')")
        return None
    return R

def erode_dilate(img: np.ndarray, loops = 5, *, count_lim = (5,3)) -> np.ndarray:
    """ 
    Applies morph() various times sequentially.
    """
    R = img.copy()
    for i in range(0, loops):
        
        R = morph(R, "ERODE", count_lim = count_lim[0])
        
        if type(R) == type(None): return None
        
        R = morph(R, "DILATE", count_lim = count_lim[1])
    if R.max() < 100:
        R = R.astype("uint8") * 255
    return R
   
    
def extract_non_zero(a: np.ndarray) -> list:
    """
    Returns a list of tuples of indices of the non zero elements.
    """
    i,j = np.nonzero(a)            
    return list(zip(i,j))


def linspace(start, stop, step=1.):
    """ Adaptation of np.linspace, so the last parameter is the step and not the number of points. """
    return np.linspace(start, stop, int((stop - start) / step + 1))


def size_filter_v2(img: np.ndarray, *, coord: list = None, n_area_m = (200, 380), colour = 0) -> np.ndarray:
    """ 
    Filters by size. Based on OpenCV implementations of the masking of a boolean image.
    """

    c_mask = np.ones(img.shape).astype("uint8")

    if type(coord) != type(None): #If I give specific coordinates then the algorithm only searches in neurons that contain such coords.
        c_mask = c_mask - 1
        c_mask[coord] = 1
    
    binary_mask = img.copy()

    Colour_check = np.isin(binary_mask, [0, colour]) # Returns True is the bit is either 0 or colour
    binary_mask[Colour_check != True] = 1

    # Now, we perform region labelling. This way, every connected component will have their own colour value.
    labelled_mask, num_labels = label(binary_mask)
    
    # Let us now remove all the too small regions.
    for l in range(1, num_labels + 1):
        m = np.zeros(img.shape).astype("uint8")
        m[labelled_mask == l] = 1
        
        # This value should be one if there is intersection and zero if not.
        Intersect = np.sum(np.bitwise_and(m, c_mask))
        
        if Intersect:
            S = np.sum(m)
            if n_area_m[0] > S or S > n_area_m[1]:
    
                binary_mask[labelled_mask == l] = colour
                

    return binary_mask


def clean_area(img: np.ndarray, coordinate: list, recursion_limit = 10000) -> bool:
    """
    This is a function that erases the pixels connected to the pixel provided by coordinate. This avoids looping over all pixels
    in the image. It modifies the image passed as a parameter. Returns True if the erasing was completed succesfully.
    """
    if recursion_limit != 0:
        sys.setrecursionlimit(recursion_limit)
    
    i,j = coordinate
        
    img[coordinate] = 0

    positions = ((i,j+1),(i+1,j),(i,j-1),(i-1,j))

    for n in positions:
        try:
            if img[n] != 0:
                clean_area(img, n, recursion_limit=0)
        except IndexError:
            pass 
    
    if recursion_limit != 0:
        sys.setrecursionlimit(1000)
    return True    


def tol_dif(img: np.ndarray) -> tuple:
    count, X = np.histogram(img.flatten(), bins = 255, range = (0,255))
    return X[1:-1], count[1:]

