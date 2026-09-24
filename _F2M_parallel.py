from multiprocessing import Pool
import os

import cv2 as cv

from _F2Mask import F2Mask
from _Preprocess import preprocess

import pickle as pkl
from datetime import datetime
import time as ti

import importlib
import _parameter

def main(img):
    print(f"Parallel of F2Mask execution of {len(img)} images\n")
    print("Working...")
    
    start = ti.time()
    #_______ This is the preprocessing stage _________
    p_image = []
    for image in img:
        prep = preprocess(image[0], sigma=p["sigma1_1"], sigma2=p["sigma1_2"])
        p_tuple = (prep.astype("uint8"), ) + image[1:]
        p_image.append(p_tuple)
    #___________________________________________________
    finish = ti.time()
    print(f"Preprocessing finished in {finish-start: .4f}s\n")
    with Pool() as pool:        
        # Starmap results keep the input order
        results = pool.starmap(F2Mask, p_image)   
    return results, p_image
        
def generate_image_list(F_folder, *, limits = (100, 550), sigma = 2.5,):
    """
    Out of the folder location of the images to be processed, it generates the list of images,
    with the corresponding parameters to be passed into F2Mask. 
    """
    
    # Saves the original working directory, to return to it when it finishes execution.
    original_dir = os.getcwd()
    
    # Gets the fluorescence directory and changes the working directory to that of the fluorescence
    files = next(os.walk(F_folder), (None, None, []))[2]  # [] if no file
    
    os.chdir(F_folder)
    
    ls = []
    for index, file in enumerate(files):            
        img = cv.imread(file, 0)
        output = file[:-5] + 'FMaskP.tif'
        ls.append((img, limits, sigma, output, file))   
    
    os.chdir(original_dir)
    return ls



if __name__ == '__main__':
    importlib.reload(_parameter)
    from _parameter import p

    path_culture = "../images/" 
    run_name = input("Please, identify this run: \n -> ")

    start = datetime.now()
    #_________________ List of images generation __________________________________
    F_folder = path_culture + "/F"   
    
    img = generate_image_list(F_folder, limits = (p["size_limits4_4low"], p["size_limits4_4high"]), sigma = p["sigma3_1"])
        
    #_____________________________________ F2Mask call________________________________
    args = []
    for i in range(len(img)):
        args.append(img[i][0:3])

    result_b, input_images = main(args)
   
    finish = datetime.now()
    #_____________________________________ Save Results _______________________________
    result = []
    save_masks = f"{path_culture}/Masks/"
    
    save = f"{save_masks}/{run_name}/"
    os.makedirs(save)

    for i in range(len(img)):
        r1, r2, r3 = result_b[i]
        result.append((r1, r2, r3, img[i][3], img[i][4]))
        cv.imwrite(r"%s\%s" %(save, img[i][3]) , r1 * 255)

    # date = datetime.now().strftime("%Y%m%d-%H%M%S")
    # with open(r'%s\Results_%s.pkl' %(save, str(date) ), 'wb') as f:
    #         pkl.dump(result, f)

    with open(r'%s\ExecutionTime.txt' %(save ), 'a') as fp:
        fp.write("Execution date: %s\nScript executed in %s (%d images)" %(finish.strftime("%Y-%m-%d"), str(finish - start), len(img)))
        fp.write(
f"""
1.1 sigma: {p["sigma1_1"]}
1.2 sigma: {p["sigma1_2"]}
2.2 init_value fraction: {p["init_value_fraction2_2"]} 
3.1 sigma: {p["sigma3_1"]}
4.1 box_half: {p["box_half4_1"]}
4.2 size_interval: {p["size_interval4_2low"]}, {p["size_interval4_2high"]}
4.3 eccentricity: {p["eccentricity4_3"]}
4.4 size_limits: {p["size_limits4_4low"]}, {p["size_limits4_4high"]}
"""
)

