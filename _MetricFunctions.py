# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:08:05 2024

@author: gerdy
"""

"""
This script has all the information needed to compute the metrics.
"""
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import cv2 as cv
import pickle as pkl

from rich.progress import track

import _HelperFunctions as hf
#_____________________________________________________________________________

class ConfusionMatrix:
    # All definitions are taken from https://www.pycm.io/doc/
    def __init__(self, CM: np.ndarray, debug = False):
        self.TP = float(CM[0,0])
        self.TN = float(CM[1,1])
        self.FP = float(CM[1,0])
        self.FN = float(CM[0,1])
        self.POP = np.cumsum(CM)[-1]
        
        self.TOP = self.TP + self.FP
        self.P = self.TP + self.FN
        self.TON = self.FN + self.TN
        self.N = self.FP + self.TN
        
        self.Tr = self.TP + self.TN
    
        self.debug = debug
    def ACC(self):
        return self.Tr / self.POP
    
    def TPR(self):
        return self.TP / self.P if self.TP != 0 else 0
    
    def TNR(self):
        return self.TN / self.N if self.TN != 0 else 0
    
    def PPV(self):
        return self.TP / self.TOP if self.TP != 0 else 0
    
    def NPV(self):
        return self.TN / self.TON if self.TN != 0 else 0
    
    def Fbeta(self, beta):
        A = 1 + beta**2
        num = A * self.TP
        den = num + self.FP + beta**2 * self.FN
        if den == 0:
            if self.debug:
                print("Matrix invalid for F_score, encountered zero in denominator")
                self.show()
            return 0
        return num / den
    
    def MCC(self):
        num = self.TP * self.TN + self.FP * self.FN
        den = (self.TOP)*(self.P)*(self.N)*(self.TON)
        if den == 0:
            if self.debug:
                print("Matrix invalid for MCC, encountered zero in denominator")
                self.show()
            return 0
        return num / np.sqrt(den)
    
    def PRE(self):
        return self.P / self.POP
    
    def G(self):
        return np.sqrt(self.PPV() * self.TPR())
    
    def J(self):  # Equals overall IoU
        return self.TP / (self.POP - self.TN)
    
    def AUC(self):
        return (self.TNR() + self.TPR()) / 2

    def AUR(self):
        return (self.PPV() + self.TPR()) / 2

    def GI(self):
        return (self.TNR() + self.TPR()) - 1
    
    def GM(self): 
        return np.sqrt(self.TPR() * self.TNR())
    
    def ADF(self):
        A = 1.25 * (self.NPV() * self.TNR()) / (0.25 * self.NPV() + self.TNR())
        return np.sqrt(self.Fbeta(2) * A)
    
    def Kappa(self):    #https://en.wikipedia.org/wiki/Cohen%27s_kappa
        num = 2 * (self.TP * self.TN - self.FN * self.FP)
        den = (self.TOP) * (self.N) + (self.P) * (self.TON)
        if den == 0:
            if self.debug:
                print("Matrix invalid for Cohen's Kappa, encountered zero in denominator")
                self.show()
            return 0
        return num / den
        
    def SE(self):
        return np.sqrt(self.ACC() * (1 - self.ACC()) / self.POP)
        
    def CI_95(self):
        ACC = self.ACC()
        SE = self.SE()
        return (ACC - 1.96 * SE, ACC + 1.96 * SE)
    
    def BennetS(self, C = 2):
        p = 1 / C
        return (self.ACC() - p) / (1 - p)
    
    def PI(self):
        sum1 = (self.TOP + self.P)**2 
        sum2 = (self.TON + self.N)**2
        p = 1 / 4 / self.POP**2 * (sum1 + sum2)
        return (self.ACC() - p) / (1 - p)
    
    def AC1(self):  # Can be shown that AC1 = 1-2*(FP + FN)/N /(1 + ((TP - TN) / N)**2)
        num = 2 * self.POP * (self.FP + self.FN)
        den = self.POP**2 + (self.TP - self.TN)**2
        return 1 - num / den

    def ReferenceEntropy(self):
        if self.P == 0:
            sum1 = 0
        else:
            sum1 = self.P * np.log2(self.P / self.POP)
        if self.N == 0:
            sum2 = 0
        else:
            sum2 = self.N * np.log2(self.N / self.POP)
        return - 1 / self.POP * (sum1 + sum2)
    
    def ResponseEntropy(self):
        if self.TOP == 0:
            sum1 = 0
        else:
            sum1 = self.TOP * np.log2(self.TOP / self.POP)
        if self.TON == 0:
            sum2 = 0
        else:
            sum2 = self.TON * np.log2(self.TON / self.POP)
        return - 1 / self.POP * (sum1 + sum2)
    
    def CrossEntropy(self):
        if self.TOP == 0:
            sum1 = self.P * np.log2(0.001 / self.POP)
        else: 
            sum1 = self.P * np.log2(self.TOP / self.POP)
        if self.TON == 0:
            sum2 = self.N * np.log2(0.001 / self.POP)
        else:
            sum2 = self.N * np.log2(self.TON / self.POP)
        return - 1 / self.POP * (sum1 + sum2)
        
    
    def ConditionalEntropy(self):
        if self.TPR() == 0: sum1 = 0
        else: sum1 = self.TPR() * np.log2(self.TPR())
        if self.TPR() == 1: sum2 = 0
        else: sum2 = (1 - self.TPR()) * np.log2(1 - self.TPR())    
        if self.FN == 0: sum3 = 0
        else: sum3 = self.FN / self.POP * np.log2(self.FN / self.POP)
        if self.TNR() == 0: sum4 = 0
        else: sum4 = self.TNR() * np.log2(self.TNR())
        return - 1 / self.POP * (self.P * (sum1 + sum2) + self.N * (sum3 + sum4))
    
    def HammingLoss(self):
        return (self.FP + self.FN) / self.POP
    
    def ZeroOneLoss(self):
        return self.FP + self.FN
    
    def NIR(self):
        return np.max((self.P, self.N)) / self.POP
    
    def PValue(self):
        from scipy.special import comb
        # Probability of the "No information rate (NIR)" being higher than the obtained "Accuracy".
        p = self.NIR()
        cumsum = 0
        for i in range(1, self.Tr + 1):
            element = comb(self.POP, i) * p**i * (1 - p)**(self.POP - i)
            cumsum += element
        return 1 - cumsum
        
    def CBA(self):
        sum1 = self.TP / np.max((self.TOP, self.P))
        sum2 = self.TN / np.max((self.TON, self.N))
        return (sum1 + sum2) / 2
    
    def ChiSquared(self):
        E11 = self.TOP * self.P / self.POP
        E12 = self.TON * self.P / self.POP
        E21 = self.TOP * self.N / self.POP
        E22 = self.TON * self.N / self.POP
        sum1 = (self.TP - E11)**2 / E11
        sum2 = (self.FP - E12)**2 / E12
        sum3 = (self.FN - E21)**2 / E21
        sum4 = (self.TN - E22)**2 / E22
        return sum1 + sum2 + sum3 + sum4
        
    def C(self):
        chi = self.ChiSquared()
        return np.sqrt(chi / (chi + self.POP))
        
    def BangdiwalaB(self):
        num = self.TP**2 + self.TN**2
        den = self.TOP * self.P + self.TON * self.N
        if den == 0:
            if self.debug:
                print("Matrix invalid for BangdiwalaB, encountered zero in denominator")
                self.show()
            return
        return num / den
        
    def KrippAlpha(self):
        e = 1 / 2 / self.POP
        Pe = ((self.TOP + self.P) * e)**2 + ((self.TON + self.N) * e)**2
        Pa = (1 - e) * self.ACC() + e
        return (Pa - Pe) / (1 - Pe)
    
    def Aickin(self, max_iterations = 10000, thr = 0.001):
        print("WARNING: Aickin's implementation is still faulty, or the index is ill defined.\nStrange unpredictable behaviour.")

        # initial conditions
        alpha_1 = self.Kappa()
        p1_A = self.TOP / self.POP;     p1_B = self.P / self.POP
        p2_A = self.TON / self.POP;     p2_B = self.N / self.POP
        pe = p1_A * p1_B + p2_A * p2_B
        
        if pe == 0:
            if self.debug:
                print("Matrix invalid for Aickin's alpha, encountered zero in denominator")
                self.show()
            return np.NaN
        #Now the loop
        iterations = 0
        Alpha = -10 # Make sure it starts the loop
        while iterations < max_iterations:
            # Los que empiezan por mayúscula son las nuevas iteraciones
            P1_A = self.TOP / self.POP /( 1 + alpha_1 * (p1_B / pe - 1))
            P1_B = self.P / self.POP /( 1 + alpha_1 * (p1_A / pe - 1))
            P2_A = self.TON / self.POP /( 1 + alpha_1 * (p2_B / pe - 1))
            P2_B = self.N / self.POP /( 1 + alpha_1 * (p2_A / pe - 1))
            Pe = P1_A * P1_B + P2_A * P2_B
            if Pe == 1 or Pe == 0:
                if self.debug:
                    print("Warning: Pe term tends to 1, which makes a denominator go to zero.")
                    self.show()
                    print("Skip further calculation for this point")
                return 
            Alpha = (self.ACC() - Pe) / (1 - Pe)
            if abs(Alpha - alpha_1) < thr: break
            
            #Reset the variable loops
            p1_A = P1_A; p1_B = P1_B; p2_A = P2_A; p2_B = P2_B
            pe = Pe
            alpha_2 = alpha_1
            alpha_1 = Alpha
            
            
            iterations += 1
        if iterations == max_iterations:
            # NOTE: For CM such that FP = FN and FP + FN + TP = 100, with TN = 0, the index behaves strangely.
            print(f"Max iteration reached {iterations}, algorithm does not converge.")
            print(Alpha, abs(Alpha - alpha_2))
            self.show()
            return
        return Alpha
          
    def scale(self, factor):
        a = self.TP * factor 
        b = self.FN * factor
        c = self.FP * factor
        d = self.TN * factor

        return ConfusionMatrix(np.array([[a, b],[c, d]]))
    
    def normal(self):
        factor = 100 / self.POP
        a = self.TP * factor 
        b = self.FN * factor
        c = self.FP * factor
        d = self.TN * factor

        return ConfusionMatrix(np.array([[a, b],[c, d]]))
    
    def space_location(self):
        if self.POP != 100:
            self = self.normal()
        
        p = (self.TP, self.TN, self.FN)
        return p
        
    def transpose(self):
        return ConfusionMatrix(np.array([[self.TP, self.FP],[self.FN, self.TN]]))

    def is_equal(self, A):
        if type(A) != ConfusionMatrix:
            print("Argument to compare against is not of the same type")
            return 

        return (self.TP == A.TP) and (self.FP == A.FP) and (self.FN == A.FN) and (self.TN == A.TN)



    def show(self):
        print(np.array([[self.TP, self.FN], [self.FP, self.TN]]))
        return
        

def CM_metrics(CM: np.ndarray, metrics: list[str]):
    """ 
    Given a Confusion Matrix in CM, as a 2x2 numpy array; it returns the indices provided in metrics
    """
    
    # Validity of the metrics inputted check
    possible_metrics = ["Acc", "Recall", "MCC", "Jaccard", "Dice", "Kappa", "Krippendorff",
                        "AC1", "Aickin", "Entropy"]
    possible_metrics_lowercase = [m.casefold() for m in possible_metrics]
    for m in metrics:
        if m.casefold() not in possible_metrics_lowercase:
            print("Error in the metric name, please, input a valid metric name")
            print(f"Possible metrics: {possible_metrics}")
            return None
    
    # Creates the list of indices that will be returned
    indices = {metric: [] for metric in metrics}
    
    m = ConfusionMatrix(CM)  
    
    # Compute the indices asked for each CM
    for metric in metrics:
        if metric.casefold() == possible_metrics_lowercase[0]: #accuracy
            indices[str(metric)].append(m.ACC()) 
            
        elif metric.casefold() == possible_metrics_lowercase[1]: #Recall
            indices[str(metric)].append(m.TPR())
            
        elif metric.casefold() == possible_metrics_lowercase[2]: #Matthew's Correlation Coefficient
            indices[str(metric)].append(m.MCC())
         
        elif metric.casefold() == possible_metrics_lowercase[3]: #Jaccard Index
            indices[str(metric)].append(m.J())
        
        elif metric.casefold() == possible_metrics_lowercase[4]: #F1 or Dice index
            indices[str(metric)].append(m.Fbeta(1))
        
        elif metric.casefold() == possible_metrics_lowercase[5]: #Cohen's Kappa
            indices[str(metric)].append(m.Kappa())
        
        elif metric.casefold() == possible_metrics_lowercase[6]: #Krippendorff's Alpha
            indices[str(metric)].append(m.KrippAlpha())

        elif metric.casefold() == possible_metrics_lowercase[7]: #Gwet's AC1 index
            indices[str(metric)].append(m.AC1())
        
        elif metric.casefold() == possible_metrics_lowercase[8]: #Aickin's Alpha
            indices[str(metric)].append(m.Aickin())
        
        elif metric.casefold() == possible_metrics_lowercase[9]: #Entropic similarity
            indices[str(metric)].append(m.ReferenceEntropy())
    
    return indices


def CM_point_metrics(points: list, metrics: list[str], show = False, MODE = ""):
    """ 
    3D points such that x + y + z <= N, for N the number of cases can represent a CM. This 
    goes through a list of points and computes the corresponding CM, returning the appropiate
    metric provided in the argument metrics.
    """
    
    # Validity of the metrics inputted check
    possible_metrics = ["Acc", "Recall", "MCC", "Jaccard", "Dice", "Kappa", "Krippendorff",
                        "AC1", "Aickin", "Entropy"]
    possible_metrics_lowercase = [m.casefold() for m in possible_metrics]
    for m in metrics:
        if m.casefold() not in possible_metrics_lowercase:
            print("Error in the metric name, please, input a valid metric name")
            print(f"Possible metrics: {possible_metrics}")
            return None
    
    # Computation code
    indices = {metric: [] for metric in metrics}
    for n, p in enumerate(points):
        # Create the confusion matrix for each point
        a, b, d = p
        e = len(points) - np.cumsum(p)[-1]
        CM = np.array([[a, d],[e, b]])
        m = ConfusionMatrix(CM)  
        
        # Compute the indices asked for each CM
        for metric in metrics:
            if metric.casefold() == possible_metrics_lowercase[0]: #accuracy
                indices[str(metric)].append(m.ACC()) 
                
            elif metric.casefold() == possible_metrics_lowercase[1]: #Recall
                indices[str(metric)].append(m.TPR())
                
            elif metric.casefold() == possible_metrics_lowercase[2]: #Matthew's Correlation Coefficient
                indices[str(metric)].append(m.MCC())
             
            elif metric.casefold() == possible_metrics_lowercase[3]: #Jaccard Index
                indices[str(metric)].append(m.J())
            
            elif metric.casefold() == possible_metrics_lowercase[4]: #F1 or Dice index
                indices[str(metric)].append(m.Fbeta(1))
            
            elif metric.casefold() == possible_metrics_lowercase[5]: #Cohen's Kappa
                indices[str(metric)].append(m.Kappa())
            
            elif metric.casefold() == possible_metrics_lowercase[6]: #Krippendorff's Alpha
                indices[str(metric)].append(m.KrippAlpha())
    
            elif metric.casefold() == possible_metrics_lowercase[7]: #Gwet's AC1 index
                indices[str(metric)].append(m.AC1())
            
            elif metric.casefold() == possible_metrics_lowercase[8]: #Aickin's Alpha
                indices[str(metric)].append(m.Aickin())
            
            elif metric.casefold() == possible_metrics_lowercase[9]: #Entropic similarity
                indices[str(metric)].append(m.ReferenceEntropy())
        
    if show:
        for metric in metrics:
            plt.title(str(metric) +"  (MODE = " +  MODE + ")")
            plt.plot(indices[str(metric)], ".")
            plt.show()
    
    return indices


def IoU(img: np.ndarray, reference: np.ndarray) -> float:
    """ Given two images, of individual instances of a class, computes the IoU index."""

    if type(img[0][0]) != np.bool_:
        component1 = np.array(img, dtype=bool)
    else: component1 = img
    
    ref_c = reference.copy()
    if reference.max() > 200: 
        ref_c[ref_c<200] = 0  
        
    component2 = np.array(ref_c, dtype=bool)

    overlap = component1 * component2 # Logical AND
    union = component1 + component2 # Logical OR
    
    IOU = overlap.sum()/float(union.sum())
    return IOU  

def IoU_Count(img: np.ndarray, reference: np.ndarray, thresh = 0.5, paint_wrong = False) -> tuple:
    """ 
    Using the IoU() function, it computes it for images that have multiple instances of a class that have to be
    compared. 
    The threshold indicates that if the IoU score for that pair is higher than it, it will be flagged as correctly identified.
    It returns the array of scores (debugging purposes), the correctly counted items and the total number of items checked.
    
    paint_wrong makes the algorithm paint the neurons that have been found to have IoU less than 0.5
    
    Returns an image that shows the generated image in blue (1), the neurons miss identified in the generation in green (2) and
    the neurons miss identified in the reference in yellow (3).
    
    WARNING: Esta función es muy gorda, habría que modularizarla y limpiarla;
    pero eso es un trabajo para mi yo del futuro.

    NOTE:
    It is only used to generate the coloured images comparing two masks. For computing the metrics, Mask2CM and InstanceMask2CM are used.
    """

    # Binarize
    img[img>0] = 1
    reference[reference>0] = 1
    
    Maxima = hf.detect_maxima(reference)[0]

    component_r = np.array(reference, dtype=bool)
    component_g = np.array(img, dtype=bool)
    img_check = img.copy() # This image will keep track of the neurons that have been analysed succesfully

    Indices = hf.extract_non_zero(Maxima)
    IntOverUnion = []; count = 0
    wrong_indices = []

    for i in Indices: 
        # This isolates the neurons one by one in the reference mask.
        Looked = hf.size_filter_v2(reference, coord = i, n_area_m = (0,0))  
        Lookedb = np.array(Looked, dtype=bool)
        Neuron_reference = Lookedb ^ component_r
 
        # This isolates the neurons one by one in the reference image.
        Looked_r = hf.size_filter_v2(img, coord = i, n_area_m = (0,0))   
        Looked_rb = np.array(Looked_r, dtype=bool)
        Neuron_identified = Looked_rb ^ component_g
        
        result = IoU(Neuron_reference, Neuron_identified)
        IntOverUnion.append(result)
        
        img_check =  hf.size_filter_v2(img_check, coord = i, n_area_m = (0,0))
        
        if result > thresh: count += 1
        else: 
            wrong_indices.append(i)
     
    wrong_indices.append(None)
            
    
    # Now it remains to check the neurons the algorithm detects but don't appear in the expert mask. "Phantom".
    Phantom_img, Phantom = hf.detect_maxima(img_check)
    if Phantom != 0:
        for i in hf.extract_non_zero(Phantom_img):
            wrong_indices.append(i)
        
    #Now i have all the indices of wronged neurons. Just have to "and" both images with just those neurons.
    in_reference_flag = True
    Looked = reference.copy(); img_check = img.copy(); Neuron_image = img.copy(); Neuron_reference_g = 0; N_R_b = 0
    for i in wrong_indices:
        # I need miss identifications, miss in identification and wrong identifications.
        # Wrong in the reference
        if type(i) == type(None): 
            in_reference_flag = False
            continue
        if in_reference_flag:
            Looked = hf.size_filter_v2(Looked, coord = i, n_area_m = (0,0))  
            Lookedb = np.array(Looked, dtype=bool)
            Neuron_reference = 3 * (Lookedb ^ component_r).astype(int)  # 2 on top of the wrong neurons.
    
            Neuron_image = hf.size_filter_v2(Neuron_image, coord = i, n_area_m = (0,0))  
            Ni_b = np.array(Neuron_image, dtype=bool)
            N_R_b = 2 * (Ni_b ^ component_g).astype(int)  # 2 on top of the wrong neurons.
    
        else:
            img_check = hf.size_filter_v2(img_check, coord = i, n_area_m = (0,0))  
            img_check_b = np.array(img_check, dtype=bool)
            Neuron_reference_g = 2 * (img_check_b ^ component_g).astype(int)
    
    
    img_wrong = np.maximum(np.maximum(np.maximum(img, Neuron_reference), Neuron_reference_g), N_R_b)

    print("Image done...")
    return IntOverUnion, ((count, int(Phantom)), len(Indices)), img_wrong  # IoU score, ((correct_neurons, neurons_invented_by_the_alg), total_count), Image with the colors that show how good or bad the identification is

def MaskComparison(expert_f: str, gen_data: str, save_path = "") -> tuple:
    """ 
    expert_f: Path to the masks that will serve as reference. 
    gen_data: Path to the masks generated
    save_path: Saves a txt file with the numbers computed
    
    Returns a tuple with the images paired and the data.
    
    """

    expert = next(os.walk(expert_f), (None, None, []))[2]  # [] if no file
    expert_num = [s.rsplit("F", 1)[0] for s in expert]  # Extracts the number image, format has to be of type XXF....tif. eg: 14F_maite.tif ---> 14  
    
    R = next(os.walk(gen_data), (None, None, []))[2]  # [] if no file
    R_num = [s.rsplit("F", 1)[0] for s in R]  # Extracts the number image, format has to be of type XXF....tif. eg: 14F_maite.tif ---> 14  

    # Pairs the files so that they are compared by pairs that actually match, if both sets are different.
    Data_pair = []
    for n, e in enumerate(expert_num):
        for m, r in enumerate(R_num):
            if e == r:
                Data_pair.append((expert[n], R[m]))
    
    # Loads all images
    imgs = [cv.imread(fr"{expert_f}\{i[0]}", 0) for i in Data_pair]
    imgs_g = [cv.imread(fr"{gen_data}\{i[1]}", 0) for i in Data_pair]
    
    if imgs == []: 
        print("Images couldn't be imported. Make sure the F identifier is in its name after the number.\n")
        print("Returning from the function call unsuccessfully")
        return None

    # Generates neuron counts
    expert_count = neuron_count(imgs)
    gen_count = neuron_count(imgs_g)

    # Compares the neuron count
    if save_path != "":
        with open(save_path, "w") as f:
            f.write("\n__________________________________NEW Execution__________________________________________\n")
        
        for n, i in enumerate(Data_pair):
            with open(save_path, "a") as f:
                f.write(f"gen: {gen_count[n]}; Maite -> {expert_count[n]} ---> Relative : {(gen_count[n] - expert_count[n])/expert_count[n]*100}%\n")
      

    return imgs, imgs_g, (expert_count, gen_count, expert_num) # list[np.ndarray] x2, tuple x2

def Mask2CM(T: np.ndarray, G: np.ndarray, normalise = False) -> ConfusionMatrix:
    """ 
    Given 2 Masks generates the corresponding CM
    """
    # Size (Total number)
    N = T.shape[0] * T.shape[1]
    
    # Binarize
    T[T != 0] = 1
    G[G != 0] = 1
    
    # Get the values
    TP = np.sum(np.bitwise_and(T, G))
    FP = np.sum(np.bitwise_and(1 - T, G))
    FN = np.sum(np.bitwise_and(T, 1 - G))
    TN = np.sum(1 - np.bitwise_or(T, G))
    
    # Generates the CM object
    CM = np.array([[TP, FN], [FP, TN]])    
    N_CM = CM / N * 100
    
    if normalise:
        return ConfusionMatrix(N_CM)
    else:
        return ConfusionMatrix(CM)
   
    
def InstanceMask2CM(i_mask1: np.ndarray, i_mask2: np.ndarray, normalise = False) -> ConfusionMatrix:
    """ 
    Given 2 Masks generates the corresponding CM in an instance basis. TP are recognised in both masks, FP is recognised in 1 but not in 2, FN is 
    recognised in 2 but not in 1 and TN doesn't make much sense in an instance context. To asses the index of agreement without influence of the TN = 0
    the Jaccard metric is better suited; while the valid count is also interesting.
    """
    TP = 0; FP = 0; FN = 0
    box = (40, 40)

    # Binarize
    mask1 = i_mask1.copy(); mask2 = i_mask2.copy()
    mask1[mask1>0] = 1; mask2[mask2>0] = 1
    
    maxima = hf.detect_maxima(mask1)[0]
    indices = hf.extract_non_zero(maxima)

    forgot_in_mask2 = mask2.copy()
    for index in indices:
        if mask1[index] == 0:
            continue

        xm = max(index[0] - box[0], 0)
        xM = min(index[0] + box[0] + 1, mask1.shape[0])
        ym = max(index[1] - box[1], 0)
        yM = min(index[1] + box[1] + 1, mask1.shape[1])

        cut1 = mask1[xm:xM, ym:yM]
        cut2 = mask2[xm:xM, ym:yM]

        # Adapt box based on the fact that it hits the border or not.
        box_a = [box[0], box[1]]
        if xm == 0 : box_a[0] = index[0]
        elif ym == 0: box_a[1] = index[1]
        box_a = (box_a[0], box_a[1])

        neuron1 = hf.size_filter_v2(cut1, coord = box_a, n_area_m=(0,0))
        neuron_in1 = (1 - neuron1) * cut1 # Now the neuron at index has been isolated.

        displaced_index2 = (box_a[0], box_a[1])
        if cut2[box_a] == 1:
            neuron2 = hf.size_filter_v2(cut2, coord = box_a, n_area_m=(0,0))
            neuron_in2 = (1 - neuron2) * cut2 
        else:  # If the location in mask1 doesn't match a location in mask2, could be that the two neurons are not aligned.
            neuron_in2 = False
            indices_extra = hf.extract_non_zero(neuron_in1)
            for i in indices_extra:
                if cut2[i] == 1:
                    neuron2 = hf.size_filter_v2(cut2, coord = i, n_area_m=(0,0))
                    neuron_in2 = (1 - neuron2) * cut2
                    displaced_index2 = i
                    break
            if type(neuron_in2) == type(False): # Neuron not found -> Mask1 has a neuron that Mask2 doesn't -> FalsePositive
                FP += 1
                hf.clean_area(mask1, coordinate = index)
                continue
        
        iou = IoU(neuron_in1, neuron_in2)
        if iou >= 0.3: # Neuron found in both and correctly identified -> True Positive
            TP += 1

            # Set the current searched neuron to black so its not detected again.
            index_d = (xm + displaced_index2[0], ym + displaced_index2[1])

            hf.clean_area(forgot_in_mask2, coordinate = index_d)  
        else: # Neuron found in Mask1 but not correctly identified in Mask2 -> False Positive
            FP += 1

        hf.clean_area(mask1, coordinate = index)
    
    # These are neurons that haven't been found in the mask1 pass; that means they have been identified in Mask2 but have no correspondence in Mask1.
    indices2 = hf.detect_maxima(forgot_in_mask2)[0] 

    FN = 0
    for index2 in hf.extract_non_zero(indices2):
        if forgot_in_mask2[index2] != 0: 
            FN += 1
            hf.clean_area(forgot_in_mask2, coordinate = index2)

    # Generates the CM object
    CM = np.array([[TP, FN], [FP, 0]])    
    N_CM = CM / (TP + FN + FP) * 100
    
    if normalise:
        return ConfusionMatrix(N_CM)
    else:
        return ConfusionMatrix(CM)
   

def neuron_count(imgs: list[np.ndarray]) -> list[int]:
    """
    Gets the neuron count for each image in imgs. For binary images
    """

    expert_count = []
    for im in imgs:
        im[im < 100] = 0; im[im >= 100] = 255

        Locations, Count = hf.detect_maxima(im)
        expert_count.append(int(Count))  
    
    return expert_count

def MetricsFromCM(e: str, g: str, save_name) -> ConfusionMatrix:
    """
    The optimal situation would be where all the elements outside the main diagonal of the confusion matrix are zero,
    resulting in a precision and recall of $1$.  e and g are the paths to the expert and generated image folders.
    """
    I, Ig, F = MaskComparison(e, g)
    
    CM = [Mask2CM(I[n], Ig[n]) for n in range(len(I))]

    MCC = [M.MCC() for M in CM]
    Jac = [M.J() for M in CM]
    TPR = [M.TPR() for M in CM]
    PPV = [M.PPV() for M in CM]

    x = np.arange(0, len(CM))

    # fig = plt.figure(constrained_layout=True)
    # axs = fig.subplot_mosaic([['TopLeft', 'TopRight'],['Bottom', 'Bottom']],
    #                          gridspec_kw={'width_ratios':[1, 1], 'height_ratios':[1, 2]})
    #
    # fig.suptitle(f"{save_name[:-4]}")
    #
    # axs["TopLeft"].plot(x, MCC,'.')
    # axs["TopLeft"].set_title("MCC")
    #
    # axs["TopRight"].plot(x, Jac,'.')
    # axs["TopRight"].set_title("Jaccard")
    #
    # axs["Bottom"].plot(TPR,PPV,  '.')
    # axs["Bottom"].set_title("Precision-Recall")
    # axs["Bottom"].axis([0, 1, 0, 1])
    #
    # plt.show()

    # with open(rf"{g}\{save_name}", "wb") as f:
    #         pkl.dump(CM, f)
    
    return CM


def GraphicalMetrics(e, g, save_folder = ""):
    """
    This function computes the difference instance-wise between two pairs of masks and returns several indices. It also
    returns a comparison image coded with colouring, based on whether or not the neuron was missed by one, by the other
    or it is correct.

    e and g are both paths to the folders that contain the images.
    """
    I, Ig, F = MaskComparison(e, g)

    metrics = [IoU_Count(I[n], Ig[n], paint_wrong=True) for n in range(len(I))] # The format is a tuple for each image, in the same order they appear in I.

    if save_folder != "":
        for n, i in enumerate(metrics):
            fig = plt.figure()
            plt.tight_layout()
            plt.title(f"{F[-1][n]}")
            plt.imshow(i[-1])
            plt.savefig(fr"{g}\{save_folder}\{F[-1][n]}_EP_comp.png", dpi = 200)
            
            plt.close(fig)
    
    print(f"Images {F[-1]} completed")

    # metrics[n] = ( IoU score, ((correct_neurons, neurons_invented_by_the_alg), total_count), Image with the colors that show how good or bad the identification is)
    # F[-1] is a list that indexes the numbers of the images in metrics, so the order can be consulted after running the script.
    return metrics, F[-1]

def insert_line(file, flags, text, once = True, jump = 0):
    """ Inserts the text [text] after the lines containing anything in [flags], in [file]"""
    index = None

    for num, line in enumerate(file):
        if type(index) != type(None) and once: break
        for flag in flags:
            if flag in line:
                index = num
                file.insert(index + 1 + jump, text)
    return 

def MetricCSV(path_to_images, expert_names: list[str], path: str):
    """
    Generates a complete .csv file with all metrics together, for just 1 image.
    path_to_images has to have the format [[image 1 dataset], [image 2 dataset], ....]
    """
    N = len(expert_names)

    f_images = ["" for i in range(0, N)]
    for n, path_images in enumerate(path_to_images):
        files = next(os.walk(path_images), (None, None, []))[2]
        files_temp = files.copy()
        for f in files_temp:
            if f[-4:] != ".tif":
                files.remove(f)
        f_images[n] = files
        

    # Generate the backbone of the file.
    naiyou = []
    for n, i in enumerate(expert_names):
        naiyou.append(f"R{n}: {i}")
    naiyou.append("Global")
    naiyou.append(" - MCC")
    naiyou.append(" - IoU")
    naiyou.append(" - (R, P)")
    naiyou.append("Instances")
    naiyou.append(" - Jaccard (instance)")
    naiyou.append(" - F1 Score (instance)")
    naiyou.append(" - Count | (missed by L, missed by R)")

    # Generate the R1-2, R1-3, R1-4, R2-3, R2-4, ... headers for each metric
    header = "   "
    for n in range(0, N):
        for m in range(n+1, N):
            header += "R%s-%-20s" %(n, m)
    header += "Image number"

    insert_line(naiyou, ["MCC", "IoU", "(R, P)", "Jaccard", "F1 Score", "Count"], header, once = False)

    # Code to import the images and write the file.

    # First make sure all paths contain the same number of images.
    if not all(len(x) == len(f_images[0]) for x in f_images):
        print("The number of images to be loaded in the paths is not the same for all raters.\n")
        for i in f_images:
            print(f"{len(i)}\n {i}")
        return False
    
    for pointer in range(len(f_images[0])):
        # Load the images to be compared.
        image_name = f_images[0][pointer].rsplit("F")[0] + "F"
        images = [cv.imread(f"{path_to_images[expert]}\\{path[pointer]}", 0) for expert, path in enumerate(f_images)]
        for i in images:
            if type(i) == type(None):
                print(f"Something went wrong loading the image {image_name}. Make sure the path is valid and the images are accessible.")
                return False

        #Global
        mcc_line = ""
        iou_line = ""
        pr_line = ""
        for n, i in enumerate(images):
            for m in range(n + 1 , len(images)):

                CM = Mask2CM(i, images[m])
                MCC = CM.MCC()            
                JAC = CM.J()
                R = np.round(CM.TPR(), 5)
                P = np.round(CM.PPV(), 5)

                mcc_line += "   %-20s" %(MCC)
                iou_line += "   %-20s" %(JAC)
                pr_line += "   (%-7s,%-7s)%3s" %(R, P, "")
        
        mcc_line += "        " + image_name
        iou_line +=  "        " + image_name
        pr_line +=  "        " + image_name
        insert_line(naiyou, ["MCC"], mcc_line, jump = 1 + pointer)
        insert_line(naiyou, ["IoU"], iou_line, jump = 1 + pointer)
        insert_line(naiyou, ["(R, P)"], pr_line, jump = 1 + pointer)

        #Instance
        iou_line = ""
        f1_line = ""
        count_line = ""
        countline = ""
        for n, i in enumerate(images):
            for m in range(n+1, len(images)):
                print(f"Working {image_name}... Done ({n},{m}) / ({len(images) - 1}, {len(images) - 1})")
                
                CM = InstanceMask2CM(i, images[m])
                JAC = CM.J()
                F1 = CM.Fbeta(1)

                iou_line += "   %-20s" %(JAC)
                f1_line += "   %-20s" %(F1)
                count_line = "%-4d | (%d,%d)" %(CM.TP, CM.FP, CM.FN) 
                countline += "   %-20s" %(count_line)

        iou_line +=  "        " + image_name
        f1_line +=  "        " + image_name
        countline +=  "        " + image_name
        insert_line(naiyou, ["Jaccard"], iou_line, jump = 1 + pointer)
        insert_line(naiyou, ["F1 Score"], f1_line, jump = 1 + pointer)
        insert_line(naiyou, ["Count"], countline, jump = 1 + pointer)

    naiyou = "\n".join(naiyou)
    with open(path, "w") as f:
        f.write(naiyou)

    return True

def MetricsParser(path: str):
    """
    Given a path to a .txt generated by the algorithm MetricsCSV(); it parses it and separates the relevant elements in the tables.
    """
    try: 
        with open(path + r"\Metrics3.txt", "r") as f:
            naiyou = f.readlines()
    except FileNotFoundError as e:
        print(e)
        return
    metrics = [[], [], [], [], [], []] # mcc | iou | rp | jac | f1 | count
    pointer = -1; process = False
    for i in naiyou:
        i = i.strip()
        if i[0:3] == "R0-":
            pointer += 1
            process = True
            continue
        elif not i[1].isnumeric() and i[1] != ".":
            process = False

        if process:
            if i[0] == "(" or ")" == "This is just to restore autoindentation in neovim.":
                
                line = i.split(")")
                for n, element in enumerate(line):
                    line[n] = element.strip() + ")"
                    if n == len(line) - 1:
                        line[n] = line[n][:-2]
                metrics[pointer].append(line)    

            else:
                line = i.split()
                line[-1] = line[-1][:-1]
                metrics[pointer].append(line)    

    # Each element is a collection of lists such that they hold the contents of a row in the file. Each list is comprised for the column elements
    # of that row, separated. This is the general structure (eg. for mcc) 
    # metrics = [file]; file = [each metric]; each metric = [rows]; row = [R0-1, R0-2, ..., Img number]
    return metrics
    
def Metrics2TEX(metrics, slice, path):
    """
    The input of the function has to come from applying the previous function MetricsParser(). This builds Tex code into a .txt such that it can
    be copied and pasted into a .tex file and build a table. This code probably has to be adapted depending on the table you want to build. 
    slice selects the actual metric to be put into the table.

    NOTE:
    It provides the basic template.
    """
    metric = metrics[slice]
    assert type(int(metric[0][-1])) == int, f"Error: The image number format ({metric[0][-1]}) is not compatible with the program."
    metric.sort(key = lambda x: int(x[-1]))
    output = []
    for i in metric:
        output.append(fr"\thead{{{i[-1]}}} & \thead{{${float(i[0]):.2f}$}} & \thead{{${float(i[1]):.2f}$}} & \thead{{${float(i[2]):.2f}$}} & \thead{{${float(i[3]):.2f}$}}& \thead{{${float(i[4]):.2f}$}} & \thead{{${float(i[5]):.2f}$}}\\")

    output = "\n".join(output)
    with open(path + r"\tex.txt", "w") as f:
        try:
            f.write(output)
            print(f"Results saved in {path}")
        except IOError:
            print("Couldn't save the file: _MetricFunctions.Metrics2TEX")

def Metrics2Plot(metrics, slice):
    """
    Gets the metrics produced by MetricsParse() and has different plot methods.
    Tbh, if, for some reason, you need to debug or tweak this function; consider rewriting it from scratch :)
    """
    try:
        metric = metrics[slice]
        comparisons = len(metric[0]) - 1
        # metric.sort(key = lambda x: int(x[-1]))
        grouped_metrics = [[] for i in range(comparisons)]
    except TypeError:
        pass
    
    if slice in [0,1,3,4]:
        # Here comes the MCC(0) and IoU(1) at a pixel level. At the instance level we got Jac(3) and F1(4).
        for i in metric:
            for n, j in enumerate(i):
                if n == comparisons: continue
                grouped_metrics[n].append(float(j))

        # The data is ready to be processed. Each component of the grouped_metrics correspond to one pair of raters; in the order (01, 02, 03, 12, 13, 23); easily generalisable for more raters.
        plt.ylim(0,1)
        plt.boxplot(grouped_metrics)
        plt.xticks([1,2,3,4,5,6],["R0 vs. R1","R0 vs. R2","R0 vs. R3","R1 vs. R2","R1 vs. R3","R2 vs. R3"])
        plt.show()

    elif slice == 2:
        fig = plt.figure()
        plt.xlim(0,1); plt.ylim(0,1)
        raters = ["R0 vs. R1","R0 vs. R2","R0 vs. R3","R1 vs. R2","R1 vs. R3","R2 vs. R3"]
        for i in metric:
            for n, j in enumerate(i):
                if n == comparisons: continue
                grouped_metrics[n].append(j)

        for m, expert in enumerate(grouped_metrics):
            for n, point in enumerate(expert):
                point = list(map(float, point.strip("()").split(",")))
                expert[n] = point
        
            expert_datapair = [[],[]]
            for i in expert:
                for n, j in enumerate(i):
                    expert_datapair[n].append(j)
            plt.plot(expert_datapair[0], expert_datapair[1], ".", label = raters[m])
        plt.legend()
        plt.show() 
        plt.close(fig)
    
    elif slice == 5:
        
        for i in metric:
            for n, j in enumerate(i):
                if n in [comparisons, 1, 4, 7, 10, 13, 16]: continue
                grouped_metrics[n].append(j)
        cleaned_metrics = [m for m in grouped_metrics if m != []] # Now the format is: [id1,(l1, r1), id2, (l2, r2), id3, (l3, r3)] 
        groups = [[] for i in range(len(cleaned_metrics) // 2 )]

        for index, m in enumerate(cleaned_metrics):
            if index % 2 == 1: continue
            comp = index // 2
            for n, j in enumerate(m):
                groups[comp].append([j, cleaned_metrics[index + 1][n]])

        # Now I have to compute the relative left and right misses for each group of raters. Current format is [R01, R02, R12,...]
        # for each R01 = [[img1], [img2], ....] and each imgn = [id, (l, r)]
        final_pass = [[] for i in range(len(cleaned_metrics) // 2 )]
        for n, group in enumerate(groups):
            rel_l, rel_r = [], []
            for img in group:
                id = int(img[0])
                lr, rr = img[1].strip("()").split(",")
                rel_l.append(1 - float(lr)/(id+float(lr)))
                rel_r.append(1 - float(rr)/(id+float(rr)))
            final_pass[n] = [rel_l, rel_r]

        # Now the data is ready to be passed into the plotbox.
        colors = ["#f0a0a0","#8080f0"]
        groups = [r"$\Delta_1$", r"$\Delta_2$"]
        legend_elements = []
        ax = plt.axes()
        for fn, f in enumerate(final_pass):
            for sln in [0,1]:
                bp = ax.boxplot(f[sln], manage_ticks=False,
                           widths=0.6,
                           patch_artist=True,
                           positions = [-0.5 + sln + 4 * fn])
                # Fill the boxes with colours (requires patch_artist=True)
                plt.setp(bp['medians'], color='black')
                k = sln % len(colors)
                for box in bp['boxes']:
                    box.set(facecolor=colors[k])
                if fn == 1: 
                    legend_elements.append(Patch(facecolor=colors[k], label=groups[sln]))

        ax.legend(handles=legend_elements)

        ax.set_ylim(0, 1)
        # ax.set_xticks([0, 4, 8, 12, 16, 20])
        # ax.set_xticklabels(["R0 vs. R1","R0 vs. R2","R0 vs. R3","R1 vs. R2","R1 vs. R3","R2 vs. R3"])
        ax.set_xticks([0, 4, 8])
        ax.set_xticklabels(["R0 vs. Alg","R1 vs. Alg","R2 vs. Alg"])
        plt.grid(axis="y", linestyle = "--")
        plt.show()
            

    elif slice == "Plot all":
        colors = ["#f0f0f0", "#f0e0b0", "#d0a0a0", "#8890ff"]
        groups = ["MCC (pixel)", "IoU (pixel)", "Jac (instance)", "F1 (instance)"]
        legend_elements = []
        ax = plt.axes()
        comparisons = len(metrics[0][0]) - 1
        for sln, sl in enumerate([0, 1, 3, 4]):
            grouped_metrics = [[] for i in range(comparisons)]
            for i in metrics[sl]:
                for n, j in enumerate(i):
                    if n == comparisons: continue
                    grouped_metrics[n].append(float(j))
            
            bp = ax.boxplot(grouped_metrics, manage_ticks=False,
                       patch_artist=True,
                       positions = [-1.5 + sln, 4.5 + sln, 10.5 + sln, 16.5 + sln, 22.5 + sln, 28.5 + sln])

            # Fill the boxes with colours (requires patch_artist=True)
            k = sln % len(colors)
            for box in bp['boxes']:
                box.set(facecolor=colors[k])
            plt.setp(bp['medians'], color='black')
            legend_elements.append(Patch(facecolor=colors[k], label=groups[sln]))

        ax.legend(handles=legend_elements)

        ax.set_ylim(0, 1)
        ax.set_xticks([0, 6, 12, 18, 24, 30])
        ax.set_xticklabels(["R0 vs. R1","R0 vs. R2","R0 vs. R3","R1 vs. R2","R1 vs. R3","R2 vs. R3"])
        plt.grid(axis="y", linestyle = "--")
        plt.show()


if __name__ == '__main__':
    if 0: # This if computes and works following the new code (sept 2026)

        # sigma1_1 has been updated to a very similar value, so other metrics should not change their behaviour too much.
        # possible names, with the baseline values: 
        names = {"sigma1_1": 2.0 , "sigma1_2": 2.0, "fraction2_2": 0.8, "sigma3_1": 2.5, "box_half4_1": 17, "s_int4_2low": 10,
                 "s_int4_2high": 30, "eccentricity4_3": 200, "s_lim4_4low": 100, "s_lim4_4high": 550,
                 "pixels5_1": 10, "loops6_1": 5}

        EXP_NAME = "new_baseline"
        # base_value = names[EXP_NAME]

        img_path = "../images"
        # [esther, maite, antonello]
        experts = [f"{img_path}/Esther/MascarasEsther/images/", 
                   f"{img_path}/Antonello/Krita-AN/Completed/",
                   f"{img_path}/Maite/1_Mascaras Maite/Selected/"]

        # algs = f"{img_path}/Masks/baseline"
        algs = f"{img_path}/Masks/{EXP_NAME}"

        # These work as the x labels for the graphs.
        sensitivity_values = [entry.name for entry in os.scandir(algs) if (entry.is_dir() and entry.name[0] != "_")]
        if len(sensitivity_values) == 0: sensitivity_values = [""]

        # sensitivity_values = os.listdir(algs)

        mccs = []; ious = []; jacs = []
            
        for alg in track(sensitivity_values):
            if alg != "" and alg[-1] == "b": continue
        
            metrics = [[] for i in experts]
            for n, exp in enumerate(experts):
                # This computes the metrics MCC, Jaccard and PR pair for every two pair of images in the folders.

                # pixel level
                I, Ig, F = MaskComparison(exp, f"{algs}/{alg}")
                CM = [Mask2CM(I[n], Ig[n]) for n in range(len(I))]
                MCC = [M.MCC() for M in CM]
                iou = [M.J() for M in CM]

                # instance level
                CM = [InstanceMask2CM(I[n], Ig[n]) for n in range(len(I))]
                Jac = [M.J() for M in CM]

                metrics_1exp = [MCC, iou, Jac]

                # This is a list where each entry is the comparison of 1 expert against the algorithm
                #  [esther, maite, anto] -> esther = [MCC, IOU, JAC] -> MCC [img1, img2, img3, ....] and so on for the others.
                metrics[n] = metrics_1exp
            
            # flatten it out in the expert dimension so all are analyzed together.
            metric_flat = [np.concatenate(metrics[:][i]) for i in range(len(experts))]

            # Append it to the corresponding senstivity value.
            mccs.append((np.mean(metric_flat[0]), np.std(metric_flat[0])))
            ious.append((np.mean(metric_flat[1]), np.std(metric_flat[1])))
            jacs.append((np.mean(metric_flat[2]), np.std(metric_flat[2])))

        # These are the baseline comparisons. I guess I could put them at the beginning of the plots for the moment.
        baseline_mcc = (0.7951821369122605, 0.08541163964642061)
        baseline_iou = (0.8048013095397356, 0.08331462431601461)
        baseline_jac = (0.8013761280718188, 0.07756901253391658)
        # NOTE: To know where the baseline goes, I should find the folder name with a "b" at the end of the name. 
        # It tells me then the index at which the baseline should be incrusted.
        index = 0
        if len(sensitivity_values) != 1:
            while sensitivity_values[index][-1] != "b" and index < len(sensitivity_values)-1: index += 1; continue
            if index == len(sensitivity_values): index = 0
        mccs.insert(index, baseline_mcc)
        ious.insert(index, baseline_iou)
        jacs.insert(index, baseline_jac)

        # Calculate mean and std for the metrics.
        mcc_mean = [M[0] for M in mccs]
        mcc_std = [M[1] for M in mccs]
        iou_mean = [M[0] for M in ious]
        iou_std = [M[1] for M in ious]
        jac_mean = [M[0] for M in jacs]
        jac_std = [M[1] for M in jacs]

        
        plt.figure(figsize=(8, 6))
        x_pos = np.arange(len(mcc_mean))
        displacement = np.array([0.15 for i in range(len(mcc_mean))])
        
        plt.errorbar(x_pos, mcc_mean, yerr=mcc_std, fmt='o', markersize=5, 
                     capsize=3, capthick=1, elinewidth=1, label='MCC(pixel)')
        plt.errorbar(x_pos + displacement, iou_mean, yerr=iou_std, fmt='o', markersize=5, 
                     capsize=3, capthick=1, elinewidth=1, label='IoU(pixel)')
        plt.errorbar(x_pos + 2*displacement, jac_mean, yerr=jac_std, fmt='o', markersize=5, 
                     capsize=3, capthick=1, elinewidth=1, label='Jac(instance)')

        x_ticks = []
        if sensitivity_values == [""]:
            x_ticks.append("b")
        for s in sensitivity_values:
            x_ticks.append(s.replace("_","."))
        plt.xticks(x_pos + displacement, x_ticks)
        # plt.ylim((0,1))
        plt.ylabel('MCC')
        plt.xlabel(EXP_NAME)
        plt.title(f'Metric sensitivity to {EXP_NAME}')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        save_name = input(f"Input the name of the file (without extension) to save the numpy data. No path means no need to save: \n -> ")
        if save_name != "":
            np.save(f"../images/Plots_sensitivity/{save_name}", [mccs, ious, jacs])

    else: # This is the old code (from 2024)
         
        # Computes the individual differences between two masks in a graphical way and represents it with colors.
        # metrics, order = GraphicalMetrics(gen_loop0, antonello, save_folder = "coloured_comp_alg")

        # Check if I'm erasing the neurons at the borders in the expert masks.!!!!
        
        # Generates the .txt file
        txt_path = r"./"

        img_path = "../images"
        experts = [f"{img_path}/Esther/MascarasEsther/images/", 
                   f"{img_path}/Antonello/Krita-AN/Completed/",
                   f"{img_path}/Maite/1_Mascaras Maite/Selected/"]

        # succeeded = MetricCSV([experts[0], experts[1], experts[2], f"{img_path}/Masks/new_baseline/exp3/"],
        #                       ["Esther", "Antonello", "Maite", "Algoritmo"],
        #                       path = rf"{txt_path}/Metrics3.txt")

        # Parses the previously generated .txt and returns the metrics into the variable m. 
        m = MetricsParser(txt_path)   

        # Generates the Tex code to copy and paste into a LaTex file and generate the big datatable.
        # Metrics2TEX(m, 4, path = txt_path)

        # Generates the plot representations for the metrics in metrics.txt
        # Metrics2Plot(m, 0)       # mcc | iou | rp | jac | f1 | count    slice = "Plot all" for a joint plot.
        Metrics2Plot(m, 5)

    
    
