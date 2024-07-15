# descriptor.py
import cv2 # traitement d'images et des vidéos
import numpy as np
from skimage.feature import graycomatrix, graycoprops #pour calculer la matrice de cooccurrence de niveaux de gris (GLCM) et pour extraire les propriétés de cette matrice
from BiT import bio_taxo

def glcm(image):#méthode pour extraire les car img avec GLCM
    co_matrix = graycomatrix(image, [1], [0], symmetric=True, normed=True)
    diss = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [diss, cont, corr, ener, asm, homo]

def bitdesc(image):
    return bio_taxo(image)#pour extraire des caractéristiques d'une image.

def glcm_bit(image_path):
    glcm_features = glcm(image_path)
    bit_features =bitdesc(image_path)
    combined_features = glcm_features + bit_features
    return combined_features
