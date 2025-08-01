# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 11:35
# @Author  : dx
# @FileName: mlknn.py
# @Function:
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
import numpy as np
from ensemble.ensembles import MLKNN
from sklearn.decomposition import PCA
import joblib


def mlknn(spts):
    x_train = load_npz(r"datasets\x_train.npz").A#训练集,type:<ndarray>
    stp = spts
    symdic = np.load('mode\symdic.npy', allow_pickle=True).item()
    meddic = np.load('mode\meddic.npy', allow_pickle=True).item()  ##
    stp = [symdic[x] for x in stp]
    stps = np.zeros((1, len(symdic)))
    for i in stp:
        stps[0, i] = 1
    pca = joblib.load(r'mode\pca.m')
    x_train[0] = pca.transform(stps)
    prob = MLKNN(90).fitmlknn(x_train)
    prob = sorted(zip(prob, range(len(prob))), reverse=True)
    meddic = dict(zip(meddic.values(), meddic.keys()))
    medpri = [meddic[prob[i][1]] for i in range(10)]
    return medpri


def onenn(spts):
    x_train = load_npz(r"datasets\x_train.npz").A
    #y_train=load_npz(r'datasets \y_train.npz') .A
    stp = spts
    symdic = np.load('mode\symdic.npy', allow_pickle=True).item()
    meddic = np.load('mode\meddic.npy', allow_pickle=True).item()  ##
    stp = [symdic[x] for x in stp]
    stps = np.zeros((1, len(symdic)))
    for i in stp:
        stps[0, i] = 1
    pca = joblib.load(r'mode\pca.m')
    x_train[0] = pca.transform(stps)
    near=MLKNN(90).fitonenn(x_train)
    data=pd.read_excel (r"datasets\data_all.xlsx")
    medpri=data.iloc[near,2].split(";")
    return medpri

#print(nearest('带下绵注;色白质稀;苔薄根微腻;头昏;腰楚;气短;疲乏;怯冷时作;脉细'.split(';') ) )