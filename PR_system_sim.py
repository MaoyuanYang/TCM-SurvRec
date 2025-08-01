#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 11:38
# @Author  : dx
# @File    : Similarity_v2.py
# @Software: PyCharm
# @Note    :
import pandas as pd
import numpy as np
import networkx as nx
from gensim.models import Word2Vec


class PrSystemSim(object):
    def __init__(self):
        # treatment data
        self.treatment_data = pd.read_excel('datasets/Treatment_202203.xlsx')
        self.treatment_symptoms = self.treatment_data['Symptom'].tolist()
        self.treatment_herbs = self.treatment_data['Herb'].tolist()
        self.treatment_formulas = self.treatment_data['Prescription'].tolist()

    @staticmethod
    def jaccard(set_x: set, set_y: set) -> float:
        """
        Calculate jaccard index with 2 string sets.
        Note that the union set can't be empty!
        :param set_x: set x
        :param set_y: set y
        :return: jaccard value, float type.
        """
        intersection = set_x.intersection(set_y)
        union = set_x.union(set_y)
        return len(intersection) / len(union)

    def main(self, symptoms: str):
        """
        main process
        :param symptoms: the symptom
        :return:
        """
        # step 1: input
        patient_symptom = set(symptoms.split(';'))  # attention for the split index !!

        # step 2: iter the treatment data, then calculate similarity between the patient sim and treatment data
        similarities = []
        similarities_append = similarities.append

        for treatment_index in range(self.treatment_data.shape[0]):
            temp_treatment_symptom = set(self.treatment_symptoms[treatment_index].split(','))
            temp_similarity = self.jaccard(patient_symptom, temp_treatment_symptom)
            similarities_append(temp_similarity)

        # step 3: form the result
        result_df = pd.DataFrame()
        assert len(similarities) == len(self.treatment_formulas)
        result_df['Symptom'] = self.treatment_symptoms
        result_df['Formula'] = self.treatment_formulas
        result_df['Herb'] = self.treatment_herbs
        result_df['Similarity'] = similarities

        # filter the result
        result_df.sort_values(by='Similarity', inplace=True, ignore_index=True, ascending=False)
        result_df.to_excel('datasets/Similarity_result.xlsx', index=False, encoding='utf-8-sig')
        top = result_df.head(1)
        return top.values.tolist()[0]


if __name__ == '__main__':
    # input: the symptom list of patient
    symptom = '咳嗽声重;咽痒;咳白稀痰;无汗;舌苔薄白;脉浮或浮紧'
    # initialization
    Sim = PrSystemSim()
    # calculate and output
    top = Sim.main(symptom)
    print(top)
