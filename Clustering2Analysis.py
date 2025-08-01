#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 9:22
# @Author  : dx
# @File    : Clustering2Analysis.py
# @Software: PyCharm
# @Note    :
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from itertools import combinations
from scipy import stats
from tqdm import tqdm
import sys
from scipy.stats import chi2_contingency, ttest_ind
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
import time
import os

plt.rcParams['font.sans-serif'] = 'Times new roman'


class Clustering2Analysis(object):
    def __init__(self, path):
        # self.data_path = os.getcwd()
        # self.basic_info = pd.read_excel(self.data_path + '\data' + '\基线.xlsx')
        # self.symptom = pd.read_excel(self.data_path + '\data' + '\症状拆分_0317.xlsx')
        # self.herb = pd.read_excel(self.data_path + '\data' + '\中药用量.xlsx')
        # self.original_data = self.process_data()
        self.original_data = pd.read_excel(path)

    @staticmethod
    def explode(df: pd.DataFrame):
        df[df.columns[1]] = df[df.columns[1]].str.split(',')
        return df.explode(df.columns[1])

    @staticmethod
    def chi2_test(ci, cj, cij, n):
        """
        Chi2 test
        :param ci:
        :param cj:
        :param cij:
        :param n:
        :return:
        """
        a1 = cij
        a2 = ci - cij
        a3 = cj - cij
        a4 = (n - ci) - (cj - cij)
        t1 = (a1 + a2) * (a1 + a3) / n
        t2 = (a1 + a2) * (a2 + a4) / n
        t3 = (a1 + a3) * (a3 + a4) / n
        t4 = (a3 + a4) * (a2 + a4) / n
        if (t1 < 5) | (t2 < 5) | (t3 < 5) | (t4 < 5) | (n < 40):
            if n >= 40 and (1 <= t1 < 5 or 1 <= t2 < 5 or 1 <= t3 < 5 or 1 <= t4 < 5):
                res = chi2_contingency([[a1, a2], [a3, a4]], True)[1]
                if res < sys.float_info.min:
                    res = sys.float_info.min
            else:
                res = stats.fisher_exact([[a1, a2], [a3, a4]])[1]
        else:
            res = chi2_contingency([[a1, a2], [a3, a4]], False)[1]
            if res < sys.float_info.min:
                res = sys.float_info.min
        return res

    def calculate_rr(self, feature_df, y_pred, feature_list, matrix_name, result_path):
        """

        Parameters
        ----------
        feature_df
        y_pred
        feature_list
        matrix_name

        Returns
        -------
        :param feature_df:
        :param y_pred:
        :param feature_list:
        :param matrix_name:
        :param result_path:

        """
        assert len(feature_df) == len(y_pred)
        feature_df['cluster'] = y_pred
        cluster_num = len(set(y_pred))
        good_res_list = []
        pbar = tqdm(feature_list, desc=f'{matrix_name}')
        for featurei in pbar:
            pbar.set_postfix({'feature': featurei})
            for c1 in combinations(range(cluster_num), r=1):
                try:
                    c1_feature = feature_df[feature_df['cluster'] == c1][featurei].dropna()
                    no_c1_feature = feature_df[feature_df['cluster'] != c1][featurei].dropna()
                    if featurei in feature_list:
                        c1_count = len(c1_feature)
                        c2_count = len(no_c1_feature)
                        c1_pos_count = c1_feature.sum()
                        c2_pos_count = no_c1_feature.sum()
                        total = c1_count + c2_count
                        cj = c1_pos_count + c2_pos_count
                        if c2_pos_count == 0:
                            rr = 0
                        else:
                            rr = (c1_pos_count / c1_count) / (c2_pos_count / c2_count)
                        if matrix_name == 'Herb':
                            p = ttest_ind(c1_feature, no_c1_feature)[1]
                        else:
                            p = self.chi2_test(c1_count, cj, c1_pos_count, total)
                        fea_str = featurei + "[" + str(c1_pos_count) + "(" + str(c1_count) + ")/" + str(
                            c2_pos_count) + "(" + str(c2_count) + ")]"

                        good_res_list.append(
                            (c1[0], featurei, matrix_name, c1_pos_count, c1_count, c2_pos_count, c2_count, rr, p,
                             fea_str)
                        )
                    else:
                        continue
                except KeyError:
                    print("Key", featurei, 'error!')
                    continue
        good_res_df = pd.DataFrame(good_res_list, columns=[
            'c1', 'featurei', 'class', 'c1_pos_count', 'c1_count', 'c2_pos_count', 'c2_count', 'RR', 'p', 'Note'])
        path = result_path + f'/RR_P_res_{matrix_name}.xlsx'
        good_res_df.to_excel(path, index=False)
        return good_res_df

    @staticmethod
    def filter_rr_result(rr_result_df: pd.DataFrame, feature_type: str, result_path):
        def func_agg(df):
            return ','.join(df.values)

        rr_result_df = rr_result_df[rr_result_df['RR'] >= 1.5]
        rr_result_df = rr_result_df[rr_result_df['p'] < 0.05]
        label_feature = rr_result_df.iloc[:, [0, 1]]
        feature = label_feature.groupby(by='c1').agg(func_agg).reset_index()
        path = result_path + f'/{feature_type}_feature.xlsx'
        feature.to_excel(path)
        return feature

    def embedding(self, item_df: pd.DataFrame):
        # first explode the dataframe
        item_df_explode = self.explode(item_df)
        # then, process the exploded dataframe to matrix
        item_df_explode['add_one'] = 1
        item_trans_df = item_df_explode.pivot_table(
            index=item_df_explode.columns[0], columns=item_df_explode.columns[1], values='add_one', aggfunc='max'
        )
        item_trans_df.fillna(0, inplace=True)
        item_trans_df = item_trans_df.astype(int)
        item_trans_df['Item_index'] = item_trans_df.index
        return item_trans_df

    def item2matrix(self, item_df: pd.DataFrame):
        """
        Transform item df to matrix
        Parameters
        ----------
        item_df: item 2 cols,
            contains id and item(e.g, symptom, herb, etc.)
            with pd.DataFrame format.

        Returns
        -------
            item matrix, with pd.DataFrame format.
        """
        # 1. form all people and item-related people
        all_people = item_df.iloc[:, [0]]
        item_df_dropna = item_df.dropna()
        # 2. form the one-hot matrix
        item_matrix = self.embedding(item_df_dropna)
        # 3. merge with all people cols
        item_merge = pd.merge(
            all_people, item_matrix, left_on=all_people.columns[0], right_on='Item_index', how='left'
        )
        # 4. fill the nan value to zero
        item_merge.fillna(0, inplace=True)
        del item_merge['Item_index']
        return item_merge

    def process_data(self):
        # process symptom, basic and herb data
        def func(df: pd.DataFrame):
            return ','.join(df.values)

        # combine three data
        # 1. symptom combine rows2row
        symptom_combine = self.symptom.groupby(by='病案号').agg(func).reset_index()
        # 2. herb drop duplicates
        herb_drop = self.herb[['住院号', '中药']].drop_duplicates(subset=['住院号', '中药'], ignore_index=True)
        herb_combine = herb_drop.groupby(by='住院号').agg(func).reset_index()
        # 3. merge
        merge_basic_symptom = pd.merge(self.basic_info, symptom_combine, left_on='住院号', right_on='病案号', how='inner')
        merge_symptom_herb = merge_basic_symptom.merge(herb_combine, on='住院号', how='left')
        merge_symptom_herb.to_excel('all.xlsx')
        return merge_symptom_herb

    @staticmethod
    def clustering(feature_matrix: pd.DataFrame, cluster_num: int):
        """

        Parameters
        ----------
        feature_matrix
        cluster_num

        Returns
        -------

        """
        kmeans = KMeans(n_clusters=cluster_num, init='k-means++').fit(feature_matrix)
        label = kmeans.labels_
        return label

    def visualization(self):
        pass

    @staticmethod
    def calculate_basic_info(basic_df: pd.DataFrame, cluster_num: int, result_path):
        """

        Parameters
        ----------
        basic_df
        cluster_num

        Returns
        -------

        """
        result_list = []

        for cluster in range(cluster_num):  # 0-9
            temp_result = []
            temp_result_append = temp_result.append

            temp_result_append(cluster)
            temp = basic_df[basic_df['label'] == cluster]
            temp_result_append(temp.shape[0])
            temp_result_append(temp[temp['性别'] == 1].shape[0])
            temp_result_append(temp[temp['性别'] == 2].shape[0])
            temp_result_append(temp['年龄'].sum() / temp.shape[0])
            temp_result_append(np.sum(list(temp['家族史'])))
            temp_result_append(np.sum(temp['过敏史']))
            temp_result_append(np.sum(temp['饮酒史']))
            temp_result_append(np.sum(temp['吸烟史']))
            temp_result_append(np.sum(temp['死亡']))
            temp_result_append(np.sum(temp['死亡']) / temp.shape[0])

            result_list.append(temp_result)

        die_rate = [i[10] for i in result_list]
        min_die = die_rate.index(min(die_rate))

        result_df = pd.DataFrame()
        result_df['Cluster'] = [i[0] for i in result_list]
        result_df['Num'] = [i[1] for i in result_list]
        result_df['Male'] = [i[2] for i in result_list]
        result_df['Female'] = [i[3] for i in result_list]
        result_df['Age_average'] = [i[4] for i in result_list]
        result_df['Family'] = [i[5] for i in result_list]
        result_df['Allergy'] = [i[6] for i in result_list]
        result_df['Drink'] = [i[7] for i in result_list]
        result_df['Smoke'] = [i[8] for i in result_list]
        result_df['Die'] = [i[9] for i in result_list]
        result_df['Die_rate'] = [i[10] for i in result_list]

        path = result_path + '/Basic_analysis.xlsx'
        result_df.to_excel(path, index=False, encoding='utf-8-sig')
        return result_df, min_die

    @staticmethod
    def survival_analysis(basic_info_df: pd.DataFrame, cluster_num: int, best_class: int, image_path):
        kmf_group_list = [(i, KaplanMeierFitter(), basic_info_df['label'] == i) for i in range(cluster_num)]  # 把每类数据取出来
        ax = plt.subplot(111)
        for i, kmf_model, group in kmf_group_list:
            kmf_model.fit(basic_info_df[group]['住院天数'], event_observed=basic_info_df[group]['死亡'],
                          label=f'c{i + 1} n={group.sum()}')
            ax = kmf_model.plot(ax=ax)
        all_path = image_path + '/pic_all.svg'
        plt.savefig(fname=all_path, dpi=200, format='svg')
        # plt.show()
        plt.close()

        for i, kmf_model, group in kmf_group_list:
            ax = plt.subplot(111)
            kmf_model.fit(basic_info_df[group]['住院天数'], event_observed=basic_info_df[group]['死亡'],
                          label=f'c{i + 1} n={group.sum()}')
            ax = kmf_model.plot(ax=ax)
            kmf_model.fit(basic_info_df[basic_info_df['label'] == best_class]['住院天数'],
                          event_observed=basic_info_df[basic_info_df['label'] == best_class]['死亡'],
                          label=f"c{best_class + 1} n={basic_info_df[basic_info_df['label'] == best_class].shape[0]}")
            ax = kmf_model.plot(ax=ax)
            path = image_path + f'/pic{i + 1}_{best_class + 1}.svg'
            plt.savefig(fname=path, dpi=200, format='svg')
            # plt.show()
            plt.close()

    def main(self, cluster_num_input, result_path, image_path):
        start = time.time()
        print('/----------------- Start! -----------------/')
        # 1. form the matrix
        print('/---------- 1. Form the matrix ----------/')
        symptom_cols = self.original_data[['住院号', '症状']]
        herb_cols = self.original_data[['住院号', '中药']]
        symptom_matrix = self.item2matrix(symptom_cols)
        herb_matrix = self.item2matrix(herb_cols)

        # 2. clustering and visualization
        print('/---------- 2. Clustering ----------/')
        class_result = self.clustering(symptom_matrix.iloc[:, 1:], cluster_num=cluster_num_input)
        symptom_matrix.insert(1, 'label', class_result, allow_duplicates=False)
        symptom_matrix.sort_values(by=['住院号'], inplace=True)
        herb_matrix.insert(1, 'label', class_result, allow_duplicates=False)
        herb_matrix.sort_values(by=['住院号'], inplace=True)
        self.original_data.insert(1, 'label', class_result, allow_duplicates=False)
        self.original_data.sort_values(by=['住院号'], inplace=True)

        # 3. calculate RR & p-value, and filtering
        print('/---------- 3. Calculate RR & p, then filtering ----------/')
        symptom_feature_list = symptom_matrix.columns[2:]
        symptom_feature_df = symptom_matrix.iloc[:, 2:]
        symptom_feature_enrichment = self.calculate_rr(
            symptom_feature_df, class_result, symptom_feature_list, matrix_name='Symptom', result_path=result_path
        )
        filtered_symptom_feature = self.filter_rr_result(symptom_feature_enrichment, 'Symptom', result_path=result_path)

        herb_feature_list = herb_matrix.columns[2:]
        herb_feature_df = herb_matrix.iloc[:, 2:]
        herb_feature_enrichment = self.calculate_rr(
            herb_feature_df, class_result, herb_feature_list, matrix_name='Herb', result_path=result_path
        )
        filtered_herb_feature = self.filter_rr_result(herb_feature_enrichment, 'Herb', result_path=result_path)

        # 4. calculate basic_info statics
        print('/---------- 4. Calculate basic_info statics ----------/')
        basic_result_df, min_die_rate = self.calculate_basic_info(
            self.original_data, cluster_num=cluster_num_input, result_path=result_path
        )
        self.survival_analysis(self.original_data[['label', '住院天数', '死亡']], cluster_num_input, min_die_rate, image_path)

        # 5. final result
        print('/---------- 5. Final result ----------/')
        filtered_symptom_feature.columns = ['Cluster', 'Symptom_Feature']
        filtered_herb_feature.columns = ['Cluster', 'Herb_Feature']
        merge_features = pd.merge(filtered_symptom_feature, filtered_herb_feature, on='Cluster', how='inner')
        merge_basics = pd.merge(merge_features, basic_result_df, on='Cluster', how='inner')
        path = result_path + '/Result_final.xlsx'
        merge_basics.to_excel(path, index=False, encoding='utf-8')
        print('/-------------- Finished! --------------/')
        end = time.time()
        print('/------ Costs:', end - start, '------/')


if __name__ == '__main__':
    clustering = Clustering2Analysis()
    clustering.main()
