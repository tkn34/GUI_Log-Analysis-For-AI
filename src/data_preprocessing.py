
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import re
import streamlit as st
from collections import defaultdict, Counter


class DataPreprocessing:
    def __init__(self, feedback_data, data, mode):
        # mode
        self.mode = mode
        # data
        if self.mode == "train":
            feedback_data["label"] = -1
            self.feedback_data = feedback_data[["log","category", "label"]]
            self.data = data[["log","category", "label"]]
        else:
            self.feedback_data = feedback_data
            self.data = data[["log","category", "label"]]
        # Compiling for optimization
        self.re_sub_1 = re.compile(r"(:(?=\s))|((?<=\s):)")
        self.re_sub_2 = re.compile(r"(\d+\.)+\d+")
        self.re_sub_3 = re.compile(r"\d{2}:\d{2}:\d{2}")
        self.re_sub_4 = re.compile(r"Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep")
        self.re_sub_5 = re.compile(r":?(\w+:)+")
        self.re_sub_6 = re.compile(r"\.|\(|\)|\<|\>|\/|\-|\=|\[|\]")
        self.p = re.compile(r"[^(A-Za-z)]")

    def __call__(self):
        if self.mode == "train":
            # RemoveParameter
            self.feedback_data["log_after"] = self._remove_parameters(self.feedback_data["log"])
            self.data["log_after"] = self._remove_parameters(self.data["log"])
            # Data Merge
            self.data = pd.concat([self.feedback_data, self.data]).reset_index()
            del self.data["index"]
        else:
            # RemoveParameter
            self.data["log_after"] = self._remove_parameters(self.data["log"])
        return self.data

    def _remove_parameters(self, logs):
        log_after_list = []
        for log in logs:
            log_tmp = log
            # Removing parameters with Regex
            log = re.sub(self.re_sub_1, "", log)
            log = re.sub(self.re_sub_2, "", log)
            log = re.sub(self.re_sub_3, "", log)
            log = re.sub(self.re_sub_4, "", log)
            log = re.sub(self.re_sub_5, "", log)
            log = re.sub(self.re_sub_6, " ", log)
            L = log.split()
            # Filtering strings that have non-letter tokens
            new_log = [k for k in L if not self.p.search(k)]
            log = " ".join(new_log)
            log_after_list.append(log)
        return log_after_list




class FeatureExtraction:
    def __init__(self, x, mode="train", fe_type="TF-ILF"):
        self.outputdir = "./model/"
        self.x = x
        self.mode = mode
        self.fe_type = fe_type # "tfilf" or "tfidf"
    
    def __call__(self):
        # Build Vocablary
        self.vocabulary = self._create_vocab()
        self._save_vocab()
        #print("Build Vocablary : Done")
        # Crete Vector
        self.vector = self._create_vector()
        #print("Crete Vector    : Done")
        #print("Create Feature  : Done")
        # Create Feature
        self.feature, self.ixf_dict = self._create_feature()
        self._save_feature()
        # ラベル付与
        
        
        return self.feature, self.vocabulary
        
    def _create_vocab(self):
        if self.mode == "train":
            vocabulary = {}
            for line in self.x:
                token_list = line.strip().split()
                for token in token_list:
                    if token not in vocabulary:
                        vocabulary[token] = len(vocabulary)
        else:
            with open(self.outputdir + "vocablary.pickle", "rb") as f:
                vocabulary = pickle.load(f)
        return vocabulary
        
    def _save_vocab(self):
        if self.mode == "train":
            with open(self.outputdir + "vocablary.pickle", "wb") as f:
                pickle.dump(self.vocabulary, f)
        
    def _create_vector(self):
        result = []
        for line in self.x:
            temp = []
            token_list = line.strip().split()
            if token_list:
                for token in token_list:
                    if token not in self.vocabulary:
                        continue
                    else:
                        temp.append(self.vocabulary[token])
            result.append(temp)
        return np.array(result)        

    def _create_feature(self):
        if self.fe_type == "TF-ILF":
            feature, ilf_dict = self._create_feature_tf_ilf()
        else:
            feature, ilf_dict = self._create_feature_tf_idf()
        return feature, ilf_dict

    def _create_feature_tf_ilf(self):
        feature_vectors = []
        # Calculate lf
        token_index_ilf_dict = defaultdict(set)
        for line in self.vector:
            for location, token in enumerate(line):
                token_index_ilf_dict[token].add(location)
        # Calculate ilf
        ilf_dict = {}
        max_length = len(max(self.vector, key=len))
        for token in token_index_ilf_dict:
            ilf_dict[token] = np.log(float(max_length) / float(len(token_index_ilf_dict[token]) + 0.01))
        # Create feature
        tfinvf = []
        for line in self.vector:
            cur_tfinvf = np.zeros(len(self.vocabulary))
            count_dict = Counter(line)
            for token_index in line:
                cur_tfinvf[token_index] = (float(count_dict[token_index]) * ilf_dict[token_index])
            tfinvf.append(cur_tfinvf)
        tfinvf = np.array(tfinvf)
        feature_vectors.append(tfinvf)
        feature = np.hstack(feature_vectors)
        feature = pd.DataFrame(feature)
        return feature, ilf_dict

    def _create_feature_tf_idf(self):
        feature_vectors = []
        # Calculate lf
        token_index_idf_dict = defaultdict(set)
        for line in self.vector:
            for location, token in enumerate(line):
                token_index_idf_dict[token].add(location)
        # Calculate idf
        idf_dict = {}
        total_log_num = len(self.vector)
        for token in token_index_idf_dict:
            idf_dict[token] = np.log(float(total_log_num) / float(len(token_index_idf_dict[token]) + 0.01))
        # Create feature    
        tfinvf = []
        for line in self.vector:
            cur_tfinvf = np.zeros(len(self.vocabulary))
            count_dict = Counter(line)
            for token_index in line:
                cur_tfinvf[token_index] = (float(count_dict[token_index]) * idf_dict[token_index])
            tfinvf.append(cur_tfinvf)
        tfinvf = np.array(tfinvf)
        feature_vectors.append(tfinvf)
        feature = np.hstack(feature_vectors)
        feature = pd.DataFrame(feature)
        return feature, idf_dict

    def _save_feature(self):
        if self.mode == "train":
            if self.fe_type == "tfilf":
                with open(self.outputdir + "tf_ilf.pickle", "wb") as f:
                    pickle.dump(self.ixf_dict, f)
            else:
                with open(self.outputdir + "tf_idf.pickle", "wb") as f:
                    pickle.dump(self.ixf_dict, f)


class CosineSimilarity:
    def __init__(self, train, train_feature, th):
        train_feature["category"] = train["category"]
        train_feature["label"] = train["label"]
        self.train_feature = train_feature[train_feature["label"]==0].reset_index()
        self.anomaly_feature = train_feature[train_feature["label"]==-1].reset_index()
        del self.train_feature["index"]
        del self.anomaly_feature["index"]
        self.th = th
    
    def __call__(self):
        # 全ログと各異常ログの類似度から異常を抽出
        all_proba_list = []
        for row_feature in self.train_feature.values:
            proba_dict = {}
            analysis_row = row_feature[:-2]
            for anomaly_row in self.anomaly_feature.values:
                category = anomaly_row[-2]
                anomaly_row = anomaly_row[:-2]
                anom_proba = np.dot(analysis_row, anomaly_row) / (np.linalg.norm(analysis_row) * np.linalg.norm(anomaly_row))
                proba_dict[category] = anom_proba
            max_kv = list(max(proba_dict.items(), key=lambda x: x[1]))
            if max_kv[1] <= self.th:
                max_kv[0] = "normal"
                max_kv.append(0)
            else:
                max_kv.append(1)
            all_proba_list.append(max_kv)
        # train_feature_label
        train_feature_label = pd.DataFrame(all_proba_list, columns=["category_pred","related", "use_label"])
        self.train_feature = pd.concat([self.train_feature, train_feature_label], axis=1)
        # anomaly_feature_label
        self.anomaly_feature["category_pred"] = self.anomaly_feature["category"]
        self.anomaly_feature["related"] = np.nan
        self.anomaly_feature["use_label"] = 1
        self.train_feature = pd.concat([self.train_feature, self.anomaly_feature])
        return self.train_feature




class RelatedDisorders:
    def __init__(self, test_feature, train_feature, feedback):
        self.train_feature = train_feature
        self.test_feature = test_feature
        self.feedback = feedback
    
    def __call__(self):
        # 全ログと各異常ログの類似度から異常を抽出
        self.anomaly_feature = self.train_feature[self.train_feature["label"]==-1].reset_index()
        del self.anomaly_feature["index"]

        all_proba_list = []
        for row_feature in self.test_feature.values:
            proba_dict = {}
            analysis_row = row_feature[:-1]
            pred = row_feature[-1]
            for anomaly_row in self.anomaly_feature.values:
                category = anomaly_row[-5]
                anomaly_row = anomaly_row[:-5]
                anom_proba = np.dot(analysis_row, anomaly_row) / (np.linalg.norm(analysis_row) * np.linalg.norm(anomaly_row))
                proba_dict[category] = anom_proba * 100
            max_kv = list(max(proba_dict.items(), key=lambda x: x[1]))
            if pred == 1:
                #max_kv[1] = str(max_kv[1])
                
                max_kv[1] = str('{:.1f}'.format(max_kv[1]))
            else:
                max_kv[0] = "-"
                max_kv[1] = "-"
            all_proba_list.append(max_kv)
        
        all_proba = pd.DataFrame(all_proba_list, columns=["関連する過去の障害","関連度(%)"])
        all_proba["正常(0)/異常(1)"] = self.test_feature["y_pred"]
        #all_proba["関連度(%)"] = all_proba["関連度(%)"].apply(lambda x: '{:.1f}'.format(x))
        return all_proba
        # self.test_feature = pd.concat([self.test_feature, all_proba], axis=1)
        # self.test_feature["正常/異常"] = self.test_feature["pred"]
        # self.test_feature = self.test_feature[["log", "正常/異常","関連する過去の障害","関連度"]]
        # return self.test_feature
