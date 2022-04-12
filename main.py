
#import
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import  GridSearchCV
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Streamlit
import streamlit as st
from PIL import Image
# import plotly.express as px
# LogAnalysis
sys.path.append(os.path.join(os.path.dirname("__file__"), "./src/"))
from data_preprocessing import DataPreprocessing
from data_preprocessing import FeatureExtraction
from data_preprocessing import CosineSimilarity
from data_preprocessing import RelatedDisorders
import warnings
warnings.simplefilter('ignore')

from st_aggrid import AgGrid


def get_file(resource, mode):
    file = st.file_uploader(mode + "データの取得", type=['csv'])
    submit = st.button(mode + "データの表示")
    if submit == True:
        st.markdown(f'{file.name} をアップロードしました.')
        df = pd.read_csv("./data/" + resource + "/" + file.name, encoding="SJIS")
        if mode == "学習":
            st.session_state["train"] = df
        else:
            st.session_state["test"] = df
    else:
        df = None
    return df

def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: red' if is_max.any() else '' for v in is_max]




if __name__ == '__main__':
    # 初期設定
    st.set_page_config(layout="wide")
    st.title("Log Analysis for AI.")
    
    # リソース選択
    st.subheader('1. リソースの選択')
    st.caption('分析対象リソースを選択してください。(現在BGLのみ)')
    resource = st.selectbox('分析対象リソース',('BGL', 'Apache'))
    
    # Feedbackの取得
    st.subheader("2. 過去の障害情報")
    st.caption('過去に発生した障害の一覧表です。')
    feedback = pd.read_csv("./data/" + resource + "/" + "feedback.csv", encoding="SJIS")
    display_feedback  = feedback[["タイトル", "障害対応情報", "作成者", "作成日時"]]
    st.session_state["feedback"] = feedback
    AgGrid(display_feedback)
    
    
    # 学習データの取得
    st.subheader("3. 学習")
    st.caption('学習データの取得/特徴量抽出/ラベル付け/学習(LightGBM)/学習結果の確認を行います。')
    train = get_file(resource, "学習")
    if train is not None:
        display_train = train[["log"]]
        AgGrid(display_train)
    st.write('')
    st.write('')
    st.write('')
    
    ### 学習の実行
    # コサイン類似度の閾値
    st.write("コサイン類似度の閾値")
    st.caption('ラベルは過去の異常データとの類似性をもとに生成します。類似度の強さを0～100で指定できます。')
    cos_threshold = st.slider('閾値', min_value=0, max_value=100, step=1, value=75) * 0.01
    st.write('')
    st.write('')
    st.write('')
    
    # 特徴量抽出方法
    st.write("特徴量抽出方法の選択")
    st.caption('特徴量抽出方法を選択してください。TF-ILFもしくはTF-IDFから選択できます。')
    feature_type = st.selectbox('特徴量抽出方法',('TF-ILF', 'TF-IDF'))
    st.write('')
    st.write('')
    st.write('')
    
    # データ分割数
    st.write("データ分割数の選択")
    st.caption('学習時のデータ分割数について指定してください。1～5までで指定できます。')
    n_splits = st.slider('学習用データ分割数', min_value=1, max_value=5, step=1, value=5)
    st.write('')
    st.write('')
    st.write('')
    
    # 学習実行
    exec_train = st.button('学習 実行')
    if exec_train == True:
        # 各ログの固有値を削除
        train = DataPreprocessing(st.session_state.feedback, st.session_state.train, "train")()
        st.session_state["train"] = train
        st.write('Remove Parameter   : Done')
        # 特徴量抽出
        train_feature, vocabulary = FeatureExtraction(st.session_state.train["log_after"].values, mode="train", fe_type=feature_type)()
        st.session_state["train_feature"] = train_feature
        st.write('Feature Extraction : Done')
        # コサイン類似度によるラベル付け
        train_feature = CosineSimilarity(st.session_state.train, st.session_state.train_feature, cos_threshold)()
        st.session_state["train_feature"] = train_feature
        st.write('Create Label       : Done')       
        # モデルのインスタンス生成&学習実行
        params = {"max_length": [5,8,10]}
        clf = GridSearchCV(lgb.LGBMClassifier(verbose=-1), params, cv=StratifiedKFold(n_splits=n_splits), scoring="recall_macro")
        model = clf.fit(train_feature.iloc[:,:-5].values, train_feature["use_label"].values) # TF-ILFで特徴量抽出したデータでモデル生成
        st.session_state["model"] = model
        
        
        #### 学習結果 ##################
        # 特徴量重要度(重要視している単語)
        keyword = pd.DataFrame({"キーワード": vocabulary.keys(), "importance":model.best_estimator_.feature_importances_})
        keyword = keyword.sort_values("importance", ascending=False)
        keyword["重要度(%)"] = (keyword["importance"]/keyword["importance"].sum()) * 100
        keyword["重要度(%)"] = keyword["重要度(%)"].apply(lambda x: '{:.1f}'.format(x))
        
        ### 学習時のスコア表示
        # 予測値のデータ生成
        y_true = train_feature["use_label"]
        y_pred = model.predict(train_feature.iloc[:,:-5].values)
        # スコア算出
        cm = confusion_matrix(y_true, y_pred)
        classification_result = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
        sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues', fmt='d')
        fig = plt.show()
        st.subheader("4. 学習結果")
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.write("混同行列")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(fig)
        with col2:
            st.write("重要単語")
            
            #st.dataframe(keyword.head(10))
            display_keyword = keyword[["キーワード", "重要度(%)"]].reset_index()
            del display_keyword["index"]
            AgGrid(display_keyword)
            #st.dataframe(display_keyword)
            #AgGrid(keyword[["キーワード", "重要度(%)"]])
        with col3:
            st.write("結果詳細")
            st.dataframe(classification_result)
    st.write('')
    st.write('')
    st.write('')
    
    # ===== 推論 ======================================================
    # 推論データの取得
    st.subheader("5. 推論")
    st.caption('推論データの取得/ログの異常検知/過去の障害情報との関連を確認できます。')
    test = get_file(resource, "推論")
    if test is not None:
        display_test = test[["log"]]
        AgGrid(display_test)
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    
    # 推論実行
    exec_test = st.button('推論 実行')
    if exec_test == True:
        # 各ログの固有値を削除
        test = DataPreprocessing(None, st.session_state.test, "test")()
        st.session_state["test"] = test
        st.write('Remove Parameter   : Done')
        # 特徴量抽出
        test_feature, _ = FeatureExtraction(st.session_state.test["log_after"].values, mode="test", fe_type=feature_type)()
        st.session_state["test_feature"] = test_feature
        st.write('Feature Extraction : Done')
        # 推論実行
        y_pred = st.session_state.model.predict(st.session_state.test_feature.values)
        st.session_state.test_feature["y_pred"] = y_pred
        # 関連重要度の表示
        st.subheader("推論結果")
        result = RelatedDisorders(st.session_state.test_feature, st.session_state.train_feature, st.session_state.feedback)().sort_index()
        result["log"] =st.session_state.test["log"]
        #result["関連度(%)"] = (result["関連度"]/result["関連度"].sum()) * 100
        result = result[["log", "正常(0)/異常(1)", "関連する過去の障害", "関連度(%)"]]
        result["【参考】本来のカテゴリ"] = st.session_state.test["category"]
        result["【参考】本来の正常/異常"] = st.session_state.test["label"]
        result = pd.merge(result, st.session_state.feedback, how = "left", left_on="関連する過去の障害", right_on="category")
        result = result[["log_x", "正常(0)/異常(1)", "タイトル", "関連度(%)", "障害対応情報","【参考】本来のカテゴリ", "【参考】本来の正常/異常"]]
        result = result.rename(columns={"log_x":"ログ", "タイトル": "関連する過去の障害"})
        result = result.fillna("-")
        st.session_state["result"] = result
        result = result.style.apply(highlight_greaterthan, threshold=1.0, column=['正常(0)/異常(1)'], axis=1)
        st.table(result)