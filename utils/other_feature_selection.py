import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(type_feature_selection,X,y,feature_number = 15):
    if type_feature_selection == 1:
        pca = PCA(n_components=feature_number)  # 选择要保留的主成分数量
        X_pca = pca.fit_transform(X)  # X是特征矩阵
        X_pca = np.expand_dims(X_pca, axis=1)
        return X_pca
    elif type_feature_selection == 2:
        selector = SelectKBest(score_func=mutual_info_classif, k=feature_number)  # 选择要保留的特征数量
        X_new = selector.fit_transform(X, y)  # X是特征矩阵，y是目标变量
        X_new = np.expand_dims(X_new, axis=1)
        return X_new
    elif type_feature_selection == 3:
        selector = SelectKBest(score_func=chi2, k=feature_number)  # 选择要保留的特征数量
        X_new = selector.fit_transform(X, y)  # X是特征矩阵，y是目标变量
        X_new = np.expand_dims(X_new, axis=1)
        return X_new
    elif type_feature_selection == 4:
        estimator = LogisticRegression(solver='lbfgs', max_iter=1000)  # 使用适当的模型
        selector = RFE(estimator, n_features_to_select=feature_number)  # 选择要保留的特征数量
        X_new = selector.fit_transform(X, y)  # X是特征矩阵，y是目标变量
        X_new = np.expand_dims(X_new, axis=1)
        return X_new
    elif type_feature_selection == 5:
        selector = SelectKBest(score_func=f_classif, k=feature_number)  # 选择要保留的特征数量
        X_new = selector.fit_transform(X, y)  # X是特征矩阵，y是目标变量
        X_new = np.expand_dims(X_new, axis=1)
        return X_new







