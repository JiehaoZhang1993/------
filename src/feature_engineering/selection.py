import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

class FeatureSelector:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.selected_features = None
        self.X = None
        self.y = None

    def prepare_data(self):
        # 选择特征和目标变量
        self.features = st.multiselect("选择特征变量（X）", self.numeric_columns)
        self.target = st.selectbox("选择目标变量（y）", self.numeric_columns)
        
        if self.features and self.target:
            self.X = self.df[self.features]
            self.y = self.df[self.target]
            return True
        return False

    def variance_threshold_selection(self):
        threshold = st.slider("选择方差阈值", 0.0, 1.0, 0.0, 0.01)
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(self.X)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        st.write("方差大于阈值的特征：", selected_features)
        return selected_features

    def mutual_information_selection(self):
        mi_scores = mutual_info_regression(self.X, self.y)
        mi_df = pd.DataFrame({
            '特征': self.features,
            '互信息得分': mi_scores
        }).sort_values('互信息得分', ascending=False)
        
        st.write("特征互信息得分：")
        st.write(mi_df)
        
        n_features = st.slider("选择保留的特征数量", 1, len(self.features), len(self.features))
        selected_features = mi_df['特征'].head(n_features).tolist()
        return selected_features

    def recursive_feature_elimination(self):
        n_features = st.slider("选择保留的特征数量", 1, len(self.features), len(self.features))
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(self.X, self.y)
        
        selected_features = self.X.columns[selector.support_].tolist()
        
        # 显示特征排名
        feature_ranking = pd.DataFrame({
            '特征': self.features,
            '排名': selector.ranking_
        }).sort_values('排名')
        st.write("特征重要性排名：")
        st.write(feature_ranking)
        
        return selected_features

    def lasso_selection(self):
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(self.X, self.y)
        
        importance_df = pd.DataFrame({
            '特征': self.features,
            '系数': np.abs(lasso.coef_)
        }).sort_values('系数', ascending=False)
        
        st.write("LASSO特征重要性：")
        st.write(importance_df)
        
        coef_threshold = st.slider(
            "选择系数阈值",
            0.0,
            float(importance_df['系数'].max()),
            0.01
        )
        selected_features = importance_df[importance_df['系数'] > coef_threshold]['特征'].tolist()
        return selected_features

    def select_features(self):
        st.subheader("特征选择")
        
        if not self.prepare_data():
            st.warning("请先选择特征变量和目标变量")
            return self.df
            
        selection_method = st.radio(
            "选择特征选择方法：",
            ["方差阈值法", "互信息法", "递归特征消除(RFE)", "LASSO特征选择"]
        )
        
        if selection_method == "方差阈值法":
            self.selected_features = self.variance_threshold_selection()
        elif selection_method == "互信息法":
            self.selected_features = self.mutual_information_selection()
        elif selection_method == "递归特征消除(RFE)":
            self.selected_features = self.recursive_feature_elimination()
        else:  # LASSO特征选择
            self.selected_features = self.lasso_selection()
        
        if self.selected_features:
            st.write("选中的特征：", self.selected_features)
            if st.button("使用选中的特征"):
                # 更新数据框，只保留选中的特征和目标变量
                selected_columns = self.selected_features + [self.target]
                self.df = self.df[selected_columns]
                st.success(f"已更新特征集，当前使用{len(self.selected_features)}个特征")
        
        return self.df 