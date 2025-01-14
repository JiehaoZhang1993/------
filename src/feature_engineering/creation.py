import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class FeatureCreator:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def create_polynomial_features(self):
        if st.checkbox("添加多项式特征"):
            degree = st.slider("选择多项式度数", 2, 5, 2)
            selected_features = st.multiselect(
                "选择要创建多项式特征的列",
                self.numeric_columns
            )
            
            if selected_features and st.button("生成多项式特征"):
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(self.df[selected_features])
                feature_names = poly.get_feature_names_out(selected_features)
                
                # 将多项式特征添加到原数据框
                for i, name in enumerate(feature_names[len(selected_features):]):
                    self.df[f'poly_{name}'] = poly_features[:, len(selected_features) + i]
                
                st.success(f"已添加{len(feature_names)-len(selected_features)}个多项式特征")

    def create_interaction_features(self):
        if st.checkbox("添加特征交互项"):
            selected_features = st.multiselect(
                "选择要创建交互项的特征（建议选择2-3个）",
                self.numeric_columns,
                key="interaction_features"
            )
            
            if len(selected_features) >= 2 and st.button("生成交互特征"):
                # 创建所有可能的两两交互项
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        feat1 = selected_features[i]
                        feat2 = selected_features[j]
                        interaction_name = f"{feat1}_x_{feat2}"
                        self.df[interaction_name] = self.df[feat1] * self.df[feat2]
                
                st.success(f"已添加{len(selected_features) * (len(selected_features)-1) // 2}个交互特征")

    def create_features(self):
        st.subheader("特征工程")
        st.write("原始特征：", list(self.numeric_columns))
        
        # 创建多项式特征
        self.create_polynomial_features()
        
        # 创建交互特征
        self.create_interaction_features()
        
        # 显示新特征预览
        if st.checkbox("查看特征预览"):
            st.write("当前所有特征：", list(self.df.columns))
            st.write("数据预览：")
            st.dataframe(self.df.head())
        
        return self.df 