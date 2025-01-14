import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class FeatureCreator:
    def __init__(self, df):
        self.df = df
        
    def create_polynomial_features(self):
        if st.checkbox("添加多项式特征"):
            degree = st.slider("选择多项式度数", 2, 5, 2)
            # 多项式特征创建的具体实现...

    def create_interaction_features(self):
        if st.checkbox("添加特征交互项"):
            # 特征交互的具体实现...

    def create_features(self):
        st.subheader("特征工程")
        self.create_polynomial_features()
        self.create_interaction_features()
        return self.df 