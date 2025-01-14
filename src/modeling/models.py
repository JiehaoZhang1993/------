import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

class ModelManager:
    def __init__(self):
        self.models = {
            "线性回归": LinearRegression,
            "岭回归": Ridge,
            "Lasso回归": Lasso,
            "支持向量回归(SVR)": SVR,
            "随机森林回归": RandomForestRegressor
        }
        
    def get_model_params(self, model_name):
        params = {}
        if model_name == "岭回归":
            alpha = st.number_input("alpha（正则化强度，建议值：0.1-10）", 0.1, 100.0, 1.0)
            params['alpha'] = alpha
            
        elif model_name == "Lasso回归":
            alpha = st.number_input("alpha（正则化强度，建议值：0.1-10）", 0.1, 100.0, 1.0)
            params['alpha'] = alpha
            
        elif model_name == "支持向量回归(SVR)":
            C = st.number_input("C（正则化参数，建议值：1-10）", 0.1, 100.0, 1.0)
            kernel = st.selectbox("核函数", ['rbf', 'linear', 'poly'])
            params['C'] = C
            params['kernel'] = kernel
            
        elif model_name == "随机森林回归":
            n_estimators = st.number_input("n_estimators（树的数量，建议值：100）", 10, 1000, 100)
            max_depth = st.number_input("max_depth（树的最大深度，建议值：None）", 1, 100, 10)
            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
            
        return params

    def create_model(self, model_name, params):
        return self.models[model_name](**params) 