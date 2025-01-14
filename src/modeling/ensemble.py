import streamlit as st
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

class EnsembleManager:
    def __init__(self):
        self.base_models = {
            "线性回归": LinearRegression(),
            "岭回归": Ridge(),
            "随机森林": RandomForestRegressor()
        }
    
    def create_voting_regressor(self):
        st.write("选择要包含的基础模型：")
        selected_models = []
        
        for name, model in self.base_models.items():
            if st.checkbox(f"使用 {name}"):
                selected_models.append((name, model))
        
        if len(selected_models) >= 2:
            return VotingRegressor(selected_models)
        else:
            st.warning("请至少选择两个基础模型")
            return None
    
    def create_stacking_regressor(self):
        st.write("选择基础模型：")
        estimators = []
        
        for name, model in self.base_models.items():
            if st.checkbox(f"使用 {name} 作为基模型"):
                estimators.append((name, model))
        
        if len(estimators) >= 2:
            final_estimator = st.selectbox(
                "选择最终模型：",
                ["线性回归", "岭回归", "随机森林"]
            )
            
            final_model = self.base_models[final_estimator]
            return StackingRegressor(
                estimators=estimators,
                final_estimator=final_model,
                cv=5
            )
        else:
            st.warning("请至少选择两个基础模型")
            return None 