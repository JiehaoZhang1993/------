import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from .models import ModelManager

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model_manager = ModelManager()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns

    def prepare_data(self):
        # 选择特征和目标变量
        features = st.multiselect("选择特征变量（X）", self.numeric_columns)
        target = st.selectbox("选择目标变量（y）", self.numeric_columns)
        
        if features and target:
            X = self.df[features]
            y = self.df[target]
            return X, y, features, target
        return None, None, None, None

    def train_model(self):
        st.subheader("模型训练")
        
        # 准备数据
        X, y, features, target = self.prepare_data()
        if X is None:
            st.warning("请先选择特征变量和目标变量")
            return None, None, None, None, None, None

        # 选择模型
        model_name = st.selectbox("选择机器学习方法", list(self.model_manager.models.keys()))
        params = self.model_manager.get_model_params(model_name)
        
        # 设置交叉验证
        cv_fold = st.number_input("交叉验证折数（0表示不使用交叉验证）", 0, 10, 0)
        
        if st.button("开始训练"):
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 创建模型
            model = self.model_manager.create_model(model_name, params)
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 如果需要交叉验证
            if cv_fold > 0:
                cv_scores = cross_val_score(model, X, y, cv=cv_fold)
                st.write(f"交叉验证平均得分: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # 训练模型
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"训练进度：{i+1}%")
                if i == 50:  # 在50%进度时训练模型
                    model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            return model, X_train, X_test, y_train, y_test, y_pred
        
        return None, None, None, None, None, None 