import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap

class ResultVisualizer:
    def __init__(self):
        self.plot_options = {
            "预测值vs实际值散点图": True,
            "残差图": True,
            "特征重要性图": True,
            "SHAP值解释": False  # 默认关闭，因为计算较慢
        }

    def plot_prediction_vs_actual(self, y_test, y_pred):
        """绘制预测值vs实际值散点图"""
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': '实际值', 'y': '预测值'},
            title='预测值 vs 实际值'
        )
        
        # 添加理想预测线（y=x）
        fig.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='理想预测线',
                line=dict(dash='dash', color='red')
            )
        )
        
        st.plotly_chart(fig)

    def plot_residuals(self, y_test, y_pred):
        """绘制残差图"""
        residuals = y_test - y_pred
        
        fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={'x': '预测值', 'y': '残差'},
            title='残差图'
        )
        
        # 添加y=0线
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="零残差线"
        )
        
        st.plotly_chart(fig)

    def plot_feature_importance(self, model, feature_names):
        """绘制特征重要性图"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                title='特征重要性'
            )
            
            st.plotly_chart(fig)
        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.abs(model.coef_)
            }).sort_values('coefficient', ascending=True)
            
            fig = px.bar(
                importance,
                x='coefficient',
                y='feature',
                orientation='h',
                title='特征系数绝对值'
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("当前模型不支持特征重要性分析")

    def plot_shap_values(self, model, X_test):
        """绘制SHAP值解释图"""
        try:
            if st.checkbox("显示SHAP值解释（可能需要较长时间）"):
                with st.spinner("正在计算SHAP值..."):
                    # 根据模型类型选择合适的SHAP解释器
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.LinearExplainer(model, X_test)
                    
                    shap_values = explainer.shap_values(X_test)
                    
                    # 如果是单输出回归问题，将shap_values转换为数组
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    # 创建SHAP摘要图
                    st.write("SHAP值摘要图：")
                    shap.summary_plot(shap_values, X_test, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                    
                    # 创建SHAP依赖图
                    if st.checkbox("显示SHAP依赖图"):
                        feature = st.selectbox("选择要分析的特征", X_test.columns)
                        shap.dependence_plot(feature, shap_values, X_test, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
        except Exception as e:
            st.error(f"SHAP值计算失败：{str(e)}")

    def plot_results(self, y_test, y_pred, model=None, X_test=None, feature_names=None):
        """绘制所有选中的可视化图表"""
        st.header("结果可视化")
        
        # 配置图表选项
        st.subheader("选择要显示的图表：")
        for option in self.plot_options:
            self.plot_options[option] = st.checkbox(option, value=self.plot_options[option])
        
        # 绘制选中的图表
        if self.plot_options["预测值vs实际值散点图"]:
            self.plot_prediction_vs_actual(y_test, y_pred)
        
        if self.plot_options["残差图"]:
            self.plot_residuals(y_test, y_pred)
        
        if self.plot_options["特征重要性图"] and model and feature_names:
            self.plot_feature_importance(model, feature_names)
        
        if self.plot_options["SHAP值解释"] and model and X_test is not None:
            self.plot_shap_values(model, X_test) 