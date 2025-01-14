import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import learning_curve

class ModelDiagnostics:
    def __init__(self, model, X_train, X_test, y_train, y_test, y_pred):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.residuals = y_test - y_pred

    def plot_residuals_analysis(self):
        st.subheader("残差分析")
        
        # 残差的正态性检验
        stat, p_value = stats.normaltest(self.residuals)
        st.write(f"残差正态性检验 p-value: {p_value:.4f}")
        if p_value < 0.05:
            st.warning("残差可能不服从正态分布")
        else:
            st.success("残差服从正态分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 残差直方图
            fig_hist = px.histogram(
                self.residuals,
                title="残差分布直方图",
                labels={'value': '残差', 'count': '频数'},
                marginal="box"
            )
            st.plotly_chart(fig_hist)
            
        with col2:
            # 残差Q-Q图
            qq_data = stats.probplot(self.residuals, dist="norm")
            fig_qq = px.scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                title="残差Q-Q图"
            )
            fig_qq.add_trace(
                go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                    mode='lines',
                    name='理论分布线'
                )
            )
            st.plotly_chart(fig_qq)
        
        # 残差vs预测值散点图
        fig_scatter = px.scatter(
            x=self.y_pred,
            y=self.residuals,
            title="残差 vs 预测值",
            labels={'x': '预测值', 'y': '残差'}
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_scatter)

    def plot_learning_curves(self):
        st.subheader("学习曲线分析")
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fig = go.Figure()
        
        # 添加训练集得分
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            name='训练集得分',
            mode='lines+markers',
            line=dict(color='blue'),
            error_y=dict(
                type='data',
                array=train_std,
                visible=True
            )
        ))
        
        # 添加验证集得分
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_mean,
            name='验证集得分',
            mode='lines+markers',
            line=dict(color='red'),
            error_y=dict(
                type='data',
                array=test_std,
                visible=True
            )
        ))
        
        fig.update_layout(
            title='学习曲线',
            xaxis_title='训练样本数',
            yaxis_title='得分',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # 分析学习曲线
        gap = np.mean(train_mean - test_mean)
        if gap > 0.3:
            st.warning("模型可能存在过拟合问题")
        elif train_mean[-1] < 0.5:
            st.warning("模型可能存在欠拟合问题")
        else:
            st.success("模型拟合程度良好")

    def analyze_prediction_errors(self):
        st.subheader("预测误差分析")
        
        error_df = pd.DataFrame({
            '实际值': self.y_test,
            '预测值': self.y_pred,
            '绝对误差': np.abs(self.y_test - self.y_pred),
            '相对误差(%)': (np.abs(self.y_test - self.y_pred) / self.y_test * 100)
        })
        
        st.write("预测误差统计：")
        st.write(error_df.describe())
        
        # 误差分布图
        col1, col2 = st.columns(2)
        with col1:
            fig_abs = px.histogram(
                error_df,
                x='绝对误差',
                title="绝对误差分布"
            )
            st.plotly_chart(fig_abs)
        
        with col2:
            fig_rel = px.histogram(
                error_df,
                x='相对误差(%)',
                title="相对误差分布"
            )
            st.plotly_chart(fig_rel)
        
        # 预测准确度分析
        error_threshold = st.slider(
            "选择可接受的相对误差阈值(%)",
            0, 100, 10
        )
        accurate_predictions = (error_df['相对误差(%)'] <= error_threshold).sum()
        accuracy = accurate_predictions / len(error_df) * 100
        
        st.metric(
            f"预测准确率 (相对误差≤{error_threshold}%)",
            f"{accuracy:.2f}%"
        )

    def run_diagnostics(self):
        st.header("模型诊断")
        
        # 残差分析
        self.plot_residuals_analysis()
        
        # 学习曲线分析
        self.plot_learning_curves()
        
        # 预测误差分析
        self.analyze_prediction_errors() 