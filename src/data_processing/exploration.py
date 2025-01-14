import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats

class DataExplorer:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns

    def show_basic_info(self):
        st.write("数据基本信息：")
        st.write(f"行数：{self.df.shape[0]}, 列数：{self.df.shape[1]}")
        st.write("数据预览：")
        st.dataframe(self.df.head())
        
        if st.checkbox("查看基础统计信息"):
            st.write("数值列统计描述：")
            st.write(self.df[self.numeric_columns].describe())

    def show_missing_values(self):
        if st.checkbox("查看缺失值信息"):
            missing_data = self.df[self.numeric_columns].isnull().sum()
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    '列名': missing_data.index,
                    '缺失值数量': missing_data.values,
                    '缺失值比例(%)': (missing_data.values / len(self.df) * 100).round(2)
                })
                st.write("缺失值统计：")
                st.write(missing_df)
            else:
                st.info("数据中没有缺失值")

    def plot_distributions(self):
        if st.checkbox("查看数据分布"):
            col = st.selectbox("选择要查看的列：", self.numeric_columns)
            
            # 直方图和箱线图
            fig = px.histogram(self.df, x=col, title=f"{col}的分布",
                             marginal="box")
            st.plotly_chart(fig)
            
            # Q-Q图
            fig_qq = px.scatter(x=stats.probplot(self.df[col].dropna(), dist="norm")[0][0],
                              y=stats.probplot(self.df[col].dropna(), dist="norm")[0][1],
                              title=f"{col}的Q-Q图")
            st.plotly_chart(fig_qq)

    def plot_correlations(self):
        if st.checkbox("查看相关性分析"):
            corr_method = st.radio("选择相关系数计算方法：",
                                 ["Pearson", "Spearman"])
            corr = self.df[self.numeric_columns].corr(method=corr_method.lower())
            
            fig = px.imshow(corr,
                           labels=dict(color="相关系数"),
                           x=corr.columns,
                           y=corr.columns,
                           color_continuous_scale="RdBu")
            st.plotly_chart(fig)

    def run_exploration(self):
        st.subheader("数据探索")
        self.show_basic_info()
        self.show_missing_values()
        self.plot_distributions()
        self.plot_correlations() 