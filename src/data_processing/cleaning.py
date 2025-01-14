import streamlit as st
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns

    def detect_outliers(self):
        if st.checkbox("异常值检测"):
            method = st.radio("选择异常值检测方法：",
                            ["IQR方法", "Z-score方法"])
            
            for col in self.numeric_columns:
                if method == "IQR方法":
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.df[(self.df[col] < lower_bound) | 
                                     (self.df[col] > upper_bound)][col]
                else:  # Z-score方法
                    z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                    outliers = self.df[col][z_scores > 3]
                
                if len(outliers) > 0:
                    st.write(f"{col} 列中检测到 {len(outliers)} 个异常值")
                    handle_method = st.radio(f"处理 {col} 列异常值：",
                                          ["替换为边界值", "删除异常值行", "不处理"],
                                          key=f"outlier_{col}")
                    
                    if handle_method == "替换为边界值":
                        if method == "IQR方法":
                            self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                            self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                        else:
                            self.df.loc[z_scores > 3, col] = self.df[col].mean()
                    elif handle_method == "删除异常值行":
                        if method == "IQR方法":
                            self.df = self.df[(self.df[col] >= lower_bound) & 
                                            (self.df[col] <= upper_bound)]
                        else:
                            self.df = self.df[z_scores <= 3]

    def run_cleaning(self):
        st.subheader("数据清洗")
        self.detect_outliers()
        return self.df 