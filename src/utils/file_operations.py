import streamlit as st
import pandas as pd
import numpy as np

class FileHandler:
    def upload_file(self):
        st.info("请确保上传的数据文件中所有需要用于分析的列都是数字格式。")
        uploaded_file = st.file_uploader("请上传数据文件", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("文件上传成功！")
                return df
            except Exception as e:
                st.error(f"文件读取错误：{str(e)}")
                return None
        return None

    def export_results(self, model, metrics, diagnostics):
        try:
            # 创建Excel文件
            output = pd.ExcelWriter('regression_results.xlsx', engine='xlsxwriter')
            
            # 保存模型信息
            model_info = pd.DataFrame({
                '项目': ['模型类型', '评估指标'],
                '值': [str(type(model).__name__), str(metrics.calculate_metrics())]
            })
            model_info.to_excel(output, sheet_name='模型信息', index=False)
            
            output.save()
            
            # 提供下载
            with open('regression_results.xlsx', 'rb') as f:
                st.download_button(
                    label="📥 下载分析结果(Excel)",
                    data=f,
                    file_name='regression_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.success("✅ 分析结果已准备完成，请点击上方按钮下载！")
            
        except Exception as e:
            st.error(f"导出结果时发生错误：{str(e)}") 