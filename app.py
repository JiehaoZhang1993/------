import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.preprocessing import StandardScaler

# 设置页面标题
st.set_page_config(page_title="机器学习回归分析工具", layout="wide")
st.title("机器学习回归分析工具")

# 添加数据格式提示
st.info("请确保上传的数据文件中所有需要用于分析的列都是数字格式。")
st.markdown("""
**数据要求：**
- 支持的文件格式：CSV、Excel (xls/xlsx)
- 所有用于分析的列必须是数字格式
- 如果存在非数字格式的列，将在数据预览时提示
""")

# 文件上传
uploaded_file = st.file_uploader("请上传数据文件", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # 读取数据
    try:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("文件上传成功！")
        st.write("数据预览：")
        st.dataframe(df.head())
        
        # 检查数据类型
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            st.warning("⚠️ 检测到以下列包含非数字格式数据，这些列不能用于回归分析：")
            for col in non_numeric_columns:
                st.write(f"- {col} (数据类型: {df[col].dtype})")
            st.write("请只选择数字格式的列作为特征变量和目标变量。")
        
        # 在特征选择之前添加
        # 数据预处理选项
        st.subheader("数据预处理")

        # 处理缺失值选项
        handle_missing = st.checkbox("处理缺失值")
        if handle_missing:
            # 只处理数值列的缺失值
            numeric_df = df[numeric_columns]
            missing_method = st.radio(
                "选择缺失值处理方法：",
                [
                    "均值填充（用该列的平均值填充）",
                    "中位数填充（用该列的中位数填充）",
                    "众数填充（用该列的最常见值填充）",
                    "删除含缺失值的行",
                ]
            )
            
        # 数据标准化选项
        normalize_data = st.checkbox("数据标准化")
        if normalize_data:
            normalize_method = st.radio(
                "选择标准化方法：",
                [
                    "标准化(StandardScaler: 均值为0，方差为1)",
                    "最小最大缩放(MinMaxScaler: 缩放到0-1之间)",
                    "稳健缩放(RobustScaler: 对异常值不敏感)",
                    "正规化(Normalizer: 将样本缩放到单位范数)"
                ]
            )
            
            # 显示将要被标准化的列
            st.write("以下数值列将被标准化：")
            st.write(list(numeric_columns))
            
        # 添加确认按钮
        if st.button("确认数据预处理"):
            # 处理缺失值
            if handle_missing:
                if missing_method == "均值填充（用该列的平均值填充）":
                    df[numeric_columns] = numeric_df.fillna(numeric_df.mean())
                    st.success("✅ 已使用均值填充数值列的缺失值")
                elif missing_method == "中位数填充（用该列的中位数填充）":
                    df[numeric_columns] = numeric_df.fillna(numeric_df.median())
                    st.success("✅ 已使用中位数填充数值列的缺失值")
                elif missing_method == "众数填充（用该列的最常见值填充）":
                    df[numeric_columns] = numeric_df.fillna(numeric_df.mode().iloc[0])
                    st.success("✅ 已使用众数填充数值列的缺失值")
                else:  # 删除含缺失值的行
                    # 只考虑数值列的缺失值
                    rows_with_missing = numeric_df.isnull().any(axis=1)
                    original_rows = len(df)
                    df = df[~rows_with_missing]
                    removed_rows = original_rows - len(df)
                    st.success(f"✅ 已删除数值列包含缺失值的行，共删除了 {removed_rows} 行数据")

            # 数据标准化
            if normalize_data:
                if normalize_method == "标准化(StandardScaler: 均值为0，方差为1)":
                    scaler = StandardScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                    st.success("✅ 已使用StandardScaler完成标准化")
                    
                elif normalize_method == "最小最大缩放(MinMaxScaler: 缩放到0-1之间)":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                    st.success("✅ 已使用MinMaxScaler完成归一化")
                    
                elif normalize_method == "稳健缩放(RobustScaler: 对异常值不敏感)":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                    st.success("✅ 已使用RobustScaler完成稳健缩放")
                    
                else:  # Normalizer
                    from sklearn.preprocessing import Normalizer
                    scaler = Normalizer()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                    st.success("✅ 已使用Normalizer完成正规化")

            # 显示预处理后的数据预览
            st.write("数据预处理后的预览：")
            st.dataframe(df[numeric_columns].head())

            # 添加标准化方法说明
            if normalize_data:
                with st.expander("查看标准化方法说明"):
                    st.markdown("""
                    **标准化方法说明：**
                    1. **标准化(StandardScaler)**
                       - 将数据转换为均值为0，方差为1的分布
                       - 适用于数据大致呈正态分布的情况
                       
                    2. **最小最大缩放(MinMaxScaler)**
                       - 将数据缩放到[0,1]区间
                       - 适用于数据分布有明显边界的情况
                       
                    3. **稳健缩放(RobustScaler)**
                       - 使用统计量（中位数和四分位数）进行缩放
                       - 适用于数据中存在异常值的情况
                       
                    4. **正规化(Normalizer)**
                       - 将样本缩放到单位范数（每个样本的范数为1）
                       - 适用于需要对样本向量的方向更敏感的情况
                    """)
        
        # 选择特征和目标变量（只显示数字格式的列）
        features = st.multiselect("选择特征变量（X）", numeric_columns)
        target = st.selectbox("选择目标变量（y）", numeric_columns)
        
        if features and target:
            X = df[features]
            y = df[target]
            
            # 选择机器学习方法
            ml_methods = {
                "线性回归": LinearRegression,
                "岭回归": Ridge,
                "Lasso回归": Lasso,
                "支持向量回归(SVR)": SVR,
                "随机森林回归": RandomForestRegressor
            }
            
            selected_method = st.selectbox("选择机器学习方法", list(ml_methods.keys()))
            
            # 根据选择的方法显示参数设置
            params = {}
            if selected_method == "岭回归":
                alpha = st.number_input("alpha（正则化强度，建议值：0.1-10）", 0.1, 100.0, 1.0)
                params['alpha'] = alpha
                
            elif selected_method == "Lasso回归":
                alpha = st.number_input("alpha（正则化强度，建议值：0.1-10）", 0.1, 100.0, 1.0)
                params['alpha'] = alpha
                
            elif selected_method == "支持向量回归(SVR)":
                C = st.number_input("C（正则化参数，建议值：1-10）", 0.1, 100.0, 1.0)
                kernel = st.selectbox("核函数", ['rbf', 'linear', 'poly'])
                params['C'] = C
                params['kernel'] = kernel
                
            elif selected_method == "随机森林回归":
                n_estimators = st.number_input("n_estimators（树的数量，建议值：100）", 10, 1000, 100)
                max_depth = st.number_input("max_depth（树的最大深度，建议值：None）", 1, 100, 10)
                params['n_estimators'] = n_estimators
                params['max_depth'] = max_depth
            
            # 选择结果呈现方式
            st.subheader("选择结果呈现方式")
            show_metrics = st.checkbox("显示评估指标（R², MSE, MAE）", value=True)
            show_scatter = st.checkbox("显示预测值vs实际值散点图", value=True)
            show_residuals = st.checkbox("显示残差图", value=True)
            show_feature_importance = st.checkbox("显示特征重要性（适用于随机森林）", value=True)
            
            # 在模型训练之前添加
            # 交叉验证设置
            cv_fold = st.number_input("交叉验证折数（0表示不使用交叉验证）", 0, 10, 0)
            
            # 开始运行按钮
            if st.button("开始运行"):
                # 分割数据
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 训练模型
                model = ml_methods[selected_method](**params)
                
                # 如果需要交叉验证，在模型创建后进行
                if cv_fold > 0:
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(model, X, y, cv=cv_fold)
                    st.write(f"交叉验证平均得分: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # 更新进度
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"训练进度：{i+1}%")
                    time.sleep(0.01)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 显示结果
                st.subheader("分析结果")
                
                # 评估指标
                if show_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² 得分", f"{r2_score(y_test, y_pred):.3f}")
                    with col2:
                        st.metric("均方误差 (MSE)", f"{mean_squared_error(y_test, y_pred):.3f}")
                    with col3:
                        st.metric("平均绝对误差 (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")
                
                # 散点图
                if show_scatter:
                    fig_scatter = px.scatter(
                        x=y_test, y=y_pred,
                        labels={'x': '实际值', 'y': '预测值'},
                        title='预测值 vs 实际值'
                    )
                    fig_scatter.add_trace(
                        go.Scatter(x=[y_test.min(), y_test.max()], 
                                 y=[y_test.min(), y_test.max()],
                                 mode='lines', name='理想线')
                    )
                    st.plotly_chart(fig_scatter)
                
                # 残差图
                if show_residuals:
                    residuals = y_test - y_pred
                    fig_residuals = px.scatter(
                        x=y_pred, y=residuals,
                        labels={'x': '预测值', 'y': '残差'},
                        title='残差图'
                    )
                    st.plotly_chart(fig_residuals)
                
                # 特征重要性
                if show_feature_importance and selected_method == "随机森林回归":
                    importance = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    })
                    fig_importance = px.bar(
                        importance.sort_values('importance', ascending=True),
                        x='importance', y='feature',
                        title='特征重要性'
                    )
                    st.plotly_chart(fig_importance)
                
                st.success("分析完成！")
                
                # 在训练完成后添加
                # 模型保存选项
                if st.button("保存模型"):
                    import pickle
                    with open('trained_model.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    st.download_button(
                        label="下载模型文件",
                        data=open('trained_model.pkl', 'rb'),
                        file_name='trained_model.pkl',
                        mime='application/octet-stream'
                    )
                
                # 创建输出结果的函数
                def create_results_excel(model, X_train, X_test, y_train, y_test, y_pred, features, target, selected_method, params):
                    # 创建一个 Excel writer 对象
                    output = pd.ExcelWriter('regression_results.xlsx', engine='xlsxwriter')
                    
                    # 1. 模型信息页
                    model_info = pd.DataFrame({
                        '项目': ['使用的模型', '目标变量', '特征变量', '模型参数'],
                        '值': [
                            selected_method,
                            target,
                            ', '.join(features),
                            str(params)
                        ]
                    })
                    model_info.to_excel(output, sheet_name='模型信息', index=False)
                    
                    # 2. 评估指标页
                    metrics = pd.DataFrame({
                        '评估指标': ['R² 得分', '均方误差 (MSE)', '平均绝对误差 (MAE)'],
                        '值': [
                            r2_score(y_test, y_pred),
                            mean_squared_error(y_test, y_pred),
                            mean_absolute_error(y_test, y_pred)
                        ]
                    })
                    metrics.to_excel(output, sheet_name='评估指标', index=False)
                    
                    # 3. 预测结果页
                    results_df = pd.DataFrame({
                        '实际值': y_test,
                        '预测值': y_pred,
                        '残差': y_test - y_pred
                    })
                    results_df.to_excel(output, sheet_name='预测结果', index=True)
                    
                    # 4. 如果是随机森林，添加特征重要性
                    if selected_method == "随机森林回归":
                        importance = pd.DataFrame({
                            '特征': features,
                            '重要性': model.feature_importances_
                        })
                        importance = importance.sort_values('重要性', ascending=False)
                        importance.to_excel(output, sheet_name='特征重要性', index=False)
                    
                    # 保存文件
                    output.save()
                    
                    # 读取文件内容用于下载
                    with open('regression_results.xlsx', 'rb') as f:
                        return f.read()

                # 添加导出结果按钮
                if st.button("导出分析结果"):
                    # 创建结果文件
                    excel_data = create_results_excel(
                        model=model,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        y_pred=y_pred,
                        features=features,
                        target=target,
                        selected_method=selected_method,
                        params=params
                    )
                    
                    # 提供下载按钮
                    st.download_button(
                        label="📥 下载分析结果(Excel)",
                        data=excel_data,
                        file_name='regression_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                    
                    st.success("✅ 分析结果已准备完成，请点击上方按钮下载！")
                    
                    # 显示结果文件内容预览
                    st.write("Excel文件包含以下内容：")
                    st.markdown("""
                    1. **模型信息**
                       - 使用的模型类型
                       - 目标变量
                       - 特征变量列表
                       - 模型参数设置
                       
                    2. **评估指标**
                       - R² 得分
                       - 均方误差 (MSE)
                       - 平均绝对误差 (MAE)
                       
                    3. **预测结果**
                       - 测试集实际值
                       - 预测值
                       - 残差
                       
                    4. **特征重要性** (仅随机森林模型)
                       - 各特征的重要性得分
                    """)
                
    except Exception as e:
        st.error(f"发生错误：{str(e)}") 