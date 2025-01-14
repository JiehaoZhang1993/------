import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, max_error
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
            
            # 在模型参数设置部分添加
            if st.checkbox("启用超参数优化"):
                st.info("将使用网格搜索寻找最优参数")
                
                if selected_method == "岭回归":
                    from sklearn.model_selection import GridSearchCV
                    param_grid = {
                        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                    }
                    grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    st.success(f"最优参数: {grid_search.best_params_}")
                    
                elif selected_method == "随机森林回归":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None]
                    }
                    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    st.success(f"最优参数: {grid_search.best_params_}")
            
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
                    from sklearn.metrics import explained_variance_score, max_error
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² 得分", f"{r2_score(y_test, y_pred):.3f}")
                        st.metric("解释方差分", f"{explained_variance_score(y_test, y_pred):.3f}")
                    with col2:
                        st.metric("均方误差 (MSE)", f"{mean_squared_error(y_test, y_pred):.3f}")
                        st.metric("最大误差", f"{max_error(y_test, y_pred):.3f}")
                    with col3:
                        st.metric("平均绝对误差 (MAE)", f"{mean_absolute_error(y_test, y_pred):.3f}")
                        st.metric("均方根误差 (RMSE)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
                
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
                
        # 在数据预览后添加数据探索部分
        if st.checkbox("数据探索与可视化"):
            st.subheader("数据探索与可视化")
            
            # 基础统计分析
            if st.checkbox("查看基础统计信息"):
                st.write("数值列统计描述：")
                st.write(df[numeric_columns].describe())
                
                # 缺失值分析
                missing_data = df[numeric_columns].isnull().sum()
                if missing_data.sum() > 0:
                    st.write("缺失值统计：")
                    missing_df = pd.DataFrame({
                        '列名': missing_data.index,
                        '缺失值数量': missing_data.values,
                        '缺失值比例(%)': (missing_data.values / len(df) * 100).round(2)
                    })
                    st.write(missing_df)
            
            # 数据分布可视化
            if st.checkbox("数据分布分析"):
                col_for_dist = st.selectbox("选择要分析的列：", numeric_columns)
                
                col1, col2 = st.columns(2)
                with col1:
                    # 直方图
                    fig_hist = px.histogram(df, x=col_for_dist, 
                                          title=f"{col_for_dist}的分布直方图",
                                          marginal="box")  # 添加箱线图
                    st.plotly_chart(fig_hist)
                    
                with col2:
                    # Q-Q图
                    from scipy import stats
                    qq_data = stats.probplot(df[col_for_dist].dropna(), dist="norm")
                    fig_qq = px.scatter(x=qq_data[0][0], y=qq_data[0][1],
                                      title=f"{col_for_dist}的Q-Q图")
                    fig_qq.add_trace(go.Scatter(x=qq_data[0][0], 
                                              y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                                              mode='lines', name='理论分布线'))
                    st.plotly_chart(fig_qq)
            
            # 相关性分析
            if st.checkbox("相关性分析"):
                corr_method = st.radio("选择相关系数计算方法：", 
                                     ["Pearson", "Spearman", "Kendall"])
                corr_matrix = df[numeric_columns].corr(method=corr_method.lower())
                
                # 热力图
                fig_heatmap = px.imshow(corr_matrix,
                                       title=f"{corr_method}相关系数热力图",
                                       color_continuous_scale="RdBu",
                                       aspect="auto")
                st.plotly_chart(fig_heatmap)
                
                # 显示具体的相关系数
                if st.checkbox("查看具体相关系数"):
                    st.write(corr_matrix.round(3))
                    
                    # 找出高相关的特征对
                    threshold = st.slider("选择相关系数阈值", 0.0, 1.0, 0.8)
                    high_corr = np.where(np.abs(corr_matrix) > threshold)
                    high_corr_list = [(corr_matrix.index[x], corr_matrix.columns[y], 
                                      corr_matrix.iloc[x, y])
                                     for x, y in zip(*high_corr) if x != y]
                    
                    if high_corr_list:
                        st.write(f"相关系数绝对值大于{threshold}的特征对：")
                        for feat1, feat2, corr in high_corr_list:
                            st.write(f"{feat1} - {feat2}: {corr:.3f}")
        
        # 在数据预览后添加数据清洗部分
        if st.checkbox("数据清洗"):
            st.subheader("数据清洗")
            
            # 异常值检测与处理
            if st.checkbox("异常值检测与处理"):
                method = st.radio(
                    "选择异常值检测方法：",
                    ["IQR方法", "Z-score方法", "隔离森林"]
                )
                
                if method == "IQR方法":
                    for col in numeric_columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                        if len(outliers) > 0:
                            st.write(f"{col} 列中检测到 {len(outliers)} 个异常值")
                            
                            handle_method = st.radio(
                                f"选择 {col} 列异常值处理方法：",
                                ["替换为边界值", "删除异常值行", "不处理"],
                                key=f"outlier_{col}"
                            )
                            
                            if handle_method == "替换为边界值":
                                df.loc[df[col] < lower_bound, col] = lower_bound
                                df.loc[df[col] > upper_bound, col] = upper_bound
                                st.success(f"{col} 列异常值已替换为边界值")
                            elif handle_method == "删除异常值行":
                                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                                st.success(f"已删除 {col} 列的异常值行")
        
        # 在模型训练完成后添加
        if st.checkbox("预测新数据"):
            st.subheader("新数据预测")
            
            # 提供两种输入方式
            input_method = st.radio("选择输入方式", ["手动输入", "上传文件"])
            
            if input_method == "手动输入":
                new_data = {}
                for feature in features:
                    new_data[feature] = st.number_input(f"输入 {feature} 的值")
                
                if st.button("进行预测"):
                    new_df = pd.DataFrame([new_data])
                    prediction = model.predict(new_df)
                    st.success(f"预测结果: {prediction[0]:.4f}")
                    
            else:
                new_file = st.file_uploader("上传新数据文件", type=['csv', 'xlsx', 'xls'])
                if new_file is not None:
                    if new_file.name.endswith('csv'):
                        new_df = pd.read_csv(new_file)
                    else:
                        new_df = pd.read_excel(new_file)
                        
                    if all(feature in new_df.columns for feature in features):
                        predictions = model.predict(new_df[features])
                        new_df['预测结果'] = predictions
                        st.write("预测结果：")
                        st.write(new_df)
                        
                        # 提供下载预测结果
                        output = pd.ExcelWriter('predictions.xlsx', engine='xlsxwriter')
                        new_df.to_excel(output, index=False)
                        output.save()
                        
                        with open('predictions.xlsx', 'rb') as f:
                            st.download_button(
                                label="下载预测结果",
                                data=f,
                                file_name='predictions.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    else:
                        st.error("新数据文件缺少必要的特征列")
        
        # 在模型训练完成后添加
        if st.checkbox("模型解释"):
            st.subheader("模型解释")
            
            if selected_method in ["线性回归", "岭回归", "Lasso回归"]:
                # 显示特征系数
                coef_df = pd.DataFrame({
                    '特征': features,
                    '系数': model.coef_
                })
                st.write("特征系数：")
                st.write(coef_df)
                
                # 可视化特征重要性
                fig_coef = px.bar(coef_df, x='特征', y='系数',
                                 title='特征系数可视化')
                st.plotly_chart(fig_coef)
            
            # 添加SHAP值解释
            if st.checkbox("显示SHAP值解释"):
                import shap
                explainer = shap.TreeExplainer(model) if selected_method == "随机森林回归" \
                    else shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test)
                
                st.write("SHAP值摘要图：")
                fig_shap = shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig_shap)
        
        # 在数据清洗后添加特征工程部分
        if st.checkbox("特征工程"):
            st.subheader("特征工程")
            
            # 多项式特征
            if st.checkbox("添加多项式特征"):
                degree = st.slider("选择多项式度数", 2, 5, 2)
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df[features])
                poly_feature_names = poly.get_feature_names_out(features)
                
                df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
                st.write("多项式特征预览：")
                st.write(df_poly.head())
                
                if st.button("使用多项式特征"):
                    features = list(poly_feature_names)
                    X = df_poly
                    st.success("已添加多项式特征")
            
            # 特征交互
            if st.checkbox("添加特征交互项"):
                selected_features = st.multiselect(
                    "选择要创建交互项的特征（最多选择2个）：",
                    features,
                    max_selections=2
                )
                
                if len(selected_features) == 2:
                    interaction_name = f"{selected_features[0]}_{selected_features[1]}"
                    df[interaction_name] = df[selected_features[0]] * df[selected_features[1]]
                    features.append(interaction_name)
                    st.success(f"已添加特征交互项：{interaction_name}")
        
        # 在机器学习方法选择部分添加
        if st.checkbox("使用模型集成"):
            st.subheader("模型集成")
            
            ensemble_method = st.radio(
                "选择集成方法：",
                ["投票回归", "堆叠回归"]
            )
            
            if ensemble_method == "投票回归":
                from sklearn.ensemble import VotingRegressor
                
                base_models = []
                if st.checkbox("使用线性回归"):
                    base_models.append(('lr', LinearRegression()))
                if st.checkbox("使用岭回归"):
                    base_models.append(('ridge', Ridge()))
                if st.checkbox("使用随机森林"):
                    base_models.append(('rf', RandomForestRegressor()))
                    
                if len(base_models) >= 2:
                    model = VotingRegressor(base_models)
                    st.success("已创建投票回归模型")
                    
            elif ensemble_method == "堆叠回归":
                from sklearn.ensemble import StackingRegressor
                
                estimators = []
                if st.checkbox("使用线性回归作为基模型"):
                    estimators.append(('lr', LinearRegression()))
                if st.checkbox("使用岭回归作为基模型"):
                    estimators.append(('ridge', Ridge()))
                if st.checkbox("使用随机森林作为基模型"):
                    estimators.append(('rf', RandomForestRegressor()))
                    
                final_estimator = st.selectbox(
                    "选择最终模型：",
                    ["线性回归", "岭回归", "随机森林"]
                )
                
                if len(estimators) >= 2:
                    if final_estimator == "线性回归":
                        final = LinearRegression()
                    elif final_estimator == "岭回归":
                        final = Ridge()
                    else:
                        final = RandomForestRegressor()
                        
                    model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=final,
                        cv=5
                    )
                    st.success("已创建堆叠回归模型")

        # 在模型评估部分添加
        if st.checkbox("显示学习曲线"):
            from sklearn.model_selection import learning_curve
            
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean,
                name='训练集得分',
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes, y=test_mean,
                name='验证集得分',
                mode='lines+markers'
            ))
            fig.update_layout(
                title='学习曲线',
                xaxis_title='训练样本数',
                yaxis_title='得分'
            )
            st.plotly_chart(fig)

        # 在特征工程部分添加
        if st.checkbox("特征选择"):
            st.subheader("特征选择")
            
            selection_method = st.radio("选择特征选择方法：",
                                      ["方差阈值法", "互信息法", "递归特征消除(RFE)", 
                                       "LASSO特征选择"])
            
            if selection_method == "方差阈值法":
                from sklearn.feature_selection import VarianceThreshold
                threshold = st.slider("选择方差阈值", 0.0, 1.0, 0.0, 0.01)
                selector = VarianceThreshold(threshold=threshold)
                X_selected = selector.fit_transform(X)
                selected_features = X.columns[selector.get_support()].tolist()
                
            elif selection_method == "互信息法":
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X, y)
                mi_df = pd.DataFrame({'特征': X.columns, '互信息得分': mi_scores})
                mi_df = mi_df.sort_values('互信息得分', ascending=False)
                
                st.write("特征互信息得分：")
                st.write(mi_df)
                
                n_features = st.slider("选择保留的特征数量", 1, len(features), len(features))
                selected_features = mi_df['特征'].head(n_features).tolist()
                
            elif selection_method == "递归特征消除(RFE)":
                from sklearn.feature_selection import RFE
                n_features = st.slider("选择保留的特征数量", 1, len(features), len(features))
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                selector = RFE(estimator=estimator, n_features_to_select=n_features)
                selector.fit(X, y)
                selected_features = X.columns[selector.support_].tolist()
                
            else:  # LASSO特征选择
                from sklearn.linear_model import LassoCV
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(X, y)
                
                importance_df = pd.DataFrame({
                    '特征': X.columns,
                    '系数': np.abs(lasso.coef_)
                }).sort_values('系数', ascending=False)
                
                st.write("LASSO特征重要性：")
                st.write(importance_df)
                
                coef_threshold = st.slider("选择系数阈值", 0.0, 
                                         float(importance_df['系数'].max()), 0.01)
                selected_features = importance_df[importance_df['系数'] > coef_threshold]['特征'].tolist()
            
            st.write("选中的特征：", selected_features)
            if st.button("使用选中的特征"):
                features = selected_features
                X = df[features]
                st.success(f"已更新特征集，当前使用{len(features)}个特征")

        # 在模型评估部分添加
        if st.checkbox("模型诊断"):
            st.subheader("模型诊断")
            
            # 残差分析
            residuals = y_test - y_pred
            
            # 残差的正态性检验
            from scipy import stats
            stat, p_value = stats.normaltest(residuals)
            st.write(f"残差正态性检验 p-value: {p_value:.4f}")
            if p_value < 0.05:
                st.warning("残差可能不服从正态分布")
            else:
                st.success("残差服从正态分布")
            
            # 残差的各种图
            col1, col2 = st.columns(2)
            with col1:
                # 残差直方图
                fig_res_hist = px.histogram(residuals, 
                                          title="残差分布直方图",
                                          marginal="box")
                st.plotly_chart(fig_res_hist)
                
            with col2:
                # 残差Q-Q图
                qq_data = stats.probplot(residuals, dist="norm")
                fig_qq = px.scatter(x=qq_data[0][0], y=qq_data[0][1],
                                  title="残差Q-Q图")
                fig_qq.add_trace(go.Scatter(x=qq_data[0][0], 
                                          y=qq_data[0][0] * qq_data[1][0] + qq_data[1][1],
                                          mode='lines', name='理论分布线'))
                st.plotly_chart(fig_qq)

        # 在预测完成后添加
        if st.checkbox("预测结果分析"):
            st.subheader("预测结果分析")
            
            # 预测误差分布
            error_df = pd.DataFrame({
                '实际值': y_test,
                '预测值': y_pred,
                '绝对误差': np.abs(y_test - y_pred),
                '相对误差(%)': (np.abs(y_test - y_pred) / y_test * 100)
            })
            
            st.write("预测误差统计：")
            st.write(error_df.describe())
            
            # 误差分布图
            col1, col2 = st.columns(2)
            with col1:
                fig_error_hist = px.histogram(error_df, x='绝对误差',
                                            title="绝对误差分布")
                st.plotly_chart(fig_error_hist)
            
            with col2:
                fig_error_rel = px.histogram(error_df, x='相对误差(%)',
                                           title="相对误差分布")
                st.plotly_chart(fig_error_rel)
            
            # 预测准确度分析
            error_threshold = st.slider("选择可接受的相对误差阈值(%)", 0, 100, 10)
            accurate_predictions = (error_df['相对误差(%)'] <= error_threshold).sum()
            accuracy = accurate_predictions / len(error_df) * 100
            
            st.metric(f"预测准确率 (相对误差≤{error_threshold}%)", 
                      f"{accuracy:.2f}%")

    except Exception as e:
        st.error(f"发生错误：{str(e)}") 