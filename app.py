import streamlit as st
from src.data_processing.exploration import DataExplorer
from src.data_processing.preprocessing import DataPreprocessor
from src.data_processing.cleaning import DataCleaner
from src.feature_engineering.selection import FeatureSelector
from src.feature_engineering.creation import FeatureCreator
from src.modeling.models import ModelManager
from src.modeling.training import ModelTrainer
from src.modeling.ensemble import EnsembleManager
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.diagnostics import ModelDiagnostics
from src.evaluation.visualization import ResultVisualizer
from src.utils.file_operations import FileHandler

def main():
    st.set_page_config(page_title="机器学习回归分析工具", layout="wide")
    st.title("机器学习回归分析工具")

    # 数据上传
    file_handler = FileHandler()
    df = file_handler.upload_file()

    if df is not None:
        # 数据探索
        explorer = DataExplorer(df)
        if st.checkbox("数据探索与可视化"):
            explorer.run_exploration()

        # 数据预处理
        preprocessor = DataPreprocessor(df)
        if st.checkbox("数据预处理"):
            df = preprocessor.run_preprocessing()

        # 数据清洗
        cleaner = DataCleaner(df)
        if st.checkbox("数据清洗"):
            df = cleaner.run_cleaning()

        # 特征工程
        feature_creator = FeatureCreator(df)
        if st.checkbox("特征工程"):
            df = feature_creator.create_features()

        # 特征选择
        feature_selector = FeatureSelector(df)
        if st.checkbox("特征选择"):
            df = feature_selector.select_features()

        # 模型训练和评估
        if st.checkbox("模型训练"):
            model_manager = ModelManager()
            trainer = ModelTrainer(df)
            ensemble_manager = EnsembleManager()
            
            # 模型训练和预测
            model, X_train, X_test, y_train, y_test, y_pred = trainer.train_model()

            # 模型评估
            metrics = MetricsCalculator(y_test, y_pred)
            diagnostics = ModelDiagnostics(model, X_train, X_test, y_train, y_test, y_pred)
            visualizer = ResultVisualizer()

            # 显示结果
            metrics.display_metrics()
            diagnostics.run_diagnostics()
            visualizer.plot_results(y_test, y_pred)

            # 导出结果
            if st.button("导出分析结果"):
                file_handler.export_results(model, metrics, diagnostics)

if __name__ == "__main__":
    main() 