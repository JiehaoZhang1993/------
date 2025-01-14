import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class MetricsCalculator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        metrics = {
            'R² 得分': r2_score(self.y_true, self.y_pred),
            'MSE': mean_squared_error(self.y_true, self.y_pred),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'RMSE': mean_squared_error(self.y_true, self.y_pred, squared=False)
        }
        return metrics

    def display_metrics(self):
        metrics = self.calculate_metrics()
        for name, value in metrics.items():
            st.metric(name, f"{value:.3f}") 