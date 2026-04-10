import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    def distribution_charts(self, df):
        """Create distribution charts for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        charts = {}
        
        if numeric_cols:
            fig = make_subplots(
                rows=1, cols=len(numeric_cols),
                subplot_titles=numeric_cols
            )
            
            for i, col in enumerate(numeric_cols, 1):
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, opacity=0.7),
                    row=1, col=i
                )
            
            fig.update_layout(height=400, title_text="Distribution of Numeric Variables")
            charts['distributions'] = fig
        
        return charts
    
    def correlation_heatmap(self, df):
        """Create correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
            return {'correlation_heatmap': fig}
        return {}
    
    def missing_data_heatmap(self, df):
        """Create missing data heatmap"""
        fig = px.imshow(
            df.isnull().astype(int),
            title="Missing Data Heatmap",
            color_continuous_scale='Reds',
            aspect="auto"
        )
        return {'missing_heatmap': fig}
    
    def outlier_boxplots(self, df):
        """Create boxplots to show outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            fig = px.box(df, x=None, y=numeric_cols, 
                        title="Boxplots - Outlier Detection")
            return {'boxplots': fig}
        return {}