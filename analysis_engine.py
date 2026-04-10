import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AnalysisEngine:
    def basic_info(self, df):
        """Basic dataset information"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': df.duplicated().sum()
        }
    
    def statistical_summary(self, df):
        """Statistical summary for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            'numeric_columns': list(numeric_cols),
            'describe': df[numeric_cols].describe().round(2).to_dict()
        }
    
    def missing_data_analysis(self, df):
        """Comprehensive missing data analysis"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        return pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct.round(2)
        }).sort_values('missing_count', ascending=False)
    
    def correlation_analysis(self, df):
        """Correlation matrix for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = corr_matrix[abs(corr_matrix) > 0.7].stack().reset_index()
            high_corr.columns = ['var1', 'var2', 'correlation']
            high_corr = high_corr[high_corr['var1'] != high_corr['var2']]
            return {
                'matrix': corr_matrix.round(3),
                'high_correlations': high_corr
            }
        return {}
    
    def outlier_detection(self, df):
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'bounds': [lower_bound, upper_bound]
            }
        
        return outliers
    
    def data_quality_score(self, df):
        """Calculate overall data quality score (0-100)"""
        total_rows = len(df)
        missing_pct = df.isnull().sum().sum() / (total_rows * len(df.columns))
        duplicate_pct = df.duplicated().sum() / total_rows
        unique_cols_pct = (df.nunique() / total_rows * 100).mean()
        
        quality_score = 100 - (missing_pct * 50 + duplicate_pct * 30 + 
                              (100 - unique_cols_pct) * 20)
        return round(max(0, min(100, quality_score)), 2)