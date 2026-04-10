import pandas as pd
import numpy as np
from analysis_engine import AnalysisEngine
from visualizer import Visualizer
from summarizer import InsightSummarizer
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class DataAnalystAgent:
    def __init__(self):
        self.engine = AnalysisEngine()
        self.visualizer = Visualizer()
        self.summarizer = InsightSummarizer()
        self.df = None
        self.analysis_results = {}
    
    def load_data(self, file_path):
        """Load dataset from various formats"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            print("\n🔍 First 5 rows:")
            print(self.df.head())
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def auto_analyze(self):
        """Perform comprehensive analysis"""
        if self.df is None:
            print("❌ No data loaded. Load data first!")
            return
        
        print("\n🔬 Starting comprehensive analysis...")
        
        # Basic info
        self.analysis_results['info'] = self.engine.basic_info(self.df)
        
        # Statistical summary
        self.analysis_results['stats'] = self.engine.statistical_summary(self.df)
        
        # Missing data analysis
        self.analysis_results['missing'] = self.engine.missing_data_analysis(self.df)
        
        # Correlation analysis
        self.analysis_results['correlation'] = self.engine.correlation_analysis(self.df)
        
        # Outlier detection
        self.analysis_results['outliers'] = self.engine.outlier_detection(self.df)
        
        # Data quality
        self.analysis_results['quality'] = self.engine.data_quality_score(self.df)
        
        print("✅ Analysis completed!")
        return self.analysis_results
    
    def create_visualizations(self):
        """Create comprehensive set of charts"""
        if self.df is None:
            print("❌ No data loaded!")
            return {}
        
        charts = {}
        print("\n📈 Creating visualizations...")
        
        # Distribution plots
        charts['distribution'] = self.visualizer.distribution_charts(self.df)
        
        # Correlation heatmap
        charts['correlation'] = self.visualizer.correlation_heatmap(self.df)
        
        # Missing data heatmap
        charts['missing'] = self.visualizer.missing_data_heatmap(self.df)
        
        # Box plots for outliers
        charts['outliers'] = self.visualizer.outlier_boxplots(self.df)
        
        print("✅ Visualizations created!")
        return charts
    
    def generate_insights(self):
        """Generate human-readable insights"""
        if not self.analysis_results:
            print("❌ Run analysis first!")
            return ""
        
        insights = self.summarizer.generate_full_report(
            self.df, self.analysis_results
        )
        return insights
    
    def full_pipeline(self, file_path):
        """Run complete analysis pipeline"""
        print("🚀 Starting FULL Data Analysis Pipeline...")
        
        # Load data
        if not self.load_data(file_path):
            return
        
        # Analyze
        self.auto_analyze()
        
        # Visualize
        charts = self.create_visualizations()
        
        # Generate insights
        insights = self.generate_insights()
        
        return {
            'data': self.df,
            'analysis': self.analysis_results,
            'charts': charts,
            'insights': insights
        }

# Demo usage
if __name__ == "__main__":
    agent = DataAnalystAgent()
    results = agent.full_pipeline('sample_data.csv')