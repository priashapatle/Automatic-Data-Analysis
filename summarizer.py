class InsightSummarizer:
    def generate_full_report(self, df, analysis_results):
        """Generate comprehensive human-readable report"""
        report = []
        
        # Dataset overview
        report.append("## 📊 DATASET OVERVIEW")
        report.append(f"- **Rows**: {df.shape[0]:,}")
        report.append(f"- **Columns**: {df.shape[1]}")
        report.append(f"- **Data Quality Score**: {analysis_results['quality']}/100")
        report.append("")
        
        # Missing data insights
        missing = analysis_results['missing']
        high_missing = missing[missing['missing_percentage'] > 10]
        if not high_missing.empty:
            report.append("## 🚨 CRITICAL ISSUES")
            for col in high_missing.index:
                pct = high_missing.loc[col, 'missing_percentage']
                report.append(f"- **{col}**: {pct:.1f}% missing ({high_missing.loc[col, 'missing_count']:,} rows)")
            report.append("")
        
        # Key statistics
        stats = analysis_results['stats']
        if stats['numeric_columns']:
            report.append("## 📈 KEY STATISTICS")
            for col in stats['numeric_columns'][:3]:  # Top 3 numeric columns
                desc = stats['describe']['describe'][col]
                report.append(f"**{col}**:")
                report.append(f"  - Mean: {desc['mean']:.2f}")
                report.append(f"  - Median: {desc['50%']:.2f}")
                report.append(f"  - Range: {desc['min']:.2f} - {desc['max']:.2f}")
            report.append("")
        
        # Correlations
        if 'high_correlations' in analysis_results['correlation']:
            corrs = analysis_results['correlation']['high_correlations']
            if not corrs.empty:
                report.append("## 🔗 STRONG CORRELATIONS")
                for _, row in corrs.iterrows():
                    report.append(f"- **{row['var1']} ↔ {row['var2']}**: {row['correlation']:.3f}")
                report.append("")
        
        # Outliers
        outliers = analysis_results['outliers']
        high_outliers = {k: v for k, v in outliers.items() if v['percentage'] > 5}
        if high_outliers:
            report.append("## ⚠️ OUTLIERS DETECTED")
            for col, info in high_outliers.items():
                report.append(f"- **{col}**: {info['count']:,} outliers ({info['percentage']:.1f}%)")
            report.append("")
        
        # Recommendations
        report.append("## 💡 RECOMMENDATIONS")
        if analysis_results['quality'] < 70:
            report.append("- Clean missing values and outliers before modeling")
        report.append("- Explore strong correlations for feature engineering")
        report.append("- Consider data transformation for skewed distributions")
        
        return "\n".join(report)