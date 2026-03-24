import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class StatsEngine:
    def __init__(self, df):
        self.df = df
        # Ensure column names are strings for reference safety
        self.df.columns = [str(c) for c in self.df.columns]

    def analyze(self):
        """
        MULTI-TEST DISCOVERY: Selects the appropriate test for every column pair.
        """
        results = []
        cols = self.df.columns
        
        # Identify numeric and categorical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        for i, col_a in enumerate(cols):
            for col_b in cols[i+1:]:
                res = None
                
                # CASE 1: Numeric vs Numeric -> Pearson Correlation
                if col_a in num_cols and col_b in num_cols:
                    res = self._pearson_test(col_a, col_b)
                
                # CASE 2: Categorical vs Numeric -> ANOVA
                elif (col_a in cat_cols and col_b in num_cols) or (col_a in num_cols and col_b in cat_cols):
                    cat, num = (col_a, col_b) if col_a in cat_cols else (col_b, col_a)
                    res = self._anova_test(cat, num)
                
                # CASE 3: Categorical vs Categorical -> Chi-Square
                elif col_a in cat_cols and col_b in cat_cols:
                    res = self._chi2_test(col_a, col_b)

                if res and res.get('p_value', 1.0) < 0.05:
                    results.append(res)

        return results

    def _pearson_test(self, col1, col2):
        """Tests linear relationship between two numbers."""
        try:
            clean_df = self.df[[col1, col2]].dropna()
            if len(clean_df) < 5: return None
            corr, p = stats.pearsonr(clean_df[col1], clean_df[col2])
            return {"pair": (col1, col2), "p_value": p, "test": "Pearson Correlation", "score": abs(corr)}
        except: return None

    def _anova_test(self, cat_col, num_col):
        """Tests if numeric values vary significantly across categories."""
        try:
            groups = [group[num_col].values for name, group in self.df.groupby(cat_col) if len(group) > 5]
            if len(groups) < 2: return None
            f_stat, p = stats.f_oneway(*groups)
            return {"pair": (cat_col, num_col), "p_value": p, "test": "ANOVA", "score": f_stat}
        except: return None

    def _chi2_test(self, col1, col2):
        """Tests dependency between two categories."""
        try:
            contingency_table = pd.crosstab(self.df[col1], self.df[col2])
            chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
            return {"pair": (col1, col2), "p_value": p, "test": "Chi-Square", "score": chi2}
        except: return None