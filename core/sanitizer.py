import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

class DataSanitizer:
    def __init__(self, df, plan=None):
        self.df = df.copy() if df is not None else None
        self.plan = plan if plan is not None else {}

    def clean(self):
        if self.df is None: return None

        # 1. Mask PII
        self._mask_pii()

        # 2. Target Encoding: Convert Binary Text to 0/1 (Crucial for Adult Dataset)
        target = self.plan.get('target')
        if target in self.df.columns and self.df[target].dtype == 'object':
            unique_vals = self.df[target].dropna().unique()
            if len(unique_vals) == 2:
                # Map sorted values so the "higher" or "positive" one is 1
                val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
                self.df[target] = self.df[target].map(val_map)
                logger.info(f"Target Encoding applied to {target}: {val_map}")

        # 3. Numeric Recovery & Cleaning
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                clean_series = self.df[col].astype(str).str.replace(r'[$,% ]', '', regex=True)
                numeric_attempt = pd.to_numeric(clean_series, errors='coerce')
                if numeric_attempt.notnull().mean() > 0.8:
                    self.df[col] = numeric_attempt

            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self.df[col] = self.df[col].clip(self.df[col].quantile(0.01), self.df[col].quantile(0.99))
            else:
                self.df[col] = self.df[col].astype(str).str.strip().str.title().fillna("Unknown")
                
        return self.df

    def _mask_pii(self):
        email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        phone_regex = r'\b\d{10}\b'
        for col in self.df.select_dtypes(include=['object']):
            self.df[col] = self.df[col].apply(lambda x: re.sub(email_regex, "[EMAIL_REDACTED]", str(x)))
            self.df[col] = self.df[col].apply(lambda x: re.sub(phone_regex, "[PHONE_REDACTED]", str(x)))

    def register_new_column(self, col_name, fingerprint):
        if col_name not in self.df.columns: return fingerprint
        fingerprint["columns"][col_name] = {
            "dtype": str(self.df[col_name].dtype),
            "samples": self.df[col_name].dropna().unique()[:3].tolist(),
            "null_count": int(self.df[col_name].isnull().sum()),
            "is_calculated": True
        }
        return fingerprint