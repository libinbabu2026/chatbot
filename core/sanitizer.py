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

        # 1. COLUMN NAME NORMALIZATION
        # Prevents KeyErrors from trailing spaces in CSV headers
        self.df.columns = [str(c).strip() for c in self.df.columns]

        # 2. MASK PII
        self._mask_pii()

        # 3. ENHANCED NUMERIC RECOVERY (PHASE 7)
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Remove currency symbols, commas, percent signs, and whitespace
                # This fixes " $ 1,200.00 " -> "1200.00"
                clean_series = self.df[col].astype(str).str.replace(r'[$,%\s]', '', regex=True)
                
                # Strip any remaining non-numeric characters except decimals
                clean_series = clean_series.str.replace(r'[^0-9.]', '', regex=True)
                
                # Attempt conversion
                numeric_attempt = pd.to_numeric(clean_series, errors='coerce')
                
                # HIGH THRESHOLD: If > 60% of data can be numeric, force it.
                if numeric_attempt.notnull().mean() > 0.6:
                    self.df[col] = numeric_attempt

        # 4. FINAL TYPE-SPECIFIC PROCESSING
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Fill NaNs with median to prevent math from returning 'NaN'
                # This is critical for the AI's .mean() and .sum() calls
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
                # Clip outliers to 1st and 99th percentile to prevent skewed math
                self.df[col] = self.df[col].clip(
                    self.df[col].quantile(0.01), 
                    self.df[col].quantile(0.99)
                )
            else:
                # Standardize categorical text
                self.df[col] = self.df[col].astype(str).str.strip().str.title().replace("Nan", "Unknown").fillna("Unknown")
                
        return self.df

    def _mask_pii(self):
        email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        phone_regex = r'\b\d{10}\b'
        for col in self.df.select_dtypes(include=['object']):
            self.df[col] = self.df[col].apply(lambda x: re.sub(email_regex, "[EMAIL_REDACTED]", str(x)))
            self.df[col] = self.df[col].apply(lambda x: re.sub(phone_regex, "[PHONE_REDACTED]", str(x)))

    def register_new_column(self, col_name, fingerprint):
        """Registers AI-created columns into the system fingerprint for future use."""
        if col_name not in self.df.columns: return fingerprint
        fingerprint["columns"][col_name] = {
            "dtype": str(self.df[col_name].dtype),
            "samples": self.df[col_name].dropna().unique()[:3].tolist(),
            "null_count": int(self.df[col_name].isnull().sum()),
            "is_calculated": True
        }
        return fingerprint