import pandas as pd
import numpy as np
import os
import logging
from core.matcher import FuzzyMatcher # Now imports from the separate file

logger = logging.getLogger(__name__)

class UniversalProfiler:
    def __init__(self, file_path, target_memory_mb=100):
        self.file_path = file_path
        self.df = None
        self.target_memory_mb = target_memory_mb
        self._load_and_sample()
        
        # 1. FORCE STRING COLUMNS: Prevents 'Value of x is not a column' errors
        self.df.columns = [str(col).strip() for col in self.df.columns]
        
        # 2. INITIALIZE MATCHER: Ready for the chatbot
        self.matcher = FuzzyMatcher(self.df.columns)

    def _load_and_sample(self):
        """Industrial Smart Loader with CSV/Excel fallback logic."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        _, file_extension = os.path.splitext(self.file_path)
        ext = file_extension.lower()
        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)

        try:
            if ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
            else:
                self._read_csv_with_sampling(file_size_mb)
        except Exception as e:
            # Automatic fallback if format is mislabeled
            if any(err in str(e) for err in ["Excel file", "BadZipFile"]):
                self._read_csv_with_sampling(file_size_mb)
            else:
                raise ValueError(f"CRITICAL: Data read failed. Error: {e}")

    def _read_csv_with_sampling(self, file_size_mb):
        if file_size_mb > self.target_memory_mb:
            sample_rate = self.target_memory_mb / file_size_mb
            self.df = pd.read_csv(self.file_path).sample(frac=sample_rate)
        else:
            self.df = pd.read_csv(self.file_path)

    def get_column_map(self):
        """Generates a dictionary mapping raw keys to readable titles."""
        return {col: str(col).replace("-", " ").replace("_", " ").title() for col in self.df.columns}

    def get_fingerprint(self):
        """Generates metadata and injects the column map for LLM context."""
        fingerprint = {
            "columns": {}, 
            "total_rows": len(self.df),
            "column_map": self.get_column_map() # Crucial for AI Naming accuracy
        }
        for col in self.df.columns:
            fingerprint["columns"][col] = {
                "dtype": str(self.df[col].dtype),
                "samples": self.df[col].dropna().unique()[:3].tolist(),
                "null_count": int(self.df[col].isnull().sum())
            }
        return fingerprint

    def get_health_score(self):
        total_cells = self.df.size
        null_cells = self.df.isnull().sum().sum()
        return round((1 - (null_cells / total_cells)) * 100, 2) if total_cells > 0 else 0

    def get_dataframe(self):
        return self.df

    def get_matcher(self):
        return self.matcher