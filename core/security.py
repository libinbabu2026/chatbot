import re
import importlib

class SecurityGuard:
    def __init__(self):
        # High-risk keywords that could lead to system compromise
        self.BLACKLIST = [
            r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bshutil\b', 
            r'\bopen\(', r'\beval\(', r'\bexec\(', r'\brequests\b', 
            r'\bsocket\b', r'\bgetattr\b', r'\bsetattr\b', r'\b__builtins__\b'
        ]
        
        self.REQUIRED_LIBS = [
            'pandas', 'numpy', 'sklearn', 'scipy', 'plotly', 'lida'
        ]

    def verify_environment(self):
        """Checks if all industrial-standard libraries are installed."""
        missing = []
        for lib in self.REQUIRED_LIBS:
            try:
                importlib.import_module(lib)
            except ImportError:
                missing.append(lib)
        return missing

    def is_code_safe(self, code: str) -> bool:
        """Scans code for blacklisted patterns."""
        for pattern in self.BLACKLIST:
            if re.search(pattern, code):
                return False
        return True

    # core/security.py

    def get_safe_scope(self, df):
        import pandas as pd
        import numpy as np
        import plotly.express as px
        
        # Define a whitelist of safe built-in functions
        # This allows math/logic but prevents system access
        safe_builtins = {
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'dict': dict,
            'list': list,
            'bool': bool,
            'range': range,
            'enumerate': enumerate,
            'zip': zip
        }
        
        return {
            "df": df,
            "pd": pd,
            "np": np,
            "px": px,
            "response": {},
            "__builtins__": safe_builtins # Pass the whitelist here
        }