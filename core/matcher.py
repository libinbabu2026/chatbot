from rapidfuzz import process, utils

class FuzzyMatcher:
    """
    STAGE 1: Maps fuzzy user terms to exact column names.
    Ensures 'sales' -> 'Total_Sales_2025' correctly.
    """
    def __init__(self, columns):
        # Convert all columns to strings to prevent matching errors
        self.columns = [str(c) for c in columns]

    def find_best_match(self, user_term, threshold=70):
        if not user_term or len(user_term) < 2:
            return None
            
        # Extract the single best match using Levenshtein distance
        match = process.extractOne(
            user_term, 
            self.columns, 
            processor=utils.default_process,
            score_cutoff=threshold
        )
        
        return match[0] if match else None