# --- core/matcher.py ---
from rapidfuzz import process, utils

class FuzzyMatcher:
    def __init__(self, columns, threshold=85): # Increased from 70 to 85
        self.columns = columns
        self.threshold = threshold

    def find_best_match(self, term):
        """
        Finds the closest actual column name for a user's search term.
        Phase 3: High-threshold filtering to stop 'near-miss' hallucinations.
        """
        if not term or len(term) < 3: 
            return None
            
        # Clean the term for better matching
        clean_term = utils.default_process(term)
        
        match = process.extractOne(
            clean_term, 
            self.columns, 
            processor=utils.default_process
        )
        
        if match and match[1] >= self.threshold:
            return match[0] # Return the actual technical column name
        return None