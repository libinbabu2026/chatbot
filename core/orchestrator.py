import json
import logging
import re
import openai
from groq import Groq

# Industrial logging
logger = logging.getLogger(__name__)

class SemanticOrchestrator:
    def __init__(self, provider="groq", api_key=None):
        """
        Phase 9 Readiness: Supports both Cloud (Groq) and Local (Ollama) endpoints.
        """
        self.provider = provider
        if provider == "groq":
            if not api_key:
                raise ValueError("Groq API Key is required for Cloud mode.")
            self.client = Groq(api_key=api_key) 
            self.model = "llama-3.3-70b-versatile"
        elif provider == "ollama":
            self.client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model = "llama3.1"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_analysis_plan(self, fingerprint):
        """Phase 2: Initial Architecting with Schema Enforcement."""
        system_prompt = "You are a Senior Data Architect. Return ONLY a JSON object with keys: 'domain', 'target'."
        user_prompt = f"Identify domain/target from fingerprint: {json.dumps(list(fingerprint.get('columns', {}).keys()))}"
        
        try:
            resp = self._call_llm(system_prompt, user_prompt, json_mode=True)
            raw_plan = self._safe_json_parse(resp)
            return {
                "domain": raw_plan.get("domain", "Business Intelligence"),
                "target": raw_plan.get("target", "General Metrics")
            }
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return {"domain": "Business Intelligence", "target": "General Metrics"}

    def handle_complex_query(self, user_input, fingerprint, dashboard_context=None, hardened_hints=None):
        """
        STAGES 1-4: Hardened Multi-Column Calculation Logic.
        Implements Dynamic Pruning and Mapping for high accuracy.
        """
        # 1. TOKEN OPTIMIZATION: Filter for relevant columns only
        all_cols = list(fingerprint.get('columns', {}).keys())
        relevant_cols = [c for c in all_cols if c.lower() in user_input.lower()]
        if hardened_hints:
            relevant_cols.extend(hardened_hints)
        
        # Determine the pruned list to send to LLM
        display_cols = list(set(relevant_cols)) if relevant_cols else all_cols

        # 2. COLUMN MAPPING: Get the Raw:Friendly dictionary
        column_map = fingerprint.get('column_map', {})
        mapping_instruction = f"\nCOLUMN DICTIONARY (Raw Name: Friendly Name): {json.dumps(column_map)}"
        
        # 3. CONTEXTUAL HINTS
        hints_str = f"\nVERIFIED HINTS: {', '.join(hardened_hints)}" if hardened_hints else ""
        context_str = f"\nCURRENT DASHBOARD CONTEXT: {json.dumps(dashboard_context)}" if dashboard_context else ""

        system_prompt = f"""
        You are a Deterministic Data Engine. Use variable 'df'.
        VALID RAW COLUMNS: {json.dumps(display_cols)}
        {mapping_instruction}
        {hints_str}
        {context_str}

        STRICT RULES:
        1. USE RAW NAMES: Always use exact column names  like 'native-country' from the VALID list. Do NOT use 'Country'.
        2. VECTORIZED MATH: Use df['New'] = df['A'] * df['B']. No loops.
        3. DEFINE BEFORE USE: You must create a new column before referencing it in a plot or filter.
        4. RESET INDEX: After grouping or aggregating, always use .reset_index().
        5. OUTPUT: response = {{'text': 'explanation', 'plot': fig, 'table': result_df}}
        6. TYPE-SAFETY: Before using .mean(), .sum(), or plotting a numeric axis, ensure the column is numeric.
        If it is not, use: df[col] = pd.to_numeric(df[col], errors='coerce')
        7. AGGREGATION: When using .groupby(), always follow it with .mean(numeric_only=True) to avoid errors.
        8. RESET INDEX: Always end aggregations with .reset_index().
        
        TASK: Write Python code. RETURN ONLY CODE. No preamble.
        """
        try:
            # Note: temperature is kept low (0.1) for deterministic code generation
            code = self._call_llm(system_prompt, user_input, temperature=0.1)
            return self._clean_code(code)
        except Exception as e:
            return f"response = {{'text': 'Logic Generation Error: {str(e)}'}}"

    def generate_follow_ups(self, user_query, result_text, domain):
        """Ensures suggestions are always a flat list of strings."""
        system_prompt = f"Consultant for {domain}. Return 3 short follow-up questions in JSON format."
        user_prompt = f"Query: {user_query}\nResult: {result_text}\nFormat: {{'suggestions': ['Q1', 'Q2', 'Q3']}}"
        try:
            resp = self._call_llm(system_prompt, user_prompt, json_mode=True, temperature=0.5)
            raw_sugs = self._safe_json_parse(resp).get("suggestions", [])
            
            # CRITICAL FIX: Ensure we only return strings, not dicts or objects
            processed = []
            for s in raw_sugs:
                if isinstance(s, dict):
                    # If AI sent [{'question': '...'}], extract the value
                    processed.append(list(s.values())[0])
                else:
                    processed.append(str(s))
            return processed[:3]
        except:
            return ["Compare with last year", "Identify outliers", "Detailed breakdown"]

    def request_fix(self, broken_code, error_msg):
        """Phase 8: Self-Healing Loop."""
        system_prompt = "Expert Python Debugger. Return ONLY corrected code."
        user_prompt = f"FIX THIS CODE.\nERROR: {error_msg}\nCODE:\n{broken_code}"
        try:
            code = self._call_llm(system_prompt, user_prompt, temperature=0.1)
            return self._clean_code(code)
        except:
            return broken_code

    def rank_insights(self, discoveries, domain, limit=5):
        """Gate 3: Ranks insights by business impact."""
        system_prompt = f"Analyst in {domain}. Rank top {limit} IDs. Format: {{'selected_ids': []}}"
        user_prompt = f"Data: {json.dumps(discoveries)}"
        try:
            resp = self._call_llm(system_prompt, user_prompt, json_mode=True)
            return self._safe_json_parse(resp).get("selected_ids", [d['id'] for d in discoveries[:limit]])
        except:
            return [d['id'] for d in discoveries[:limit]]

    def categorize_insights(self, discoveries, domain):
        """Phase 7: Strategic categorization."""
        system_prompt = "Strategist. Categorize IDs into STRATEGIC/OPERATIONAL/TACTICAL. Return JSON."
        user_prompt = f"DATA: {json.dumps([{'id': d['id'], 'q': d['question']} for d in discoveries])}"
        try:
            resp = self._call_llm(system_prompt, user_prompt, json_mode=True)
            data = self._safe_json_parse(resp).get("categorization", [])
            return [c for c in data if isinstance(c, dict) and "id" in c]
        except:
            return [{"id": d['id'], "level": "TACTICAL"} for d in discoveries]

    def get_business_story(self, col1, col2, test, domain):
        """Generates the impact summary."""
        system_prompt = f"Analyst for {domain}."
        user_prompt = f"Impact of link between {col1} and {col2} ({test}). 2 sentences."
        try:
            return self._call_llm(system_prompt, user_prompt, temperature=0.4)
        except:
            return "Relationship detected with potential business impact."

    def _call_llm(self, system, user, json_mode=False, temperature=0.1):
        """Unified LLM interface."""
        params = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature
        }
        if json_mode and self.provider == "groq":
            params["response_format"] = {"type": "json_object"}
            
        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content

    def _safe_json_parse(self, text):
        """Defensive JSON parsing using regex."""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except:
            return {}

    def _clean_code(self, raw_code):
        """Removes markdown artifacts."""
        code = re.sub(r'```python|```', '', raw_code)
        return code.strip()