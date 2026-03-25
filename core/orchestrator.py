import json
import logging
import re
import openai
from groq import Groq

logger = logging.getLogger(__name__)

class SemanticOrchestrator:
    def __init__(self, provider="groq", api_key=None):
        """
        Unified Orchestrator: Supports Cloud (Groq) and Local (Ollama) endpoints.
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

    def _call_llm(self, system, user, json_mode=False, temperature=0.1):
        params = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature
        }
        if json_mode and self.provider == "groq":
            params["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content

    def generate_analysis_plan(self, fingerprint):
        """Generates initial domain and target for the dashboard."""
        system_prompt = "You are a Senior Data Architect. Return ONLY a JSON object with keys: 'domain', 'target'."
        user_prompt = f"Identify domain/target from fingerprint: {json.dumps(list(fingerprint.get('columns', {}).keys()))}"
        
        try:
            resp = self._call_llm(system_prompt, user_prompt, json_mode=True)
            raw_plan = self._safe_json_parse(resp)
            return {
                "domain": raw_plan.get("domain", "Business Intelligence"),
                "target": raw_plan.get("target", "General Metrics")
            }
        except Exception:
            return {"domain": "Business Intelligence", "target": "General Metrics"}

    def handle_complex_query(self, user_input, fingerprint, dashboard_context=None, hardened_hints=None):
        """
        NL to Python conversion with Phase 5 Data Grounding.
        """
        # 1. INTENT GATE
        data_indicators = ['total', 'average', 'mean', 'sum', 'count', 'show', 'list', 'plot', 'chart', 'compare', 'percentage']
        is_data_query = any(kw in user_input.lower() for kw in data_indicators) or (hardened_hints and len(hardened_hints) > 0)

        if not is_data_query:
            general_resp = self._call_llm("Answer directly. No code.", user_input)
            return f"response = {{'text': \"{general_resp}\", 'table': None}}"

        # 2. SCHEMA GROUNDING (Phase 5)
        schema_context = ""
        cols_info = fingerprint.get('columns', {})
        for col, info in cols_info.items():
            schema_context += f"- {col} ({info.get('dtype')}). Samples: {info.get('samples')}\n"

        # 3. DYNAMIC SYSTEM PROMPT
        system_prompt = f"""
        Act as a Strict Forensic Data Auditor. 
        
        STRICT RULES FOR MATHEMATICS:
        1. **Pre-Check**: Before calculating, print the count of non-null rows: `print(df[col].count())`.
        2. **Type Enforcement**: You MUST use `pd.to_numeric(df[col], errors='coerce')` on any column used for math. 
        3. **No Percentage Sums**: Never sum percentages. Calculate the raw numerator and denominator first, then divide.
        4. **Verification**: After your calculation, write a check: 
           `if result > total_rows: # Logic is likely wrong, adjust.`
        5. **Narrative Grounding**: The 'text' MUST state the sample size. 
           Example: "Out of {{len(df)}} total records, {{match_count}} were analyzed..."

        OUTPUT FORMAT:
        response = {{
            'text': f"### Math Audit\\n* Total Rows: {{len(df)}}\\n* Matches Found: {{len(res_df)}}\\n\\n**Final Answer**: {{calculation_result}}",
            'table': res_df
        }}
        7. GROUNDING: If columns are missing or the query is impossible, 
           you MUST still return a dictionary:
           response = {{'text': 'I cannot find that data.', 'table': None}}
           NEVER set response = None.
        """
        return self._call_llm(system_prompt, user_input)

    def rank_insights(self, stats_results, plan, limit=5):
        """Ranks raw findings by business impact. Required by SynthesisFilter."""
        if isinstance(plan, str): plan = {"domain": plan}
        findings = [{"id": i, "vars": r.get("vars", [])} for i, r in enumerate(stats_results[:15])]
        
        system = "Rank findings by business impact. Return JSON: {'ranked_ids': [0, 1, 2]}"
        user = f"Domain: {plan.get('domain')}\nFindings: {json.dumps(findings)}"
        
        try:
            resp = self._call_llm(system, user, json_mode=True)
            data = self._safe_json_parse(resp)
            return data.get("ranked_ids", list(range(min(len(stats_results), limit))))
        except:
            return list(range(min(len(stats_results), limit)))

    def categorize_insights(self, discoveries, domain):
        """
        Categorizes insights into strategic levels. Required by SynthesisFilter.
        """
        system_prompt = "Strategist. Categorize IDs into STRATEGIC/OPERATIONAL/TACTICAL. Return JSON."
        user_prompt = f"DATA: {json.dumps([{'id': d['id'], 'q': d['question']} for d in discoveries])}"
        try:
            resp = self._call_llm(system_prompt, user_prompt, json_mode=True)
            data = self._safe_json_parse(resp).get("categorization", [])
            return [c for c in data if isinstance(c, dict) and "id" in c]
        except:
            return [{"id": d['id'], "level": "TACTICAL"} for d in discoveries]

    def generate_follow_ups(self, user_query, result_text, domain):
        """Generates 3 suggestion chips."""
        system = f"Consultant for {domain}. Return 3 follow-ups in JSON: {{'suggestions': []}}"
        try:
            resp = self._call_llm(system, f"Query: {user_query}", json_mode=True)
            return self._safe_json_parse(resp).get("suggestions", ["Show trends", "Breakdown", "Compare"])
        except:
            return ["Show trends", "Breakdown", "Compare"]

    def request_fix(self, broken_code, error_msg):
        """Self-healing loop."""
        system = "Expert Python Debugger. Return ONLY corrected code."
        user = f"FIX THIS CODE.\nERROR: {error_msg}\nCODE:\n{broken_code}"
        try:
            code = self._call_llm(system, user, temperature=0.1)
            return self._clean_code(code)
        except:
            return broken_code

    def get_business_story(self, col1, col2, test, domain):
        """Generates impact summary for correlation results."""
        system = f"Analyst for {domain}."
        user = f"Impact of link between {col1} and {col2} ({test}). 2 sentences."
        try:
            return self._call_llm(system, user, temperature=0.4)
        except:
            return "Relationship detected with potential business impact."

    def _safe_json_parse(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except:
            return {}

    def _clean_code(self, raw_code):
        code = re.sub(r'```python|```', '', raw_code)
        return code.strip()