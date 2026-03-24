import logging
import plotly.express as px
import pandas as pd
import json
import re

logger = logging.getLogger(__name__)

class ChartResult:
    def __init__(self, code, business_story=""):
        self.code = code
        self.business_story = business_story

class HybridVisualizer:
    def __init__(self, provider="groq", api_key=None):
        """
        STAGE 6: Type-Aware Visual Logic & LIDA Integration.
        """
        self.provider = provider
        from lida import Manager, llm
        
        # 1. Initialize self.df as None so the attribute exists immediately
        self.df = None 
        
        # Setup LIDA LLM Backend
        text_gen = llm(provider="openai", api_key=api_key)
        if provider == "groq":
            text_gen.client.base_url = "https://api.groq.com/openai/v1"
            self.model_name = "llama-3.3-70b-versatile"
        else:
            text_gen.client.base_url = "http://localhost:11434/v1"
            self.model_name = "llama3"
            
        self.lida = Manager(text_gen=text_gen)
        self.summary = None

    def set_data(self, df):
        """Captures both the raw dataframe and the LIDA summary."""
        self.df = df
        try:
            self.summary = self.lida.summarize(df)
        except Exception as e:
            logger.error(f"LIDA Summary Generation Failed: {e}")

    def get_template_code(self, goal):
        """
        DETERMINISTIC ROUTER: Uses Metadata to force stable Plotly code.
        """
        if self.df is None:
            return "response = {'text': 'Visualizer Error: No data loaded.'}"

        # Extract metadata
        pair = goal.get('metadata', {}).get('pair', [None, None])
        col1, col2 = pair[0], pair[1]
        dtype1 = goal.get('metadata', {}).get('dtype1', 'object').lower()
        cardinality = goal.get('metadata', {}).get('cardinality1', 0)
        agg_type = goal.get('metadata', {}).get('agg_type', 'mean')
        
        # --- TYPE-SAFETY CHECK ---
        # Handle single-column cases and prevent math on string columns
        is_numeric = False
        if col2 and col2 in self.df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if not is_numeric:
            agg_type = 'count'
            agg_label = "Count of"
        else:
            agg_label = str(agg_type).capitalize()

        # --- CHART TYPE ROUTING ---
        if col1 and ("date" in dtype1 or "datetime" in dtype1 or "time" in str(col1).lower()):
            tid = "line_trend"
        elif cardinality > 1 and cardinality <= 8:
            tid = "pie_share"
        elif col1 is None or col1 == col2 or col1 == "Unnamed: 0":
            tid = "histogram_dist"
        else:
            tid = "bar_ranking"

        # Sanitize for code injection
        c1 = str(col1).replace("'", "\\'") if col1 else None
        c2 = str(col2).replace("'", "\\'") if col2 else None

        # Templates with numeric_only and reset_index safety
        if agg_type in ['mean', 'sum', 'median', 'std', 'var']:
            agg_arg = "numeric_only=True"
        else:
            agg_arg = "" # count() takes no arguments

        # --- UPDATED TEMPLATES ---
        templates = {
            "bar_ranking": f"""
top_df = df.groupby('{c1}')['{c2}'].{agg_type}({agg_arg}).sort_values(ascending=False).head(10).reset_index()
fig = px.bar(top_df, x='{c2}', y='{c1}', orientation='h', color='{c1}', template='plotly_dark', title='{agg_label} {c2} by {c1}')
fig.update_layout(showlegend=False, yaxis={{'categoryorder':'total ascending'}})
""",
            "line_trend": f"""
df['{c1}'] = pd.to_datetime(df['{c1}'])
trend_df = df.groupby('{c1}')['{c2}'].{agg_type}({agg_arg}).reset_index().sort_values('{c1}')
fig = px.line(trend_df, x='{c1}', y='{c2}', markers=True, template='plotly_dark', title='Trend: {c2}')
""",
            "pie_share": f"""
share_df = df.groupby('{c1}')['{c2}'].{agg_type}({agg_arg}).reset_index()
fig = px.pie(share_df, names='{c1}', values='{c2}', hole=0.4, template='plotly_dark', title='{c2} Distribution')
""",
            "histogram_dist": f"""
fig = px.histogram(df, x='{c2}', nbins=30, marginal='box', template='plotly_dark', title='Distribution: {c2}')
"""
        }

        code = templates.get(tid, templates["histogram_dist"])
        code += "\nfig.update_layout(margin=dict(l=20, r=20, t=50, b=20))"
        code += "\nresponse = {'plot': fig}"
        return code

    def generate_dashboard_charts(self, lida_goals):
        """Generates a list of ChartResult objects for the dashboard loop."""
        charts = []
        for goal in lida_goals:
            code = self.get_template_code(goal)
            chart_obj = ChartResult(code=code, business_story=goal.get('business_impact', 'Insight detected.'))
            charts.append(chart_obj)
        return charts

    def generate_ai_chart(self, user_query, library="plotly"):
        """Pure AI Generation using LIDA for custom chatbot requests."""
        from lida import TextGenerationConfig
        if self.summary is None: return None
        try:
            # Create a goal based on user query
            from lida.datamodel import Goal
            goal = Goal(question=user_query, visualization=user_query, rationale="")
            
            visuals = self.lida.visualize(summary=self.summary, goal=goal, library=library, 
                                          textgen_config=TextGenerationConfig(n=1, model=self.model_name, temperature=0.2))
            return visuals[0] if visuals else None
        except Exception as e:
            logger.error(f"AI Visualization Failed: {e}")
            return None