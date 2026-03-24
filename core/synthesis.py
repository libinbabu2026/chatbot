import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class SynthesisFilter:
    def __init__(self, stats_results, plan, df, orchestrator):
        self.stats_results = stats_results
        self.plan = plan
        self.df = df
        self.orchestrator = orchestrator
        self.target = plan.get('target', '').lower()

    def package_for_lida(self, limit=5):
        """
        Hardened Synthesis: Extracts strict metadata for the Visualizer Router.
        """
        discoveries = []
        
        # GATE 1: Filter by Statistical Significance
        valid_results = [res for res in self.stats_results if res.get('p_value', 1.0) < 0.05]

        for i, res in enumerate(valid_results):
            c1, c2 = res['pair']
            priority_score = 10 if (self.target in str(c1).lower() or self.target in str(c2).lower()) else 1

            # CALCULATE METADATA FOR ROUTER
            dtype1 = str(self.df[c1].dtype) if c1 in self.df.columns else "object"
            # Uniqueness count (Cardinality)
            unique_count = self.df[c1].nunique() if c1 in self.df.columns else 0

            discoveries.append({
                "id": i,
                "question": f"How does {c1} affect {c2}?",
                "priority": priority_score,
                "metadata": {
                    "pair": (c1, c2),
                    "p_value": round(res['p_value'], 4),
                    "test_used": res['test'],
                    "dtype1": dtype1,
                    "cardinality1": unique_count,
                    "agg_type": "sum" if any(k in str(c2).lower() for k in ["price", "sales", "volume", "total"]) else "mean"
                }
            })

        discoveries = sorted(discoveries, key=lambda x: (-x['priority'], x['metadata']['p_value']))

        top_ids = self.orchestrator.rank_insights(
            discoveries[:limit*2], 
            self.plan.get('domain', 'Business'), 
            limit=limit
        )
        final_discoveries = [d for d in discoveries if d['id'] in top_ids]

        categorized_raw = self.orchestrator.categorize_insights(
            final_discoveries, 
            self.plan.get('domain', 'Business')
        )

        level_map = {cat["id"]: cat.get("level", "TACTICAL") for cat in categorized_raw if isinstance(cat, dict) and "id" in cat}

        for goal in final_discoveries:
            goal['level'] = level_map.get(goal['id'], "TACTICAL")
            try:
                goal['metadata']['business_impact'] = self.orchestrator.get_business_story(
                    goal['metadata']['pair'][0], 
                    goal['metadata']['pair'][1], 
                    goal['metadata']['test_used'],
                    self.plan.get('domain', 'Business')
                )
            except Exception:
                goal['metadata']['business_impact'] = "Significant trend detected."

        return final_discoveries