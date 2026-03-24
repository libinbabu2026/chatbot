import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.visualizer import HybridVisualizer

# 1. CREATE MOCK DATA (Simulating a real dataset)
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'E'] * 20,
    'Date': pd.date_range(start='2026-01-01', periods=100),
    'Values': np.random.randint(10, 100, size=100),
    'Growth': np.random.uniform(0.1, 5.0, size=100)
})

# 2. DEFINE TEST GOALS (Covering all 4 new templates)
test_goals = [
    {
        "template_id": "composite_ranking",
        "question": "Top Categories by Value",
        "metadata": {"pair": ("Category", "Values"), "agg_type": "sum"}
    },
    {
        "template_id": "composite_trend",
        "question": "Value Trend over Time",
        "metadata": {"pair": ("Date", "Values"), "agg_type": "mean"}
    },
    {
        "template_id": "composite_share",
        "question": "Value Share by Category",
        "metadata": {"pair": ("Category", "Values"), "agg_type": "sum"}
    },
    {
        "template_id": "composite_distribution",
        "question": "Value Spread Analysis",
        "metadata": {"pair": (None, "Values")}
    }
]

# 3. INITIALIZE VISUALIZER
# Note: Providing a dummy key as we are testing template logic, not AI generation
viz = HybridVisualizer(provider="groq", api_key="dummy_key")

print("🚀 Starting Multi-Viz Stress Test...\n")

for goal in test_goals:
    tid = goal['template_id']
    print(f"Testing Template: [{tid}]...")
    
    try:
        # Get the Python string from your Visualizer
        code = viz.get_template_code(goal)
        
        # Prepare the "Execution Fridge" (The Scope)
        # This MUST match the scope in your app.py
        scope = {
            "df": df,
            "pd": pd,
            "px": px,
            "np": np,
            "go": go,
            "make_subplots": make_subplots,
            "response": {}
        }
        
        # EXECUTE
        exec(code, scope)
        
        # VERIFY
        if "plot" in scope['response']:
            print(f"✅ SUCCESS: {tid} generated a valid Plotly Figure.")
            # Verify it's a subplot (has more than 1 trace usually)
            trace_count = len(scope['response']['plot'].data)
            print(f"   - Found {trace_count} data traces in subplot.\n")
        else:
            print(f"❌ FAILURE: {tid} executed but 'response[plot]' is empty.\n")
            
    except Exception as e:
        print(f"💥 CRASH in {tid}: {str(e)}")
        print(f"--- FAILED CODE ---\n{code}\n-------------------\n")

print("🏁 Test Suite Complete.")