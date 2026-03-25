import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import traceback
import ast 
import re

# Advanced Plotly Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core Engine Imports
from core.profiler import UniversalProfiler
from core.orchestrator import SemanticOrchestrator
from core.sanitizer import DataSanitizer
from core.stats_engine import StatsEngine
from core.synthesis import SynthesisFilter
from core.visualizer import HybridVisualizer

# --- STAGE 5: ADAPTIVE KILL-GATE VALIDATOR (BYPASSED) ---
def validate_code_integrity(code, valid_columns):
    """
    WARNING: Kill-Gate Disabled. 
    Validation bypassed. Arbitrary code execution is now possible.
    """
    # Retain leading zero fix to prevent SyntaxError: invalid decimal literal
    cleaned_code = re.sub(r'\b0+(\d+)\b', r'\1', code)
    
    # Always return True to bypass security checks
    return True, "Security Gate: BYPASSED", cleaned_code

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="InsightEngine Pro", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetric"] { background-color: #ffffff !important; border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0; }
    [data-testid="stMetricLabel"] p { color: #555555 !important; font-weight: bold !important; }
    [data-testid="stMetricValue"] div { color: #000000 !important; }
    .stExpander { background-color: #ffffff; color: #000000 !important; border-radius: 10px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PIPELINE ---
# --- app.py (Update this function) ---
@st.cache_data(show_spinner=False)
def run_heavy_pipeline(temp_path, api_key):
    profiler = UniversalProfiler(temp_path)
    raw_df = profiler.get_dataframe()
    
    # 1. Clean the data first
    sanitizer = DataSanitizer(raw_df) 
    clean_df = sanitizer.clean()
    
    # 2. Generate a FRESH fingerprint from the CLEANED data
    # This ensures the AI sees the actual columns and actual values
    clean_fingerprint = {
        "columns": {
            col: {
                "dtype": str(clean_df[col].dtype),
                "samples": clean_df[col].dropna().unique()[:5].tolist(), # Send 5 real examples
                "stats": clean_df[col].describe().to_dict() if clean_df[col].dtype != 'object' else "categorical"
            } for col in clean_df.columns
        },
        "row_count": len(clean_df)
    }
    
    temp_orch = SemanticOrchestrator(provider="groq", api_key=api_key)
    dp = {
        "clean_df": clean_df,
        "fingerprint": clean_fingerprint,
        "health_score": profiler.get_health_score(),
        "column_map": profiler.get_column_map(),
        "matcher": profiler.get_matcher(),
        "columns": list(clean_df.columns)
    }
    dp["plan"] = temp_orch.generate_analysis_plan(dp["fingerprint"])
    stats_engine = StatsEngine(dp["clean_df"])
    dp["stats_results"] = stats_engine.analyze()
    return dp

# --- 3. SESSION INITIALIZATION ---
for key in ["messages", "suggestions", "data_pack", "lineage_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "log" in key or "mess" in key or "sugg" in key else None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    api_key = st.text_input("Groq API Key", type="password")
    insight_limit = st.slider("Max Insights", 1, 10, 5)
    
    if st.session_state.data_pack:
        st.divider()
        with st.expander("🔍 Data Dictionary"):
            cmap = st.session_state.data_pack.get('column_map', {})
            df_map = pd.DataFrame(list(cmap.items()), columns=["Raw Name", "Friendly Name"])
            st.dataframe(df_map, hide_index=True, use_container_width=True)

    if st.button("🔄 Reset System"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

# --- 5. DATA INGESTION ---
uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])

if uploaded_file and api_key:
    orchestrator = SemanticOrchestrator(provider="groq", api_key=api_key)
    visualizer = HybridVisualizer(provider="groq", api_key=api_key)

    if st.session_state.data_pack is None or st.session_state.get("current_file") != uploaded_file.name:
        temp_path = f"data/{uploaded_file.name}"
        if not os.path.exists("data"): os.makedirs("data")
        with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

        with st.status("🏗️ Building Intelligence...", expanded=True) as status:
            try:
                dp = run_heavy_pipeline(temp_path, api_key)
                synthesis = SynthesisFilter(dp['stats_results'], dp['plan'], dp['clean_df'], orchestrator)
                lida_goals = synthesis.package_for_lida(limit=insight_limit)
                st.session_state.data_pack = dp
                st.session_state.lida_goals = lida_goals
                st.session_state.current_file = uploaded_file.name
                status.update(label="✅ Intelligence Ready", state="complete")
            except Exception as e:
                st.error(f"Build Error: {traceback.format_exc()}"); st.stop()

    # Load persistent data & define GLOBAL variables
    data_pack = st.session_state.data_pack
    lida_goals = st.session_state.lida_goals
    plan = data_pack.get('plan', {}) 
    valid_cols = data_pack['columns']
    visualizer.set_data(data_pack['clean_df'])

    mode = st.radio("Interface:", ["📊 Dashboard", "💬 Assistant", "📜 Lineage"], horizontal=True)
    st.divider()

    # --- 6. MODE A: DASHBOARD ---
    if mode == "📊 Dashboard":
        st.header(f"📊 {plan.get('domain', 'Business')} Command Center")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Health", f"{data_pack['health_score']}%")
        m2.metric("Rows", f"{len(data_pack['clean_df']):,}")
        m3.metric("Target", plan.get('target', 'General'))
        m4.metric("Insights", len(lida_goals))

        for i, goal in enumerate(lida_goals):
            with st.expander(goal['question']):
                c_chart, c_text = st.columns([2, 1])
                with c_chart:
                    code = visualizer.get_template_code(goal)
                    is_valid, err, healed_code = validate_code_integrity(code, valid_cols)
                    if is_valid:
                        scope = {
                            "df": data_pack['clean_df'].copy(), 
                            "pd": pd, "px": px, "np": np, "go": go, 
                            "make_subplots": make_subplots, "response": {}
                        }
                        try:
                            exec(healed_code, scope)
                            res = scope.get("response", {})
                            if "plot" in res and res["plot"] is not None:
                                st.plotly_chart(res["plot"], use_container_width=True, key=f"db_plot_{i}")
                        except Exception as e: st.error(f"Logic Error: {e}")
                    else:
                        st.warning(err)
                with c_text: st.info(goal['metadata'].get('business_impact', 'Analyzing...'))

    # --- 7. MODE B: ASSISTANT (STABILITY FIX) ---
    elif mode == "💬 Assistant":
        st.header("💬 AI Data Assistant")
        
        # 1. LOOP: Show previous chat history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("table") is not None:
                    with st.container(border=True):
                        st.caption("📋 **Data Verification Table**")
                        # Use to_string() failsafe if markdown fails
                        st.dataframe(msg["table"], use_container_width=True, hide_index=True)
                if msg.get("plot") is not None: 
                    st.plotly_chart(msg["plot"], use_container_width=True, key=f"chat_plot_{i}")

        # 2. INPUT: Get user query
        user_input = st.chat_input("Ask about your data...")
        if st.session_state.suggestions:
            st.caption("Suggested:")
            chips = st.columns(len(st.session_state.suggestions))
            for i, sug in enumerate(st.session_state.suggestions):
                if chips[i].button(str(sug), key=f"s_{i}", use_container_width=True): user_input = sug

        if user_input:
            st.session_state.suggestions = []
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

        # 3. LOGIC: Generate and execute answer
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_query = st.session_state.messages[-1]["content"]
            
            with st.chat_message("assistant"):
                # Create a placeholder so the answer appears LIVE
                container = st.container() 
                
                with st.status("🧠 Analyzing Data...", expanded=False) as status:
                    matcher = data_pack['matcher']
                    hints = [matcher.find_best_match(w) for w in user_query.split() if matcher.find_best_match(w)]
                    query_code = orchestrator.handle_complex_query(user_query, data_pack['fingerprint'], hardened_hints=hints)
                    
                    success = False
                    result_data = {}

                    for attempt in range(3):
                        _, _, cleaned_code = validate_code_integrity(query_code, valid_cols)
                        scope = {
                            "df": data_pack['clean_df'].copy(), 
                            "pd": pd, "px": px, "np": np, "go": go, 
                            "make_subplots": make_subplots, "response": {}
                        }
                        try:
                            exec(cleaned_code, scope)
                            # FORCE result_data to be a dictionary even if scope['response'] is None
                            result_data = scope.get("response")
                            if not isinstance(result_data, dict):
                                result_data = {}
            
                            success = True
                            status.update(label="✅ Analysis Complete", state="complete")
                            break
                        except Exception as e:
                            err_msg = traceback.format_exc()
                            status.update(label=f"🔄 Healing (Attempt {attempt+1})...", state="running")
                            query_code = orchestrator.request_fix(query_code, err_msg)

                # --- RENDER THE RESULT ---
                if success:
                    txt = result_data.get("text", "")
                    tbl = result_data.get("table")
                    plt_obj = result_data.get("plot")

                    # Failsafe for "Calculation complete" / Blank answers
                    if ("Calculation complete" in txt or not txt) and tbl is not None:
                        # Use to_string() to avoid the 'tabulate' error you had
                        table_preview = tbl.head(10).to_string(index=False)
                        txt = f"### Data Summary\nI've processed the numbers for you:\n\n```text\n{table_preview}\n```"

                    with container:
                        st.markdown(txt)
                        if tbl is not None:
                            with st.container(border=True):
                                st.caption("📋 Verification Data")
                                st.dataframe(tbl, use_container_width=True, hide_index=True)
                        if plt_obj is not None:
                            st.plotly_chart(plt_obj, use_container_width=True)

                    # CRITICAL: Save to history BEFORE the rerun happens
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": txt, 
                        "plot": plt_obj, 
                        "table": tbl
                    })
                    st.session_state.lineage_log.append({"query": user_query, "logic": cleaned_code})
                    st.session_state.suggestions = orchestrator.generate_follow_ups(user_query, txt, plan.get('domain', 'Business'))
                    
                    # Final refresh to lock the chat
                    st.rerun()
                else:
                    st.error("I'm sorry, I couldn't calculate that after 3 attempts.")
    # --- 8. MODE C: LINEAGE ---
    elif mode == "📜 Lineage":
        st.header("📜 Data Lineage & Logic Audit")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🔄 Column Mapping")
            cmap = data_pack.get('column_map', {})
            mapping_df = pd.DataFrame([{"Friendly Name": v, "Technical Raw Name": k} for k, v in cmap.items()])
            st.table(mapping_df)
        with col_b:
            st.subheader("🧠 Execution History")
            if not st.session_state.lineage_log: st.info("No queries executed yet.")
            for entry in reversed(st.session_state.lineage_log):
                with st.expander(f"Query: {entry['query']}", expanded=False):
                    st.code(entry['logic'], language='python')

elif not api_key:
    st.info("🗝️ Enter Groq API Key to begin.")