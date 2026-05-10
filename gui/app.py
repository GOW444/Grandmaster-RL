import sys
from pathlib import Path

# 1. FIX: Calculate the project root and handle paths correctly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt

# Internal imports
from env.chess_env import ChessPuzzleEnv
from env.eval_env import EvalChessPuzzleEnv
from agents.baselines import RandomAgent, RatingMatchAgent, FixedProgressionAgent
from evaluation.evaluate import evaluate_agent, evaluate_all
from evaluation.visualize import plot_all

# --- Path Configuration ---
CHECKPOINT_DIR = ROOT / "checkpoints"
PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# --- Logic: Better Model Loading & Status ---
@st.cache_resource
def load_models():
    models = {}
    status = {}
    
    # Check for PPO
    ppo_path = CHECKPOINT_DIR / "ppo" / "ppo_final"
    if ppo_path.with_suffix(".zip").exists() or ppo_path.exists():
        models["PPO"] = PPO.load(ppo_path)
        status["PPO"] = "✅ Loaded"
    else:
        status["PPO"] = "❌ Missing Checkpoint"
    
    # Check for SAC
    sac_path = CHECKPOINT_DIR / "sac" / "sac_final"
    if sac_path.with_suffix(".zip").exists() or sac_path.exists():
        models["SAC"] = SAC.load(sac_path)
        status["SAC"] = "✅ Loaded"
    else:
        status["SAC"] = "❌ Missing Checkpoint"
        
    models["Random"] = RandomAgent()
    models["RatingMatch"] = RatingMatchAgent()
    models["FixedProgression"] = FixedProgressionAgent()
    
    return models, status

# --- Page Setup ---
st.set_page_config(page_title="Grandmaster-RL Dashboard", layout="wide", page_icon="♟️")
st.title("♟️ Grandmaster-RL: Adaptive Chess Curriculum")

# --- Feature 1: Model Status Dashboard ---
models, model_status = load_models()
with st.expander("📡 Model Integrity & Status"):
    cols = st.columns(len(model_status))
    for i, (name, msg) in enumerate(model_status.items()):
        cols[i].metric(name, msg)

# --- Sidebar: Advanced Settings ---
with st.sidebar:
    st.header("🛠️ Evaluation Settings")
    n_episodes = st.slider("Number of Episodes", 5, 100, 20)
    seed = st.number_input("Random Seed", value=42)
    
    st.subheader("🧪 Environment Stress Tests")
    # Feature 2: Dynamic Fatigue/Jitter control
    fatigue_level = st.select_slider("Learner Fatigue", options=["None", "Low", "High"], value="Low")
    jitter = st.slider("IRT Temperature Jitter (Noise)", 0.0, 1.0, 0.2)
    
    run_eval = st.button("🚀 Run Multi-Agent Evaluation", use_container_width=True)

# --- Execution ---
if run_eval:
    with st.spinner("Simulating learner trajectories and generating plots..."):
        train_env = ChessPuzzleEnv(seed=seed)
        eval_env = EvalChessPuzzleEnv(seed=seed)
        
        trajectories = {}
        all_metrics = []
        
        # Import THEMES to ensure we map theme gains correctly
        from env.learner_model import THEMES

        for name, model in models.items():
            metrics = evaluate_agent(model, train_env, eval_env, n_episodes=n_episodes)
            
            # 1. Capture Trajectories for line plots
            trajectories[name] = {
                "success": metrics["success_rate_trajectories_train"],
                "difficulty": metrics["difficulty_trajectories_train"]
            }

            # 2. Build a complete row for the DataFrame
            row = {
                "agent": name,
                "lei_train": metrics["lei_train"],
                "lei_eval": metrics["lei_eval"],
                "robustness": metrics["robustness"],
                "mean_delta_rho_train": metrics["mean_delta_rho_train"],  # FIXED: Added this
                "mean_delta_rho_eval": metrics["mean_delta_rho_eval"],
                "mean_success_rate_train": metrics["mean_success_rate_train"],
                "mean_success_rate_eval": metrics["mean_success_rate_eval"],
                "difficulty_variance_train": metrics["difficulty_variance_train"],
                "difficulty_variance_eval": metrics["difficulty_variance_eval"],
            }
            
            # 3. Add per-theme gains (Required for the Heatmap plot)
            for i, theme in enumerate(THEMES):
                row[f"theme_{theme}_gain_train"] = float(metrics["per_theme_skill_gain_train"][i])
                row[f"theme_{theme}_gain_eval"] = float(metrics["per_theme_skill_gain_eval"][i])
                
            all_metrics.append(row)
        
        results_df = pd.DataFrame(all_metrics)
        
        # 4. Generate all plots
        plot_all(results_df, trajectories, output_dir=PLOT_DIR)
        
        st.session_state["results"] = results_df
    st.success("Evaluation & Visualization Complete!")
# --- Results Display ---
if "results" in st.session_state:
    df = st.session_state["results"]
    
    tab1, tab2 = st.tabs(["📊 Performance Metrics", "📈 Publication Plots"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Metric Comparison")
            st.dataframe(df.style.background_gradient(cmap='Blues', subset=['lei_eval', 'robustness']), use_container_width=True)
        with c2:
            st.subheader("Robustness Factor")
            st.bar_chart(df.set_index("agent")["robustness"])

    with tab2:
        plots = {
            "LEI Comparison": "lei_comparison.png",
            "Robustness Map": "robustness.png",
            "Skill Gain": "skill_improvement.png",
            "Theme Heatmap": "theme_heatmap.png",
            "Success Trajectory": "success_trajectory.png",
            "Difficulty Curve": "difficulty_progression.png"
        }
        
        p_tabs = st.tabs(list(plots.keys()))
        for p_tab, (title, filename) in zip(p_tabs, plots.items()):
            with p_tab:
                img_path = PLOT_DIR / filename
                if img_path.exists():
                    st.image(str(img_path), caption=title, use_container_width=True)
                else:
                    st.error(f"Plot file {filename} could not be generated. Check logs.")
