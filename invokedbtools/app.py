#!/usr/bin/env python3
"""InvokeDB Tools - Streamlit Database Browser"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# Load config
config_path = Path(__file__).parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

db_path = Path(config["InvokeDBPath"]) / "databases" / "invokeai.db"

# Streamlit page config
st.set_page_config(
    page_title="InvokeDB Browser",
    page_icon="🗄️",
    layout="wide"
)

# Reduce top padding and customize title
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("InvokeDB Model Browser")

# Check if database exists
if not db_path.exists():
    st.error(f"Database not found at: {db_path}")
    st.stop()

# Connect to database
@st.cache_resource
def get_connection():
    return sqlite3.connect(db_path, check_same_thread=False)

conn = get_connection()

# Load models
@st.cache_data(ttl=60)
def load_models():
    query = """
    SELECT
        id,
        name,
        type,
        base,
        format,
        path
    FROM models
    ORDER BY name COLLATE NOCASE
    """
    df = pd.read_sql_query(query, conn)
    return df

# Load and display
try:
    df = load_models()

    # Stats
    st.write(f"**Total Models:** {len(df)}")

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        height=600,
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "name": st.column_config.TextColumn("Name", width="large"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "base": st.column_config.TextColumn("Base", width="small"),
            "format": st.column_config.TextColumn("Format", width="small"),
            "path": st.column_config.TextColumn("Path", width="large"),
        }
    )

except Exception as e:
    st.error(f"Error loading models: {e}")
    import traceback
    st.code(traceback.format_exc())
