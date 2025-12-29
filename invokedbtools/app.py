#!/usr/bin/env python3
"""InvokeDB Tools - Streamlit Database Browser"""

import json
import sqlite3
from pathlib import Path

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

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .model-row {
        padding: 0.5rem;
        border-bottom: 1px solid #333;
        cursor: pointer;
    }
    .model-row:hover {
        background-color: #2a2a2a;
    }
    .model-name {
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    .model-meta {
        font-size: 0.85rem;
        color: #999;
    }
    .badge {
        display: inline-block;
        padding: 0.1rem 0.4rem;
        margin-right: 0.3rem;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-format {
        background-color: #3b82f6;
        color: white;
    }
    .badge-base {
        background-color: #8b5cf6;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Model Manager")

# Check if database exists
if not db_path.exists():
    st.error(f"Database not found at: {db_path}")
    st.stop()

# Connect to database
@st.cache_resource
def get_connection():
    return sqlite3.connect(db_path, check_same_thread=False)

conn = get_connection()

# Initialize session state
if 'selected_model_id' not in st.session_state:
    st.session_state.selected_model_id = None

# Load all models
def load_models(filter_text="", filter_type="All Models"):
    query = """
    SELECT
        id,
        name,
        type,
        base,
        format,
        path,
        trigger_phrases,
        source,
        file_size
    FROM models
    WHERE 1=1
    """
    params = []

    if filter_text:
        query += " AND name LIKE ? COLLATE NOCASE"
        params.append(f"%{filter_text}%")

    if filter_type != "All Models":
        query += " AND type = ?"
        params.append(filter_type.lower())

    query += " ORDER BY name COLLATE NOCASE"

    cursor = conn.cursor()
    cursor.execute(query, params)

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]

# Get model types for filter
def get_model_types():
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT type FROM models ORDER BY type")
    types = [row[0] for row in cursor.fetchall() if row[0]]
    return ["All Models"] + types

# Delete model
def delete_model(model_id):
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        st.success("Model deleted from database")
        st.session_state.selected_model_id = None
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting model: {e}")

# Update trigger phrases
def update_trigger_phrases(model_id, trigger_phrases):
    try:
        cursor = conn.cursor()
        # Get current config
        cursor.execute("SELECT config FROM models WHERE id = ?", (model_id,))
        result = cursor.fetchone()
        if result:
            config_json = json.loads(result[0])
            config_json['trigger_phrases'] = trigger_phrases
            # Update config
            cursor.execute(
                "UPDATE models SET config = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(config_json), model_id)
            )
            conn.commit()
            st.success("Trigger phrases updated")
            st.rerun()
    except Exception as e:
        st.error(f"Error updating trigger phrases: {e}")

# Format file size
def format_size(size_bytes):
    if not size_bytes:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

# Main layout
try:
    # Get model types
    model_types = get_model_types()

    # Filter controls
    col1, col2 = st.columns([3, 1])
    with col1:
        search_text = st.text_input("Search", placeholder="Search models...", label_visibility="collapsed")
    with col2:
        filter_type = st.selectbox("Type", model_types, label_visibility="collapsed")

    # Load models with filters
    models = load_models(search_text, filter_type)

    st.write(f"**{len(models)} Selected**")

    # Two-column layout
    left_col, right_col = st.columns([2, 3])

    with left_col:
        # Model list
        st.markdown("### LoRAs" if filter_type == "lora" or filter_type == "All Models" else f"### {filter_type.upper()}")

        for model in models:
            with st.container():
                col_name, col_delete = st.columns([10, 1])

                with col_name:
                    # Clickable model row
                    if st.button(
                        model['name'],
                        key=f"select_{model['id']}",
                        use_container_width=True,
                        type="secondary" if st.session_state.selected_model_id != model['id'] else "primary"
                    ):
                        st.session_state.selected_model_id = model['id']
                        st.rerun()

                    # Badges and file size
                    badge_html = ""
                    if model.get('format'):
                        badge_html += f'<span class="badge badge-format">{model["format"]}</span>'
                    if model.get('base'):
                        badge_html += f'<span class="badge badge-base">{model["base"]}</span>'

                    file_size = format_size(model.get('file_size'))
                    meta_html = f'{badge_html}<span class="model-meta">{file_size}</span>'

                    st.markdown(meta_html, unsafe_allow_html=True)

                with col_delete:
                    if st.button("🗑️", key=f"delete_{model['id']}", help="Delete model"):
                        delete_model(model['id'])

    with right_col:
        # Model details
        if st.session_state.selected_model_id:
            # Find selected model
            selected = next((m for m in models if m['id'] == st.session_state.selected_model_id), None)

            if selected:
                st.markdown(f"## {selected['name']}")

                # Source
                if selected.get('source'):
                    st.caption(f"Source: {selected['source']}")

                st.markdown("---")

                # Model details
                st.markdown("**Base Model**")
                st.write(selected.get('base', 'Unknown'))

                st.markdown("**Model Type**")
                st.write(selected.get('type', 'Unknown'))

                st.markdown("**Model Format**")
                st.write(selected.get('format', 'Unknown'))

                st.markdown("**Path**")
                st.code(selected.get('path', 'Unknown'), language=None)

                st.markdown("**File Size**")
                st.write(format_size(selected.get('file_size')))

                st.markdown("---")

                # Trigger phrases
                st.markdown("**Trigger Phrases**")
                current_triggers = selected.get('trigger_phrases', '')
                new_triggers = st.text_input(
                    "Trigger Phrases",
                    value=current_triggers or '',
                    placeholder="Type phrase here",
                    label_visibility="collapsed",
                    key=f"triggers_{selected['id']}"
                )

                if new_triggers != (current_triggers or ''):
                    if st.button("Save Trigger Phrases"):
                        update_trigger_phrases(selected['id'], new_triggers)

                st.markdown("---")

                # Related Models section (placeholder)
                st.markdown("**Related Models**")
                st.caption("No related models")
            else:
                st.info("Model not found or was deleted")
        else:
            st.info("Select a model to view details")

except Exception as e:
    st.error(f"Error loading models: {e}")
    import traceback
    st.code(traceback.format_exc())
