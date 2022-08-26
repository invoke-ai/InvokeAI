import logging
import pandas as pd
from PIL import Image
import os
import streamlit as st
import transformers
import uuid
from streamlit_drawable_canvas import st_canvas
from ldm.simplet2i import T2I

st.set_page_config(layout="wide")

if "model" not in st.session_state:
    print("Loading model...")
    model = T2I(sampler_name="k_lms")

    # to get rid of annoying warning messages from pytorch
    transformers.logging.set_verbosity_error()
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    print("Initializing model, be patient...")
    model.load_model()
    st.session_state.model = model
    print("Loading done")

if "canvas_image_init" not in st.session_state:
    st.session_state.canvas_image_init = {0: Image.open("static/white_background.png")}

if "generated_images" not in st.session_state:
    st.session_state.generated_images = {}

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 3)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 100, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

if "iteration" not in st.session_state:
    st.session_state.iteration = 0

if st.button("Reset iterations"):
    st.session_state.iteration = 0
    st.session_state.canvas_image_init = {0: Image.open("static/white_background.png")}
    st.experimental_rerun()
    
project_name = st.text_input("Project name", "Fields with trees")
project_name_path = project_name.replace(" ", "_")

canvas_image_init = st.session_state.canvas_image_init[st.session_state.iteration]

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=canvas_image_init,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    initial_drawing=None,
    point_display_radius=point_display_radius if drawing_mode == "point" else 0,
    key="canvas",
)

prompt = st.text_input(
    "Enter your prompt", "Painting of a field with trees in the style of William Blake"
)
generate = st.button("Generate new images")
back = st.button("Go back to last generation")
outdir = "outputs/streamlit"

if back:
    st.session_state.iteration -= 1
    canvas_image_init = st.session_state.canvas_image_init[st.session_state.iteration]
    st.experimental_rerun()

if (st.session_state.iteration + 1) in st.session_state.canvas_image_init:
    forward = st.button("Go to next generation")
    if forward:
        st.session_state.iteration += 1
        canvas_image_init = st.session_state.canvas_image_init[st.session_state.iteration]
        st.experimental_rerun()

if generate:
    # save canvas image
    canvas_path = os.path.join(
        outdir, f"canvas_image_{project_name_path}_{st.session_state.iteration}.png"
    )
    canvas_image = Image.fromarray(canvas_result.image_data)
    st.session_state.canvas_image_init[st.session_state.iteration].paste(
        canvas_image,
        (0, 0),
        canvas_image,
    )
    st.session_state.canvas_image_init[st.session_state.iteration].save(canvas_path)

    for i in range(2):
        st.session_state.model.prompt2png( #prompt2img
            prompt=prompt,
            outdir=outdir,
            iterations=1,
            steps=100,
            filename=f"{project_name_path}_{st.session_state.iteration}_generated_{i}.png",
            init_img=canvas_path,
            strength=0.6,
        )
    # iterate over files in
    # that directory

for filename in os.listdir(outdir):
    f = os.path.join(outdir, filename)
    # checking if it is a file
    if f"{project_name_path}_{st.session_state.iteration}_generated_" in f:
        c1, c2 = st.columns((1, 1))
        image = Image.open(f)
        c1.image(image)
        select_image = c2.button("Select this image", key=f)
        if select_image:
            st.session_state.iteration += 1
            st.session_state.canvas_image_init[st.session_state.iteration] = image
            st.experimental_rerun()

# # Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
