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

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid1().hex

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
bg_image = st.sidebar.file_uploader(
    "Background image:", type=["png", "jpg"], key="uploader"
)

realtime_update = st.sidebar.checkbox("Update in realtime", True)

if "iteration" not in st.session_state:
    st.session_state.iteration = 0

if st.button("Reset"):
    st.session_state.id = uuid.uuid1().hex
    st.session_state.iteration = 0
    st.session_state.canvas_image_init = {0: Image.open("static/white_background.png")}
    st.session_state.pop("uploader")
    st.experimental_rerun()

project_name = st.text_input("Project name", "Castle on a cliff")
project_name_path = project_name.replace(" ", "_")

canvas_image_init = (
    Image.open(bg_image)
    if bg_image
    else st.session_state.canvas_image_init[st.session_state.iteration]
)

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
    "Enter your prompt",
    "(painting of a castle on a cliff, lightining, seas, moonlight) by Ivan Aivazovsky and Hans Baluschek, elegant, dynamic lighting, beautiful, poster, trending on artstation, poster, anato finnstark, wallpaper, 4 k, award winning, digital art, imperial colors, fascinate view",
)
strength = st.slider("strength", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
steps = st.slider("steps", min_value=0, max_value=200, value=100, step=1)
nb_images = st.slider("nb_images", min_value=0, max_value=10, value=4, step=1)

col1, col2, col3, col4, col5 = st.columns(5)

with col2:
    generate = st.button("Generate new images")

with col3:
    back = st.button("Go back to last generation")
outdir = "outputs/streamlit"

if back:
    st.session_state.iteration -= 1
    canvas_image_init = st.session_state.canvas_image_init[st.session_state.iteration]
    st.experimental_rerun()

with col4:
    if (st.session_state.iteration + 1) in st.session_state.canvas_image_init:
        forward = st.button("Go to next generation")
        if forward:
            st.session_state.iteration += 1
            canvas_image_init = st.session_state.canvas_image_init[
                st.session_state.iteration
            ]
            st.experimental_rerun()

if generate:
    # save canvas image
    canvas_path = os.path.join(
        outdir,
        f"canvas_image_{project_name_path}_{st.session_state.iteration}_{st.session_state.id}.png",
    )
    canvas_image = Image.fromarray(canvas_result.image_data)
    canvas_image_init.paste(
        canvas_image,
        (0, 0),
        canvas_image,
    )
    canvas_image_init.save(canvas_path)

    progress_bar = st.progress(0)
    for i in range(nb_images):
        st.session_state.model.prompt2png(  # prompt2img
            prompt=prompt,
            outdir=outdir,
            iterations=1,
            steps=steps,
            filename=f"{project_name_path}_{st.session_state.iteration}_generated_{st.session_state.id}_{i}.png",
            init_img=canvas_path,
            strength=strength,
        )
        progress_bar.progress(float(i + 1) / nb_images)
    # iterate over files in
    # that directory

for filename in os.listdir(outdir):
    f = os.path.join(outdir, filename)
    # checking if it is a file
    if (
        f"{project_name_path}_{st.session_state.iteration}_generated_{st.session_state.id}"
        in f
    ):
        c1, c2 = st.columns((1, 1))
        image = Image.open(f)
        c1.image(image)
        select_image = c2.button("Select this image", key=f)
        if select_image:
            st.session_state.iteration += 1
            st.session_state.canvas_image_init[st.session_state.iteration] = image

            # clear uploaded files
            st.session_state.pop("uploader")

            st.experimental_rerun()
