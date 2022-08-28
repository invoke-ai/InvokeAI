import argparse
import string
import os
import cv2
from tqdm import tqdm
from ldm.simplet2i import T2I
from random import choice
import PIL
from PIL.Image import Resampling

_t2i = T2I()

def get_vid_path():
    return os.path.join(".", "outputs", "vid-samples")

def prompt2vid(**config):
    prompt = config["prompt"]
    n_frames = config["n_frames"]
    initial_image = config["init_img"]
    fps = config["fps"] if "fps" in config else 30.0
    cfg_scale = config["cfg_scale"] if "cfg_scale" in config else 7.5
    strength = config["strength"] if "strength" in config else 0.8

    vid_path = get_vid_path()
    frames_path = os.path.join(vid_path, "frames")
    os.makedirs(frames_path, exist_ok=True)
    
    if initial_image:
        next_frame = PIL.Image.open(initial_image)
    else:
        next_frame, _seed = _t2i.prompt2image(prompt, steps=50, cfg_scale=cfg_scale)[0]
    
    w, h = next_frame.size
    video_writer = cv2.VideoWriter(os.path.join(vid_path, "video.mp4"), 1, fps, (w, h))

    next_frame_filename = os.path.join(frames_path, "0.png")
    next_frame.save(next_frame_filename)
    video_writer.write(cv2.imread(next_frame_filename))
    
    for i in tqdm(range(n_frames), desc="Creating Video"):
        images = _t2i.prompt2image(prompt, init_img=next_frame_filename, strength=strength, cfg_scale=cfg_scale)
        
        next_frame, _seed = choice(images)
        w, h = next_frame.size
        crop_box = (int(w * 0.01), int(h * 0.01), int(w * 0.99), int(h * 0.99))
        next_frame = next_frame.crop(crop_box).resize((w, h), resample=Resampling.BICUBIC)
        
        next_frame_filename = os.path.join(frames_path, f"{i}.png")
        next_frame.save(next_frame_filename)
        video_writer.write(cv2.imread(next_frame_filename))
    
    cv2.destroyAllWindows()
    video_writer.release()

def create_parser():
    parser = argparse.ArgumentParser(description="Parse Arguments for dream_video")
    parser.add_argument("prompt")
    parser.add_argument(
        "-F",
        "--n_frames",
        type=int,
        help="Number of Video Frames to Generate"
    )
    parser.add_argument(
        "-I",
        "--init_img",
        type=str,
        default=None
    )
    parser.add_argument(
        "-f",
        "--strength",
        type=float,
        default=0.8,
        help="The strength applied to img2img per frame"
    )
    parser.add_argument(
        "-P",
        "--fps",
        type=float,
        default=30.0,
        help="Frames per Second of the result video"
    )
    parser.add_argument(
        "-C",
        "--cfg_scale",
        type=float,
        default=7.5,
        help="Prompt configuration scale"
    )
    return parser

if __name__ == "__main__":
    parser = create_parser()
    opt = parser.parse_args()

    print("Initializing Model, please wait...")
    _t2i.load_model()

    print(opt)
    print(vars(opt))
    prompt2vid(**vars(opt))