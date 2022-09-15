import argparse
import os
import string
import cv2
from datetime import datetime
from tqdm import tqdm
from ldm.generate import Generate
from random import choice
import PIL
from PIL.Image import Resampling

_gen = Generate()

def get_folder_name(prompt = ""):
    now = datetime.now()
    
    prompt_parts = prompt.split()
    first_word = prompt_parts[0] if len(prompt_parts) > 0 else ""
    
    h, m, s = (str(t).ljust(2, "0") for t in [now.hour, now.minute, now.second])

    return f"{first_word}-{now.year}-{now.month}-{now.day}-{h}{m}{s}"

def get_vid_path(prompt = ""):
    return os.path.join(".", "outputs", "vid-samples", get_folder_name(prompt))

def prompt2vid(
        prompt,
        n_frames,
        init_img = None,
        fps = 30.0,
        cfg_scale = 7.5,
        strength = 0.8,
        zoom_speed = 2.0,
        seed = None
    ):

    vid_path = get_vid_path(prompt)
    frames_path = os.path.join(vid_path, "frames")
    os.makedirs(frames_path, exist_ok=True)
    
    if init_img:
        next_frame = PIL.Image.open(init_img)
    else:
        next_frame, _seed = _gen.prompt2image(prompt, steps=50, cfg_scale=cfg_scale, seed=seed)[0]
    
    w, h = next_frame.size
    video_tag = cv2.VideoWriter_fourcc(*"MPEG")
    video_writer = cv2.VideoWriter(os.path.join(vid_path, "video.mp4"), video_tag, fps, (w, h))

    # write first frame
    next_frame_filename = os.path.join(frames_path, "0.png")
    next_frame.save(next_frame_filename)
    video_writer.write(cv2.imread(next_frame_filename))
    
    for i in tqdm(range(1, n_frames), desc="Creating Video"):
        images = _gen.prompt2image(prompt, init_img=next_frame_filename, strength=strength, cfg_scale=cfg_scale, seed=seed)
        
        next_frame, _seed = choice(images)
        
        # calculate the area to crop for the generated image
        w, h = next_frame.size
        crop_w, crop_h = int(w / zoom_speed ** (1 / fps)), int(h / zoom_speed ** (1 / fps)) # magn. by (zoom_speed)x per second
        inset_x, inset_y = int((w - crop_w) / 2), int((h - crop_h) / 2)
        crop_box = (inset_x, inset_y, w - inset_x, h - inset_y)
        
        # resize to original size
        next_frame = next_frame.crop(crop_box).resize((w, h), resample=Resampling.BICUBIC)
        
        # save and write to video
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
    parser.add_argument(
        "-z",
        "--zoom_speed",
        type=float,
        default=2.0,
        help="Factor to zoom in by each second"
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=None,
        help="Seed to use. If not given, use a different seed for each frame."
    )
    return parser

if __name__ == "__main__":
    parser = create_parser()
    opt = vars(parser.parse_args())

    prompt2vid(**opt)