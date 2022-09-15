## Creating Videos from Prompts

dream_video.py creates a video by running img2img for each movie frame, cropping and rescaling the image (zooming in) and running img2img on the rescaled image again.

dream_video.py can be called on the CLI like this:

`python dream_video.py [-h] [-F N_FRAMES] [-I INIT_IMG] [-f STRENGTH] [-P FPS] [-C CFG_SCALE] [-z ZOOM_SPEED] [-S SEED] prompt`

- Zoom Speed is the Zoom Factor applied per second. This is normalized to FPS.
- If an initial image is given, it will use img2img for the first frame, otherwise txt2img
- If no seed is given, every frame will use a different seed. This will give a lot of motion in the result. If a seed is given, it will be the same for all frames.
    - Creating a video with random seeds and low FPS could yield interesting results when morphed together

It can also be used as a module:

```py
import dream_video

dream_video.prompt2vid(prompt="Prompt", n_frames=120)
```