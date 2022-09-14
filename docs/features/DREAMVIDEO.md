## Creating Videos from Prompts

dream_video.py creates a video by running img2img for each movie frame, cropping and rescaling the image (zooming in) and running img2img on the rescaled image again.

dream_video.py can be called on the CLI like this:

`python scripts/dream_video.py [-h] [-F N_FRAMES] [-I INIT_IMG] [-f STRENGTH] [-P FPS] [-C CFG_SCALE] [-z ZOOM_SPEED] prompt`

It can also be used as a module:

```py
import dream_video

dream_video.prompt2vid(prompt="Prompt", n_frames=120)
```