# FAQs

**Where do I get started? How can I install Invoke?**

- You can download the latest installers [here](https://github.com/invoke-ai/InvokeAI/releases) - Note that any releases marked as *pre-release* are in a beta state. You may experience some issues, but we appreciate your help testing those! For stable/reliable installations, please install the **[Latest Release](https://github.com/invoke-ai/InvokeAI/releases/latest)**

**How can I download models? Can I use models I already have downloaded?**

- Models can be downloaded through the model manager, or through option [4] in the invoke.bat/invoke.sh launcher script. To download a model through the Model Manager, use the HuggingFace Repo ID by pressing the “Copy” button next to the repository name. Alternatively, to download a model from CivitAi, use the download link in the Model Manager.
- Models that are already downloaded can be used by creating a symlink to the model location in the `autoimport` folder or by using the Model Manger’s “Scan for Models” function.

**My images are taking a long time to generate. How can I speed up generation?** 

- A common solution is to reduce the size of your RAM & VRAM cache to 0.25. This ensures your system has enough memory to generate images.
- Additionally, check the [hardware requirements](https://invoke-ai.github.io/InvokeAI/#hardware-requirements) to ensure that your system is capable of generating images.
- Lastly, double check your generations are happening on your GPU (if you have one). InvokeAI will log what is being used for generation upon startup. 

**I’ve installed Python on Windows but the installer says it can’t find it?**

- Then ensure that you checked  **'Add python.exe to PATH'** when installing Python. This can be found at the bottom of the Python Installer window. If you already have Python installed, this can be done with the modify / repair feature of the installer.

**I’ve installed everything successfully but I still get an error about Triton when starting Invoke?**

- This can be safely ignored. InvokeAI doesn't use Triton, but if you are on Linux and wish to dismiss the error, you can install Triton.

**I updated to 3.4.0 and now xFormers can’t load C++/CUDA?**

- An issue occurred with your PyTorch update. Follow these steps to fix :
    1. Launch your invoke.bat / invoke.sh and select the option to open the developer console
    2. Run:`pip install ".[xformers]" --upgrade --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121`
    - If you run into an error with `typing_extensions`, re-open the developer console and run:  `pip install -U typing-extensions`

**It says my pip is out of date - is that why my install isn't working?**
- An out of date won't cause an installation to fail. The cause of the error can likely be found above the message that says pip is out of date.
- If you saw that warning but the install went well, don't worry about it (but you can update pip afterwards if you'd like).   

**How can I generate the exact same that I found on the internet?**
Most example images with prompts that you'll find on the internet have been generated using different software, so you can't expect to get identical results. In order to reproduce an image, you need to replicate the exact settings and processing steps, including (but not limited to) the model, the positive and negative prompts, the seed, the sampler, the exact image size, any upscaling steps, etc.


**Where can I get more help?** 

- Create an issue on [GitHub](https://github.com/invoke-ai/InvokeAI/issues) or post in the [#help channel](https://discord.com/channels/1020123559063990373/1149510134058471514) of the InvokeAI Discord