---
title: Manual Installation, Windows
---

# :fontawesome-brands-windows: Windows

## **Notebook install (semi-automated)**

We have a
[Jupyter notebook](https://github.com/invoke-ai/InvokeAI/blob/main/notebooks/Stable-Diffusion-local-Windows.ipynb)
with cell-by-cell installation steps. It will download the code in this repo as
one of the steps, so instead of cloning this repo, simply download the notebook
from the link above and load it up in VSCode (with the appropriate extensions
installed)/Jupyter/JupyterLab and start running the cells one-by-one.

Note that you will need NVIDIA drivers, Python 3.10, and Git installed
beforehand - simplified
[step-by-step instructions](https://github.com/invoke-ai/InvokeAI/wiki/Easy-peasy-Windows-install)
are available in the wiki (you'll only need steps 1, 2, & 3 ).

## **Manual Install**

### **pip**

See
[Easy-peasy Windows install](https://github.com/invoke-ai/InvokeAI/wiki/Easy-peasy-Windows-install)
in the wiki

---

### **Conda**

1. Install Anaconda3 (miniconda3 version) from [here](https://docs.anaconda.com/anaconda/install/windows/)

2. Install Git from [here](https://git-scm.com/download/win)

3. Launch Anaconda from the Windows Start menu. This will bring up a command
   window. Type all the remaining commands in this window.

4. Run the command:

    ```batch
    git clone https://github.com/invoke-ai/InvokeAI.git
    ```

    This will create stable-diffusion folder where you will follow the rest of
    the steps.

5. Enter the newly-created InvokeAI folder. From this step forward make sure that you are working in the InvokeAI directory!

    ```batch
    cd InvokeAI
    ```

6. Run the following two commands:

    ```batch title="step 6a"
    conda env create
    ```

    ```batch title="step 6b"
    conda activate invokeai
    ```

    This will install all python requirements and activate the "invokeai" environment
    which sets PATH and other environment variables properly.

    Note that the long form of the first command is `conda env create -f environment.yml`. If the
    environment file isn't specified, conda will default to `environment.yml`. You will need
    to provide the `-f` option if you wish to load a different environment file at any point.

7. Run the command:

    ```batch
    python scripts\preload_models.py
    ```

    This installs several machine learning models that stable diffusion requires.

    Note: This step is required. This was done because some users may might be
    blocked by firewalls or have limited internet connectivity for the models to
    be downloaded just-in-time.

8. Now you need to install the weights for the big stable diffusion model.

   - Sign up at https://huggingface.co
   - Go to the [Stable diffusion diffusion model page](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
   - Accept the terms and click Access Repository
   - Download [v1-5-pruned-emaonly.ckpt (4.27 GB)](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt)
     and move it into this directory under `models/ldm/stable_diffusion_v1/v1-5-pruned-emaonly.ckpt`

   There are many other models that you can use. Please see [../features/INSTALLING_MODELS.md]
   for details.

9. Start generating images!

    ```batch title="for the pre-release weights"
    python scripts\invoke.py -l
    ```

    ```batch title="for the post-release weights"
    python scripts\invoke.py
    ```

10. Subsequently, to relaunch the script, first activate the Anaconda command window (step 3),enter the InvokeAI directory (step 5, `cd \path\to\InvokeAI`), run `conda activate invokeai` (step 6b), and then launch the invoke script (step 9).

!!! tip "Tildebyte has written an alternative"

    ["Easy peasy Windows install"](https://github.com/invoke-ai/InvokeAI/wiki/Easy-peasy-Windows-install)
    which uses the Windows Powershell and pew. If you are having trouble with
    Anaconda on Windows, give this a try (or try it first!)

---

This distribution is changing rapidly. If you used the `git clone` method
(step 5) to download the stable-diffusion directory, then to update to the
latest and greatest version, launch the Anaconda window, enter
`stable-diffusion`, and type:

```bash
git pull
conda env update
```

This will bring your local copy into sync with the remote one.
