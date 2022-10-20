---
title: Linux
---

# :fontawesome-brands-linux: Linux

## Installation

1. You will need to install the following prerequisites if they are not already
   available. Use your operating system's preferred installer.

    - Python (version 3.8.5 recommended; higher may work)
    - git

2. Install the Python Anaconda environment manager.

    ```bash
    ~$  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    ~$  chmod +x Anaconda3-2022.05-Linux-x86_64.sh
    ~$  ./Anaconda3-2022.05-Linux-x86_64.sh
    ```

    After installing anaconda, you should log out of your system and log back in. If
    the installation worked, your command prompt will be prefixed by the name of the
    current anaconda environment - `(base)`.

3. Copy the InvokeAI source code from GitHub:

    ```bash
    (base) ~$ git clone https://github.com/invoke-ai/InvokeAI.git
    ```

    This will create InvokeAI folder where you will follow the rest of the steps.

4. Enter the newly-created InvokeAI folder. From this step forward make sure that you are working in the InvokeAI directory!

    ```bash
    (base) ~$ cd InvokeAI
    (base) ~/InvokeAI$
    ```

5. Use anaconda to copy necessary python packages, create a new python
   environment named `invokeai` and activate the environment.

    ```bash
    (base) ~/InvokeAI$ conda env create
    (base) ~/InvokeAI$ conda activate invokeai
    (invokeai) ~/InvokeAI$
    ```

    After these steps, your command prompt will be prefixed by `(invokeai)` as shown
    above.

6. Now you need to install the model weights for the Stable Diffusion and the other models.

      - You will first need to set up an account
        with [Hugging Face](https://huggingface.co).
      - [login](https://huggingface.co/docs/huggingface_hub/quick-start#login) with your account: 
        ```bash
        (invokeai) ~/InvokeAI$ huggingface-cli login
        ```
      - Visit the pages of the Stable Diffusion models to accept their terms:
          + [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) 
          + [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
              + [with inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
      - Download the models:
        ```bash
        (invokeai) ~/InvokeAI$ python3 scripts/preload_models.py
        ```

7. Start generating images!

    ```bash
    # for the pre-release weights use the -l or --liaon400m switch
    (invokeai) ~/InvokeAI$ python3 scripts/invoke.py -l

    # for the post-release weights do not use the switch
    (invokeai) ~/InvokeAI$ python3 scripts/invoke.py

    # for additional configuration switches and arguments, use -h or --help
    (invokeai) ~/InvokeAI$ python3 scripts/invoke.py -h
    ```

8. Subsequently, to relaunch the script, be sure to run "conda activate invokeai" (step 5, second command), enter the `InvokeAI` directory, and then launch the invoke script (step 7). If you forget to activate the 'invokeai' environment, the script will fail with multiple `ModuleNotFound` errors.

## Updating to newer versions of the script

This distribution is changing rapidly. If you used the `git clone` method (step 5) to download the InvokeAI directory, then to update to the latest and greatest version, launch the Anaconda window, enter `InvokeAI` and type:

```bash
(invokeai) ~/InvokeAI$ git pull
(invokeai) ~/InvokeAI$ conda env update -f environment.yml
```

This will bring your local copy into sync with the remote one.
