There is a [Docker](#docker) version as well, which allows for an easier setup.
## Docker
Once you have installed docker, and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for GPU support, you can simply run one of these versions:
### 1. Docker native
```
docker run -it --platform linux/amd64 --gpus all --entrypoint bash -v "${PWD}/weights/:/app/weights/" -v "${PWD}/outputs/:/app/outputs/" guestros/stable-diffusion-lstein:latest
```
### 2. Linux/Mac helper script:
`./runDocker.sh`
### 3. Windows Powershell
`.\runDocker-Windows.ps1`
or
Right Click on "runDocker-Windows.ps1" -> "Run with PowerShell"
### Usage
Either one will open an interactive shell command, in which you can run your commands as usual. The "outputs" and "weights" folders will be mounted locally in the current working directory, meaning that the created images will appear in the "outputs" folder. If you want to use specific/new weights you can copy them into the "weights" folder and they will appear in the docker container in the "/app/weights/" path.
### Building the docker image yourself
Just run `docker compose build` or `docker build -t [IMAGE_NAME] .`