# InvokeAI - A Stable Diffusion Toolkit

Stable Diffusion distribution by InvokeAI: https://github.com/invoke-ai

The Docker image tracks the `main` branch of the InvokeAI project, which means it includes the latest features, but may contain some bugs.

Your working directory is mounted under the `/workspace` path inside the pod. The models are in `/workspace/invokeai/models`, and outputs are in `/workspace/invokeai/outputs`.

> **Only the /workspace directory will persist between pod restarts!**

> **If you _terminate_ (not just _stop_) the pod, the /workspace will be lost.**

## Quickstart

1. Launch a pod from this template. **It will take about 5-10 minutes to run through the initial setup**. Be patient.
1. Wait for the application to load.
    - TIP: you know it's ready when the CPU usage goes idle
    - You can also check the logs for a line that says "_Point your browser at..._"
1. Open the Invoke AI web UI: click the `Connect` => `connect over  HTTP` button.
1. Generate some art!

## Other things you can do

At any point you may edit the pod configuration and set an arbitrary Docker command. For example, you could run a command to downloads some models using `curl`, or fetch some images and place them into your outputs to continue a working session.

If you need to run *multiple commands*, define them in the Docker Command field like this:

`bash -c "cd ${INVOKEAI_ROOT}/outputs; wormhole receive 2-foo-bar; invoke.py --web --host 0.0.0.0"`

### Copying your data in and out of the pod

This image includes a couple of handy tools to help you get the data into the pod (such as your custom models or embeddings), and out of the pod (such as downloading your outputs). Here are your options for getting your data in and out of the pod:

- **SSH server**:
    1. Make sure to create and set your Public Key in the RunPod settings (follow the official instructions)
    1. Add an exposed port 22 (TCP) in the pod settings!
    1. When your pod restarts, you will see a new entry in the `Connect` dialog. Use this SSH server to `scp` or `sftp` your files as necessary, or SSH into the pod using the fully fledged SSH server.

- [**Magic Wormhole**](https://magic-wormhole.readthedocs.io/en/latest/welcome.html):
    1. On your computer, `pip install magic-wormhole` (see above instructions for details)
    1. Connect to the command line **using the "light" SSH client** or the browser-based console. _Currently there's a bug where `wormhole` isn't available when connected to "full" SSH server, as described above_.
    1. `wormhole send /workspace/invokeai/outputs` will send the entire `outputs` directory. You can also send individual files.
    1. Once packaged, you will see a `wormhole receive <123-some-words>` command. Copy it
    1. Paste this command into the terminal on your local machine to securely download the payload.
    1. It works the same in reverse: you can `wormhole send` some models from your computer to the pod. Again, save your files somewhere in `/workspace` or they will be lost when the pod is stopped.

- **RunPod's Cloud Sync feature** may be used to sync the persistent volume to cloud storage. You could, for example, copy the entire `/workspace` to S3, add some custom models to it, and copy it back from S3 when launching new pod configurations. Follow the Cloud Sync instructions.


### Disable the NSFW checker

The NSFW checker is enabled by default. To disable it, edit the pod configuration and set the following command:

```
invoke --web --host 0.0.0.0 --no-nsfw_checker
```

---

Template ©2023 Eugene Brodsky [ebr](https://github.com/ebr)