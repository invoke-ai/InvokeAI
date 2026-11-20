# InvokeAI - A Stable Diffusion Toolkit

Stable Diffusion distribution by InvokeAI: https://github.com/invoke-ai

The Docker image tracks the `main` branch of the InvokeAI project, which means it includes the latest features, but may contain some bugs.

Your working directory is mounted under the `/workspace` path inside the pod. The models are in `/workspace/invokeai/models`, and outputs are in `/workspace/invokeai/outputs`.

> **Only the /workspace directory will persist between pod restarts!**

> **If you _terminate_ (not just _stop_) the pod, the /workspace will be lost.**

## Quickstart

1. Before launching, choose a pod with a public IP for SSH. In the template's port settings, clear the exposed HTTP ports and expose only TCP port 22. Anyone who can reach InvokeAI on a fresh installation can claim the first administrator account.
2. Set the pod's `PUBLIC_KEY` environment variable to your SSH **public** key so that the image starts its SSH server. Set `INVOKEAI_ROOT=/workspace/invokeai` so the administrator account persists on the volume. Launch the pod and wait for InvokeAI to start.
3. In the pod's **Connect** tab, copy the **SSH over exposed TCP** command. Insert `-N -L 127.0.0.1:9090:127.0.0.1:9090` immediately after `ssh` and run it on your computer. Leave that terminal open. See [RunPod's full SSH setup](https://docs.runpod.io/pods/configuration/use-ssh).
4. Open `http://localhost:9090` on your computer, create the administrator account and sign in.
5. Keep using the tunnel for private access. If you want to share InvokeAI through the pod's HTTP proxy, enable HTTP port 9090 only after setup, with multi-user mode enabled and the same `/workspace/invokeai` volume attached.

If the pod already has a nonempty `/root/.ssh/authorized_keys`, each public key in `PUBLIC_KEY` must match a complete line in that file, including its comment. A mismatch stops startup and leaves the file intact. To rotate a key, add the new key from a trusted session, remove the retired key, then set `PUBLIC_KEY` to the new line and restart. To recover from a mismatch, you can also set `PUBLIC_KEY` back to an authorized line for which you have the private key.

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
