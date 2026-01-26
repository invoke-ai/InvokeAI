# Show Image

Displays a provided image using your operating system's image viewer and forwards the image through the pipeline unchanged. Use this node when you want to quickly inspect an image while building or debugging a workflow.

## Inputs

- `image`: The image to show. This is typically an image produced earlier in the pipeline or a loaded image resource.

## Outputs

- `image`: The same image, passed through so downstream nodes can continue processing it.

---

## Notes

- The node launches the system image viewer; behavior depends on the host OS.  
- This uses python `PIL.Image.show()`, which opens a viewer on the host system, not within the InvokeAI UI. If you are running InvokeAI on a remote server, the image will open on the server's display instead of your local machine.