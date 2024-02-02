# Progress

We have 3 different places to display progress images:

- TextToImage & ImageToImage
- Canvas
- Workflow

The progress slice tracks the latest denoising progress events, latest image output, and active batch ids for each of the workspaces.

Each of these have different requirements for displaying progress images, but much of the logic around tracking progress is the same, so it is consolidated here.

It also holds the latest progress event separately, which is used for the progress bar.
