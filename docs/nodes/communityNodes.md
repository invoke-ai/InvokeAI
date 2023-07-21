# Community Nodes

These are nodes that have been developed by the community, for the community. If you're not sure what a node is, you can learn more about nodes [here](overview.md).

If you'd like to submit a node for the community, please refer to the [node creation overview](./overview.md#contributing-nodes).

To download a node, simply download the `.py` node file from the link and add it to the `invokeai/app/invocations/` folder in your Invoke AI install location. Along with the node, an example node graph should be provided to help you get started with the node. 

To use a community node graph, download the the `.json` node graph file and load it into Invoke AI via the **Load Nodes** button on the Node Editor. 

## Disclaimer

The nodes linked below have been developed and contributed by members of the Invoke AI community. While we strive to ensure the quality and safety of these contributions, we do not guarantee the reliability or security of the nodes. If you have issues or concerns with any of the nodes below, please raise it on GitHub or in the Discord.

## List of Nodes

### Face Mask

**Description:** This node autodetects a face in the image using MediaPipe and masks it by making it transparent. Via outpainting you can swap faces with other faces, or invert the mask and swap things around the face with other things. Additionally, you can supply X and Y offset values to scale/change the shape of the mask for finer control. The node also outputs an all-white mask in the same dimensions as the input image. This is needed by the inpaint node (and unified canvas) for outpainting.

**Node Link:** https://github.com/ymgenesis/InvokeAI/blob/facemaskmediapipe/invokeai/app/invocations/facemask.py

**Example Node Graph:** https://www.mediafire.com/file/gohn5sb1bfp8use/21-July_2023-FaceMask.json/file

**Output Examples** 

![2e3168cb-af6a-475d-bfac-c7b7fd67b4c2](https://github.com/ymgenesis/InvokeAI/assets/25252829/a5ad7d44-2ada-4b3c-a56e-a21f8244a1ac)
![2_annotated](https://github.com/ymgenesis/InvokeAI/assets/25252829/53416c8a-a23b-4d76-bb6d-3cfd776e0096)
![2fe2150c-fd08-4e26-8c36-f0610bf441bb](https://github.com/ymgenesis/InvokeAI/assets/25252829/b0f7ecfe-f093-4147-a904-b9f131b41dc9)
![831b6b98-4f0f-4360-93c8-69a9c1338cbe](https://github.com/ymgenesis/InvokeAI/assets/25252829/fc7b0622-e361-4155-8a76-082894d084f0)

--------------------------------
### Super Cool Node Template

**Description:** This node allows you to do super cool things with InvokeAI.

**Node Link:** https://github.com/invoke-ai/InvokeAI/fake_node.py

**Example Node Graph:**  https://github.com/invoke-ai/InvokeAI/fake_node_graph.json

**Output Examples** 

![Invoke AI](https://invoke-ai.github.io/InvokeAI/assets/invoke_ai_banner.png)

### Ideal Size

**Description:** This node calculates an ideal image size for a first pass of a multi-pass upscaling. The aim is to avoid duplication that results from choosing a size larger than the model is capable of.

**Node Link:** https://github.com/JPPhoto/ideal-size-node

## Help
If you run into any issues with a node, please post in the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy). 
