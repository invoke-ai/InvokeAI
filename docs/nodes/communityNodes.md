# Community Nodes

These are nodes that have been developed by the community, for the community. If you're not sure what a node is, you can learn more about nodes [here](overview.md).

If you'd like to submit a node for the community, please refer to the [node creation overview](./overview.md#contributing-nodes).

To download a node, simply download the `.py` node file from the link and add it to the `invokeai/app/invocations/` folder in your Invoke AI install location. Along with the node, an example node graph should be provided to help you get started with the node. 

To use a community node graph, download the the `.json` node graph file and load it into Invoke AI via the **Load Nodes** button on the Node Editor. 

## Disclaimer

The nodes linked below have been developed and contributed by members of the Invoke AI community. While we strive to ensure the quality and safety of these contributions, we do not guarantee the reliability or security of the nodes. If you have issues or concerns with any of the nodes below, please raise it on GitHub or in the Discord.

## List of Nodes

### FaceTools

**Description:** FaceTools is a collection of nodes created to manipulate faces as you would in Unified Canvas. It includes FaceMask, FaceOff, and FacePlace. FaceMask autodetects a face in the image using MediaPipe and creates a mask from it. FaceOff similarly detects a face, then takes the face off of the image by adding a square bounding box around it and cropping/scaling it. FacePlace puts the bounded face image from FaceOff back onto the original image. Using these nodes with other inpainting node(s), you can put new faces on existing things, put new things around existing faces, and work closer with a face as a bounded image. Additionally, you can supply X and Y offset values to scale/change the shape of the mask for finer control on FaceMask and FaceOff. See GitHub repository below for usage examples.

**Node Link:** https://github.com/ymgenesis/FaceTools/

**FaceMask Output Examples** 

![5cc8abce-53b0-487a-b891-3bf94dcc8960](https://github.com/invoke-ai/InvokeAI/assets/25252829/43f36d24-1429-4ab1-bd06-a4bedfe0955e)
![b920b710-1882-49a0-8d02-82dff2cca907](https://github.com/invoke-ai/InvokeAI/assets/25252829/7660c1ed-bf7d-4d0a-947f-1fc1679557ba)
![71a91805-fda5-481c-b380-264665703133](https://github.com/invoke-ai/InvokeAI/assets/25252829/f8f6a2ee-2b68-4482-87da-b90221d5c3e2)

<hr>

### Ideal Size

**Description:** This node calculates an ideal image size for a first pass of a multi-pass upscaling. The aim is to avoid duplication that results from choosing a size larger than the model is capable of.

**Node Link:** https://github.com/JPPhoto/ideal-size-node

--------------------------------
### Super Cool Node Template

**Description:** This node allows you to do super cool things with InvokeAI.

**Node Link:** https://github.com/invoke-ai/InvokeAI/fake_node.py

**Example Node Graph:**  https://github.com/invoke-ai/InvokeAI/fake_node_graph.json

**Output Examples** 

![Invoke AI](https://invoke-ai.github.io/InvokeAI/assets/invoke_ai_banner.png)

## Help
If you run into any issues with a node, please post in the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy). 
