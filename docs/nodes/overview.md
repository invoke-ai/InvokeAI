# Nodes
## What are Nodes?
An Node is simply a single operation that takes in some inputs and gives
out some outputs. We can then chain multiple nodes together to create more
complex functionality. All InvokeAI features are added through nodes.

This means nodes can be used to easily extend the image generation capabilities of InvokeAI, and allow you build workflows to suit your needs. 

You can read more about nodes and the node editor [here](../features/NODES.md). 


## Downloading Nodes
To download a new node, visit our list of [Community Nodes](communityNodes.md). These are codes that have been created by the community, for the community. 


## Contributing Nodes

To learn about creating a new node, please visit our [Node creation documenation](../contributing/INVOCATIONS.md). 

Once you’ve created a node and confirmed that it behaves as expected locally, follow these steps: 
- Make sure the node is contained in a new Python (.py) file 
- Submit a pull request with a link to your node in GitHub against the `nodes` branch to add the node to the [Community Nodes](Community Nodes) list
    - Make sure you are following the template below and have provided all relevant details about the node and what it does.
- A maintainer will review the pull request and node. If the node is aligned with the direction of the project, you might be asked for permission to include it in the core project.

### Community Node Template

```markdown
--------------------------------
### Super Cool Node Template

**Description:** This node allows you to do super cool things with InvokeAI.

**Node Link:** https://github.com/invoke-ai/InvokeAI/fake_node.py

**Example Node Graph:**  https://github.com/invoke-ai/InvokeAI/fake_node_graph.json

**Output Examples** 

![InvokeAI](https://invoke-ai.github.io/InvokeAI/assets/invoke_ai_banner.png)
```
