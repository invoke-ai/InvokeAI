# Contributing Nodes

To learn about the specifics of creating a new node, please visit our [Node creation documentation](../contributing/INVOCATIONS.md). 

Once you’ve created a node and confirmed that it behaves as expected locally, follow these steps: 

- Make sure the node is contained in a new Python (.py) file. Preferably, the node is in a repo with a README detailing the nodes usage & examples to help others more easily use your node. Including the tag "invokeai-node" in your repository's README can also help other users find it more easily. 
- Submit a pull request with a link to your node(s) repo in GitHub against the `main` branch to add the node to the [Community Nodes](communityNodes.md) list
    - Make sure you are following the template below and have provided all relevant details about the node and what it does. Example output images and workflows are very helpful for other users looking to use your node.
- A maintainer will review the pull request and node. If the node is aligned with the direction of the project, you may be asked for permission to include it in the core project.

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
