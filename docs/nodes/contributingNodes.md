# Contributing Nodes

To learn about the specifics of creating a new node, please visit our [Node creation documentation](../contributing/INVOCATIONS.md). 

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
