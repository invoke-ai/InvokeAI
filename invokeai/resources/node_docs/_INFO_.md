This folder contains developer-authored node documentation to be displayed in the Workflow Editor.

## Naming:

- One Markdown file per invocation, named exactly after its invocation_type with a .md suffix (e.g., "img_crop.md" for the "img_crop" invocation).
- Files live in this folder (and in language subfolders such as en/).

## Authoring:

- Description: Explain the intended use case(s) for the node, and any important details about its behavior. The intention here is to explain to the user **why** and **how** they would use this node. The description should not be a repeat of the node's technical specification, but rather a user-focused explanation of its purpose and functionality, written in clear, non-technical language.

- Inputs: List and describe each input port, including expected data types and any important details about how the input affects the node's behavior. Use code formatting for input names (e.g., `Input Name`) when listing and referring to them.

- Outputs: (if applicable) List and describe each output port, including data types and any important details about the output data. If the node has a single output that is already explained in the description, this section can be omitted.

- Examples: Provide one or more example usages of the node, including images where applicable. Each example should include a brief description of the scenario being demonstrated.

## Images:

- Place image files in the `images/` subfolder next to the markdown file. Reference them using relative paths in the markdown.
- Ensure image names are unique and descriptive. Images can be reused across multiple docs if applicable, e.g., SD1.5 denoise and prompt nodes can all be shown in a single image since their usage is tied together.
- Images can be screenshots of the node in use, example outputs, or diagrams illustrating concepts.
- When displaying node usage examples, keep the example focused on the node and its immediate upstream/downstream connections. For best readability, keep the image width approximately two nodes wide.

[Use IMAGE_PLACEHOLDER for any images at this time. We will replace these with actual images later.]

## Submitting:

- Because these docs are included as an installed module and served through API, new files will only be included after a `uv pip install`. This ensures parity between dev and user installs.
- Check that your markdown renders correctly in a markdown viewer in the UI.

## Doc Template:

# [NODE NAME]

[Node description goes here.]

## Inputs

- [Input 1 Name]: [Description of input 1]
- [Input 2 Name]: [Description of input 2]
- ...

## Outputs

- [Output 1 Name]: [Description of output 1]
- [Output 2 Name]: [Description of output 2]
- ...

## Example Usage

### [Example Scenario 1]

![Descriptive Alt Text for Example 1](./images/[image_file_name_1].png)  
[Brief description of Example Scenario 1.]

### [Example Scenario 2]

![Descriptive Alt Text for Example 2](./images/[image_file_name_2].png)  
[Brief description of Example Scenario 2.]

## Notes:

- [Any additional notes or important details about the node's behavior.]
