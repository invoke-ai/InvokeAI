import { Connection, HandleType } from 'reactflow';
import { Node } from 'reactflow';
import { FieldType } from 'features/nodes/types/types';

export const findConnectionToValidHandle = (
  node: Node,
  handleCurrentNodeId: string,
  handleCurrentName: string,
  handleCurrentType: HandleType,
  handleCurrentFieldType: FieldType
): Connection | null => {
  const handles =
    handleCurrentType == 'source' ? node.data.inputs : node.data.outputs;
  for (const handleName in handles) {
    const handle = handles[handleName];

    //TODO: This is a hack, and needs to properly check if the connection is valid
    //outside of just checking the type.
    //Can I use the useIsValidConnection hook/makeConnectionIsValid selector?
    //That would require changing how I'm currently calling this, I think?
    const isValidConnection = handle.type == handleCurrentFieldType;

    if (isValidConnection) {
      const sourceID =
        handleCurrentType == 'source' ? handleCurrentNodeId : node.id;
      const targetID =
        handleCurrentType == 'source' ? node.id : handleCurrentNodeId;
      const sourceHandle =
        handleCurrentType == 'source' ? handleCurrentName : handle.name;
      const targetHandle =
        handleCurrentType == 'source' ? handle.name : handleCurrentName;

      return {
        source: sourceID,
        sourceHandle: sourceHandle,
        target: targetID,
        targetHandle: targetHandle,
      };
    }
  }
  return null;
};
