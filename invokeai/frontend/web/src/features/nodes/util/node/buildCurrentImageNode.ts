import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { CurrentImageNode } from 'features/nodes/types/invocation';
import type { XYPosition } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

export const buildCurrentImageNode = (position: XYPosition): CurrentImageNode => {
  const nodeId = uuidv4();
  const node: CurrentImageNode = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'current_image',
    position,
    data: {
      id: nodeId,
      type: 'current_image',
      isOpen: true,
      label: 'Current Image',
    },
  };
  return node;
};
