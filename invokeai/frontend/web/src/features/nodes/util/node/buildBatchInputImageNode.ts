import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { BatchImageInputNode } from 'features/nodes/types/invocation';
import type { XYPosition } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

export const buildBatchInputImageNode = (position: XYPosition): BatchImageInputNode => {
  const nodeId = uuidv4();
  const node: BatchImageInputNode = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'image_batch',
    position,
    data: {
      id: nodeId,
      isOpen: true,
      label: 'Image Batch',
      type: 'image_batch',
      images: [],
    },
  };
  return node;
};
