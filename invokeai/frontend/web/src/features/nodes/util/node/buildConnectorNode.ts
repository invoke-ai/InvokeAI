import type { XYPosition } from '@xyflow/react';
import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import type { ConnectorNode } from 'features/nodes/types/invocation';
import { v4 as uuidv4 } from 'uuid';

export const buildConnectorNode = (position: XYPosition): ConnectorNode => {
  const nodeId = uuidv4();
  const node: ConnectorNode = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'connector',
    position,
    data: {
      id: nodeId,
      type: 'connector',
      isOpen: true,
      label: 'Connector',
    },
  };
  return node;
};
