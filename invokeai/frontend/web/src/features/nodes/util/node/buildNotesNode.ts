import { SHARED_NODE_PROPERTIES } from 'features/nodes/types/constants';
import { NotesNode } from 'features/nodes/types/invocation';
import { XYPosition } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

export const buildNotesNode = (position: XYPosition): NotesNode => {
  const nodeId = uuidv4();
  const node: NotesNode = {
    ...SHARED_NODE_PROPERTIES,
    id: nodeId,
    type: 'notes',
    position,
    data: {
      id: nodeId,
      isOpen: true,
      label: 'Notes',
      notes: '',
      type: 'notes',
    },
  };
  return node;
};
