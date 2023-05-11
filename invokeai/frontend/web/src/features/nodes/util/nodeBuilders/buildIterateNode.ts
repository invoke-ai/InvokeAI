import { v4 as uuidv4 } from 'uuid';

import { IterateInvocation } from 'services/api';

export const buildIterateNode = (): IterateInvocation => {
  const nodeId = uuidv4();
  return {
    id: nodeId,
    type: 'iterate',
    // collection: [],
    // index: 0,
  };
};
