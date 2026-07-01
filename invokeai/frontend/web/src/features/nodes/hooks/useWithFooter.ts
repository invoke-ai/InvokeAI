import { useIsExecutableNode } from 'features/nodes/hooks/useIsBatchNode';

import { useNodeHasImageOutput } from './useNodeHasImageOutput';

export const useWithFooter = () => {
  const hasImageOutput = useNodeHasImageOutput();
  const isExecutableNode = useIsExecutableNode();
  return isExecutableNode && hasImageOutput;
};
