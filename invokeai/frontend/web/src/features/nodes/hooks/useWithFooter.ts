import { useIsExecutableNode } from 'features/nodes/hooks/useIsBatchNode';

import { useNodeHasGalleryOutput } from './useNodeHasGalleryOutput';

export const useWithFooter = () => {
  const hasGalleryOutput = useNodeHasGalleryOutput();
  const isExecutableNode = useIsExecutableNode();
  return isExecutableNode && hasGalleryOutput;
};
