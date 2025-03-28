import { isBatchNodeType, isGeneratorNodeType } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useIsExecutableNode = (nodeId: string) => {
  const template = useNodeTemplateOrThrow(nodeId);
  const isExecutableNode = useMemo(
    () => !isBatchNodeType(template.type) && !isGeneratorNodeType(template.type),
    [template]
  );
  return isExecutableNode;
};
