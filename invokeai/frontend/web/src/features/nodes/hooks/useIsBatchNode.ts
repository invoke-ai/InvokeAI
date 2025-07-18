import { isBatchNodeType, isGeneratorNodeType } from 'features/nodes/types/invocation';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useIsExecutableNode = () => {
  const template = useNodeTemplateOrThrow();
  const isExecutableNode = useMemo(
    () => !isBatchNodeType(template.type) && !isGeneratorNodeType(template.type),
    [template]
  );
  return isExecutableNode;
};
