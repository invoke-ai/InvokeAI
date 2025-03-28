import type { Classification } from 'features/nodes/types/common';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useNodeClassification = (nodeId: string): Classification => {
  const template = useNodeTemplateOrThrow(nodeId);
  const classification = useMemo(() => template.classification, [template]);
  return classification;
};
