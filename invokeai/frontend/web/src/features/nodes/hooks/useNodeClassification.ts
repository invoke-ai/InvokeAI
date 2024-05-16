import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { Classification } from 'features/nodes/types/common';
import { useMemo } from 'react';

export const useNodeClassification = (nodeId: string): Classification => {
  const template = useNodeTemplate(nodeId);
  const classification = useMemo(() => template.classification, [template]);
  return classification;
};
