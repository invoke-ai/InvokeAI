import type { Classification } from 'features/nodes/types/common';
import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useNodeClassification = (): Classification => {
  const template = useNodeTemplateOrThrow();
  const classification = useMemo(() => template.classification, [template]);
  return classification;
};
