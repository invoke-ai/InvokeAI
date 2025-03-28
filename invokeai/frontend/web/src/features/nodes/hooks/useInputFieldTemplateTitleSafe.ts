import { useMemo } from 'react';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useInputFieldTemplateTitleSafe = (nodeId: string, fieldName: string): string => {
  const template = useNodeTemplateOrThrow(nodeId);
  const title = useMemo(() => template.inputs[fieldName]?.title ?? '', [fieldName, template.inputs]);
  return title;
};
