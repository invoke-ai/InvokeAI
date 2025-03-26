import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useMemo } from 'react';

export const useInputFieldTemplateTitleSafe = (nodeId: string, fieldName: string): string => {
  const template = useNodeTemplate(nodeId);
  const title = useMemo(() => template.inputs[fieldName]?.title ?? '', [fieldName, template.inputs]);
  return title;
};
