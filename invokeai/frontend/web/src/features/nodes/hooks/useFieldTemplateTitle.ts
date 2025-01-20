import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { useMemo } from 'react';

export const useFieldTemplateTitle = (nodeId: string, fieldName: string, kind: 'inputs' | 'outputs'): string => {
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, kind);
  const fieldTemplateTitle = useMemo(() => fieldTemplate.title, [fieldTemplate]);
  return fieldTemplateTitle;
};
