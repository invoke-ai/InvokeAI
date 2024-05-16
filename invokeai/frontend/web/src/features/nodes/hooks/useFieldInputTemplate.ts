import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldInputTemplate = (nodeId: string, fieldName: string): FieldInputTemplate | null => {
  const template = useNodeTemplate(nodeId);
  const fieldTemplate = useMemo(() => template.inputs[fieldName] ?? null, [fieldName, template.inputs]);
  return fieldTemplate;
};
