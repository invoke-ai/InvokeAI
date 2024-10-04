import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldOutputTemplate = (nodeId: string, fieldName: string): FieldOutputTemplate | null => {
  const template = useNodeTemplate(nodeId);
  const fieldTemplate = useMemo(() => template.outputs[fieldName] ?? null, [fieldName, template.outputs]);
  return fieldTemplate;
};
