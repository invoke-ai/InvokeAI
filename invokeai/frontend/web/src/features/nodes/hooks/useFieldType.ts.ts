import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import type { FieldType } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useFieldType = (nodeId: string, fieldName: string, kind: 'inputs' | 'outputs'): FieldType => {
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, kind);
  const fieldType = useMemo(() => fieldTemplate.type, [fieldTemplate]);
  return fieldType;
};
