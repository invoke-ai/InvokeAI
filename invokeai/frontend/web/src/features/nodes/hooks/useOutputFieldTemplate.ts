import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useOutputFieldTemplate = (nodeId: string, fieldName: string): FieldOutputTemplate => {
  const template = useNodeTemplate(nodeId);
  const fieldTemplate = useMemo(() => {
    const _fieldTemplate = template.outputs[fieldName];
    assert(_fieldTemplate, `Template for output field ${fieldName} not found`);
    return _fieldTemplate;
  }, [fieldName, template.outputs]);
  return fieldTemplate;
};
