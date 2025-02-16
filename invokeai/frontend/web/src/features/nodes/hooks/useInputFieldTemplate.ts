import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useInputFieldTemplate = (nodeId: string, fieldName: string): FieldInputTemplate => {
  const template = useNodeTemplate(nodeId);
  const fieldTemplate = useMemo(() => {
    const _fieldTemplate = template.inputs[fieldName];
    assert(_fieldTemplate, `Template for input field ${fieldName} not found.`);
    return _fieldTemplate;
  }, [fieldName, template.inputs]);
  return fieldTemplate;
};
