import type { FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { assert } from 'tsafe';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useOutputFieldTemplate = (nodeId: string, fieldName: string): FieldOutputTemplate => {
  const template = useNodeTemplateOrThrow(nodeId);
  const fieldTemplate = useMemo(() => {
    const _fieldTemplate = template.outputs[fieldName];
    assert(_fieldTemplate, `Template for output field ${fieldName} not found`);
    return _fieldTemplate;
  }, [fieldName, template.outputs]);
  return fieldTemplate;
};
