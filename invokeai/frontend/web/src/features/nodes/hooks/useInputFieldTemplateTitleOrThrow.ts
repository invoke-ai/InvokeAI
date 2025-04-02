import { useMemo } from 'react';
import { assert } from 'tsafe';

import { useNodeTemplateOrThrow } from './useNodeTemplateOrThrow';

export const useInputFieldTemplateTitleOrThrow = (nodeId: string, fieldName: string): string => {
  const template = useNodeTemplateOrThrow(nodeId);

  const title = useMemo(() => {
    const fieldTemplate = template.inputs[fieldName];
    assert(fieldTemplate, `Template for input field ${fieldName} not found.`);
    return fieldTemplate.title;
  }, [fieldName, template.inputs]);

  return title;
};
