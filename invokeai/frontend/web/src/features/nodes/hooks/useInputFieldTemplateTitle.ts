import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useInputFieldTemplateTitle = (nodeId: string, fieldName: string): string => {
  const template = useNodeTemplate(nodeId);

  const title = useMemo(() => {
    const fieldTemplate = template.inputs[fieldName];
    assert(fieldTemplate, `Template for input field ${fieldName} not found.`);
    return fieldTemplate.title;
  }, [fieldName, template.inputs]);

  return title;
};
