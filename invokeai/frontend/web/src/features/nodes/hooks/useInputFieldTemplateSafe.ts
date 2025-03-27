import { useNodeTemplateSafe } from 'features/nodes/hooks/useNodeTemplateSafe';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

/**
 * Returns the template for a specific input field of a node.
 *
 * **Note:** This function is a safe version of `useInputFieldTemplate` and will not throw an error if the template is not found.
 *
 * @param nodeId - The ID of the node.
 * @param fieldName - The name of the input field.
 */
export const useInputFieldTemplateSafe = (nodeId: string, fieldName: string): FieldInputTemplate | null => {
  const template = useNodeTemplateSafe(nodeId);
  const fieldTemplate = useMemo(() => template?.inputs[fieldName] ?? null, [fieldName, template?.inputs]);
  return fieldTemplate;
};
