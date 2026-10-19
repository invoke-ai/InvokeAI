import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

/**
 * Returns the template for a specific input field of a node.
 *
 * **Note:** This hook will throw an error if the template for the input field is not found.
 *
 * @param fieldName - The name of the input field.
 * @throws Will throw an error if the template for the input field is not found.
 */
export const useInputFieldTemplateOrThrow = (fieldName: string): FieldInputTemplate => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(() => ctx.buildSelectInputFieldTemplateOrThrow(fieldName), [ctx, fieldName]);
  return useAppSelector(selector);
};
