import { useStore } from '@nanostores/react';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { FieldError } from 'features/nodes/store/util/fieldValidators';
import { $nodeErrors } from 'features/nodes/store/util/fieldValidators';
import { computed } from 'nanostores';
import { useMemo } from 'react';

/**
 * A hook that returns the errors for a given input field. The errors calculation is debounced.
 *
 * @param fieldName The name of the field
 * @returns An array of FieldError objects
 */
export const useInputFieldErrors = (fieldName: string): FieldError[] => {
  const ctx = useInvocationNodeContext();
  const $errors = useMemo(
    () =>
      computed($nodeErrors, (nodeErrors) => {
        const thisNodeErrors = nodeErrors[ctx.nodeId];
        if (!thisNodeErrors) {
          return EMPTY_ARRAY;
        }
        const errors = thisNodeErrors.filter((error) => {
          error.type === 'field-error' && error.fieldName === fieldName;
        });
        if (errors.length === 0) {
          return EMPTY_ARRAY;
        }
        return errors as FieldError[];
      }),
    [ctx, fieldName]
  );

  return useStore($errors);
};
