import { useStore } from '@nanostores/react';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { $nodeErrors } from 'features/nodes/store/util/fieldValidators';
import { computed } from 'nanostores';
import { useMemo } from 'react';

/**
 * A hook that returns a boolean representing whether the field is invalid. A field is invalid if it has any errors.
 * The errors calculation is debounced.
 *
 * @param fieldName The name of the field
 *
 * @returns A boolean representing whether the field is invalid
 */
export const useInputFieldIsInvalid = (fieldName: string) => {
  const ctx = useInvocationNodeContext();
  const $isInvalid = useMemo(
    () =>
      computed($nodeErrors, (nodeErrors) => {
        const thisNodeErrors = nodeErrors[ctx.nodeId];
        if (!thisNodeErrors) {
          return false;
        }
        const isFieldInvalid = thisNodeErrors.some((error) => {
          error.type === 'field-error' && error.fieldName === fieldName;
        });
        return isFieldInvalid;
      }),
    [ctx, fieldName]
  );
  return useStore($isInvalid);
};
