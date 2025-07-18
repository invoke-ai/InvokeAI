import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { FieldInputInstance } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useInputFieldInstance = <T extends FieldInputInstance>(fieldName: string): T => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(() => {
    return ctx.buildSelectInputFieldOrThrow(fieldName);
  }, [ctx, fieldName]);
  return useAppSelector(selector) as T;
};
