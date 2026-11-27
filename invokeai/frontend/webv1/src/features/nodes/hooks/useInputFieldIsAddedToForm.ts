import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useInputFieldIsAddedToForm = (fieldName: string) => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(() => ctx.buildSelectIsInputFieldAddedToForm(fieldName), [ctx, fieldName]);
  return useAppSelector(selector);
};
