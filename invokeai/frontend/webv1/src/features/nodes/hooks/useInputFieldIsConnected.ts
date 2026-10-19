import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { useMemo } from 'react';

export const useInputFieldIsConnected = (fieldName: string) => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(() => ctx.buildSelectIsInputFieldConnected(fieldName), [fieldName, ctx]);

  return useAppSelector(selector);
};
