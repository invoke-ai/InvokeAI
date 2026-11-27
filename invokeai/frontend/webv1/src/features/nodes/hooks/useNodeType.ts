import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';

export const useNodeType = (): string => {
  const ctx = useInvocationNodeContext();
  return useAppSelector(ctx.selectNodeTypeOrThrow);
};
