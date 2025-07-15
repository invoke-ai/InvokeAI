import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { InvocationNodeData } from 'features/nodes/types/invocation';

export const useNodeData = (): InvocationNodeData => {
  const ctx = useInvocationNodeContext();
  return useAppSelector(ctx.selectNodeDataOrThrow);
};
