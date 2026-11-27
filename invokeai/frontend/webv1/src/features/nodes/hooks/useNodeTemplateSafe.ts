import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { InvocationTemplate } from 'features/nodes/types/invocation';

export const useNodeTemplateSafe = (): InvocationTemplate | null => {
  const ctx = useInvocationNodeContext();
  return useAppSelector(ctx.selectNodeTemplateSafe);
};
