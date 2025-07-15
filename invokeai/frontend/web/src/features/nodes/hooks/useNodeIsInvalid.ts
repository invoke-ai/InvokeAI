import { useStore } from '@nanostores/react';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { $nodeErrors } from 'features/nodes/store/util/fieldValidators';

export const useNodeIsInvalid = () => {
  const ctx = useInvocationNodeContext();
  const hasErrors = useStore($nodeErrors, { keys: [ctx.nodeId] })[ctx.nodeId];
  return hasErrors;
};
