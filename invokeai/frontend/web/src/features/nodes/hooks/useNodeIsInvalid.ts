import { useStore } from '@nanostores/react';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { $nodeErrors } from 'features/nodes/store/util/fieldValidators';

export const useNodeHasErrors = () => {
  const ctx = useInvocationNodeContext();
  const errors = useStore($nodeErrors, { keys: [ctx.nodeId] })[ctx.nodeId];
  return errors ? errors.length > 0 : false;
};
