import { useAppSelector } from 'app/store/storeHooks';
import { isInvocationNode } from 'features/nodes/types/invocation';

export const useDoesFieldExist = (nodeId: string, fieldName?: string) => {
  const doesFieldExist = useAppSelector((s) => {
    const node = s.nodes.present.nodes.find((n) => n.id === nodeId);
    if (!isInvocationNode(node)) {
      return false;
    }
    if (fieldName === undefined) {
      return true;
    }
    if (!node.data.inputs[fieldName]) {
      return false;
    }
    return true;
  });

  return doesFieldExist;
};
