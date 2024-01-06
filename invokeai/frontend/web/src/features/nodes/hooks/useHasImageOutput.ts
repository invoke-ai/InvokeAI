import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { some } from 'lodash-es';
import { useMemo } from 'react';

export const useHasImageOutput = (nodeId: string) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const node = nodes.nodes.find((node) => node.id === nodeId);
        if (!isInvocationNode(node)) {
          return false;
        }
        return some(
          node.data.outputs,
          (output) =>
            output.type.name === 'ImageField' &&
            // the image primitive node (node type "image") does not actually save the image, do not show the image-saving checkboxes
            node.data.type !== 'image'
        );
      }),
    [nodeId]
  );

  const hasImageOutput = useAppSelector(selector);
  return hasImageOutput;
};
