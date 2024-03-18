import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplate } from 'features/nodes/store/selectors';
import { some } from 'lodash-es';
import { useMemo } from 'react';

export const useHasImageOutput = (nodeId: string): boolean => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const template = selectNodeTemplate(nodes, nodeId);
        return some(
          template?.outputs,
          (output) =>
            output.type.name === 'ImageField' &&
            // the image primitive node (node type "image") does not actually save the image, do not show the image-saving checkboxes
            template?.type !== 'image'
        );
      }),
    [nodeId]
  );

  const hasImageOutput = useAppSelector(selector);
  return hasImageOutput;
};
