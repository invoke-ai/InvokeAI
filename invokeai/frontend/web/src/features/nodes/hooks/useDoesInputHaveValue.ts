import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeData } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useDoesInputHaveValue = (nodeId: string, fieldName: string): boolean => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const data = selectNodeData(nodes, nodeId);
        if (!data) {
          return false;
        }
        return data.inputs[fieldName]?.value !== undefined;
      }),
    [fieldName, nodeId]
  );

  const doesFieldHaveValue = useAppSelector(selector);

  return doesFieldHaveValue;
};
