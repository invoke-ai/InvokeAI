import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { useMemo } from 'react';

export const useOutputFieldIsConnected = (nodeId: string, fieldName: string) => {
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const firstConnectedEdge = nodes.edges.find((edge) => {
          return edge.source === nodeId && edge.sourceHandle === fieldName;
        });
        return firstConnectedEdge !== undefined;
      }),
    [fieldName, nodeId]
  );

  const isConnected = useAppSelector(selector);

  return isConnected;
};
