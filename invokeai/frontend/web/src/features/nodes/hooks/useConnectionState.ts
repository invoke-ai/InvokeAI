import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { makeConnectionErrorSelector } from 'features/nodes/store/util/makeIsConnectionValidSelector';
import { useMemo } from 'react';

import { useFieldType } from './useFieldType.ts';

const selectIsConnectionInProgress = createSelector(
  selectNodesSlice,
  (nodes) => nodes.connectionStartFieldType !== null && nodes.connectionStartParams !== null
);

export type UseConnectionStateProps = {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
};

export const useConnectionState = ({ nodeId, fieldName, kind }: UseConnectionStateProps) => {
  const fieldType = useFieldType(nodeId, fieldName, kind);

  const selectIsConnected = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) =>
        Boolean(
          nodes.edges.filter((edge) => {
            return (
              (kind === 'input' ? edge.target : edge.source) === nodeId &&
              (kind === 'input' ? edge.targetHandle : edge.sourceHandle) === fieldName
            );
          }).length
        )
      ),
    [fieldName, kind, nodeId]
  );

  const selectConnectionError = useMemo(
    () => makeConnectionErrorSelector(nodeId, fieldName, kind === 'input' ? 'target' : 'source', fieldType),
    [nodeId, fieldName, kind, fieldType]
  );

  const selectIsConnectionStartField = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) =>
        Boolean(
          nodes.connectionStartParams?.nodeId === nodeId &&
            nodes.connectionStartParams?.handleId === fieldName &&
            nodes.connectionStartParams?.handleType === { input: 'target', output: 'source' }[kind]
        )
      ),
    [fieldName, kind, nodeId]
  );

  const isConnected = useAppSelector(selectIsConnected);
  const isConnectionInProgress = useAppSelector(selectIsConnectionInProgress);
  const isConnectionStartField = useAppSelector(selectIsConnectionStartField);
  const connectionError = useAppSelector(selectConnectionError);

  const shouldDim = useMemo(
    () => Boolean(isConnectionInProgress && connectionError && !isConnectionStartField),
    [connectionError, isConnectionInProgress, isConnectionStartField]
  );

  return {
    isConnected,
    isConnectionInProgress,
    isConnectionStartField,
    connectionError,
    shouldDim,
  };
};
