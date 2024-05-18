import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $edgePendingUpdate, $pendingConnection, $templates, selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { makeConnectionErrorSelector } from 'features/nodes/store/util/makeConnectionErrorSelector';
import { useMemo } from 'react';

type UseConnectionStateProps = {
  nodeId: string;
  fieldName: string;
  kind: 'inputs' | 'outputs';
};

export const useConnectionState = ({ nodeId, fieldName, kind }: UseConnectionStateProps) => {
  const pendingConnection = useStore($pendingConnection);
  const templates = useStore($templates);
  const edgePendingUpdate = useStore($edgePendingUpdate);

  const selectIsConnected = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) =>
        Boolean(
          nodes.edges.filter((edge) => {
            return (
              (kind === 'inputs' ? edge.target : edge.source) === nodeId &&
              (kind === 'inputs' ? edge.targetHandle : edge.sourceHandle) === fieldName
            );
          }).length
        )
      ),
    [fieldName, kind, nodeId]
  );

  const selectConnectionError = useMemo(
    () => makeConnectionErrorSelector(templates, nodeId, fieldName, kind === 'inputs' ? 'target' : 'source'),
    [templates, nodeId, fieldName, kind]
  );

  const isConnected = useAppSelector(selectIsConnected);
  const isConnectionInProgress = useMemo(() => Boolean(pendingConnection), [pendingConnection]);
  const isConnectionStartField = useMemo(() => {
    if (!pendingConnection) {
      return false;
    }
    return (
      pendingConnection.node.id === nodeId &&
      pendingConnection.fieldTemplate.name === fieldName &&
      pendingConnection.fieldTemplate.fieldKind === { inputs: 'input', outputs: 'output' }[kind]
    );
  }, [fieldName, kind, nodeId, pendingConnection]);
  const connectionError = useAppSelector((s) => selectConnectionError(s, pendingConnection, edgePendingUpdate));

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
