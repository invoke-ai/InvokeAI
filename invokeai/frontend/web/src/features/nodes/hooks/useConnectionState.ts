import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $pendingConnection, $templates, selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { makeConnectionErrorSelector } from 'features/nodes/store/util/connectionValidation.js';
import { useMemo } from 'react';

import { useFieldType } from './useFieldType.ts';

type UseConnectionStateProps = {
  nodeId: string;
  fieldName: string;
  kind: 'inputs' | 'outputs';
};

export const useConnectionState = ({ nodeId, fieldName, kind }: UseConnectionStateProps) => {
  const pendingConnection = useStore($pendingConnection);
  const templates = useStore($templates);
  const fieldType = useFieldType(nodeId, fieldName, kind);

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
    () =>
      makeConnectionErrorSelector(
        templates,
        pendingConnection,
        nodeId,
        fieldName,
        kind === 'inputs' ? 'target' : 'source',
        fieldType
      ),
    [templates, pendingConnection, nodeId, fieldName, kind, fieldType]
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
