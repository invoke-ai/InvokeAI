import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $edgePendingUpdate, $pendingConnection, $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
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

  const selectValidationResult = useMemo(
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
      pendingConnection.nodeId === nodeId &&
      pendingConnection.handleId === fieldName &&
      pendingConnection.fieldTemplate.fieldKind === { inputs: 'input', outputs: 'output' }[kind]
    );
  }, [fieldName, kind, nodeId, pendingConnection]);
  const validationResult = useAppSelector((s) => selectValidationResult(s, pendingConnection, edgePendingUpdate));

  const shouldDim = useMemo(
    () => Boolean(isConnectionInProgress && !validationResult.isValid && !isConnectionStartField),
    [validationResult, isConnectionInProgress, isConnectionStartField]
  );

  return {
    isConnected,
    isConnectionInProgress,
    isConnectionStartField,
    validationResult,
    shouldDim,
  };
};
