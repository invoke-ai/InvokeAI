import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { makeConnectionErrorSelector } from 'features/nodes/store/util/makeIsConnectionValidSelector';
import { InputFieldValue, OutputFieldValue } from 'features/nodes/types/types';
import { useMemo } from 'react';

const selectIsConnectionInProgress = createSelector(
  stateSelector,
  ({ nodes }) =>
    nodes.currentConnectionFieldType !== null &&
    nodes.connectionStartParams !== null
);

export type UseConnectionStateProps =
  | {
      nodeId: string;
      field: InputFieldValue;
      kind: 'input';
    }
  | {
      nodeId: string;
      field: OutputFieldValue;
      kind: 'output';
    };

export const useConnectionState = ({
  nodeId,
  field,
  kind,
}: UseConnectionStateProps) => {
  const selectIsConnected = useMemo(
    () =>
      createSelector(stateSelector, ({ nodes }) =>
        Boolean(
          nodes.edges.filter((edge) => {
            return (
              (kind === 'input' ? edge.target : edge.source) === nodeId &&
              (kind === 'input' ? edge.targetHandle : edge.sourceHandle) ===
                field.name
            );
          }).length
        )
      ),
    [field.name, kind, nodeId]
  );

  const selectConnectionError = useMemo(
    () =>
      makeConnectionErrorSelector(
        nodeId,
        field.name,
        kind === 'input' ? 'target' : 'source',
        field.type
      ),
    [nodeId, field.name, field.type, kind]
  );

  const selectIsConnectionStartField = useMemo(
    () =>
      createSelector(stateSelector, ({ nodes }) =>
        Boolean(
          nodes.connectionStartParams?.nodeId === nodeId &&
            nodes.connectionStartParams?.handleId === field.name &&
            nodes.connectionStartParams?.handleType ===
              { input: 'target', output: 'source' }[kind]
        )
      ),
    [field.name, kind, nodeId]
  );

  const isConnected = useAppSelector(selectIsConnected);
  const isConnectionInProgress = useAppSelector(selectIsConnectionInProgress);
  const isConnectionStartField = useAppSelector(selectIsConnectionStartField);
  const connectionError = useAppSelector(selectConnectionError);

  const shouldDim = useMemo(
    () =>
      Boolean(
        isConnectionInProgress && connectionError && !isConnectionStartField
      ),
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
