import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $edgePendingUpdate, $pendingConnection, $templates } from 'features/nodes/store/nodesSlice';
import { makeConnectionErrorSelector } from 'features/nodes/store/util/makeConnectionErrorSelector';
import { useMemo } from 'react';

export const useInputFieldConnectionState = (nodeId: string, fieldName: string) => {
  const pendingConnection = useStore($pendingConnection);
  const templates = useStore($templates);
  const edgePendingUpdate = useStore($edgePendingUpdate);

  const selectValidationResult = useMemo(
    () => makeConnectionErrorSelector(templates, nodeId, fieldName, 'target'),
    [templates, nodeId, fieldName]
  );

  const isConnectionInProgress = useMemo(() => Boolean(pendingConnection), [pendingConnection]);
  const isConnectionStartField = useMemo(() => {
    if (!pendingConnection) {
      return false;
    }
    return (
      pendingConnection.nodeId === nodeId &&
      pendingConnection.handleId === fieldName &&
      pendingConnection.fieldTemplate.fieldKind === 'input'
    );
  }, [fieldName, nodeId, pendingConnection]);
  const validationResult = useAppSelector((s) => selectValidationResult(s, pendingConnection, edgePendingUpdate));

  return {
    isConnectionInProgress,
    isConnectionStartField,
    validationResult,
  };
};
