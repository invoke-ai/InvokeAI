import { useStore } from '@nanostores/react';
import type { HandleType } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import {
  $edgePendingUpdate,
  $isConnectionInProgress,
  $pendingConnection,
  $templates,
} from 'features/nodes/store/nodesSlice';
import { makeConnectionErrorSelector } from 'features/nodes/store/util/makeConnectionErrorSelector';
import { useMemo } from 'react';

export const useConnectionValidationResult = (nodeId: string, fieldName: string, handleType: HandleType) => {
  const pendingConnection = useStore($pendingConnection);
  const templates = useStore($templates);
  const edgePendingUpdate = useStore($edgePendingUpdate);

  const selectValidationResult = useMemo(
    () => makeConnectionErrorSelector(templates, nodeId, fieldName, handleType, pendingConnection, edgePendingUpdate),
    [templates, nodeId, fieldName, handleType, pendingConnection, edgePendingUpdate]
  );

  const validationResult = useAppSelector(selectValidationResult);
  return validationResult;
};

export const useIsConnectionStartField = (nodeId: string, fieldName: string, handleType: HandleType) => {
  const pendingConnection = useStore($pendingConnection);

  const isConnectionStartField = useMemo(() => {
    if (!pendingConnection) {
      return false;
    }
    if (pendingConnection.nodeId !== nodeId || pendingConnection.handleId !== fieldName) {
      return false;
    }
    if (handleType === 'source' && pendingConnection.fieldTemplate.fieldKind === 'output') {
      return true;
    }
    if (handleType === 'target' && pendingConnection.fieldTemplate.fieldKind === 'input') {
      return true;
    }
    return false;
  }, [fieldName, handleType, nodeId, pendingConnection]);

  return isConnectionStartField;
};

export const useIsConnectionInProgress = () => {
  const isConnectionInProgress = useStore($isConnectionInProgress);

  return isConnectionInProgress;
};
