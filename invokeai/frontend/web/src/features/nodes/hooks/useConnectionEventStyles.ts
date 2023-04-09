import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { CSSProperties, useMemo } from 'react';
import { HandleType, useReactFlow } from 'reactflow';
import { FieldType, Invocation } from '../types';

const invalidTargetStyles: CSSProperties = {
  opacity: 0.3,
};

const validTargetStyles: CSSProperties = {};

export const useConnectionEventStyles = (
  nodeId: string,
  fieldType: FieldType,
  handleType: HandleType
) => {
  const flow = useReactFlow();
  const pendingConnection = useAppSelector(
    (state: RootState) => state.nodes.pendingConnection
  );

  return useMemo(() => {
    if (!pendingConnection) {
      return;
    }

    const {
      handleId,
      handleType: sourceHandleType,
      nodeId: sourceNodeId,
    } = pendingConnection;

    // default to connectable if these are not present - unsure why they ever would not be present...
    if (!handleId || !sourceNodeId || !handleType) {
      return validTargetStyles;
    }

    if (
      // cannot connect a node's input to its own output
      nodeId === sourceNodeId
    ) {
      return invalidTargetStyles;
    }

    if (
      // cannot connect inputs to inputs or outputs to outputs
      handleType === sourceHandleType
    ) {
      return invalidTargetStyles;
    }

    const node = flow.getNode(sourceNodeId)?.data as Invocation;

    // handle field types must be the same
    if (
      fieldType !==
      (sourceHandleType === 'target'
        ? node.inputs[handleId].type
        : node.outputs[handleId].type)
    ) {
      return invalidTargetStyles;
    }

    return validTargetStyles;
  }, [pendingConnection, nodeId, flow, fieldType, handleType]);
};
