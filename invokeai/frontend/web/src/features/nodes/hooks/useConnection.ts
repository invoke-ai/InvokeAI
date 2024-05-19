import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/storeHooks';
import { $mouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import {
  $edgePendingUpdate,
  $isAddNodePopoverOpen,
  $pendingConnection,
  $templates,
  connectionMade,
} from 'features/nodes/store/nodesSlice';
import { getFirstValidConnection } from 'features/nodes/store/util/getFirstValidConnection';
import { isString } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import type { OnConnect, OnConnectEnd, OnConnectStart } from 'reactflow';
import { useUpdateNodeInternals } from 'reactflow';
import { assert } from 'tsafe';

export const useConnection = () => {
  const store = useAppStore();
  const templates = useStore($templates);
  const updateNodeInternals = useUpdateNodeInternals();

  const onConnectStart = useCallback<OnConnectStart>(
    (event, { nodeId, handleId, handleType }) => {
      assert(nodeId && handleId && handleType, 'Invalid connection start event');
      const nodes = store.getState().nodes.present.nodes;

      const node = nodes.find((n) => n.id === nodeId);
      if (!node) {
        return;
      }

      const template = templates[node.data.type];
      if (!template) {
        return;
      }

      const fieldTemplates = template[handleType === 'source' ? 'outputs' : 'inputs'];
      const fieldTemplate = fieldTemplates[handleId];
      if (!fieldTemplate) {
        return;
      }

      $pendingConnection.set({ nodeId, handleId, handleType, fieldTemplate });
    },
    [store, templates]
  );
  const onConnect = useCallback<OnConnect>(
    (connection) => {
      const { dispatch } = store;
      dispatch(connectionMade(connection));
      const nodesToUpdate = [connection.source, connection.target].filter(isString);
      updateNodeInternals(nodesToUpdate);
      $pendingConnection.set(null);
    },
    [store, updateNodeInternals]
  );
  const onConnectEnd = useCallback<OnConnectEnd>(() => {
    const { dispatch } = store;
    const pendingConnection = $pendingConnection.get();
    const edgePendingUpdate = $edgePendingUpdate.get();
    const mouseOverNodeId = $mouseOverNode.get();

    // If we are in the middle of an edge update, and the mouse isn't over a node, we should just bail so the edge
    // update logic can finish up
    if (edgePendingUpdate && !mouseOverNodeId) {
      $pendingConnection.set(null);
      return;
    }

    if (!pendingConnection) {
      return;
    }
    const { nodes, edges } = store.getState().nodes.present;
    if (mouseOverNodeId) {
      const { handleType } = pendingConnection;
      const source = handleType === 'source' ? pendingConnection.nodeId : mouseOverNodeId;
      const sourceHandle = handleType === 'source' ? pendingConnection.handleId : null;
      const target = handleType === 'target' ? pendingConnection.nodeId : mouseOverNodeId;
      const targetHandle = handleType === 'target' ? pendingConnection.handleId : null;

      const connection = getFirstValidConnection(
        source,
        sourceHandle,
        target,
        targetHandle,
        nodes,
        edges,
        templates,
        edgePendingUpdate
      );
      if (connection) {
        dispatch(connectionMade(connection));
        const nodesToUpdate = [connection.source, connection.target].filter(isString);
        updateNodeInternals(nodesToUpdate);
      }
      $pendingConnection.set(null);
    } else {
      // The mouse is not over a node - we should open the add node popover
      $isAddNodePopoverOpen.set(true);
    }
  }, [store, templates, updateNodeInternals]);

  const api = useMemo(() => ({ onConnectStart, onConnect, onConnectEnd }), [onConnectStart, onConnect, onConnectEnd]);
  return api;
};
