import { useStore } from '@nanostores/react';
import type { EdgeChange, OnConnect, OnConnectEnd, OnConnectStart } from '@xyflow/react';
import { useUpdateNodeInternals } from '@xyflow/react';
import { useAppStore } from 'app/store/storeHooks';
import { $mouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import {
  $addNodeCmdk,
  $didUpdateEdge,
  $edgePendingUpdate,
  $pendingConnection,
  $templates,
  edgesChanged,
} from 'features/nodes/store/nodesSlice';
import { selectNodes, selectNodesSlice } from 'features/nodes/store/selectors';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  resolveConnectorSourceFieldType,
} from 'features/nodes/store/util/connectorTopology';
import { getFirstValidConnection } from 'features/nodes/store/util/getFirstValidConnection';
import { connectionToEdge } from 'features/nodes/store/util/reactFlowUtil';
import type { AnyEdge } from 'features/nodes/types/invocation';
import { isConnectorNode } from 'features/nodes/types/invocation';
import { useCallback, useMemo } from 'react';
import { assert } from 'tsafe';

export const useConnection = () => {
  const store = useAppStore();
  const templates = useStore($templates);
  const updateNodeInternals = useUpdateNodeInternals();

  const onConnectStart = useCallback<OnConnectStart>(
    (event, { nodeId, handleId, handleType }) => {
      assert(nodeId && handleId && handleType, 'Invalid connection start event');
      const nodes = selectNodes(store.getState());

      const node = nodes.find((n) => n.id === nodeId);
      if (!node) {
        return;
      }

      if (isConnectorNode(node)) {
        if (handleType === 'source' && handleId !== CONNECTOR_OUTPUT_HANDLE) {
          return;
        }
        if (handleType === 'target' && handleId !== CONNECTOR_INPUT_HANDLE) {
          return;
        }

        const resolvedSourceType =
          handleType === 'source'
            ? resolveConnectorSourceFieldType(nodeId, nodes, selectNodesSlice(store.getState()).edges, templates)
            : null;
        $pendingConnection.set({
          nodeId,
          handleId,
          handleType,
          fieldTemplate:
            handleType === 'source'
              ? {
                  name: CONNECTOR_OUTPUT_HANDLE,
                  title: 'Connector Output',
                  description: '',
                  fieldKind: 'output',
                  ui_hidden: false,
                  type: resolvedSourceType ?? { name: 'AnyField', cardinality: 'SINGLE', batch: false },
                }
              : {
                  name: CONNECTOR_INPUT_HANDLE,
                  title: 'Connector Input',
                  description: '',
                  fieldKind: 'input',
                  input: 'connection',
                  required: false,
                  default: undefined,
                  ui_hidden: false,
                  type: { name: 'AnyField', cardinality: 'SINGLE', batch: false },
                },
        });
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
      const newEdge = connectionToEdge(connection);
      dispatch(edgesChanged([{ type: 'add', item: newEdge }]));
      updateNodeInternals([newEdge.source, newEdge.target]);
      $pendingConnection.set(null);
    },
    [store, updateNodeInternals]
  );
  const onConnectEnd = useCallback<OnConnectEnd>(() => {
    const { dispatch } = store;
    const pendingConnection = $pendingConnection.get();
    const edgePendingUpdate = $edgePendingUpdate.get();
    const mouseOverNodeId = $mouseOverNode.get();

    // If we are in the middle of an edge update, and the mouse isn't over a node, OR we have just updated the edge,
    // we should just bail and let the edge update (i.e. reconnect) logic handle the connection
    if ((edgePendingUpdate && !mouseOverNodeId) || $didUpdateEdge.get()) {
      $pendingConnection.set(null);
      return;
    }

    if (!pendingConnection) {
      return;
    }
    const { nodes, edges } = selectNodesSlice(store.getState());
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
        const newEdge = connectionToEdge(connection);
        const edgeChanges: EdgeChange<AnyEdge>[] = [{ type: 'add', item: newEdge }];

        const nodesToUpdate = [newEdge.source, newEdge.target];
        if (edgePendingUpdate) {
          $didUpdateEdge.set(true);
          edgeChanges.push({ type: 'remove', id: edgePendingUpdate.id });
          nodesToUpdate.push(edgePendingUpdate.source, edgePendingUpdate.target);
        }
        dispatch(edgesChanged(edgeChanges));
        updateNodeInternals(nodesToUpdate);
      }
      $pendingConnection.set(null);
    } else {
      // The mouse is not over a node - we should open the add node popover
      $addNodeCmdk.set(true);
    }
  }, [store, templates, updateNodeInternals]);

  const api = useMemo(() => ({ onConnectStart, onConnect, onConnectEnd }), [onConnectStart, onConnect, onConnectEnd]);
  return api;
};
