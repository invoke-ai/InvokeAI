import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/storeHooks';
import { $mouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { $pendingConnection, $templates, connectionMade } from 'features/nodes/store/nodesSlice';
import { getFirstValidConnection } from 'features/nodes/store/util/findConnectionToValidHandle';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useCallback, useMemo } from 'react';
import type { OnConnect, OnConnectEnd, OnConnectStart } from 'reactflow';
import { assert } from 'tsafe';

export const useConnection = () => {
  const store = useAppStore();
  const templates = useStore($templates);

  const onConnectStart = useCallback<OnConnectStart>(
    (event, params) => {
      const nodes = store.getState().nodes.present.nodes;
      const { nodeId, handleId, handleType } = params;
      assert(nodeId && handleId && handleType, `Invalid connection start params: ${JSON.stringify(params)}`);
      const node = nodes.find((n) => n.id === nodeId);
      assert(isInvocationNode(node), `Invalid node during connection: ${JSON.stringify(node)}`);
      const template = templates[node.data.type];
      assert(template, `Template not found for node type: ${node.data.type}`);
      const fieldTemplate = handleType === 'source' ? template.outputs[handleId] : template.inputs[handleId];
      assert(fieldTemplate, `Field template not found for field: ${node.data.type}.${handleId}`);
      $pendingConnection.set({
        node,
        template,
        fieldTemplate,
      });
    },
    [store, templates]
  );
  const onConnect = useCallback<OnConnect>(
    (connection) => {
      const { dispatch } = store;
      dispatch(connectionMade(connection));
      $pendingConnection.set(null);
    },
    [store]
  );
  const onConnectEnd = useCallback<OnConnectEnd>(() => {
    const { dispatch } = store;
    const pendingConnection = $pendingConnection.get();
    if (!pendingConnection) {
      return;
    }
    const mouseOverNodeId = $mouseOverNode.get();
    const { nodes, edges } = store.getState().nodes.present;
    if (mouseOverNodeId) {
      const candidateNode = nodes.filter(isInvocationNode).find((n) => n.id === mouseOverNodeId);
      if (!candidateNode) {
        // The mouse is over a non-invocation node - bail
        return;
      }
      const candidateTemplate = templates[candidateNode.data.type];
      assert(candidateTemplate, `Template not found for node type: ${candidateNode.data.type}`);
      const connection = getFirstValidConnection(nodes, edges, pendingConnection, candidateNode, candidateTemplate);
      if (connection) {
        dispatch(connectionMade(connection));
      }
    }
    $pendingConnection.set(null);
  }, [store, templates]);

  const api = useMemo(() => ({ onConnectStart, onConnect, onConnectEnd }), [onConnectStart, onConnect, onConnectEnd]);
  return api;
};
