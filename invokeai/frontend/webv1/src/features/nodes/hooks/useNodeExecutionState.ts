import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { selectNodes } from 'features/nodes/store/selectors';
import type { NodeExecutionStates } from 'features/nodes/store/types';
import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { map } from 'nanostores';
import { useEffect, useMemo } from 'react';

/**
 * A nanostore that holds the ephemeral execution state of nodes in the graph. The execution state includes
 * the status, error, progress, progress image, and outputs of each node.
 *
 * Note that, because a node can be duplicated by an iterate node, it can have multiple outputs recorded, one for each
 * iteration. For example, consider a collection of 3 images that are passed to an iterate node, which then passes each
 * image to a resize node. The resize node will have 3 outputs - one for each image.
 */
export const $nodeExecutionStates = map<NodeExecutionStates>({});

const initialNodeExecutionState: Omit<NodeExecutionState, 'nodeId'> = {
  status: zNodeStatus.enum.PENDING,
  error: null,
  progress: null,
  progressImage: null,
  outputs: [],
};

export const useNodeExecutionState = (nodeId?: string) => {
  const executionStates = useStore($nodeExecutionStates, nodeId ? { keys: [nodeId] } : undefined);
  const executionState = useMemo(() => (nodeId ? executionStates[nodeId] : undefined), [executionStates, nodeId]);
  return executionState;
};

const removeNodeExecutionState = (nodeId: string) => {
  $nodeExecutionStates.setKey(nodeId, undefined);
};

export const upsertExecutionState = (nodeId: string, updates?: Partial<NodeExecutionState>) => {
  const state = $nodeExecutionStates.get()[nodeId];
  if (!state) {
    $nodeExecutionStates.setKey(nodeId, { ...deepClone(initialNodeExecutionState), nodeId, ...updates });
  } else {
    $nodeExecutionStates.setKey(nodeId, { ...state, ...updates });
  }
};

const selectNodeIds = createMemoizedSelector(selectNodes, (nodes) => nodes.map((node) => node.id));

/**
 * Keeps the ephemeral store of node execution states in sync with the nodes in the graph.
 *
 * For example, if a node is deleted from the graph, its execution state is removed from the store, and
 * if a new node is added to the graph, an initial execution state is added to the store.
 *
 * Node execution states are stored in $nodeExecutionStates nanostore.
 */
export const useSyncExecutionState = () => {
  const nodeIds = useAppSelector(selectNodeIds);
  useEffect(() => {
    const nodeExecutionStates = $nodeExecutionStates.get();
    const nodeIdsToAdd = nodeIds.filter((id) => !nodeExecutionStates[id]);
    const nodeIdsToRemove = Object.keys(nodeExecutionStates).filter((id) => !nodeIds.includes(id));
    for (const id of nodeIdsToAdd) {
      upsertExecutionState(id);
    }
    for (const id of nodeIdsToRemove) {
      removeNodeExecutionState(id);
    }
  }, [nodeIds]);
};
