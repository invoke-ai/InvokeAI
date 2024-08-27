import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodeExecutionStates } from 'features/nodes/store/types';
import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { map } from 'nanostores';
import { useEffect, useMemo } from 'react';

export const $nodeExecutionStates = map<NodeExecutionStates>({});

const initialNodeExecutionState: Omit<NodeExecutionState, 'nodeId'> = {
  status: zNodeStatus.enum.PENDING,
  error: null,
  progress: null,
  progressImage: null,
  outputs: [],
};

export const useExecutionState = (nodeId?: string) => {
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

const selectNodeIds = createMemoizedSelector(selectNodesSlice, (nodesSlice) => nodesSlice.nodes.map((node) => node.id));

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
