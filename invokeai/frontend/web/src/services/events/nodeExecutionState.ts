import { deepClone } from 'common/util/deepClone';
import type { NodeExecutionStates } from 'features/nodes/store/types';
import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { S } from 'services/api/types';
import { hasCompletedInvocationKey, markInvocationAsCompleted } from 'services/events/invocationTracking';

type CompletedInvocationKeysByItemId = Map<number, Set<string>>;

const getInitialNodeExecutionState = (nodeId: string): NodeExecutionState => ({
  nodeId,
  status: zNodeStatus.enum.PENDING,
  progress: null,
  progressImage: null,
  outputs: [],
  error: null,
});

export const getResetNodeExecutionStatesOnQueueItemStarted = (
  nodeExecutionStates: NodeExecutionStates,
  itemId: number,
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId
): NodeExecutionStates | undefined => {
  if (completedInvocationKeysByItemId.has(itemId)) {
    return;
  }

  const next: NodeExecutionStates = {};
  for (const [nodeId, nodeExecutionState] of Object.entries(nodeExecutionStates)) {
    if (!nodeExecutionState) {
      next[nodeId] = nodeExecutionState;
      continue;
    }
    const clone = deepClone(nodeExecutionState);
    clone.status = zNodeStatus.enum.PENDING;
    clone.error = null;
    clone.progress = null;
    clone.progressImage = null;
    clone.outputs = [];
    next[nodeId] = clone;
  }
  return next;
};

export const getUpdatedNodeExecutionStateOnInvocationStarted = (
  nodeExecutionState: NodeExecutionState | undefined,
  data: S['InvocationStartedEvent'],
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId
) => {
  if (hasCompletedInvocationKey(completedInvocationKeysByItemId, data.item_id, data.invocation.id)) {
    return;
  }

  const _nodeExecutionState = deepClone(nodeExecutionState ?? getInitialNodeExecutionState(data.invocation_source_id));
  _nodeExecutionState.status = zNodeStatus.enum.IN_PROGRESS;

  return _nodeExecutionState;
};

export const getUpdatedNodeExecutionStateOnInvocationProgress = (
  nodeExecutionState: NodeExecutionState | undefined,
  data: S['InvocationProgressEvent'],
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId
) => {
  if (hasCompletedInvocationKey(completedInvocationKeysByItemId, data.item_id, data.invocation.id)) {
    return;
  }

  const _nodeExecutionState = deepClone(nodeExecutionState ?? getInitialNodeExecutionState(data.invocation_source_id));
  _nodeExecutionState.status = zNodeStatus.enum.IN_PROGRESS;
  _nodeExecutionState.progress = data.percentage ?? null;
  _nodeExecutionState.progressImage = data.image ?? null;

  return _nodeExecutionState;
};

export const getUpdatedNodeExecutionStateOnInvocationComplete = (
  nodeExecutionState: NodeExecutionState | undefined,
  data: S['InvocationCompleteEvent'],
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId
) => {
  if (hasCompletedInvocationKey(completedInvocationKeysByItemId, data.item_id, data.invocation.id)) {
    return;
  }

  const _nodeExecutionState = deepClone(nodeExecutionState ?? getInitialNodeExecutionState(data.invocation_source_id));
  _nodeExecutionState.status = zNodeStatus.enum.COMPLETED;
  if (_nodeExecutionState.progress !== null) {
    _nodeExecutionState.progress = 1;
  }
  _nodeExecutionState.outputs.push(data.result);
  markInvocationAsCompleted(completedInvocationKeysByItemId, data.item_id, data.invocation.id);

  return _nodeExecutionState;
};

export const getUpdatedNodeExecutionStateOnInvocationError = (
  nodeExecutionState: NodeExecutionState | undefined,
  data: S['InvocationErrorEvent']
) => {
  const _nodeExecutionState = deepClone(nodeExecutionState ?? getInitialNodeExecutionState(data.invocation_source_id));
  _nodeExecutionState.status = zNodeStatus.enum.FAILED;
  _nodeExecutionState.progress = null;
  _nodeExecutionState.progressImage = null;
  _nodeExecutionState.error = {
    error_type: data.error_type,
    error_message: data.error_message,
    error_traceback: data.error_traceback,
  };

  return _nodeExecutionState;
};

export const getNodeExecutionStatesFromCompletedSession = (
  session: S['SessionQueueItem']['session']
): NodeExecutionState[] => {
  const nodeExecutionStates: NodeExecutionState[] = [];

  for (const [nodeId, preparedNodeIds] of Object.entries(session.source_prepared_mapping)) {
    const outputs = preparedNodeIds.flatMap((preparedNodeId) => {
      const result = session.results[preparedNodeId];
      return result ? [result] : [];
    });

    if (outputs.length === 0) {
      continue;
    }

    nodeExecutionStates.push({
      ...getInitialNodeExecutionState(nodeId),
      status: zNodeStatus.enum.COMPLETED,
      outputs,
    });
  }

  return nodeExecutionStates;
};

export const getCompletedInvocationIdsFromCompletedSession = (session: S['SessionQueueItem']['session']): string[] => {
  const completedInvocationIds: string[] = [];

  for (const preparedNodeIds of Object.values(session.source_prepared_mapping)) {
    for (const preparedNodeId of preparedNodeIds) {
      if (session.results[preparedNodeId]) {
        completedInvocationIds.push(preparedNodeId);
      }
    }
  }

  return completedInvocationIds;
};
