import { deepClone } from 'common/util/deepClone';
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
