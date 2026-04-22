import { deepClone } from 'common/util/deepClone';
import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import type { S } from 'services/api/types';

const getInvocationKey = (data: { item_id: number; invocation: { id: string } }) =>
  `${data.item_id}:${data.invocation.id}`;

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
  completedInvocationKeys: Set<string>
) => {
  if (completedInvocationKeys.has(getInvocationKey(data))) {
    return;
  }

  const _nodeExecutionState = deepClone(nodeExecutionState ?? getInitialNodeExecutionState(data.invocation_source_id));
  _nodeExecutionState.status = zNodeStatus.enum.IN_PROGRESS;

  return _nodeExecutionState;
};

export const getUpdatedNodeExecutionStateOnInvocationProgress = (
  nodeExecutionState: NodeExecutionState | undefined,
  data: S['InvocationProgressEvent'],
  completedInvocationKeys: Set<string>
) => {
  if (completedInvocationKeys.has(getInvocationKey(data))) {
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
  completedInvocationKeys: Set<string>
) => {
  const completedInvocationKey = getInvocationKey(data);

  if (completedInvocationKeys.has(completedInvocationKey)) {
    return;
  }

  const _nodeExecutionState = deepClone(nodeExecutionState ?? getInitialNodeExecutionState(data.invocation_source_id));
  _nodeExecutionState.status = zNodeStatus.enum.COMPLETED;
  if (_nodeExecutionState.progress !== null) {
    _nodeExecutionState.progress = 1;
  }
  _nodeExecutionState.outputs.push(data.result);
  completedInvocationKeys.add(completedInvocationKey);

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
