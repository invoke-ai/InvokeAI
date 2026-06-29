import { createKeyedTransientStore } from '@workbench/externalStore';

import type { InvocationCompleteEvent, InvocationErrorEvent, InvocationStartedEvent } from './events';

import { buildApiUrl } from './http';

/**
 * Ephemeral per-node execution state, keyed by the invocation's source node id
 * (the workflow editor's node id). Like the queue-item progress store, this is
 * high-frequency transient data that deliberately lives outside the workbench
 * reducer; the editor's nodes subscribe per id and only re-render when their
 * own node's state moves.
 */

export type NodeExecutionStatus = 'running' | 'completed' | 'failed';

export interface NodeExecutionState {
  status: NodeExecutionStatus;
  /** 0..1, or null while indeterminate. Only meaningful while running. */
  progress: number | null;
  progressMessage: string | null;
  /** Thumbnail of the node's most recent image output, when it produced one. */
  outputImageUrl: string | null;
  error: string | null;
}

const stateByNodeId = createKeyedTransientStore<string, NodeExecutionState>();

/** Pull the produced image out of an invocation output, whatever the node type. */
const getResultImageName = (result: InvocationCompleteEvent['result']): string | null => {
  const image = (result as { image?: { image_name?: unknown } }).image;

  return typeof image?.image_name === 'string' ? image.image_name : null;
};

export const nodeExecutionStore = {
  clearAll(): void {
    stateByNodeId.clear();
  },
  completed(event: InvocationCompleteEvent): void {
    const imageName = getResultImageName(event.result);
    const previous = stateByNodeId.get(event.invocation_source_id);

    stateByNodeId.set(event.invocation_source_id, {
      error: null,
      outputImageUrl: imageName
        ? buildApiUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`)
        : (previous?.outputImageUrl ?? null),
      progress: null,
      progressMessage: null,
      status: 'completed',
    });
  },
  failed(event: InvocationErrorEvent): void {
    const previous = stateByNodeId.get(event.invocation_source_id);

    stateByNodeId.set(event.invocation_source_id, {
      error: event.error_message,
      outputImageUrl: previous?.outputImageUrl ?? null,
      progress: null,
      progressMessage: null,
      status: 'failed',
    });
  },
  progress(nodeId: string, percentage: number | null, message: string): void {
    const previous = stateByNodeId.get(nodeId);

    stateByNodeId.set(nodeId, {
      error: null,
      outputImageUrl: previous?.outputImageUrl ?? null,
      progress: percentage,
      progressMessage: message,
      status: 'running',
    });
  },
  /** A queue item reached a terminal state: nothing can still be running. */
  settleRunning(): void {
    for (const [nodeId, state] of stateByNodeId.entries()) {
      if (state.status === 'running') {
        stateByNodeId.set(nodeId, { ...state, progress: null, progressMessage: null, status: 'completed' });
      }
    }
  },
  started(event: InvocationStartedEvent): void {
    const previous = stateByNodeId.get(event.invocation_source_id);

    stateByNodeId.set(event.invocation_source_id, {
      error: null,
      outputImageUrl: previous?.outputImageUrl ?? null,
      progress: null,
      progressMessage: null,
      status: 'running',
    });
  },
};

export type NodeExecutionSink = typeof nodeExecutionStore;

export const useNodeExecutionState = (nodeId: string): NodeExecutionState | null =>
  stateByNodeId.useValue(nodeId) ?? null;
