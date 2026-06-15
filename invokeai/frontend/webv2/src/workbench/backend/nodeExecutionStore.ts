import { useSyncExternalStore } from 'react';

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

const stateByNodeId = new Map<string, NodeExecutionState>();
const listeners = new Set<() => void>();

const emit = (): void => {
  for (const listener of listeners) {
    listener();
  }
};

const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

/** Pull the produced image out of an invocation output, whatever the node type. */
const getResultImageName = (result: InvocationCompleteEvent['result']): string | null => {
  const image = (result as { image?: { image_name?: unknown } }).image;

  return typeof image?.image_name === 'string' ? image.image_name : null;
};

export const nodeExecutionStore = {
  clearAll(): void {
    if (stateByNodeId.size > 0) {
      stateByNodeId.clear();
      emit();
    }
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
    emit();
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
    emit();
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
    emit();
  },
  /** A queue item reached a terminal state: nothing can still be running. */
  settleRunning(): void {
    let changed = false;

    for (const [nodeId, state] of stateByNodeId) {
      if (state.status === 'running') {
        stateByNodeId.set(nodeId, { ...state, progress: null, progressMessage: null, status: 'completed' });
        changed = true;
      }
    }

    if (changed) {
      emit();
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
    emit();
  },
};

export type NodeExecutionSink = typeof nodeExecutionStore;

export const useNodeExecutionState = (nodeId: string): NodeExecutionState | null =>
  useSyncExternalStore(subscribe, () => stateByNodeId.get(nodeId) ?? null);
