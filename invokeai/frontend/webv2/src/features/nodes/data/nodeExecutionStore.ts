import type {
  NodeInvocationCompleteEvent,
  NodeInvocationErrorEvent,
  NodeInvocationStartedEvent,
} from '@features/nodes/core/executionContracts';

import { getFirstOutputImageName } from '@platform/core/outputImages';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore, createKeyedTransientStore } from '@platform/state/externalStore';

import { browserNodesDataPort } from './transport';

/**
 * Keep transient execution state outside the workbench reducer; subscribe per source node ID to isolate frequent
 * renders.
 */

export type NodeExecutionStatus = 'running' | 'completed' | 'failed';

/** How the queue item that was running these nodes ended. */
export type NodeExecutionOutcome = 'completed' | 'failed' | 'canceled';

export interface NodeExecutionState {
  status: NodeExecutionStatus;
  /** 0..1, or null while indeterminate. Only meaningful while running. */
  progress: number | null;
  progressMessage: string | null;
  /** Thumbnail of the node's most recent image output, when it produced one. */
  outputImageUrl: string | null;
  /** The node's most recent invocation result in the current run (a loop body runs many times). */
  latestOutput: unknown;
  error: string | null;
}

/** The project workflow whose run the store reflects; editors showing another copy ignore the node states. */
export interface NodeExecutionOrigin {
  projectId: string;
  workflowId: string;
}

const stateByNodeId = createKeyedTransientStore<string, NodeExecutionState>();
const originStore = createExternalStore<{ origin: NodeExecutionOrigin | null }>({ origin: null });

export const nodeExecutionStore = {
  clearAll(): void {
    stateByNodeId.clear();
    if (originStore.getSnapshot().origin !== null) {
      originStore.setSnapshot({ origin: null });
    }
  },
  getOrigin(): NodeExecutionOrigin | null {
    return originStore.getSnapshot().origin;
  },
  setOrigin(origin: NodeExecutionOrigin | null): void {
    const current = originStore.getSnapshot().origin;
    if (current?.projectId !== origin?.projectId || current?.workflowId !== origin?.workflowId) {
      originStore.setSnapshot({ origin });
    }
  },
  subscribeOrigin(listener: () => void): () => void {
    return originStore.subscribe(listener);
  },
  get(nodeId: string): NodeExecutionState | null {
    return stateByNodeId.get(nodeId) ?? null;
  },
  subscribe(nodeId: string, listener: () => void): () => void {
    return stateByNodeId.subscribeKey(nodeId, listener);
  },
  completed(event: NodeInvocationCompleteEvent): void {
    const imageName = getFirstOutputImageName(event.result);

    stateByNodeId.set(event.invocation_source_id, {
      error: null,
      outputImageUrl: imageName
        ? browserNodesDataPort.buildUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`)
        : null,
      latestOutput: event.result,
      progress: null,
      progressMessage: null,
      status: 'completed',
    });
  },
  failed(event: NodeInvocationErrorEvent): void {
    const previous = stateByNodeId.get(event.invocation_source_id);

    stateByNodeId.set(event.invocation_source_id, {
      error: event.error_message,
      outputImageUrl: previous?.outputImageUrl ?? null,
      latestOutput: previous?.latestOutput ?? null,
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
      latestOutput: previous?.latestOutput ?? null,
      progress: percentage,
      progressMessage: message,
      status: 'running',
    });
  },
  /**
   * The queue item running these nodes reached a terminal state: a node still marked running
   * finished with it, or never will (its failure/cancel event was lost or never sent).
   */
  settleRunning(nodeIds: Iterable<string>, outcome: NodeExecutionOutcome, error?: string): void {
    for (const nodeId of nodeIds) {
      const state = stateByNodeId.get(nodeId);

      if (state?.status !== 'running') {
        continue;
      }

      if (outcome === 'completed') {
        stateByNodeId.set(nodeId, { ...state, progress: null, progressMessage: null, status: 'completed' });
      } else if (outcome === 'failed') {
        stateByNodeId.set(nodeId, {
          ...state,
          error: error ?? state.error,
          progress: null,
          progressMessage: null,
          status: 'failed',
        });
      } else {
        stateByNodeId.delete(nodeId);
      }
    }
  },
  started(event: NodeInvocationStartedEvent): void {
    const previous = stateByNodeId.get(event.invocation_source_id);

    stateByNodeId.set(event.invocation_source_id, {
      error: null,
      outputImageUrl: previous?.outputImageUrl ?? null,
      latestOutput: previous?.latestOutput ?? null,
      progress: null,
      progressMessage: null,
      status: 'running',
    });
  },
};

export interface NodeExecutionSink {
  clearAll(): void;
  completed(event: NodeInvocationCompleteEvent): void;
  failed(event: NodeInvocationErrorEvent): void;
  get(nodeId: string): NodeExecutionState | null;
  getOrigin(): NodeExecutionOrigin | null;
  setOrigin(origin: NodeExecutionOrigin | null): void;
  subscribeOrigin(listener: () => void): () => void;
  progress(nodeId: string, percentage: number | null, message: string): void;
  settleRunning(nodeIds: Iterable<string>, outcome: NodeExecutionOutcome, error?: string): void;
  started(event: NodeInvocationStartedEvent): void;
  subscribe(nodeId: string, listener: () => void): () => void;
}

registerAccountOwnedResource({
  clear: nodeExecutionStore.clearAll,
  name: 'node-execution',
});

export const useNodeExecutionState = (nodeId: string): NodeExecutionState | null =>
  stateByNodeId.useValue(nodeId) ?? null;
