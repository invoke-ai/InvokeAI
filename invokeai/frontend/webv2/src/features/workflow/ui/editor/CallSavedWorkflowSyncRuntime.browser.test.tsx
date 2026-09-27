import type { ProjectGraphState, WorkflowNode } from '@features/workflow/core/types';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { ProjectGraphAction } from '@features/workflow/utility';

import { invalidateWorkflowLibraryCache } from '@features/workflow/data/libraryCache';
import { createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { getLibraryWorkflowRecordMock } = vi.hoisted(() => ({ getLibraryWorkflowRecordMock: vi.fn() }));

vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLibraryWorkflowRecord: getLibraryWorkflowRecordMock,
}));

// The reconciler only needs a loaded snapshot to run; the child record never
// arrives in these tests, so the template contents are never read.
vi.mock('@features/workflow/data/templates', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => ({ error: null, status: 'loaded', templates: {} }),
  subscribeInvocationTemplates: () => () => undefined,
}));

import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';

import { CallSavedWorkflowSyncRuntime } from './CallSavedWorkflowSyncRuntime';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const MISSING_WORKFLOW_ID = 'missing-workflow';

/** A call node as `parseWorkflowJson` produces it for a reloaded parent: an id, and status `loading`. */
const buildCallNode = (
  id: string,
  workflowId: string = MISSING_WORKFLOW_ID,
  withCapturedFields = false
): WorkflowNode =>
  ({
    data: {
      callSavedWorkflowStatus: 'loading',
      inputs: {
        ...(withCapturedFields ? { captured: { label: 'Captured', name: 'captured', value: 'persisted' } } : {}),
        workflow_id: { label: '', name: 'workflow_id', value: workflowId },
      },
      ...(withCapturedFields
        ? {
            dynamicInputTemplates: {
              captured: {
                description: 'Captured field',
                fieldKind: 'input',
                input: 'any',
                name: 'captured',
                required: false,
                title: 'Captured',
                type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
                uiHidden: false,
              },
            },
          }
        : {}),
      isIntermediate: false,
      isOpen: true,
      label: '',
      nodePack: 'invokeai',
      notes: '',
      type: 'call_saved_workflow',
      useCache: true,
      version: '1.0.0',
    },
    id,
    position: { x: 0, y: 0 },
    type: 'invocation',
  }) as unknown as WorkflowNode;

const readStatuses = (graph: ProjectGraphState): (string | undefined)[] =>
  graph.nodes.map((node) => (node.type === 'invocation' ? node.data.callSavedWorkflowStatus : undefined));

const settle = async (ms: number) => {
  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  });
};

/** Waits until the request count stops changing, so an early window cannot read as a pass. */
const settleUntilRequestsStop = async (getCount: () => number, sample = 25, maxSamples = 80) => {
  let previous = -1;
  let unchangedSamples = 0;

  for (let index = 0; index < maxSamples; index += 1) {
    const current = getCount();

    if (current === previous) {
      unchangedSamples += 1;
    } else {
      unchangedSamples = 0;
    }

    if (unchangedSamples >= 3) {
      return;
    }

    previous = current;
    await settle(sample);
  }
};

/**
 * Shared failing child-workflow IDs must settle every node to error without ping-ponging retries or remaining
 * loading forever.
 */
describe('CallSavedWorkflowSyncRuntime with an unreachable child workflow', () => {
  let host: HTMLDivElement;
  let root: Root;
  let queryClient: QueryClient;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    queryClient = new QueryClient();
    getLibraryWorkflowRecordMock.mockReset();
    // The delay matters: the reconciler coalesces everything scheduled before
    // its next macrotask, so a synchronous rejection folds the `fetch` and
    // `error` cache events into a single pass. A real request separates them.
    getLibraryWorkflowRecordMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          setTimeout(() => reject(new Error('not found')), 10);
        })
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    queryClient.clear();
    host.remove();
  });

  const mountWith = async (nodes: WorkflowNode[]) => {
    let graph: ProjectGraphState = { ...createProjectGraph('parent'), nodes };
    const listeners = new Set<() => void>();
    const snapshot = () => ({
      activeWorkflow: { document: graph },
      activeWorkflowId: graph.id,
      galleryValues: {},
      id: 'project-1',
      isWorkflowRunning: false,
      projectGraph: graph,
      workflowValues: {},
      workflows: [{ document: graph }],
    });
    let current = snapshot();
    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally stable for this render lifetime
    const adapter = {
      commands: {
        editGraph: (action: ProjectGraphAction) => {
          graph = projectGraphReducer(graph, action);
          current = snapshot();
          for (const listener of listeners) {
            listener();
          }
        },
        redo: vi.fn(),
        undo: vi.fn(),
      },
      getProjectGraph: () => graph,
      notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
      project: {
        getSnapshot: () => current,
        subscribe: (listener: () => void) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      widgets: { open: vi.fn(), patchValues: vi.fn() },
    } as unknown as WorkflowUiAdapter;

    root = createRoot(host);

    await act(() => {
      root.render(
        <QueryClientProvider client={queryClient}>
          <WorkflowUiProvider adapter={adapter}>
            <CallSavedWorkflowSyncRuntime />
          </WorkflowUiProvider>
        </QueryClientProvider>
      );
    });

    return {
      readGraph: () => graph,
      updateGraph: (nextGraph: ProjectGraphState) => {
        graph = nextGraph;
        current = snapshot();
        for (const listener of listeners) {
          listener();
        }
      },
    };
  };

  /** Every node reports the failure, and no further requests go out once they do. */
  const expectSettled = async (readGraph: () => ProjectGraphState, expected: string[], expectedCalls?: number) => {
    await settle(250);

    expect(readStatuses(readGraph())).toEqual(expected);

    const settledCalls = getLibraryWorkflowRecordMock.mock.calls.length;
    if (expectedCalls !== undefined) {
      expect(settledCalls).toBe(expectedCalls);
    }

    await settle(250);

    expect(getLibraryWorkflowRecordMock.mock.calls.length).toBe(settledCalls);
  };

  it('settles a single node', async () => {
    const { readGraph } = await mountWith([buildCallNode('call-1')]);

    await expectSettled(readGraph, ['error']);
  });

  // Distinct workflow IDs isolate retry bookkeeping for otherwise identical nodes.
  it('settles two nodes naming different unreachable workflows', async () => {
    const { readGraph } = await mountWith([buildCallNode('call-1', 'missing-a'), buildCallNode('call-2', 'missing-b')]);

    await expectSettled(readGraph, ['error', 'error']);
  });

  it('settles two nodes naming the same unreachable workflow', async () => {
    const { readGraph } = await mountWith([buildCallNode('call-1'), buildCallNode('call-2')]);

    await expectSettled(readGraph, ['error', 'error'], 1);
  });

  it('refetches an invalidated workflow once for multiple call nodes', async () => {
    const { readGraph } = await mountWith([buildCallNode('call-1'), buildCallNode('call-2'), buildCallNode('call-3')]);

    await expectSettled(readGraph, ['error', 'error', 'error'], 1);

    invalidateWorkflowLibraryCache(MISSING_WORKFLOW_ID);
    await settle(250);

    expect(getLibraryWorkflowRecordMock).toHaveBeenCalledTimes(2);
    expect(readStatuses(readGraph())).toEqual(['error', 'error', 'error']);
  });

  it('keeps an invalidation armed while an existing detail request settles', async () => {
    let resolveFirstRequest:
      | ((record: { name: string; workflow: { edges: never[]; nodes: never[] }; workflow_id: string }) => void)
      | undefined;
    let attempts = 0;
    getLibraryWorkflowRecordMock.mockImplementation((workflowId: string) => {
      attempts += 1;

      if (attempts === 1) {
        return new Promise((resolve) => {
          resolveFirstRequest = resolve;
        });
      }

      return Promise.resolve({ name: workflowId, workflow: { edges: [], nodes: [] }, workflow_id: workflowId });
    });

    const { readGraph } = await mountWith([buildCallNode('call-1')]);
    await settle(20);
    invalidateWorkflowLibraryCache(MISSING_WORKFLOW_ID);
    resolveFirstRequest?.({
      name: MISSING_WORKFLOW_ID,
      workflow: { edges: [], nodes: [] },
      workflow_id: MISSING_WORKFLOW_ID,
    });
    await settle(80);

    expect(getLibraryWorkflowRecordMock).toHaveBeenCalledTimes(2);
    expect(readStatuses(readGraph())).toEqual(['ready']);
  });

  it('keeps an invalidation retry available when the first matching node changes', async () => {
    const attempts = mockWorkflowThatRecoversAfterOneFailure(MISSING_WORKFLOW_ID);
    const { readGraph, updateGraph } = await mountWith([buildCallNode('call-1'), buildCallNode('call-2')]);

    await settle(60);
    invalidateWorkflowLibraryCache(MISSING_WORKFLOW_ID);
    selectWorkflow(readGraph, updateGraph, 'other-workflow');
    await settle(60);

    expect(attempts.filter((id) => id === MISSING_WORKFLOW_ID)).toHaveLength(2);
    expect(readStatuses(readGraph())).toEqual(['error', 'ready']);
  });

  it('keeps recalled dynamic fields visible when the child workflow is unavailable', async () => {
    const { readGraph } = await mountWith([buildCallNode('call-1', MISSING_WORKFLOW_ID, true)]);

    await expectSettled(readGraph, ['error']);

    const node = readGraph().nodes[0];
    expect(node.type === 'invocation' && node.data.dynamicInputTemplates).toHaveProperty('captured');
    expect(node.type === 'invocation' && node.data.inputs.captured?.value).toBe('persisted');
  });

  it('keeps a switched workflow retryable when the previous request settles late', async () => {
    let rejectWorkflowA: ((error: Error) => void) | undefined;
    getLibraryWorkflowRecordMock.mockImplementation((workflowId: string) => {
      if (workflowId === 'workflow-a') {
        return new Promise((_resolve, reject) => {
          rejectWorkflowA = reject;
        });
      }

      return Promise.reject(new Error('not found'));
    });

    const { readGraph, updateGraph } = await mountWith([buildCallNode('call-1', 'workflow-a')]);
    await settle(25);

    updateGraph(
      projectGraphReducer(readGraph(), {
        fieldName: 'workflow_id',
        nodeId: 'call-1',
        type: 'setFieldValue',
        value: 'workflow-b',
      })
    );
    await settle(25);
    expect(readStatuses(readGraph())).toEqual(['error']);

    invalidateWorkflowLibraryCache('workflow-b');
    rejectWorkflowA?.(new Error('workflow A settled late'));
    await settle(25);

    expect(getLibraryWorkflowRecordMock).toHaveBeenCalledTimes(3);
  });

  it('does not retain removed-node state when its id is reused', async () => {
    const { readGraph, updateGraph } = await mountWith([buildCallNode('call-1', 'workflow-a')]);
    await settle(60);
    expect(readStatuses(readGraph())).toEqual(['error']);

    updateGraph(
      projectGraphReducer(readGraph(), {
        fieldName: 'workflow_id',
        nodeId: 'call-1',
        type: 'setFieldValue',
        value: 'workflow-b',
      })
    );
    await settle(60);
    expect(readStatuses(readGraph())).toEqual(['error']);
    expect(getLibraryWorkflowRecordMock.mock.calls.map(([workflowId]) => workflowId)).toEqual([
      'workflow-a',
      'workflow-b',
    ]);

    updateGraph({ ...readGraph(), nodes: [] });
    await settle(20);

    updateGraph({ ...createProjectGraph('parent'), nodes: [buildCallNode('call-1', 'workflow-a')] });
    await settle(60);

    expect(readStatuses(readGraph())).toEqual(['error']);
    expect(getLibraryWorkflowRecordMock.mock.calls.map(([workflowId]) => workflowId)).toEqual([
      'workflow-a',
      'workflow-b',
    ]);
  });

  /**
   * Explicit reselection must retry indefinitely cached failures because automatic retries and expiration are
   * disabled.
   */
  const mockWorkflowThatRecoversAfterOneFailure = (workflowId: string) => {
    const attempts: string[] = [];

    getLibraryWorkflowRecordMock.mockImplementation((requestedId: string) => {
      attempts.push(requestedId);
      const hasFailedOnce = attempts.filter((id) => id === workflowId).length > 1;

      return new Promise((resolve, reject) => {
        setTimeout(() => {
          if (requestedId === workflowId && hasFailedOnce) {
            resolve({ name: 'Recovered', workflow: { edges: [], nodes: [] }, workflow_id: requestedId });
            return;
          }

          reject(new Error('not found'));
        }, 10);
      });
    });

    return attempts;
  };

  const selectWorkflow = (
    readGraph: () => ProjectGraphState,
    updateGraph: (next: ProjectGraphState) => void,
    value: string
  ) => {
    updateGraph(
      projectGraphReducer(readGraph(), { fieldName: 'workflow_id', nodeId: 'call-1', type: 'setFieldValue', value })
    );
  };

  it('retries a workflow that errored earlier when it is selected again', async () => {
    const attempts = mockWorkflowThatRecoversAfterOneFailure(MISSING_WORKFLOW_ID);
    const { readGraph, updateGraph } = await mountWith([buildCallNode('call-1')]);

    await settle(60);

    selectWorkflow(readGraph, updateGraph, 'other-workflow');
    await settle(60);

    selectWorkflow(readGraph, updateGraph, MISSING_WORKFLOW_ID);
    await settle(60);

    expect(attempts.filter((id) => id === MISSING_WORKFLOW_ID)).toHaveLength(2);
    expect(readStatuses(readGraph())).toEqual(['ready']);
  });

  it('retries a workflow that errored earlier after the selection is cleared and remade', async () => {
    const attempts = mockWorkflowThatRecoversAfterOneFailure(MISSING_WORKFLOW_ID);
    const { readGraph, updateGraph } = await mountWith([buildCallNode('call-1')]);

    await settle(60);

    selectWorkflow(readGraph, updateGraph, '');
    await settle(60);

    selectWorkflow(readGraph, updateGraph, MISSING_WORKFLOW_ID);
    await settle(60);

    expect(attempts.filter((id) => id === MISSING_WORKFLOW_ID)).toHaveLength(2);
    expect(readStatuses(readGraph())).toEqual(['ready']);
  });
  /**
   * One invalidation refreshes one shared cache entry, so it costs one request
   * however many nodes name that workflow. Selecting the workflow through the
   * picker arms a per-node retry. A cache invalidation supersedes those stale
   * node authorizations with one shared retry, so one invalidation costs one
   * request even when several nodes name the workflow.
   */
  it('refetches an invalidated workflow once for nodes that selected it through the picker', async () => {
    const SHARED_WORKFLOW_ID = 'shared-workflow';
    let isAvailable = true;

    getLibraryWorkflowRecordMock.mockImplementation(
      () =>
        new Promise((resolve, reject) => {
          setTimeout(() => {
            if (isAvailable) {
              resolve({ name: 'Child', workflow: { edges: [], nodes: [] }, workflow_id: SHARED_WORKFLOW_ID });
              return;
            }

            reject(new Error('not found'));
          }, 10);
        })
    );

    const { readGraph, updateGraph } = await mountWith([
      buildCallNode('call-1', ''),
      buildCallNode('call-2', ''),
      buildCallNode('call-3', ''),
    ]);

    await settle(60);

    for (const nodeId of ['call-1', 'call-2', 'call-3']) {
      updateGraph(
        projectGraphReducer(readGraph(), {
          fieldName: 'workflow_id',
          nodeId,
          type: 'setFieldValue',
          value: SHARED_WORKFLOW_ID,
        })
      );
      await settle(60);
    }

    const callsBeforeInvalidation = getLibraryWorkflowRecordMock.mock.calls.length;

    isAvailable = false;
    invalidateWorkflowLibraryCache(SHARED_WORKFLOW_ID);
    // Poll to quiescence rather than trusting a fixed window: the failure here
    // is extra requests, so a window that expired early would read as a pass.
    await settleUntilRequestsStop(() => getLibraryWorkflowRecordMock.mock.calls.length);

    expect(getLibraryWorkflowRecordMock.mock.calls.length - callsBeforeInvalidation).toBe(1);
  });
});
