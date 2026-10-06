import type {
  FieldInputTemplate,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowNode,
} from '@features/workflow/core/types';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type { ProjectGraphAction } from '@features/workflow/utility';

import { invalidateWorkflowLibraryCache } from '@features/workflow/data/libraryCache';
import {
  getSavedWorkflowPickerOwnedQuery,
  getSavedWorkflowPickerSharedQuery,
} from '@features/workflow/data/savedWorkflowFieldUtils';
import { savedWorkflowPickerQueryOptions } from '@features/workflow/data/savedWorkflowQueries';
import {
  buildInvocationNode,
  createProjectGraph,
  projectGraphReducer,
  serializeWorkflowJson,
} from '@features/workflow/utility';
import { queryClient } from '@platform/query/client';
import { QueryClientProvider, useInfiniteQuery } from '@tanstack/react-query';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { getInvocationTemplatesSnapshotMock, getLibraryWorkflowRecordMock, listLibraryWorkflowsMock } = vi.hoisted(
  () => ({
    getInvocationTemplatesSnapshotMock: vi.fn(),
    getLibraryWorkflowRecordMock: vi.fn(),
    listLibraryWorkflowsMock: vi.fn(),
  })
);

vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLibraryWorkflowRecord: getLibraryWorkflowRecordMock,
  listLibraryWorkflows: listLibraryWorkflowsMock,
}));

vi.mock('@features/workflow/data/templates', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: getInvocationTemplatesSnapshotMock,
  subscribeInvocationTemplates: () => () => undefined,
}));

import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';

import { CallSavedWorkflowSyncRuntime } from './CallSavedWorkflowSyncRuntime';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const MISSING_WORKFLOW_ID = 'missing-workflow';
const PICKER_QUERY_KEY = savedWorkflowPickerQueryOptions(getSavedWorkflowPickerOwnedQuery()).queryKey;
const CHILD_INVOCATION_TYPE = 'text_input';
const CHILD_NODE_ID = 'input-node';
const dynamicFieldName = (fieldName: string) => `saved_workflow_input::${CHILD_NODE_ID}::${fieldName}`;

const makeStringField = (name: string, title: string, description: string): FieldInputTemplate => ({
  default: '',
  description,
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: false,
  title,
  type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
});

const childInputTemplate: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: 'Test inputs',
  inputs: {
    negative_prompt: makeStringField('negative_prompt', 'Negative prompt', 'Avoid these details'),
    prompt: makeStringField('prompt', 'Prompt', 'Describe the image'),
  },
  nodePack: 'invokeai',
  outputType: 'image_output',
  outputs: {},
  tags: [],
  title: 'Text input',
  type: CHILD_INVOCATION_TYPE,
  useCache: true,
  version: '1.0.0',
};

const buildLibraryChildWorkflow = (promptLabel: string, promptDescription: string, exposeNegativePrompt: boolean) => {
  let document = createProjectGraph('library-child', 'Library child');
  const node = buildInvocationNode(childInputTemplate, { x: 0, y: 0 });
  node.id = CHILD_NODE_ID;
  node.data.inputs.prompt = {
    description: promptDescription,
    label: promptLabel,
    name: 'prompt',
    value: 'initial prompt',
  };
  node.data.inputs.negative_prompt = {
    label: 'Negative prompt',
    name: 'negative_prompt',
    value: 'initial negative prompt',
  };
  document = projectGraphReducer(document, { node, type: 'addNode' });
  document = projectGraphReducer(document, {
    fieldIdentifier: { fieldName: 'prompt', nodeId: CHILD_NODE_ID },
    type: 'exposeField',
  });

  if (exposeNegativePrompt) {
    document = projectGraphReducer(document, {
      fieldIdentifier: { fieldName: 'negative_prompt', nodeId: CHILD_NODE_ID },
      type: 'exposeField',
    });
  }

  return serializeWorkflowJson(document);
};

const WorkflowPickerQueryProbe = () => {
  useInfiniteQuery(savedWorkflowPickerQueryOptions(getSavedWorkflowPickerOwnedQuery()));
  return null;
};

const WorkflowPickerVariantsProbe = () => {
  const ownedQuery = useInfiniteQuery(savedWorkflowPickerQueryOptions(getSavedWorkflowPickerOwnedQuery('updated')));
  const sharedQuery = useInfiniteQuery(savedWorkflowPickerQueryOptions(getSavedWorkflowPickerSharedQuery()));

  return (
    <>
      <button
        aria-label="Fetch next owned picker page"
        data-testid="owned-next-page"
        onClick={() => void ownedQuery.fetchNextPage()}
        type="button"
      />
      <button
        aria-label="Fetch next shared picker page"
        data-testid="shared-next-page"
        onClick={() => void sharedQuery.fetchNextPage()}
        type="button"
      />
    </>
  );
};

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
describe('CallSavedWorkflowSyncRuntime', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    queryClient.clear();
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
    listLibraryWorkflowsMock.mockReset();
    getInvocationTemplatesSnapshotMock.mockReturnValue({ error: null, status: 'loaded', templates: {} });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    queryClient.clear();
    host.remove();
  });

  const mountWith = async (
    nodes: WorkflowNode[],
    withPicker = false,
    withRuntime = true,
    withPickerVariants = false,
    edges: WorkflowEdge[] = []
  ) => {
    let graph: ProjectGraphState = { ...createProjectGraph('parent'), edges, nodes };
    const notifications = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
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
      notifications,
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
            {withRuntime ? <CallSavedWorkflowSyncRuntime /> : null}
            {withPicker ? <WorkflowPickerQueryProbe /> : null}
            {withPickerVariants ? <WorkflowPickerVariantsProbe /> : null}
          </WorkflowUiProvider>
        </QueryClientProvider>
      );
    });

    return {
      notifications,
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

  it.each([
    ['saving a new workflow', () => invalidateWorkflowLibraryCache('new-workflow')],
    ['deleting a workflow', () => invalidateWorkflowLibraryCache()],
    ['renaming a workflow', () => invalidateWorkflowLibraryCache('renamed-workflow')],
  ])('refreshes picker results after %s', async (_operation, invalidateLibrary) => {
    listLibraryWorkflowsMock
      .mockResolvedValueOnce({ items: [{ workflow_id: 'old-item' }], page: 0, pages: 1, per_page: 20, total: 1 })
      .mockResolvedValueOnce({ items: [{ workflow_id: 'new-item' }], page: 0, pages: 1, per_page: 20, total: 1 });

    // Publication can occur outside the editor widget, so picker invalidation must work without its runtime mounted.
    await mountWith([], true, false);
    await settleUntilRequestsStop(() => listLibraryWorkflowsMock.mock.calls.length);

    expect(listLibraryWorkflowsMock).toHaveBeenCalledTimes(1);

    invalidateLibrary();
    await settleUntilRequestsStop(() => listLibraryWorkflowsMock.mock.calls.length);

    expect(listLibraryWorkflowsMock).toHaveBeenCalledTimes(2);
    expect(queryClient.getQueryData(PICKER_QUERY_KEY)).toMatchObject({
      pages: [{ items: [{ workflow_id: 'new-item' }] }],
    });
  });

  it('refreshes every loaded owned and shared picker page', async () => {
    let version = 1;
    listLibraryWorkflowsMock.mockImplementation(({ isPublic, page }: { isPublic?: boolean; page: number }) =>
      Promise.resolve({
        items: [{ workflow_id: `v${version}-${isPublic ? 'shared' : 'owned'}-${page}` }],
        page,
        pages: 2,
        per_page: 20,
        total: 2,
      })
    );

    await mountWith([], false, false, true);
    await settleUntilRequestsStop(() => listLibraryWorkflowsMock.mock.calls.length);
    expect(listLibraryWorkflowsMock).toHaveBeenCalledTimes(2);

    await act(() => host.querySelector<HTMLButtonElement>('[data-testid="owned-next-page"]')?.click());
    await settleUntilRequestsStop(() => listLibraryWorkflowsMock.mock.calls.length);
    await act(() => host.querySelector<HTMLButtonElement>('[data-testid="shared-next-page"]')?.click());
    await settleUntilRequestsStop(() => listLibraryWorkflowsMock.mock.calls.length);
    expect(listLibraryWorkflowsMock).toHaveBeenCalledTimes(4);

    version = 2;
    invalidateWorkflowLibraryCache();
    await settleUntilRequestsStop(() => listLibraryWorkflowsMock.mock.calls.length);

    const ownedKey = savedWorkflowPickerQueryOptions(getSavedWorkflowPickerOwnedQuery('updated')).queryKey;
    const sharedKey = savedWorkflowPickerQueryOptions(getSavedWorkflowPickerSharedQuery()).queryKey;
    expect(listLibraryWorkflowsMock).toHaveBeenCalledTimes(8);
    expect(queryClient.getQueryData(ownedKey)).toMatchObject({
      pages: [{ items: [{ workflow_id: 'v2-owned-0' }] }, { items: [{ workflow_id: 'v2-owned-1' }] }],
    });
    expect(queryClient.getQueryData(sharedKey)).toMatchObject({
      pages: [{ items: [{ workflow_id: 'v2-shared-0' }] }, { items: [{ workflow_id: 'v2-shared-1' }] }],
    });
  });

  it('propagates a selected library workflow update into its call node inputs', async () => {
    const childTemplates: InvocationTemplates = { [CHILD_INVOCATION_TYPE]: childInputTemplate };
    getInvocationTemplatesSnapshotMock.mockReturnValue({ error: null, status: 'loaded', templates: childTemplates });
    let currentRecord = {
      name: 'Child workflow',
      workflow: buildLibraryChildWorkflow('Prompt v1', 'Description v1', false),
      workflow_id: 'updated-child',
    };
    getLibraryWorkflowRecordMock.mockImplementation(() => Promise.resolve(currentRecord));

    const { readGraph, updateGraph } = await mountWith([buildCallNode('call-1', 'updated-child')]);
    await settleUntilRequestsStop(() => getLibraryWorkflowRecordMock.mock.calls.length);

    const promptInputName = dynamicFieldName('prompt');
    let callNode = readGraph().nodes.find((node) => node.id === 'call-1');
    expect(callNode?.type === 'invocation' && callNode.data.inputs[promptInputName]).toMatchObject({
      description: 'Description v1',
      label: 'Prompt v1',
      value: 'initial prompt',
    });

    updateGraph(
      projectGraphReducer(readGraph(), {
        fieldName: promptInputName,
        nodeId: 'call-1',
        type: 'setFieldValue',
        value: 'parent value',
      })
    );
    currentRecord = {
      ...currentRecord,
      workflow: buildLibraryChildWorkflow('Prompt v2', 'Description v2', true),
    };
    invalidateWorkflowLibraryCache('updated-child');
    await settleUntilRequestsStop(() => getLibraryWorkflowRecordMock.mock.calls.length);

    callNode = readGraph().nodes.find((node) => node.id === 'call-1');
    expect(getLibraryWorkflowRecordMock).toHaveBeenCalledTimes(2);
    expect(callNode?.type === 'invocation' && callNode.data.inputs[promptInputName]).toMatchObject({
      description: 'Description v2',
      label: 'Prompt v2',
      value: 'parent value',
    });
    expect(callNode?.type === 'invocation' && callNode.data.inputs[dynamicFieldName('negative_prompt')]).toMatchObject({
      label: 'Negative prompt',
      value: 'initial negative prompt',
    });
  });

  /** A string source wired into both of `child-a`'s inputs; `child-b` was saved from it without the negative prompt. */
  const mountConnectedCall = async () => {
    const stringSourceTemplate: InvocationTemplate = {
      ...childInputTemplate,
      inputs: {},
      outputType: 'string_output',
      outputs: {
        value: {
          description: '',
          name: 'value',
          title: 'Value',
          type: { batch: false, cardinality: 'SINGLE', name: 'StringField' },
        },
      },
      title: 'String',
      type: 'string_source',
    };
    getInvocationTemplatesSnapshotMock.mockReturnValue({
      error: null,
      status: 'loaded',
      templates: { [CHILD_INVOCATION_TYPE]: childInputTemplate, string_source: stringSourceTemplate },
    });
    // `child-b` was saved from `child-a`, so the prompt keeps its field identity; it no longer exposes the negative.
    const library: Record<string, Record<string, unknown>> = {
      'child-a': buildLibraryChildWorkflow('Prompt', 'Describe the image', true),
      'child-b': buildLibraryChildWorkflow('Prompt', 'Describe the image', false),
    };
    getLibraryWorkflowRecordMock.mockImplementation((workflowId: string) =>
      library[workflowId]
        ? Promise.resolve({ name: workflowId, workflow: library[workflowId], workflow_id: workflowId })
        : Promise.reject(new Error('not found'))
    );
    const source = buildInvocationNode(stringSourceTemplate, { x: 0, y: 0 });
    source.id = 'source-1';
    const edges: WorkflowEdge[] = ['prompt', 'negative_prompt'].map((fieldName) => ({
      id: `edge-${fieldName}`,
      source: source.id,
      sourceHandle: 'value',
      target: 'call-1',
      targetHandle: dynamicFieldName(fieldName),
      type: 'default',
    }));

    const mounted = await mountWith([source, buildCallNode('call-1', 'child-a')], false, true, false, edges);
    await settleUntilRequestsStop(() => getLibraryWorkflowRecordMock.mock.calls.length);

    // Syncing a signature that accepts every connection removes nothing and says nothing.
    expect(mounted.readGraph().edges.map((edge) => edge.id)).toEqual(['edge-prompt', 'edge-negative_prompt']);
    expect(mounted.notifications.info).not.toHaveBeenCalled();

    return mounted;
  };
  const readCallNode = (graph: ProjectGraphState) => {
    const node = graph.nodes.find((candidate) => candidate.id === 'call-1');

    return node?.type === 'invocation' ? node : undefined;
  };

  it('keeps the connections a newly selected workflow still accepts and reports the ones it drops', async () => {
    const { notifications, readGraph, updateGraph } = await mountConnectedCall();

    selectWorkflow(readGraph, updateGraph, 'child-b');
    await settleUntilRequestsStop(() => getLibraryWorkflowRecordMock.mock.calls.length);

    expect(readCallNode(readGraph())?.data.callSavedWorkflowStatus).toBe('ready');
    expect(Object.keys(readCallNode(readGraph())?.data.dynamicInputTemplates ?? {})).toEqual([
      dynamicFieldName('prompt'),
    ]);
    expect(readGraph().edges.map((edge) => edge.id)).toEqual(['edge-prompt']);
    expect(notifications.info).toHaveBeenCalledTimes(1);
    expect(notifications.info).toHaveBeenCalledWith(expect.stringContaining('savedWorkflowDroppedEdges'));
  });

  it("removes the previous workflow's inputs and reports their connections when the new selection fails", async () => {
    const { notifications, readGraph, updateGraph } = await mountConnectedCall();

    selectWorkflow(readGraph, updateGraph, 'missing-child');
    await settleUntilRequestsStop(() => getLibraryWorkflowRecordMock.mock.calls.length);

    expect(readCallNode(readGraph())?.data.callSavedWorkflowStatus).toBe('error');
    expect(readCallNode(readGraph())?.data.dynamicInputTemplates).toEqual({});
    expect(readGraph().edges).toEqual([]);
    expect(notifications.info).toHaveBeenCalledTimes(1);
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
