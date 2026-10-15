import type { InvocationTemplatesSnapshot } from '@features/workflow/react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { ReactFlowProvider, useStoreApi, type Node } from '@xyflow/react';
/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- test injects stable fakes into the runtime */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { requestWorkflowFitView, workflowFitViewRequestStore, type WorkflowFlowInstance } from './flowInstanceStore';
import { requestNodeSelection, workflowSelectionStore } from './selectionStore';
import { WorkflowSelectionRequestRuntime } from './WorkflowSelectionRequestRuntime';

const templates = vi.hoisted(() => ({
  catalog: {} as Record<string, unknown>,
  listeners: new Set<() => void>(),
  status: 'loaded' as InvocationTemplatesSnapshot['status'],
}));

vi.mock('@features/workflow/react', () => ({
  getInvocationTemplatesSnapshot: () => ({ error: null, status: templates.status, templates: templates.catalog }),
  subscribeInvocationTemplates: (listener: () => void) => {
    templates.listeners.add(listener);
    return () => templates.listeners.delete(listener);
  },
}));

const SCOPE = { projectId: 'project-1', workflowId: 'workflow-1' };

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const waitForPendingSelection = () =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, 0);
  });

/** A ready fit lands two frames later, after the resize observer has measured the rebuilt nodes. */
const settleFrames = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
      })
  );

describe('WorkflowSelectionRequestRuntime', () => {
  let host: HTMLDivElement;
  let root: Root | null;

  beforeEach(() => {
    workflowSelectionStore.patchSnapshot({ hoveredNodeId: null, selectedNodeIds: [], selectionRequest: null });
    workflowFitViewRequestStore.setSnapshot({ request: null });
    templates.status = 'loaded';
    templates.catalog = {};
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    if (root) {
      await act(() => root?.unmount());
    }
    host.remove();
  });

  it('applies pending and later requests with current motion preferences, then unsubscribes', async () => {
    const fitView = vi.fn(() => Promise.resolve(true));
    const flowInstance = { fitView } as unknown as WorkflowFlowInstance;
    const selectNodes = vi.fn();
    requestNodeSelection(['pending-node']);

    await act(() => {
      root?.render(
        <ReactFlowProvider>
          <WorkflowSelectionRequestRuntime
            {...SCOPE}
            fitOnMount={null}
            flowInstance={flowInstance}
            isLargeGraph={false}
            reduceMotion={false}
            selectNodes={selectNodes}
          />
        </ReactFlowProvider>
      );
    });
    await act(waitForPendingSelection);

    expect(selectNodes).toHaveBeenLastCalledWith(['pending-node']);
    expect(fitView).toHaveBeenLastCalledWith({
      duration: 300,
      maxZoom: 1.25,
      nodes: [{ id: 'pending-node' }],
    });

    await act(() => {
      root?.render(
        <ReactFlowProvider>
          <WorkflowSelectionRequestRuntime
            {...SCOPE}
            fitOnMount={null}
            flowInstance={flowInstance}
            isLargeGraph={false}
            reduceMotion
            selectNodes={selectNodes}
          />
        </ReactFlowProvider>
      );
    });
    act(() => requestNodeSelection(['later-node']));

    expect(selectNodes).toHaveBeenLastCalledWith(['later-node']);
    expect(fitView).toHaveBeenLastCalledWith({ duration: 0, maxZoom: 1.25, nodes: [{ id: 'later-node' }] });

    await act(() => root?.unmount());
    root = null;
    act(() => requestNodeSelection(['after-unmount']));

    expect(selectNodes).toHaveBeenCalledTimes(2);
    expect(fitView).toHaveBeenCalledTimes(2);
  });

  const target = (id: string, x: number, y: number) => ({ id, position: { x, y } });
  const flowNode = (id: string, x: number, y: number, measured = true): Node => ({
    data: {},
    id,
    measured: measured ? { height: 40, width: 80 } : undefined,
    position: { x, y },
  });
  const fakeFlow = () => {
    const fitBounds = vi.fn(() => Promise.resolve(true));

    return { fitBounds, flowInstance: { fitBounds } as unknown as WorkflowFlowInstance };
  };

  let providerKey = 0;

  // `initialNodes` only seed a provider on mount, so each step renders a fresh provider.
  const renderFit = async (
    flowInstance: WorkflowFlowInstance,
    initialNodes: Node[],
    {
      fitOnMount = null,
      isLargeGraph = false,
      workflowId = SCOPE.workflowId,
    }: { fitOnMount?: ReturnType<typeof target>[] | null; isLargeGraph?: boolean; workflowId?: string } = {}
  ) => {
    providerKey += 1;
    await act(() => {
      root?.render(
        <ReactFlowProvider key={providerKey} initialNodes={initialNodes}>
          <WorkflowSelectionRequestRuntime
            fitOnMount={fitOnMount}
            flowInstance={flowInstance}
            isLargeGraph={isLargeGraph}
            projectId={SCOPE.projectId}
            reduceMotion
            selectNodes={vi.fn()}
            workflowId={workflowId}
          />
        </ReactFlowProvider>
      );
    });
    await act(waitForPendingSelection);
    await settleFrames();
  };

  it('fits a request once the flow shows the requested nodes measured at their positions, then clears it', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    requestWorkflowFitView(SCOPE, [target('a', 0, 0), target('b', 100, 60)]);

    await renderFit(flowInstance, [flowNode('a', 0, 0), flowNode('b', 100, 60)]);

    expect(fitBounds).toHaveBeenCalledWith({ height: 100, width: 180, x: 0, y: 0 }, { duration: 0 });
    expect(workflowFitViewRequestStore.getSnapshot().request).toBeNull();
  });

  it('waits while the flow still shows other nodes, stale positions, or unmeasured nodes', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    requestWorkflowFitView(SCOPE, [target('a', 10, 10)]);

    await renderFit(flowInstance, [flowNode('a', 0, 0)]);
    expect(fitBounds).not.toHaveBeenCalled();

    await renderFit(flowInstance, [flowNode('a', 10, 10, false)]);
    expect(fitBounds).not.toHaveBeenCalled();

    await renderFit(flowInstance, [flowNode('a', 10, 10), flowNode('b', 0, 0)]);
    expect(fitBounds).not.toHaveBeenCalled();
    expect(workflowFitViewRequestStore.getSnapshot().request).not.toBeNull();

    await renderFit(flowInstance, [flowNode('a', 10, 10)]);
    expect(fitBounds).toHaveBeenCalledTimes(1);
  });

  // The editor being replaced can show the same ids and positions (another copy of the same template).
  it('leaves a request for another workflow to that workflow’s editor', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    requestWorkflowFitView(SCOPE, [target('a', 0, 0)]);

    await renderFit(flowInstance, [flowNode('a', 0, 0)], { workflowId: 'workflow-2' });

    expect(fitBounds).not.toHaveBeenCalled();
    expect(workflowFitViewRequestStore.getSnapshot().request).not.toBeNull();
  });

  it('drops a fit request for an empty document instead of leaving it armed', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    requestWorkflowFitView(SCOPE, []);

    await renderFit(flowInstance, []);

    expect(fitBounds).not.toHaveBeenCalled();
    expect(workflowFitViewRequestStore.getSnapshot().request).toBeNull();
  });

  it('fits a newly shown workflow once on mount without animating, and not again as the flow changes', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    // A zero-duration fit moves the viewport synchronously, so the flow store notifies before the fit returns.
    fitBounds.mockImplementation(() => {
      templates.listeners.forEach((listener) => listener());
      return Promise.resolve(true);
    });

    await renderFit(flowInstance, [flowNode('a', 0, 0)], { fitOnMount: [target('a', 0, 0)] });

    expect(fitBounds).toHaveBeenCalledExactlyOnceWith({ height: 40, width: 80, x: 0, y: 0 }, { duration: 0 });

    await act(() => templates.listeners.forEach((listener) => listener()));
    await settleFrames();

    expect(fitBounds).toHaveBeenCalledTimes(1);
  });

  // Nodes rendered before templates arrive are placeholder cards that grow once their template is known.
  it('holds the fit until the node templates have settled', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    templates.status = 'loading';

    await renderFit(flowInstance, [flowNode('a', 0, 0)], { fitOnMount: [target('a', 0, 0)] });
    expect(fitBounds).not.toHaveBeenCalled();

    templates.status = 'error';
    await act(() => templates.listeners.forEach((listener) => listener()));
    await settleFrames();

    expect(fitBounds).toHaveBeenCalledTimes(1);
  });

  // Templates load before the rebuilt nodes carry them; the placeholder's size must not be what gets fitted.
  it('waits for a node with a known template to be rebuilt with it', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    const invocation = (template: object | null): Node => ({
      ...flowNode('a', 0, 0),
      data: { documentNode: { data: { type: 'noise' } }, template },
      type: 'invocation',
    });
    templates.catalog = { noise: {} };

    await renderFit(flowInstance, [invocation(null)], { fitOnMount: [target('a', 0, 0)] });
    expect(fitBounds).not.toHaveBeenCalled();

    await renderFit(flowInstance, [invocation({})], { fitOnMount: [target('a', 0, 0)] });
    expect(fitBounds).toHaveBeenCalledTimes(1);
  });

  it('gives up the mount fit once the user moves the view first', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    templates.status = 'loading';
    const captured: { store: ReturnType<typeof useStoreApi> | null } = { store: null };
    // No pane is mounted here, so the move is written where a pan would land: the flow store's transform.
    const CaptureStore = () => {
      const store = useStoreApi();

      useMountEffect(() => {
        captured.store = store;
      });

      return null;
    };

    await act(() => {
      root?.render(
        <ReactFlowProvider initialNodes={[flowNode('a', 0, 0)]}>
          <CaptureStore />
          <WorkflowSelectionRequestRuntime
            fitOnMount={[target('a', 0, 0)]}
            flowInstance={flowInstance}
            isLargeGraph={false}
            projectId={SCOPE.projectId}
            reduceMotion
            selectNodes={vi.fn()}
            workflowId={SCOPE.workflowId}
          />
        </ReactFlowProvider>
      );
    });
    await act(waitForPendingSelection);
    await act(() => captured.store?.setState({ transform: [200, 100, 1.5] }));

    templates.status = 'loaded';
    await act(() => templates.listeners.forEach((listener) => listener()));
    await settleFrames();

    expect(fitBounds).not.toHaveBeenCalled();
  });

  it('counts the unmeasured nodes of a large graph at a typical node size', async () => {
    const { fitBounds, flowInstance } = fakeFlow();
    const nodes = [flowNode('a', 0, 0), flowNode('far', 5000, 3000, false)];
    const targets = [target('a', 0, 0), target('far', 5000, 3000)];

    await renderFit(flowInstance, nodes, { fitOnMount: targets });
    expect(fitBounds).not.toHaveBeenCalled();

    await renderFit(flowInstance, nodes, { fitOnMount: targets, isLargeGraph: true });
    expect(fitBounds).toHaveBeenCalledWith({ height: 3160, width: 5288, x: 0, y: 0 }, { duration: 0 });
  });
});
