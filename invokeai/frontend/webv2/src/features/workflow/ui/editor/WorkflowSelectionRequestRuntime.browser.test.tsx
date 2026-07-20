/* eslint-disable react-perf/jsx-no-new-object-as-prop -- test injects stable fakes into the runtime */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkflowFlowInstance } from './flowInstanceStore';

import { requestNodeSelection, workflowSelectionStore } from './selectionStore';
import { WorkflowSelectionRequestRuntime } from './WorkflowSelectionRequestRuntime';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const waitForPendingSelection = () =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, 0);
  });

describe('WorkflowSelectionRequestRuntime', () => {
  let host: HTMLDivElement;
  let root: Root | null;

  beforeEach(() => {
    workflowSelectionStore.patchSnapshot({ hoveredNodeId: null, selectedNodeIds: [], selectionRequest: null });
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
        <WorkflowSelectionRequestRuntime flowInstance={flowInstance} reduceMotion={false} selectNodes={selectNodes} />
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
        <WorkflowSelectionRequestRuntime flowInstance={flowInstance} reduceMotion selectNodes={selectNodes} />
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
});
