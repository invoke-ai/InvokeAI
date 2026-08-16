import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph, serializeWorkflowJson } from '@features/workflow/utility';
import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { registerLibraryGraphSyncedHandler, releaseLibraryGraphSyncedHandler } from './librarySyncBridge';
import { useSaveWorkflowToLibrary } from './useSaveWorkflowToLibrary';

const { createLibraryWorkflowMock, invalidateWorkflowLibraryCacheMock, updateLibraryWorkflowMock } = vi.hoisted(() => ({
  createLibraryWorkflowMock: vi.fn(),
  invalidateWorkflowLibraryCacheMock: vi.fn(),
  updateLibraryWorkflowMock: vi.fn(),
}));

vi.mock('@features/workflow/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  createLibraryWorkflow: createLibraryWorkflowMock,
  invalidateWorkflowLibraryCache: invalidateWorkflowLibraryCacheMock,
  updateLibraryWorkflow: updateLibraryWorkflowMock,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  return {
    port: {
      getSnapshot: () => snapshot,
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

/**
 * Regression coverage for the "save as new" echo-autosave bug: on the create
 * path, `bindLibraryWorkflow` synchronously adds `libraryWorkflowId` to the
 * stored project graph, but the hook used to mark the autosaver's baseline
 * with the JSON it had serialized *before* the bind — which
 * `serializeWorkflowJson` does not include `id` (added only once
 * `libraryWorkflowId` is set). The library autosaver's own `read()`
 * re-serializes the *current* (now-bound) graph, which does include `id`, so
 * the mismatch looked like a dirty edit and queued a redundant PUT of
 * otherwise-identical content on the next debounce. The fix re-serializes
 * from the post-bind store snapshot before marking synced.
 */
describe('useSaveWorkflowToLibrary bind-then-sync', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    createLibraryWorkflowMock.mockReset();
    createLibraryWorkflowMock.mockResolvedValue('library-workflow-99');
    updateLibraryWorkflowMock.mockReset();
    invalidateWorkflowLibraryCacheMock.mockReset();
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('marks the autosaver baseline with the post-bind serialized graph, matching what read() will produce', async () => {
    const initialGraph = createProjectGraph('workflow-1');
    const project = createMutablePort({
      galleryValues: {},
      graphHistory: [],
      id: 'project-1',
      isWorkflowRunning: false,
      projectGraph: initialGraph,
      workflowValues: {},
    });

    const bindLibraryWorkflow = vi.fn((libraryWorkflowId: string) => {
      project.setSnapshot({
        ...project.port.getSnapshot(),
        projectGraph: { ...project.port.getSnapshot().projectGraph, libraryWorkflowId },
      });
    });

    // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop -- intentionally stable for this render lifetime
    const adapter = {
      commands: {
        bindLibraryWorkflow,
        editGraph: vi.fn(),
        redo: vi.fn(),
        replace: vi.fn(),
        restoreSnapshot: vi.fn(),
        saveSnapshot: vi.fn(),
        undo: vi.fn(),
      },
      notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
      project: project.port,
    } as unknown as WorkflowUiAdapter;

    let saveAsNew: (() => Promise<string | null>) | null = null;
    const Harness = () => {
      const hook = useSaveWorkflowToLibrary();

      useEffect(() => {
        saveAsNew = hook.saveAsNew;
      });

      return null;
    };

    await act(() => {
      root.render(
        <WorkflowUiProvider adapter={adapter}>
          <Harness />
        </WorkflowUiProvider>
      );
    });

    const syncedCalls: Record<string, unknown>[] = [];
    const handler = (serialized: Record<string, unknown>) => syncedCalls.push(serialized);

    registerLibraryGraphSyncedHandler(handler);

    try {
      await act(async () => {
        await saveAsNew?.();
      });

      expect(bindLibraryWorkflow).toHaveBeenCalledWith('library-workflow-99');
      expect(syncedCalls).toHaveLength(1);

      // What the library autosaver's own read() would produce right now,
      // from the bound store — the synced baseline must match this exactly
      // (same content AND key order, since the autosaver dedupes on the
      // stringified JSON) or the bind alone is read back as a dirty edit.
      const expectedPostBindSerialized = serializeWorkflowJson(project.port.getSnapshot().projectGraph);

      expect(JSON.stringify(syncedCalls[0])).toBe(JSON.stringify(expectedPostBindSerialized));
      expect(syncedCalls[0]).toMatchObject({ id: 'library-workflow-99' });
    } finally {
      releaseLibraryGraphSyncedHandler(handler);
    }
  });
});
