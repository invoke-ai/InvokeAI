import type { Project } from '@workbench/projectContracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createGalleryBoardReferenceCheck } from './WorkbenchRuntime';
import { createDraftProject } from './workbenchState';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const withGallery = (project: Project, values: Record<string, unknown>, queueBoardIds: string[] = []): Project => {
  const state = { ...createInitialWorkbenchState(), activeProjectId: project.id, projects: [project] };
  const [patched] = workbenchReducer(state, {
    projectId: project.id,
    type: 'patchWidgetValues',
    values,
    widgetId: 'gallery',
  }).projects;

  return {
    ...patched!,
    queue: {
      ...patched!.queue,
      items: queueBoardIds.map(
        (galleryBoardId, index) =>
          ({
            snapshot: { galleryBoardId },
            status: index === 0 ? 'pending' : 'completed',
          }) as Project['queue']['items'][number]
      ),
    },
  };
};

const createStore = (projects: Project[], hasHydrated = true) => {
  let snapshot = { hasHydrated, projects };
  const listeners = new Set<() => void>();
  const reconcileDeletedBoardOutcome = vi.fn();

  return {
    commands: { gallery: { reconcileDeletedBoardOutcome } },
    getSnapshot: () => snapshot,
    reconcileDeletedBoardOutcome,
    set: (next: Partial<typeof snapshot>) => {
      snapshot = { ...snapshot, ...next };
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };
};

const settle = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
const reconciled = (store: ReturnType<typeof createStore>) =>
  store.reconcileDeletedBoardOutcome.mock.calls.map(([outcome]) => (outcome as { boardId: string }).boardId);

beforeEach(() => {
  accountLifecycle.activate('board-reference-check-test');
});

describe('createGalleryBoardReferenceCheck', () => {
  it('clears the boards loaded projects name that the server says are gone, asking about each once', async () => {
    const first = createDraftProject([]);
    const second = createDraftProject([first]);
    const store = createStore(
      [
        withGallery(first, { autoAddBoardId: 'kept', projectBoardId: 'own-inbox', selectedBoardId: 'gone' }),
        // Its own inbox is not asked about, nor the board a finished result went to.
        withGallery(second, { projectBoardId: 'own-inbox-2', selectedBoardId: 'own-inbox-2' }, ['gone', 'finished']),
      ],
      false
    );
    const boardExists = vi.fn((boardId: string) =>
      boardId === 'unreachable' ? Promise.reject(new TypeError('offline')) : Promise.resolve(boardId === 'kept')
    );

    createGalleryBoardReferenceCheck(store as never, boardExists);
    // Nothing is asked before the session is loaded.
    expect(boardExists).not.toHaveBeenCalled();

    store.set({ hasHydrated: true });
    await settle();

    expect(boardExists.mock.calls.map(([boardId]) => boardId).sort()).toEqual(['gone', 'kept']);
    expect(reconciled(store)).toEqual(['gone']);
    expect(store.reconcileDeletedBoardOutcome).toHaveBeenCalledWith(
      expect.objectContaining({ deletedImageNames: [], deletedVideoNames: [] })
    );

    // A project opened later is checked as it arrives; an unanswered question clears nothing.
    const third = withGallery(createDraftProject([first, second]), { autoAddBoardId: 'unreachable' }, ['also-gone']);
    boardExists.mockClear();
    store.set({ projects: [...store.getSnapshot().projects, third] });
    await settle();

    expect(boardExists.mock.calls.map(([boardId]) => boardId).sort()).toEqual(['also-gone', 'unreachable']);
    expect(reconciled(store)).toEqual(['gone', 'also-gone']);
  });

  it('asks again for a project loaded later, or reopened, rather than trusting an earlier answer', async () => {
    const first = withGallery(createDraftProject([]), { autoAddBoardId: 'shared-board' });
    const later = withGallery(createDraftProject([first]), { selectedBoardId: 'shared-board' });
    const store = createStore([first]);
    let exists = true;
    const boardExists = vi.fn(() => Promise.resolve(exists));

    createGalleryBoardReferenceCheck(store as never, boardExists);
    await settle();
    expect(reconciled(store)).toEqual([]);

    // The board goes, with a project deleted in this session; a project opened afterwards names it.
    exists = false;
    store.set({ projects: [first, later] });
    await settle();
    expect(reconciled(store)).toEqual(['shared-board']);

    // Closed and opened again, the first project is asked about afresh.
    store.set({ projects: [later] });
    store.set({ projects: [later, first] });
    await settle();
    expect(boardExists).toHaveBeenCalledTimes(3);
  });

  it('acts on no answer that arrives after the workbench is gone', async () => {
    const project = withGallery(createDraftProject([]), { selectedBoardId: 'gone' });
    const store = createStore([project]);
    let answer: (exists: boolean) => void = () => undefined;
    const check = createGalleryBoardReferenceCheck(
      store as never,
      () =>
        new Promise((resolve) => {
          answer = resolve;
        })
    );

    check.dispose();
    answer(false);
    await settle();

    expect(store.reconcileDeletedBoardOutcome).not.toHaveBeenCalled();
  });
});
