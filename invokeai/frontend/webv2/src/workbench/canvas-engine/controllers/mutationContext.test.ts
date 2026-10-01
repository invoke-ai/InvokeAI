import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createHistory, NO_HELD_ASSET_REFS } from '@workbench/canvas-engine/history/history';
import { describe, expect, it, vi } from 'vitest';

import { createCanvasMutationContext, type CanvasMutationContextDeps } from './mutationContext';

interface LockStore {
  get(): boolean;
  subscribe(listener: () => void): () => void;
  setLocked(next: boolean): void;
}

const createLockStore = (): LockStore => {
  let locked = false;
  const listeners: (() => void)[] = [];
  return {
    get: () => locked,
    setLocked: (next) => {
      locked = next;
      for (const listener of listeners.slice()) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.push(listener);
      return () => {
        const index = listeners.indexOf(listener);
        if (index >= 0) {
          listeners.splice(index, 1);
        }
      };
    },
  };
};

const createHarness = (overrides: Partial<CanvasMutationContextDeps> = {}) => {
  const lock = createLockStore();
  const editOwner = Symbol('test-edit-owner');
  const deps: CanvasMutationContextDeps = {
    commitEdit: vi.fn(),
    createLayerId: () => 'layer-new',
    dispatch: vi.fn(() => true),
    editOwner,
    projectId: 'p1',
    editingLocked: lock,
    getDocument: () => null,
    getReducerDocument: () => null,
    history: createHistory(),
    installPrepared: () => undefined,
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    preparePixels: (layerId, rect) => ({ layerId, rect, surface: {} }) as PreparedLayerCacheReplacement,
    refreshMirror: vi.fn(),
    reserveRaster: () => ({ lease: { release: () => undefined }, status: 'ok' }),
    subscribeReducer: () => () => undefined,
    ...overrides,
  };
  const context = createCanvasMutationContext(deps);
  return { context, deps, editOwner, lock };
};

describe('createCanvasMutationContext', () => {
  describe('document edit permits', () => {
    it('exposes the engine project id on the concurrency surface', () => {
      const { context } = createHarness();

      expect(context.projectId).toBe('p1');
      expect(context.getEditRevision()).toBe(0);
    });

    it('captures a permit while unlocked and keeps it current until the lock transitions', () => {
      const { context } = createHarness();
      const permit = context.capturePermit();
      expect(permit).not.toBeNull();
      expect(context.canEdit()).toBe(true);
      expect(context.isPermitCurrent(permit!)).toBe(true);
    });

    it('refuses anonymous permits while locked but grants them to the edit owner', () => {
      const { context, editOwner, lock } = createHarness();
      lock.setLocked(true);
      expect(context.canEdit()).toBe(false);
      expect(context.capturePermit()).toBeNull();
      expect(context.capturePermit(Symbol('someone-else'))).toBeNull();
      expect(context.canEdit(editOwner)).toBe(true);
      expect(context.capturePermit(editOwner)).not.toBeNull();
    });

    it('invalidates an anonymous permit after the lock toggles on and back off', () => {
      const { context, lock } = createHarness();
      const permit = context.capturePermit()!;
      lock.setLocked(true);
      lock.setLocked(false);
      expect(context.canEdit()).toBe(true);
      expect(context.isPermitCurrent(permit)).toBe(false);
    });

    it('reports anonymous permits stale while the lock is held', () => {
      const { context, lock } = createHarness();
      const permit = context.capturePermit()!;
      lock.setLocked(true);
      expect(context.isPermitCurrent(permit)).toBe(false);
    });

    it('keeps an owner-held permit current across lock transitions', () => {
      const { context, editOwner, lock } = createHarness();
      const permit = context.capturePermit(editOwner)!;
      lock.setLocked(true);
      expect(context.isPermitCurrent(permit)).toBe(true);
      lock.setLocked(false);
      expect(context.isPermitCurrent(permit)).toBe(true);
    });
  });

  describe('transactions', () => {
    const entry = (bytes = 10) => ({ bytes, heldAssetRefs: NO_HELD_ASSET_REFS, redo: vi.fn(), undo: vi.fn() });
    const select = (id: string): CanvasProjectMutation => ({ id, type: 'setCanvasSelectedLayer' });

    /** A reducer whose documents record the selected id; the mirror follows unless told to lag. */
    const createDocuments = () => {
      const state = {
        dispatchError: null as Error | null,
        mirror: { selectedLayerId: null } as unknown as CanvasDocumentContractV3,
        mirrorLags: false,
        reducer: { selectedLayerId: null } as unknown as CanvasDocumentContractV3,
        rejects: false,
      };
      const dispatch = vi.fn((mutation: CanvasProjectMutation) => {
        if (!state.rejects && mutation.type === 'setCanvasSelectedLayer') {
          state.reducer = { selectedLayerId: mutation.id } as unknown as CanvasDocumentContractV3;
          if (!state.mirrorLags) {
            state.mirror = state.reducer;
          }
        }
        if (state.dispatchError) {
          throw state.dispatchError;
        }
        return !state.rejects;
      });
      return {
        deps: {
          dispatch,
          getDocument: () => state.mirror,
          getReducerDocument: () => state.reducer,
          refreshMirror: vi.fn(),
        },
        state,
      };
    };
    const selected = (id: string) => (document: CanvasDocumentContractV3 | null) => document?.selectedLayerId === id;

    const begin = (context: ReturnType<typeof createHarness>['context'], historyBytes = 10) => {
      const txn = context.begin({ historyBytes });
      if (!('publish' in txn)) {
        throw new Error(`refused: ${txn.status}`);
      }
      return txn;
    };

    it('refuses before anything is admitted or dispatched', () => {
      const documents = createDocuments();
      const history = createHistory({ byteBudget: 100 });
      const { context, deps, lock } = createHarness({ ...documents.deps, history });

      expect(context.begin({ historyBytes: 101 })).toEqual({ status: 'over-budget' });
      lock.setLocked(true);
      expect(context.begin({ historyBytes: 10 })).toEqual({ status: 'busy' });
      lock.setLocked(false);
      const gesture = createHarness({ ...documents.deps, isGestureActive: () => true });
      expect(gesture.context.begin({ historyBytes: 10 })).toEqual({ status: 'gesture-active' });
      const empty = createHarness({ getReducerDocument: () => null });
      expect(empty.context.begin({ historyBytes: 10 })).toEqual({ status: 'not-ready' });
      context.dispose();
      expect(context.begin({ historyBytes: 10 })).toEqual({ status: 'not-ready' });

      expect(deps.dispatch).not.toHaveBeenCalled();
      expect(history.canUndo()).toBe(false);
    });

    it('publishes a verified step once, records it and routes a user edit after it landed', () => {
      const documents = createDocuments();
      const history = createHistory();
      const order: string[] = [];
      const { context, deps } = createHarness({
        ...documents.deps,
        commitEdit: vi.fn(() => order.push('route')),
        history,
      });
      const txn = begin(context);

      const result = txn.publish(
        'Select',
        {
          accepted: selected('a'),
          install: () => order.push('install'),
          mutation: select('a'),
          notify: () => order.push('notify'),
        },
        entry()
      );
      txn.end();

      expect(result.status).toBe('committed');
      expect(history.entries().past).toEqual(['Select']);
      expect(deps.dispatch).toHaveBeenCalledWith(select('a'), 'system');
      expect(order).toEqual(['install', 'route', 'notify']);
      expect(context.historyTop()).toBe('token' in result ? result.token : null);
    });

    it('routes only user edits, never system-originated ones or history replays', async () => {
      const documents = createDocuments();
      const history = createHistory();
      const { context, deps } = createHarness({ ...documents.deps, history });
      const system = begin(context);
      system.publish('Select', { accepted: selected('a'), mutation: select('a') }, entry(), { origin: 'system' });
      system.end();
      expect(deps.commitEdit).not.toHaveBeenCalled();

      const user = begin(context);
      user.publish(
        'Select',
        { accepted: selected('b'), mutation: select('b') },
        {
          ...entry(),
          redo: () => context.applyStep({ accepted: selected('b'), mutation: select('b') }),
          undo: () => context.applyStep({ accepted: selected('a'), mutation: select('a') }),
        }
      );
      user.end();
      expect(deps.commitEdit).toHaveBeenCalledOnce();
      await history.undo();
      await history.redo();
      expect(deps.commitEdit).toHaveBeenCalledOnce();
      expect(documents.deps.dispatch).toHaveBeenLastCalledWith(select('b'), 'system');
    });

    it('records nothing when the reducer leaves the document unchanged', () => {
      const documents = createDocuments();
      documents.state.rejects = true;
      const history = createHistory();
      const { context, deps } = createHarness({ ...documents.deps, history });
      const txn = begin(context);

      expect(txn.publish('Select', { accepted: selected('a'), mutation: select('a') }, entry())).toEqual({
        status: 'dispatch-rejected',
      });
      txn.end();
      expect(history.canUndo()).toBe(false);
      expect(deps.commitEdit).not.toHaveBeenCalled();
    });

    it('rolls back an accepted mutation whose postconditions fail and reports an unmirrored rollback', () => {
      const documents = createDocuments();
      const report = vi.fn();
      const { context } = createHarness({ ...documents.deps, report });
      const step = {
        accepted: () => false,
        mutation: select('a'),
        rollback: { mutation: select('origin'), restored: selected('origin') },
      };

      const reverted = begin(context);
      expect(reverted.publish('Select', step, entry())).toEqual({
        recovered: 'reverted',
        status: 'postcondition-failed',
      });
      reverted.end();
      expect(documents.state.reducer.selectedLayerId).toBe('origin');
      expect(report).not.toHaveBeenCalled();

      documents.state.mirrorLags = true;
      const unmirrored = begin(context);
      expect(unmirrored.publish('Select', step, entry())).toEqual({
        recovered: 'reverted-unmirrored',
        status: 'postcondition-failed',
      });
      unmirrored.end();
      expect(report).toHaveBeenCalledOnce();
    });

    it('rolls back when the document cannot be read after the mutation', () => {
      const documents = createDocuments();
      let unreadable = false;
      const { context } = createHarness({
        ...documents.deps,
        dispatch: (mutation) => {
          const dispatched = documents.deps.dispatch(mutation);
          unreadable = mutation.type === 'setCanvasSelectedLayer' && mutation.id === 'a';
          return dispatched;
        },
        getReducerDocument: () => {
          if (unreadable) {
            throw new Error('state read failed');
          }
          return documents.state.reducer;
        },
      });
      const txn = begin(context);

      expect(
        txn.publish(
          'Select',
          {
            accepted: selected('a'),
            mutation: select('a'),
            rollback: { mutation: select('origin'), restored: selected('origin') },
          },
          entry()
        )
      ).toEqual({ recovered: 'reverted', status: 'postcondition-failed' });
      txn.end();
      expect(documents.state.reducer.selectedLayerId).toBe('origin');
    });

    it('accepts a landed mutation whose observer threw once the mirror reconciles', () => {
      const documents = createDocuments();
      documents.state.dispatchError = new Error('observer exploded');
      documents.state.mirrorLags = true;
      documents.deps.refreshMirror.mockImplementation(() => {
        documents.state.mirror = documents.state.reducer;
      });
      const { context } = createHarness(documents.deps);
      const txn = begin(context);

      expect(txn.publish('Select', { accepted: selected('a'), mutation: select('a') }, entry()).status).toBe(
        'committed'
      );
      txn.end();
      expect(documents.deps.refreshMirror).toHaveBeenCalledOnce();
    });

    it('contains routing and notification failures once the step is recorded', () => {
      const documents = createDocuments();
      const history = createHistory();
      const { context } = createHarness({
        ...documents.deps,
        commitEdit: () => {
          throw new Error('router exploded');
        },
        history,
      });
      const txn = begin(context);

      const result = txn.publish(
        'Select',
        {
          accepted: selected('a'),
          mutation: select('a'),
          notify: () => {
            throw new Error('listener exploded');
          },
        },
        entry()
      );
      txn.end();
      expect(result.status).toBe('committed');
      expect(history.canUndo()).toBe(true);
    });

    it('refuses publication once the permit is stale, without dispatching', () => {
      const documents = createDocuments();
      const { context, deps, lock } = createHarness(documents.deps);
      const txn = begin(context);
      lock.setLocked(true);
      lock.setLocked(false);

      expect(txn.publish('Select', { mutation: select('a') }, entry())).toEqual({ status: 'busy' });
      txn.end();
      expect(deps.dispatch).not.toHaveBeenCalled();
    });

    it('grows its admission for a larger entry and refuses one history could never keep', () => {
      const documents = createDocuments();
      const history = createHistory({ byteBudget: 100 });
      const { context, deps } = createHarness({ ...documents.deps, history });
      const txn = begin(context, 10);

      expect(txn.growHistory(50)).toBe(true);
      expect(txn.publish('Select', { mutation: select('a') }, entry(101))).toEqual({ status: 'over-budget' });
      expect(deps.dispatch).not.toHaveBeenCalled();
      expect(txn.publish('Select', { accepted: selected('a'), mutation: select('a') }, entry(60)).status).toBe(
        'committed'
      );
      txn.end();
    });

    it('keeps its admission when an observer of the step ends it mid-publication', () => {
      const documents = createDocuments();
      const history = createHistory();
      let txn: ReturnType<typeof begin> | null = null;
      const { context } = createHarness({
        ...documents.deps,
        dispatch: (mutation) => {
          const dispatched = documents.deps.dispatch(mutation);
          txn?.end();
          return dispatched;
        },
        history,
      });
      txn = begin(context);

      expect(txn.publish('Select', { accepted: selected('a'), mutation: select('a') }, entry()).status).toBe(
        'committed'
      );
      expect(history.entries().past).toEqual(['Select']);
    });

    it('releases admission, raster reservations and held resources when it ends', () => {
      const documents = createDocuments();
      const history = createHistory({ byteBudget: 100 });
      const lease = { release: vi.fn() };
      const held = { release: vi.fn() };
      const reserveRaster = vi.fn((bytes: number) =>
        bytes > 50
          ? ({ availableBytes: 50, requestedBytes: bytes, status: 'over-budget' } as const)
          : ({ lease, status: 'ok' } as const)
      );
      const { context } = createHarness({ ...documents.deps, history, reserveRaster });
      const txn = begin(context, 100);

      expect(txn.reserveRaster(60)).toBe(false);
      expect(txn.reserveRaster(40)).toBe(true);
      txn.hold(held);
      expect(context.begin({ historyBytes: 1 })).toEqual({ status: 'over-budget' });
      txn.end();
      txn.end();

      expect(lease.release).toHaveBeenCalledOnce();
      expect(held.release).toHaveBeenCalledOnce();
      expect('publish' in context.begin({ historyBytes: 100 })).toBe(true);
    });

    it('coalesces only into the entry it names while that entry is still newest', () => {
      const documents = createDocuments();
      const history = createHistory();
      const { context } = createHarness({ ...documents.deps, history });
      const first = begin(context);
      const recorded = first.publish('Nudge', { accepted: selected('a'), mutation: select('a') }, entry());
      first.end();
      const token = 'token' in recorded ? recorded.token : undefined;

      const second = begin(context);
      expect(
        second.publish('Nudge', { accepted: selected('b'), mutation: select('b') }, entry(), { replacing: token })
          .status
      ).toBe('committed');
      second.end();
      expect(history.entries().past).toEqual(['Nudge']);

      const stale = begin(context);
      expect(stale.publish('Nudge', { mutation: select('c') }, entry(), { replacing: token })).toEqual({
        status: 'busy',
      });
      stale.end();
    });

    it('refuses edits while a replay is running', async () => {
      const documents = createDocuments();
      const history = createHistory();
      const { context } = createHarness({ ...documents.deps, history });
      let finish!: () => void;
      const txn = begin(context);
      txn.publish(
        'Slow',
        { accepted: selected('a'), mutation: select('a') },
        {
          ...entry(),
          undo: () =>
            new Promise<void>((resolve) => {
              finish = resolve;
            }),
        }
      );
      txn.end();

      const replay = history.undo();
      expect(context.canEdit()).toBe(false);
      expect(context.begin({ historyBytes: 1 })).toEqual({ status: 'busy' });
      finish();
      await replay;
      expect(context.canEdit()).toBe(true);
    });
  });

  describe('dispose', () => {
    it('makes every permit stale and refuses later edits', () => {
      const { context } = createHarness();
      const permit = context.capturePermit()!;
      context.dispose();
      expect(context.isPermitCurrent(permit)).toBe(false);
      expect(context.canEdit()).toBe(false);
    });
  });
});

describe('edit revision', () => {
  it('advances once per reducer document identity and never on notification alone', () => {
    let document = { layers: [] } as unknown as CanvasDocumentContractV3;
    const listeners: (() => void)[] = [];
    const { context } = createHarness({
      getReducerDocument: () => document,
      subscribeReducer: (listener) => {
        listeners.push(listener);
        return () => undefined;
      },
    });

    expect(context.getEditRevision()).toBe(0);
    listeners.forEach((listener) => listener());
    expect(context.getEditRevision()).toBe(0);

    document = { ...document };
    listeners.forEach((listener) => listener());
    expect(context.getEditRevision()).toBe(1);

    document = { ...document };
    expect(context.getEditRevision()).toBe(2);
  });
});
