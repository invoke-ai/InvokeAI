import type { PreparedLayerCacheReplacement } from '@workbench/canvas-engine/render/layerCache';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { describe, expect, it, vi } from 'vitest';

import { createCanvasMutationContext, type CanvasMutationContextDeps } from './mutationContext';

const action: CanvasProjectMutation = { id: 'layer-1', type: 'setCanvasSelectedLayer' };

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
    createLayerId: () => 'layer-new',
    dispatch: vi.fn(() => true),
    editOwner,
    editingLocked: lock,
    endBurst: () => undefined,
    getDocument: () => null,
    getReducerDocument: () => null,
    history: createHistory(),
    installPrepared: () => undefined,
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    preparePixels: (layerId, rect) => ({ layerId, rect, surface: {} }) as PreparedLayerCacheReplacement,
    refreshMirror: vi.fn(),
    ...overrides,
  };
  const context = createCanvasMutationContext(deps);
  return { context, deps, editOwner, lock };
};

describe('createCanvasMutationContext', () => {
  describe('document edit permits', () => {
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

  describe('dispatchPrepared', () => {
    it('throws when dispatch rejects the mutation and the reducer shows no postcondition', () => {
      const { context } = createHarness({ dispatch: () => false });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => false,
          () => false
        )
      ).toThrow('Canvas document mutation was rejected');
    });

    it('rethrows a dispatch error when the reducer did not apply the mutation', () => {
      const failure = new Error('subscriber exploded');
      const { context } = createHarness({
        dispatch: () => {
          throw failure;
        },
      });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => false,
          () => true
        )
      ).toThrow(failure);
    });

    it('swallows a dispatch error when the mutation is applied and mirrored', () => {
      const failure = new Error('subscriber exploded');
      const refreshMirror = vi.fn();
      const { context } = createHarness({
        dispatch: () => {
          throw failure;
        },
        refreshMirror,
      });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => true,
          () => true
        )
      ).not.toThrow();
      expect(refreshMirror).toHaveBeenCalledTimes(1);
    });

    it('reconciles the mirror after a dispatch error and succeeds once it converges', () => {
      const failure = new Error('subscriber exploded');
      let mirrored = false;
      const refreshMirror = vi.fn(() => {
        mirrored = true;
      });
      const { context } = createHarness({
        dispatch: () => {
          throw failure;
        },
        refreshMirror,
      });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => true,
          () => mirrored
        )
      ).not.toThrow();
      expect(refreshMirror).toHaveBeenCalledTimes(1);
    });

    it('rethrows the original dispatch error when the mirror never converges', () => {
      const failure = new Error('subscriber exploded');
      const refreshMirror = vi.fn();
      const { context } = createHarness({
        dispatch: () => {
          throw failure;
        },
        refreshMirror,
      });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => true,
          () => false
        )
      ).toThrow(failure);
      expect(refreshMirror).toHaveBeenCalledTimes(1);
    });

    it('swallows a refresh error when the mirror converged anyway', () => {
      const failure = new Error('subscriber exploded');
      let mirrored = false;
      const refreshMirror = vi.fn(() => {
        mirrored = true;
        throw new Error('refresh exploded');
      });
      const { context } = createHarness({
        dispatch: () => {
          throw failure;
        },
        refreshMirror,
      });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => true,
          () => mirrored
        )
      ).not.toThrow();
    });

    it('detects a reducer that rejects by returning unchanged state without throwing', () => {
      const { context } = createHarness();
      expect(() =>
        context.dispatchPrepared(
          action,
          () => false,
          () => false
        )
      ).toThrow('Canvas document mutation was rejected');
    });

    it('reconciles an unmirrored accepted mutation and throws when the mirror stays behind', () => {
      const refreshMirror = vi.fn();
      const { context } = createHarness({ refreshMirror });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => true,
          () => false
        )
      ).toThrow('Canvas document mutation was not mirrored');
      expect(refreshMirror).toHaveBeenCalledTimes(1);
    });

    it('accepts an unmirrored mutation once the mirror refresh converges', () => {
      let mirrored = false;
      const refreshMirror = vi.fn(() => {
        mirrored = true;
      });
      const { context } = createHarness({ refreshMirror });
      expect(() =>
        context.dispatchPrepared(
          action,
          () => true,
          () => mirrored
        )
      ).not.toThrow();
      expect(refreshMirror).toHaveBeenCalledTimes(1);
    });
  });

  describe('dispose', () => {
    it('stops tracking lock transitions so pre-dispose permits stay current', () => {
      const { context, lock } = createHarness();
      const permit = context.capturePermit()!;
      context.dispose();
      lock.setLocked(true);
      lock.setLocked(false);
      expect(context.isPermitCurrent(permit)).toBe(true);
    });
  });
});
