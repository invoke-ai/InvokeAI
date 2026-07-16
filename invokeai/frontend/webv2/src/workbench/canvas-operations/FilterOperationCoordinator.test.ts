import type { FilterOperationSession } from '@workbench/canvas-operations/contracts';

import { describe, expect, it, vi } from 'vitest';

import { createFilterOperationCoordinator } from './FilterOperationCoordinator';
import { createCanvasOperationStores } from './operationStores';

const layer = {
  filter: null,
  id: 'layer-1',
  isEnabled: true,
  isLocked: false,
  name: 'Layer 1',
  type: 'raster',
} as const;

const createSession = (): FilterOperationSession => ({
  blockCommit: vi.fn(),
  cancel: vi.fn(),
  commit: vi.fn(() => Promise.resolve('committed' as const)),
  dispose: vi.fn(),
  getSnapshot: vi.fn(() => ({ layerId: layer.id, status: 'ready' }) as never),
  interruptProcessing: vi.fn(),
  process: vi.fn(() => Promise.resolve('published' as const)),
  reset: vi.fn(),
  setAutoProcess: vi.fn(),
  subscribe: vi.fn(() => () => undefined),
  updateDraft: vi.fn(),
});

const createHarness = () => {
  const session = createSession();
  const stores = createCanvasOperationStores();
  const deps = {
    captureGuard: vi.fn(() => ({ layer, layerId: layer.id }) as never),
    clearOtherOperation: vi.fn(),
    clearPreview: vi.fn(),
    controller: {
      getSnapshot: vi.fn(() => ({ status: 'idle' })),
      subscribe: vi.fn(() => () => undefined),
    },
    createSession: vi.fn(() => session),
    getDocument: vi.fn<() => any>(() => ({ layers: [layer], selectedLayerId: null })),
    getInitialDraft: vi.fn(() => ({ settings: {}, type: 'canny' })),
    getSessionDeps: vi.fn(() => ({})),
    isInteractionLocked: vi.fn(() => false),
    selectLayer: vi.fn(),
    setViewTool: vi.fn(),
    stores,
  };
  return { coordinator: createFilterOperationCoordinator(deps as never), deps, session, stores };
};

describe('FilterOperationCoordinator', () => {
  it('owns session start, publication, and cancellation', () => {
    const { coordinator, deps, session, stores } = createHarness();

    expect(coordinator.start(layer.id)).toBe('started');
    expect(deps.clearOtherOperation).toHaveBeenCalledOnce();
    expect(deps.selectLayer).toHaveBeenCalledWith(layer.id);
    expect(deps.setViewTool).toHaveBeenCalledOnce();
    expect(stores.filterSession.get()).toEqual({ layerId: layer.id, status: 'ready' });

    coordinator.cancel();
    expect(session.dispose).toHaveBeenCalledOnce();
    expect(stores.filterSession.get()).toBeNull();
  });

  it('delegates mutation, process, and commit to the owned session', async () => {
    const { coordinator, session } = createHarness();
    coordinator.start(layer.id);

    expect(coordinator.updateDraft({ settings: { strength: 1 }, type: 'canny' })).toBe('updated');
    expect(coordinator.setAutoProcess(true)).toBe('updated');
    expect(await coordinator.process()).toBe('completed');
    expect(await coordinator.commit('apply', vi.fn())).toBe('committed');

    expect(session.updateDraft).toHaveBeenCalledOnce();
    expect(session.setAutoProcess).toHaveBeenCalledWith(true);
    expect(session.process).toHaveBeenCalledOnce();
    expect(session.commit).toHaveBeenCalledWith('apply');
  });

  it('rejects missing, unsupported, disabled, locked, and busy starts deterministically', () => {
    const { coordinator, deps } = createHarness();
    deps.getDocument.mockReturnValueOnce(null);
    expect(coordinator.start(layer.id)).toBe('missing');
    deps.getDocument.mockReturnValueOnce({ layers: [{ ...layer, type: 'inpaint_mask' }], selectedLayerId: null });
    expect(coordinator.start(layer.id)).toBe('unsupported');
    deps.getDocument.mockReturnValueOnce({ layers: [{ ...layer, isEnabled: false }], selectedLayerId: null });
    expect(coordinator.start(layer.id)).toBe('disabled');
    deps.getDocument.mockReturnValueOnce({ layers: [{ ...layer, isLocked: true }], selectedLayerId: null });
    expect(coordinator.start(layer.id)).toBe('locked');
    deps.isInteractionLocked.mockReturnValueOnce(true);
    expect(coordinator.start(layer.id)).toBe('locked');
  });
});
