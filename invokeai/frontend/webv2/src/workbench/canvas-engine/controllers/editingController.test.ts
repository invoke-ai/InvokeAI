import type { SelectionState, SelectionStateDeps } from '@workbench/canvas-engine/selection/selectionState';

import { describe, expect, it, vi } from 'vitest';

import { EditingController } from './editingController';

const createSelection = (): SelectionState => ({
  antsPaths: () => [],
  bounds: () => null,
  clear: vi.fn(),
  commit: vi.fn(),
  containsPoint: () => false,
  dispose: vi.fn(),
  hasSelection: () => false,
  invert: vi.fn(),
  mask: () => null,
  replaceMask: vi.fn(),
  selectAll: vi.fn(),
});

const createTextOptions = () => ({
  canEdit: () => true,
  commitStructural: vi.fn(),
  createLayerId: () => 'text-1',
  getDocument: () => null,
  invalidate: vi.fn(),
  isGestureActive: () => false,
  options: { get: () => ({}) as never },
  session: { get: () => null, set: vi.fn() },
});

const createTransformOptions = () => ({
  backend: {} as never,
  canEdit: () => true,
  dispatch: vi.fn(),
  endBurst: vi.fn(),
  getCache: () => null,
  getDocument: () => null,
  invalidate: vi.fn(),
  isGestureActive: () => false,
  pushHistory: vi.fn(),
  replaceCache: vi.fn(),
  restoreCache: vi.fn(),
  session: { get: () => null, set: vi.fn() },
  setOverride: vi.fn(),
});

const createSelectionPixelOptions = () => ({
  applyImagePatch: vi.fn(),
  backend: {} as never,
  beginControlEdit: () => null,
  canEdit: () => true,
  deleteDerived: vi.fn(),
  endBurst: vi.fn(),
  getDocument: () => null,
  getFillColor: () => '#000',
  history: {} as never,
  invalidateLayer: vi.fn(),
  isGestureActive: () => false,
  layers: {} as never,
  markDirty: vi.fn(),
  notifyPainted: vi.fn(),
});

const createSelectionImageOptions = () => ({
  capturePermit: () => null,
  decodeImage: vi.fn(),
  getDocument: () => null,
  isGestureActive: () => false,
  isGuardCurrent: () => false,
  isPermitCurrent: () => false,
});

describe('EditingController', () => {
  it('owns selection state and invalidates exclusive leases with document lifecycle', () => {
    const selection = createSelection();
    const createSelectionState = vi.fn((_deps: SelectionStateDeps) => selection);
    const controller = new EditingController({
      getDocument: () => null,
      selection: {} as SelectionStateDeps,
      selectionPixels: createSelectionPixelOptions(),
      selectionImage: createSelectionImageOptions(),
      createSelectionState,
      text: createTextOptions(),
      transform: createTransformOptions(),
    });

    expect(controller.selection).toBe(selection);
    const lease = controller.edits.tryAcquire({ kind: 'filter', layerId: 'layer-1' });
    expect(lease?.isCurrent()).toBe(true);

    controller.invalidateDocument();
    expect(lease?.signal.aborted).toBe(true);
    expect(lease?.isCurrent()).toBe(false);
    expect(controller.edits.tryAcquire({ kind: 'filter' })?.isCurrent()).toBe(true);
  });

  it('disposes selection and leases idempotently and cannot reactivate afterward', () => {
    const selection = createSelection();
    const controller = new EditingController({
      getDocument: () => null,
      selection: {} as SelectionStateDeps,
      selectionPixels: createSelectionPixelOptions(),
      selectionImage: createSelectionImageOptions(),
      createSelectionState: () => selection,
      text: createTextOptions(),
      transform: createTransformOptions(),
    });
    const lease = controller.edits.tryAcquire({ kind: 'select-object' });

    controller.dispose();
    controller.dispose();
    controller.activate();

    expect(selection.dispose).toHaveBeenCalledTimes(1);
    expect(lease?.signal.aborted).toBe(true);
    expect(controller.edits.tryAcquire({ kind: 'filter' })).toBeNull();
  });
});
