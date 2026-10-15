import type { SelectionState, SelectionStateDeps } from '@workbench/canvas-engine/selection/selectionState';

import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createTestEditConcurrency } from '@workbench/canvas-engine/editConcurrency.testStub';
import { createHistory } from '@workbench/canvas-engine/history/history';
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
  restore: vi.fn(),
  selectAll: vi.fn(),
  snapshot: vi.fn(() => ({ alpha: null, bounds: null, commits: [], rect: null, selected: false })),
});

const createTextOptions = () => ({
  canEdit: () => true,
  captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
  commitStructural: vi.fn(),
  createLayerId: () => 'text-1',
  getDocument: () => null,
  invalidate: vi.fn(),
  colors: { get: () => ({ background: '#ffffff', foreground: '#000000' }) },
  isGestureActive: () => false,
  options: { get: () => ({}) as never },
  session: { get: () => null, set: vi.fn() },
});

const createTransformOptions = () => ({
  backend: {} as never,
  canEdit: () => true,
  commitStructural: vi.fn(),
  ctx: {} as never,
  getCache: () => null,
  getDocument: () => null,
  invalidate: vi.fn(),
  isGestureActive: () => false,
  reportRefusal: vi.fn(),
  restoreCache: vi.fn(),
  session: { get: () => null, set: vi.fn() },
  setOverride: vi.fn(),
});

const createSelectionPixelOptions = () => ({
  applyImagePatch: vi.fn(),
  backend: {} as never,
  beginPixelEdit: () => null,
  canEdit: () => true,
  ctx: {} as never,
  deleteDerived: vi.fn(),
  getDocument: () => null,
  getFillColor: () => '#000',
  invalidateLayer: vi.fn(),
  isRasterCacheReady: () => true,
  isGestureActive: () => false,
  layers: {} as never,
  markDirty: vi.fn(),
  notifyPainted: vi.fn(),
  reportRefusal: vi.fn(),
  requestRasterization: vi.fn(),
});

const createFloatingSelectionOptions = () => ({
  applyImagePatch: vi.fn(),
  backend: {} as never,
  ctx: {} as never,
  getDocument: () => null,
  invalidateLayer: vi.fn(),
  layers: {} as never,
  markDirty: vi.fn(),
  notifyPainted: vi.fn(),
  onChange: vi.fn(),
  reportRefusal: vi.fn(),
  suspendPersistence: () => () => undefined,
});

const createSelectionImageOptions = () => ({
  concurrency: createTestEditConcurrency({ capturePermit: () => null, isPermitCurrent: () => false }),
  decodeImage: vi.fn(),
  getDocument: () => null,
  isGuardCurrent: () => false,
});

describe('EditingController', () => {
  it('owns selection state and invalidates exclusive leases with document lifecycle', () => {
    const selection = createSelection();
    const createSelectionState = vi.fn((_deps: SelectionStateDeps) => selection);
    const controller = new EditingController({
      floatingSelection: createFloatingSelectionOptions(),
      getDocument: () => null,
      history: createHistory(),
      selection: {} as SelectionStateDeps,
      selectionPixels: createSelectionPixelOptions(),
      selectionImage: createSelectionImageOptions(),
      createSelectionState,
      text: createTextOptions(),
      transform: createTransformOptions(),
    });

    // The exposed selection records history; floats and document swaps access the underlying state.
    controller.selection.clear();
    expect(selection.clear).toHaveBeenCalledTimes(1);
    controller.discardSelection();
    expect(selection.clear).toHaveBeenCalledTimes(2);
    expect(controller.floatingSelection).toBeDefined();
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
      floatingSelection: createFloatingSelectionOptions(),
      getDocument: () => null,
      history: createHistory(),
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
