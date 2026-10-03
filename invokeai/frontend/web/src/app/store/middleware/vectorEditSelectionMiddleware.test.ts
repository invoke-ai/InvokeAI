import { configureStore } from '@reduxjs/toolkit';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { canvasReset } from 'features/controlLayers/store/actions';
import {
  allEntitiesDeleted,
  canvasSliceConfig,
  canvasSnapshotRestored,
  entityDeleted,
  entityReset,
  entitySelected,
  rasterLayerAdded,
} from 'features/controlLayers/store/canvasSlice';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import { getInitialCanvasState } from 'features/controlLayers/store/types';
import { getRasterLayerState, getVectorLayerState } from 'features/controlLayers/store/util';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { vectorEditSelectionMiddleware } from './vectorEditSelectionMiddleware';

const source = { id: 'source', type: 'vector_layer' } as const;
const target = { id: 'target', type: 'raster_layer' } as const;

const setup = (hasSession: boolean) => {
  let sessionActive = hasSession;
  const acceptEditSession = vi.fn(() => {
    sessionActive = false;
  });
  let pending: (() => void) | undefined;
  const requestEditExit = vi.fn((action: () => void) => {
    pending = action;
  });
  $canvasManager.set({
    tool: {
      tools: {
        path: {
          $editSession: { get: () => (sessionActive ? { entityIdentifier: source } : null) },
          requestEditExit,
          acceptEditSession,
        },
      },
    },
  } as unknown as CanvasManager);
  const initial = getInitialCanvasState();
  initial.selectedEntityIdentifier = source;
  initial.vectorLayers.entities.push(getVectorLayerState(source.id));
  initial.rasterLayers.entities.push(getRasterLayerState(target.id));
  const store = configureStore({
    reducer: (state = { canvas: { present: initial } }, action) => ({
      canvas: { present: canvasSliceConfig.slice.reducer(state.canvas.present, action) },
    }),
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().prepend(vectorEditSelectionMiddleware),
  });
  return {
    store,
    requestEditExit,
    acceptEditSession,
    confirm: () => {
      acceptEditSession();
      pending?.();
    },
  };
};

afterEach(() => $canvasManager.set(null));

describe('vector edit layer selection middleware', () => {
  it('keeps the current layer selected until the pending action is confirmed', () => {
    const { store, requestEditExit, confirm } = setup(true);
    store.dispatch(entitySelected({ entityIdentifier: target }));
    expect(store.getState().canvas.present.selectedEntityIdentifier).toEqual(source);
    expect(requestEditExit).toHaveBeenCalledOnce();
    confirm();
    expect(store.getState().canvas.present.selectedEntityIdentifier).toEqual(target);
  });

  it('does not prompt for selecting the edited layer again', () => {
    const { store, requestEditExit } = setup(true);
    store.dispatch(entitySelected({ entityIdentifier: source }));
    expect(requestEditExit).not.toHaveBeenCalled();
    expect(store.getState().canvas.present.selectedEntityIdentifier).toEqual(source);
  });

  it('selects immediately when no edit session is active', () => {
    const { store, requestEditExit } = setup(false);
    store.dispatch(entitySelected({ entityIdentifier: target }));
    expect(requestEditExit).not.toHaveBeenCalled();
    expect(store.getState().canvas.present.selectedEntityIdentifier).toEqual(target);
  });

  it('passes unrelated actions through without prompting', () => {
    const { store, requestEditExit } = setup(true);
    store.dispatch({ type: 'unrelated' });
    expect(requestEditExit).not.toHaveBeenCalled();
    expect(store.getState().canvas.present.selectedEntityIdentifier).toEqual(source);
  });

  it.each([
    entityDeleted({ entityIdentifier: source }),
    entityReset({ entityIdentifier: source }),
    allEntitiesDeleted(),
    canvasReset(),
    canvasSnapshotRestored(getInitialCanvasState()),
  ])('ends the session for $type', (action) => {
    const { store, acceptEditSession } = setup(true);
    store.dispatch(action);
    expect(acceptEditSession).toHaveBeenCalledOnce();
  });

  it('does not end editing when another layer is deleted', () => {
    const { store, acceptEditSession } = setup(true);
    store.dispatch(entityDeleted({ entityIdentifier: target }));
    expect(acceptEditSession).not.toHaveBeenCalled();
  });

  it('ends editing if a new layer automatically becomes selected', () => {
    const { store, acceptEditSession } = setup(true);
    store.dispatch(rasterLayerAdded({ isSelected: true }));
    expect(acceptEditSession).toHaveBeenCalledOnce();
  });
});
