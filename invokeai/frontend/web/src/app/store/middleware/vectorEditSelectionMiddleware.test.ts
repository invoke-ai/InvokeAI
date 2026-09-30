import { configureStore } from '@reduxjs/toolkit';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { vectorEditSelectionMiddleware } from './vectorEditSelectionMiddleware';

const source = { id: 'source', type: 'vector_layer' } as const;
const target = { id: 'target', type: 'raster_layer' } as const;

const setup = (hasSession: boolean) => {
  let pending: (() => void) | undefined;
  const requestEditExit = vi.fn((action: () => void) => {
    pending = action;
  });
  $canvasManager.set({
    tool: {
      tools: {
        path: {
          $editSession: { get: () => (hasSession ? { entityIdentifier: source } : null) },
          requestEditExit,
        },
      },
    },
  } as unknown as CanvasManager);
  const store = configureStore({
    reducer: (state: CanvasEntityIdentifier = source, action) =>
      entitySelected.match(action) ? action.payload.entityIdentifier : state,
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().prepend(vectorEditSelectionMiddleware),
  });
  return { store, requestEditExit, confirm: () => pending?.() };
};

afterEach(() => $canvasManager.set(null));

describe('vector edit layer selection middleware', () => {
  it('keeps the current layer selected until the pending action is confirmed', () => {
    const { store, requestEditExit, confirm } = setup(true);
    store.dispatch(entitySelected({ entityIdentifier: target }));
    expect(store.getState()).toEqual(source);
    expect(requestEditExit).toHaveBeenCalledOnce();
    confirm();
    expect(store.getState()).toEqual(target);
  });

  it('does not prompt for selecting the edited layer again', () => {
    const { store, requestEditExit } = setup(true);
    store.dispatch(entitySelected({ entityIdentifier: source }));
    expect(requestEditExit).not.toHaveBeenCalled();
    expect(store.getState()).toEqual(source);
  });

  it('selects immediately when no edit session is active', () => {
    const { store, requestEditExit } = setup(false);
    store.dispatch(entitySelected({ entityIdentifier: target }));
    expect(requestEditExit).not.toHaveBeenCalled();
    expect(store.getState()).toEqual(target);
  });

  it('passes unrelated actions through without prompting', () => {
    const { store, requestEditExit } = setup(true);
    store.dispatch({ type: 'unrelated' });
    expect(requestEditExit).not.toHaveBeenCalled();
    expect(store.getState()).toEqual(source);
  });
});
