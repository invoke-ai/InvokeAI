import { configureStore, createListenerMiddleware } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { autoSwitchedImages } from 'features/gallery/store/autoSwitchedImages';
import {
  boardIdSelected,
  gallerySliceConfig,
  imageSelected,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { beforeEach, describe, expect, it } from 'vitest';

import { addAutoSwitchedSelectionListener } from './autoSwitchedSelection';

// A store with the real gallery reducer and the real listener, so the predicate is exercised
// against actual selection-writing actions rather than a hand-built state pair.
const buildStore = () => {
  const listenerMiddleware = createListenerMiddleware();
  addAutoSwitchedSelectionListener(listenerMiddleware.startListening as unknown as AppStartListening);
  return configureStore({
    reducer: { gallery: gallerySliceConfig.slice.reducer },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().prepend(listenerMiddleware.middleware),
  });
};

describe('addAutoSwitchedSelectionListener', () => {
  beforeEach(() => {
    // The marker is a module singleton; drop anything a previous test left on it.
    autoSwitchedImages.settle(null);
  });

  it('keeps the marker when the auto-switch selection lands', () => {
    const store = buildStore();
    autoSwitchedImages.record('a.png');
    store.dispatch(imageSelected('a.png'));
    expect(autoSwitchedImages.consume('a.png')).toBe(true);
  });

  it('drops the marker once the user selects something else', () => {
    // The dead click this exists to prevent: the auto-switch to A never rendered because the user
    // clicked B first, so their later click on A must still get its reveal.
    const store = buildStore();
    autoSwitchedImages.record('a.png');
    store.dispatch(imageSelected('a.png'));
    store.dispatch(imageSelected('b.png'));
    store.dispatch(imageSelected('a.png'));
    expect(autoSwitchedImages.consume('a.png')).toBe(false);
  });

  it('settles on every action that writes the selection, not just imageSelected', () => {
    const store = buildStore();

    autoSwitchedImages.record('a.png');
    store.dispatch(imageSelected('a.png'));
    store.dispatch(selectionChanged(['b.png']));
    expect(autoSwitchedImages.consume('a.png')).toBe(false);

    autoSwitchedImages.record('c.png');
    store.dispatch(imageSelected('c.png'));
    store.dispatch(boardIdSelected({ boardId: 'other', select: { selection: ['d.png'], galleryView: 'images' } }));
    expect(autoSwitchedImages.consume('c.png')).toBe(false);
  });

  it('leaves the marker alone when an action does not move the selection', () => {
    const store = buildStore();
    autoSwitchedImages.record('a.png');
    store.dispatch(imageSelected('a.png'));
    // Selecting the same item again, and a board switch that carries no selection, must not
    // discard a marker whose image has not rendered yet.
    store.dispatch(imageSelected('a.png'));
    store.dispatch(boardIdSelected({ boardId: 'other' }));
    expect(autoSwitchedImages.consume('a.png')).toBe(true);
  });
});
