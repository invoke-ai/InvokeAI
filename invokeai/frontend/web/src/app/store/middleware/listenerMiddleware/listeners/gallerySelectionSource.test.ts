import { configureStore, createListenerMiddleware } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import {
  $gallerySelection,
  markNextSelectionAutoSwitched,
  resetGallerySelectionSource,
} from 'features/gallery/store/gallerySelectionSource';
import {
  boardIdSelected,
  gallerySliceConfig,
  imageSelected,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { beforeEach, describe, expect, it } from 'vitest';

import { addGallerySelectionSourceListener } from './gallerySelectionSource';

const buildStore = () => {
  const listenerMiddleware = createListenerMiddleware();
  addGallerySelectionSourceListener(listenerMiddleware.startListening as unknown as AppStartListening);
  return configureStore({
    reducer: { gallery: gallerySliceConfig.slice.reducer },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware().prepend(listenerMiddleware.middleware),
  });
};

describe('addGallerySelectionSourceListener', () => {
  beforeEach(() => {
    resetGallerySelectionSource();
  });

  it('publishes a click made through the gallery grid', () => {
    const store = buildStore();
    store.dispatch(imageSelected('clicked.png'));
    expect($gallerySelection.get()).toMatchObject({ name: 'clicked.png', isAutoSwitch: false });
  });

  it('does not publish a multi-select mutation that leaves the active item in place', () => {
    // Active item b, selection [a, b]: ctrl-clicking `a` off the selection dispatches
    // selectionChanged([b]). Nothing the viewer shows changes — publishing it would flash the
    // progress overlay off for bookkeeping aimed at a different item.
    const store = buildStore();
    store.dispatch(selectionChanged(['a.png', 'b.png']));
    const beforeDeselect = $gallerySelection.get().generation;

    store.dispatch(selectionChanged(['b.png']));

    expect($gallerySelection.get().generation).toBe(beforeDeselect);
  });

  it('publishes a multi-select mutation that moves the active item', () => {
    // Ctrl-clicking an unselected item appends it and makes it active. selectionChanged is not in
    // the pick list, so this relies on the change-of-active-item clause.
    const store = buildStore();
    store.dispatch(imageSelected('a.png'));
    store.dispatch(selectionChanged(['a.png', 'b.png']));
    expect($gallerySelection.get()).toMatchObject({ name: 'b.png', isAutoSwitch: false });
  });

  it('publishes a re-selection of the item already active as a new selection', () => {
    // Nothing in the state changes, so a state-transition-only predicate would miss it — and the
    // viewer would have no way to make a repeat click on the displayed item visible.
    const store = buildStore();
    store.dispatch(imageSelected('a.png'));
    const first = $gallerySelection.get().generation;
    store.dispatch(imageSelected('a.png'));
    expect($gallerySelection.get().generation).toBeGreaterThan(first);
  });

  it('attributes an auto-switch that carries its own board change', () => {
    const store = buildStore();
    markNextSelectionAutoSwitched();
    store.dispatch(boardIdSelected({ boardId: 'other', select: { selection: ['auto.png'], galleryView: 'images' } }));
    expect($gallerySelection.get()).toMatchObject({ name: 'auto.png', isAutoSwitch: true });
  });

  it('publishes a selection cleared by an action it does not name', () => {
    // logout is not in the action list; the active-item clause is what covers it.
    const store = buildStore();
    store.dispatch(imageSelected('a.png'));
    store.dispatch({ type: 'auth/logout' });
    expect($gallerySelection.get().name).toBeNull();
  });

  it('does not treat a bare board click as the user picking the item that stays selected', () => {
    // Clicking a board in the boards list dispatches boardIdSelected with no selection payload and
    // leaves the selection alone. Counting it would make the viewer reveal an item the user never
    // clicked, over the live progress preview.
    const store = buildStore();
    store.dispatch(imageSelected('a.png'));
    const beforeBoardClick = $gallerySelection.get().generation;

    store.dispatch(boardIdSelected({ boardId: 'some-other-board' }));

    expect($gallerySelection.get().generation).toBe(beforeBoardClick);
  });
});
