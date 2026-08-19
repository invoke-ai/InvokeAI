import { configureStore, createListenerMiddleware } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { gallerySliceConfig, galleryViewChanged, imageSelected } from 'features/gallery/store/gallerySlice';
import { api } from 'services/api';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { addBoardIdSelectedListener } from './boardIdSelected';

// The listener waits for the board's item list before auto-selecting, so the store needs the API
// slice present (the query is never fulfilled here — the point is what happens meanwhile).
const buildStore = () => {
  const listenerMiddleware = createListenerMiddleware();
  addBoardIdSelectedListener(listenerMiddleware.startListening as unknown as AppStartListening);
  return configureStore({
    reducer: {
      gallery: gallerySliceConfig.slice.reducer,
      [api.reducerPath]: api.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({ serializableCheck: false }).prepend(listenerMiddleware.middleware),
  });
};

describe('addBoardIdSelectedListener', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('does not overwrite a selection made while it was waiting for the board list', async () => {
    // The gallery's auto-switch dispatches galleryViewChanged immediately before imageSelected.
    // Without the cancel, the probe this starts wakes up on that very selection and re-selects
    // from a stale (or empty) list, undoing the auto-switch — and the viewer then reveals the
    // wrong item over the live preview, the flash the auto-switch marker exists to prevent.
    const store = buildStore();

    store.dispatch(galleryViewChanged('images'));
    store.dispatch(imageSelected('new.png'));

    // Past the probe's 5s give-up, which would otherwise clear the selection outright.
    await vi.advanceTimersByTimeAsync(6000);

    expect(store.getState().gallery.selection).toEqual(['new.png']);
  });

  it('still clears the selection when a board switch finds nothing to show', async () => {
    // The auto-select probe itself must keep working: a board change with no items selects
    // nothing rather than leaving the previous board's item highlighted.
    const store = buildStore();
    store.dispatch(imageSelected('from-previous-board.png'));

    store.dispatch(galleryViewChanged('assets'));
    await vi.advanceTimersByTimeAsync(6000);

    expect(store.getState().gallery.selection).toEqual([]);
  });
});
