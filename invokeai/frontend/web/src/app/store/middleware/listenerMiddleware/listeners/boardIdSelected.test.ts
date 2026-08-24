import { configureStore, createListenerMiddleware } from '@reduxjs/toolkit';
import type { AppStartListening, RootState } from 'app/store/store';
import { $gallerySelection, resetGallerySelectionSource } from 'features/gallery/store/gallerySelectionSource';
import { selectGalleryItemNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  boardIdSelected,
  gallerySliceConfig,
  galleryViewChanged,
  imageSelected,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { api } from 'services/api';
import { galleryApi } from 'services/api/endpoints/gallery';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { addBoardIdSelectedListener } from './boardIdSelected';
import { addGallerySelectionSourceListener } from './gallerySelectionSource';

// The listener waits for the board's item list before auto-selecting, so the store needs the API
// slice present (in most tests here the query is never fulfilled — the point is what happens
// meanwhile). `withSelectionSource` also registers the listener that publishes selections to the
// viewer, for the tests that care whether the probe's own write reads as a user pick.
const buildStore = ({ withSelectionSource = false }: { withSelectionSource?: boolean } = {}) => {
  const listenerMiddleware = createListenerMiddleware();
  addBoardIdSelectedListener(listenerMiddleware.startListening as unknown as AppStartListening);
  if (withSelectionSource) {
    addGallerySelectionSourceListener(listenerMiddleware.startListening as unknown as AppStartListening);
  }
  return configureStore({
    reducer: {
      gallery: gallerySliceConfig.slice.reducer,
      [api.reducerPath]: api.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({ serializableCheck: false }).prepend(listenerMiddleware.middleware).concat(api.middleware),
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

  it('does not overwrite a selection made through the gallery grid either', () => {
    // Ctrl/shift-clicks and the delete flow's selection pruning dispatch selectionChanged, not
    // imageSelected, so matching on the action type alone leaves those paths exposed.
    const store = buildStore();

    store.dispatch(galleryViewChanged('images'));
    store.dispatch(selectionChanged(['picked.png']));

    return vi.advanceTimersByTimeAsync(6000).then(() => {
      expect(store.getState().gallery.selection).toEqual(['picked.png']);
    });
  });

  it('does not overwrite a multi-selection made while the probe was waiting', () => {
    const store = buildStore();

    store.dispatch(galleryViewChanged('images'));
    store.dispatch(selectionChanged(['first.png', 'second.png']));

    return vi.advanceTimersByTimeAsync(6000).then(() => {
      expect(store.getState().gallery.selection).toEqual(['first.png', 'second.png']);
    });
  });

  it('does not overwrite a multi-selection narrowed while the probe was waiting', () => {
    // Removing one of two selected thumbnails leaves the *last* selected item unchanged, so a
    // predicate watching only the active item never fired and the probe survived to replace the
    // whole selection when it woke.
    const store = buildStore();
    store.dispatch(selectionChanged(['first.png', 'second.png']));

    store.dispatch(galleryViewChanged('images'));
    store.dispatch(selectionChanged(['second.png']));

    return vi.advanceTimersByTimeAsync(6000).then(() => {
      expect(store.getState().gallery.selection).toEqual(['second.png']);
    });
  });

  it('does not overwrite a re-selection of the item already active', () => {
    // Same shape: the active item does not change, but the user has just said what they want.
    const store = buildStore();
    store.dispatch(imageSelected('a.png'));

    store.dispatch(galleryViewChanged('images'));
    store.dispatch(imageSelected('a.png'));

    return vi.advanceTimersByTimeAsync(6000).then(() => {
      expect(store.getState().gallery.selection).toEqual(['a.png']);
    });
  });

  it("auto-selects the board's first item without that reading as a user pick", async () => {
    // The probe picks *for* the user, so its write must not publish as a pick when it lands on the
    // item already displayed: NoBoardBoard re-dispatches boardIdSelected even when its board is
    // already selected, and the viewer answers a pick by lifting a running generation's progress
    // overlay off the item for two seconds — a stale flash for a click on the current board.
    resetGallerySelectionSource();
    const store = buildStore({ withSelectionSource: true });

    store.dispatch(boardIdSelected({ boardId: 'none' }));
    // Fulfil the item-name query the probe is waiting on, under the same cache key it computes.
    // The store here carries only the two slices this listener needs, so the selector — typed
    // against the whole RootState — has to be told that is enough.
    const queryArgs = selectGalleryItemNamesQueryArgs(store.getState() as unknown as RootState);
    // The upsert has to be flushed through the fake timers before it is awaited, or its fulfilled
    // action lands after the probe's 5s give-up.
    const upsert = store.dispatch(
      galleryApi.util.upsertQueryData('listGalleryItemNames', queryArgs, {
        item_names: ['already-showing.png'],
        starred_count: 0,
        total_count: 1,
      })
    );
    await vi.advanceTimersByTimeAsync(0);
    await upsert;
    await vi.advanceTimersByTimeAsync(6000);

    // The probe really does select for the user — this is also the only coverage of that path.
    expect(store.getState().gallery.selection).toEqual(['already-showing.png']);
    const generationAfterFirstProbe = $gallerySelection.get().generation;
    expect(generationAfterFirstProbe, 'moving the viewer to a new item is worth publishing').toBeGreaterThan(0);

    // Re-select the same board. The probe runs again and lands on the item already displayed.
    store.dispatch(boardIdSelected({ boardId: 'none' }));
    // The listener's `condition` only re-evaluates its predicate when an action is dispatched, so a
    // test that merely advances time would watch this second probe time out and clear the selection
    // — the give-up path, not the path under test. Any action wakes it; this one touches nothing.
    store.dispatch({ type: 'test/tick' });
    await vi.advanceTimersByTimeAsync(6000);

    expect(store.getState().gallery.selection).toEqual(['already-showing.png']);
    expect($gallerySelection.get().generation, 'nothing moved, so there is nothing to reveal').toBe(
      generationAfterFirstProbe
    );
  });
});
