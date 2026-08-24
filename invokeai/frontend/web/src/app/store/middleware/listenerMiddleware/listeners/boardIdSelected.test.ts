import { configureStore, createListenerMiddleware } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
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
  // Records every gallery action the store sees, so a test can tell "the probe wrote the selection
  // without publishing it as a pick" from "the probe never wrote anything at all" — the resulting
  // state is the same either way.
  const galleryActions: string[] = [];
  const recorder = () => (next: (action: unknown) => unknown) => (action: unknown) => {
    const type = (action as { type?: string }).type;
    if (type?.startsWith('gallery/')) {
      galleryActions.push(type);
    }
    return next(action);
  };
  const store = configureStore({
    reducer: {
      gallery: gallerySliceConfig.slice.reducer,
      [api.reducerPath]: api.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({ serializableCheck: false })
        .prepend(listenerMiddleware.middleware)
        .concat(recorder)
        .concat(api.middleware),
  });
  return Object.assign(store, { galleryActions });
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

  it("auto-selects the board's first item, and leaves a still-valid selection alone", async () => {
    // Both halves of the probe's job. It must select *for* the user when the viewer has nothing
    // valid to show — but a board or view "change" that changed nothing (clicking the board you
    // are already on, or the tab already showing: both dispatch unconditionally) must not replace
    // the user's pick with the newest item. That would discard their selection, and moving the
    // displayed item mid-generation also lifts the progress overlay off it for two seconds.
    resetGallerySelectionSource();
    const store = buildStore({ withSelectionSource: true });

    store.dispatch(boardIdSelected({ boardId: 'none' }));
    // Fulfil the item-name query the probe is waiting on. The upsert has to be flushed through the
    // fake timers before it is awaited, or its fulfilled action lands after the probe's 5s give-up.
    const upsert = store.dispatch(
      galleryApi.util.upsertQueryData('listGalleryItemNames', selectGalleryItemNamesQueryArgs(store.getState()), {
        item_names: ['newest.png', 'older.png'],
        starred_count: 0,
        total_count: 2,
      })
    );
    await vi.advanceTimersByTimeAsync(0);
    await upsert;
    await vi.advanceTimersByTimeAsync(6000);

    // Nothing was selected, so the probe picks the first item — the only coverage of that path.
    expect(store.getState().gallery.selection).toEqual(['newest.png']);
    const generationAfterFirstProbe = $gallerySelection.get().generation;
    expect(generationAfterFirstProbe, 'moving the viewer to a new item is worth publishing').toBeGreaterThan(0);

    // The user scrolls down and picks something else, then clicks the board they are already on.
    store.dispatch(imageSelected('older.png'));
    const generationAfterUserPick = $gallerySelection.get().generation;
    store.galleryActions.length = 0;
    store.dispatch(boardIdSelected({ boardId: 'none' }));
    await vi.advanceTimersByTimeAsync(6000);

    // `older.png` is still in this list, so the probe has nothing to fix and writes nothing at all
    // — not even a no-op write, which would still have to be checked for a spurious reveal. (The
    // click itself shows up in galleryActions; what must not is a write to the selection.)
    expect(store.galleryActions).not.toContain('gallery/selectionChanged');
    expect(store.galleryActions).not.toContain('gallery/imageSelected');
    expect(store.getState().gallery.selection).toEqual(['older.png']);
    expect($gallerySelection.get().generation, 'nothing moved, so there is nothing to reveal').toBe(
      generationAfterUserPick
    );
  });

  it('does not report an empty board as the user picking something', async () => {
    // The probe writes for the user, so it uses the mutation action rather than `imageSelected`.
    // Everywhere else that is belt-and-braces — the check above means any write it does make moves
    // the displayed item, which publishes under either action. Here is where the choice shows:
    // an empty board with nothing selected is a write that changes nothing, and as `imageSelected`
    // it would still publish, telling the viewer the user asked to see something.
    resetGallerySelectionSource();
    const store = buildStore({ withSelectionSource: true });

    store.dispatch(boardIdSelected({ boardId: 'none' }));
    const upsert = store.dispatch(
      galleryApi.util.upsertQueryData('listGalleryItemNames', selectGalleryItemNamesQueryArgs(store.getState()), {
        item_names: [],
        starred_count: 0,
        total_count: 0,
      })
    );
    await vi.advanceTimersByTimeAsync(0);
    await upsert;
    await vi.advanceTimersByTimeAsync(6000);

    expect(store.getState().gallery.selection).toEqual([]);
    expect($gallerySelection.get().generation, 'nothing to show is not a selection').toBe(0);
  });

  it('still selects for the user when the viewer has nothing left to show', async () => {
    // The mirror case, and the reason the "did anything change?" test lives in the probe rather
    // than in each click handler: deleting the last item, or the probe's own give-up, leaves the
    // selection empty, and clicking the board is how the user asks for something to look at. A
    // handler that simply swallowed the click for the already-selected board made that inert.
    const store = buildStore();

    store.dispatch(boardIdSelected({ boardId: 'none' }));
    const upsert = store.dispatch(
      galleryApi.util.upsertQueryData('listGalleryItemNames', selectGalleryItemNamesQueryArgs(store.getState()), {
        item_names: ['newest.png'],
        starred_count: 0,
        total_count: 1,
      })
    );
    await vi.advanceTimersByTimeAsync(0);
    await upsert;
    await vi.advanceTimersByTimeAsync(6000);
    store.dispatch(selectionChanged([]));

    // Clicking the board again, with the list already cached.
    store.dispatch(boardIdSelected({ boardId: 'none' }));
    await vi.advanceTimersByTimeAsync(6000);

    expect(store.getState().gallery.selection).toEqual(['newest.png']);
  });

  it('does not clear a good selection just because the store went quiet', async () => {
    // `condition` only re-evaluates on a dispatched action, so with the list already cached and
    // nothing else happening there is no wake-up at all: the probe used to sit through its 5s
    // deadline and then clear the selection it should have kept.
    const store = buildStore();

    store.dispatch(boardIdSelected({ boardId: 'none' }));
    const upsert = store.dispatch(
      galleryApi.util.upsertQueryData('listGalleryItemNames', selectGalleryItemNamesQueryArgs(store.getState()), {
        item_names: ['a.png', 'b.png'],
        starred_count: 0,
        total_count: 2,
      })
    );
    await vi.advanceTimersByTimeAsync(0);
    await upsert;
    await vi.advanceTimersByTimeAsync(6000);
    store.dispatch(imageSelected('b.png'));

    // A second board click, then silence — no further dispatch to wake `condition`.
    store.dispatch(boardIdSelected({ boardId: 'none' }));
    await vi.advanceTimersByTimeAsync(6000);

    expect(store.getState().gallery.selection).toEqual(['b.png']);
  });
});
