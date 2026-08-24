import type { UnknownAction } from '@reduxjs/toolkit';
import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening, RootState } from 'app/store/store';
import {
  selectGalleryItemNamesQueryArgs,
  selectGalleryView,
  selectLastSelectedItem,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { galleryApi } from 'services/api/endpoints/gallery';

/** The actions that ask this listener to pick an item for the user. */
const startsProbe = isAnyOf(boardIdSelected, galleryViewChanged);

/**
 * Whether this navigation has nothing for the probe to do: it did not actually change the board or
 * the view, and the item the viewer is showing is in the list that board/view is already displaying.
 *
 * Clicking the board you are already on, or the tab already showing, dispatches regardless
 * (NoBoardBoard and the view tabs don't guard, unlike GalleryBoard and VirtualBoardItem). The probe
 * picks the list's first item unconditionally, so left alone it throws the user's selection away —
 * and moving the displayed item mid-generation also lifts the progress overlay off it for a couple
 * of seconds.
 *
 * Both halves are load-bearing:
 *
 * - Comparing the *navigation* against the previous state, not just the item against the list: a
 *   real board switch can land on a list that contains the displayed item, which virtual date
 *   boards guarantee — their query args drop `board_id` and filter on `created_date` alone, so the
 *   list is a superset of every board's items for that day. Skipping that strands the viewer on the
 *   previous board's item.
 * - Requiring the displayed item to be in the list: the selection can be non-empty and still not be
 *   in what the grid shows — a search term narrows the list without starting a probe, and so do
 *   `starredFirst`, `orderDir` and the archived-boards toggle. Clicking the board is how the user
 *   gets unstuck from that, so the click must fall through and re-pick. Likewise when the selection
 *   is empty, after deleting the last item or hiding date boards while one is selected.
 *
 * Answered synchronously off the cached list rather than after the query wait, because that wait is
 * not safe for a click that should do nothing: `condition` re-evaluates only on a dispatched
 * action, so a quiet store lets its 5s deadline expire and the give-up branch clears the selection.
 * An uncached list simply falls through and probes as before.
 */
const isNoOpNavigation = (action: UnknownAction, state: RootState, previousState: RootState): boolean => {
  const changedNothing =
    (boardIdSelected.match(action) && selectSelectedBoardId(previousState) === action.payload.boardId) ||
    (galleryViewChanged.match(action) && selectGalleryView(previousState) === action.payload);

  if (!changedNothing) {
    return false;
  }

  const activeItem = selectLastSelectedItem(state);
  const cached = galleryApi.endpoints.listGalleryItemNames.select(selectGalleryItemNamesQueryArgs(state))(state);
  return !!activeItem && !!cached.data?.item_names.includes(activeItem);
};

export const addBoardIdSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    // Two jobs, so this cannot be a plain action matcher. The probe below is started by a board or
    // view change — but it must also be *cancelled* by any selection that lands while it waits,
    // and a selection arrives through several actions: imageSelected from the gallery's auto-switch,
    // plain thumbnail clicks and next/prev navigation, selectionChanged from ctrl/shift-clicks, the
    // grid's own arrow-key handler, the delete flow's pruning and this listener's probe,
    // boardIdSelected carrying a selection. Matching the resulting change of the selection covers
    // all of them, including any writer added later — an action list would silently miss it.
    //
    // The whole selection, not just its active item: removing one of several selected thumbnails,
    // or re-picking the one already active, leaves the last item unchanged while still being the
    // user settling what they want. Comparing only that item left the probe running through those,
    // to overwrite their selection when it woke. The state is immutable, so a new array reference
    // is exactly "the selection was written", and cancelling a probe more often than strictly
    // needed costs nothing.
    predicate: (action, currentState, previousState) =>
      startsProbe(action) || selectSelection(currentState) !== selectSelection(previousState),
    effect: async (action, { getState, getOriginalState, dispatch, condition, cancelActiveListeners }) => {
      // Decided before cancelling anything, and before the first await (the only point
      // getOriginalState is valid at). A navigation that changes nothing has no business killing
      // the probe of one that did: the app dispatches `boardIdSelected` and `galleryViewChanged`
      // back to back in several places (deleting the selected board, uploading to another board),
      // and the second of those pairs is routinely a no-op — cancelling there and returning left
      // nothing to select the new board's item, stranding the viewer on the old one.
      if (startsProbe(action) && !(boardIdSelected.match(action) && action.payload.select)) {
        if (isNoOpNavigation(action, getState(), getOriginalState())) {
          return;
        }
      }

      // Cancel any in-progress instances of this listener, we don't want to select an item from a previous board
      cancelActiveListeners();

      if (!startsProbe(action)) {
        // A selection landed. It settles what should be displayed, so a probe still waiting on a
        // board's items must not overwrite it when it resolves. The gallery's auto-switch dispatches
        // galleryViewChanged immediately before its selection: without this the probe that view
        // change starts wakes on the selection that follows it, re-selects the first name in a
        // possibly stale cached list, and undoes the switch — and the viewer then reveals that
        // wrong image over the live preview, which is the flash the auto-switch marker exists to
        // prevent. Cancelling above is the whole effect; there is nothing to auto-select here.
        return;
      }

      if (boardIdSelected.match(action) && action.payload.select) {
        // This action already has a resource selection - skip the below auto-selection logic
        return;
      }

      // The grid is backed by the polymorphic listGalleryItemNames endpoint (the legacy
      // getImageNames query is no longer dispatched), so the auto-select probe must read its
      // cache or it will time out and clear the user's selection on every board switch. The
      // selector already maps a virtual board id to its `created_date` filter.
      const selectQuery = galleryApi.endpoints.listGalleryItemNames.select(selectGalleryItemNamesQueryArgs(getState()));
      // wait until the board has some items - maybe it already has some from a previous fetch
      // must use getState() to ensure we do not have stale state
      const isSuccess = await condition(() => selectQuery(getState()).isSuccess, 5000);

      if (!isSuccess) {
        dispatch(selectionChanged([]));
        return;
      }

      // the board was just changed - we can select the first gallery item (image or video)
      const itemNames = selectQuery(getState()).data?.item_names;

      // The probe picks *for* the user, so it writes with the mutation action rather than
      // `imageSelected`. The state is identical either way, but `imageSelected` means "the user
      // asked to see this", which the viewer answers by lifting the progress overlay. This write
      // always moves the displayed item (the check above returned otherwise), so it still
      // publishes through the change-of-active-item clause. See gallerySelectionSource.
      const firstItemName = itemNames?.[0];

      dispatch(selectionChanged(firstItemName ? [firstItemName] : []));
    },
  });
};
