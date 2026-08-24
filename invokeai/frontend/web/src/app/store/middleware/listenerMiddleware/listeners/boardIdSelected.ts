import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import {
  selectGalleryItemNamesQueryArgs,
  selectGalleryView,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { galleryApi } from 'services/api/endpoints/gallery';

/** The actions that ask this listener to pick an item for the user. */
const startsProbe = isAnyOf(boardIdSelected, galleryViewChanged);

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

      // Nothing to probe for if this "change" changed nothing and the user already has a
      // selection. Clicking the board you are already on, or the tab already showing, dispatches
      // regardless (NoBoardBoard and the view tabs don't guard, unlike GalleryBoard and
      // VirtualBoardItem), and the probe picks the list's first item unconditionally — so it would
      // throw the user's selection away, and moving the displayed item mid-generation also lifts
      // the progress overlay off it for a couple of seconds.
      //
      // Decided here rather than after the query, because the wait itself is not safe for a click
      // that should do nothing: `condition` re-evaluates only on a dispatched action, so a quiet
      // store lets its 5s deadline expire and the give-up branch below clears the selection.
      //
      // And decided on whether the *navigation* changed anything rather than on whether the
      // displayed item is in the new list: a real board switch can land on a list that contains it
      // — a virtual date board's args drop `board_id` and filter on `created_date` alone, so its
      // list is a superset of every board's items for that day — and skipping that would strand
      // the viewer on the previous board's item.
      //
      // An empty selection still probes: with nothing to show, clicking the board is how the user
      // asks for something, and it is reachable — deleting the last item, hiding date boards while
      // one is selected, or this listener's own give-up all leave the selection empty.
      const previousState = getOriginalState();
      const isNoOpNavigation =
        (boardIdSelected.match(action) && selectSelectedBoardId(previousState) === action.payload.boardId) ||
        (galleryViewChanged.match(action) && selectGalleryView(previousState) === action.payload);

      if (isNoOpNavigation && selectSelection(getState()).length > 0) {
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
