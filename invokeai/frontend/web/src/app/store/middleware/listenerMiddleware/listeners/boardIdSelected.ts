import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { selectGalleryItemNamesQueryArgs, selectSelection } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { galleryApi } from 'services/api/endpoints/gallery';

/** The actions that ask this listener to pick an item for the user. */
const startsProbe = isAnyOf(boardIdSelected, galleryViewChanged);

export const addBoardIdSelectedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    // Two jobs, so this cannot be a plain action matcher. The probe below is started by a board or
    // view change — but it must also be *cancelled* by any selection that lands while it waits,
    // and a selection arrives through several actions: imageSelected from the gallery's auto-switch,
    // plain thumbnail clicks and keyboard navigation, selectionChanged from ctrl/shift-clicks, the
    // delete flow's pruning and this listener's own probe, boardIdSelected carrying a selection.
    // Matching the resulting change of the selection covers all of them, including any writer added
    // later — an action list would silently miss it.
    //
    // The whole selection, not just its active item: removing one of several selected thumbnails,
    // or re-picking the one already active, leaves the last item unchanged while still being the
    // user settling what they want. Comparing only that item left the probe running through those,
    // to overwrite their selection when it woke. The state is immutable, so a new array reference
    // is exactly "the selection was written", and cancelling a probe more often than strictly
    // needed costs nothing.
    predicate: (action, currentState, previousState) =>
      startsProbe(action) || selectSelection(currentState) !== selectSelection(previousState),
    effect: async (action, { getState, dispatch, condition, cancelActiveListeners }) => {
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

      // The probe picks an item *for* the user, so it writes the selection with the mutation
      // action rather than `imageSelected`. The state is identical either way, but `imageSelected`
      // means "the user asked to see this", and while a generation is running the viewer answers
      // that by lifting the progress overlay off the item for a couple of seconds — so a write
      // that changes nothing must not announce itself as a pick. NoBoardBoard and the view tabs
      // dispatch even when nothing changed (unlike GalleryBoard and VirtualBoardItem), which
      // re-runs this probe; when it lands back on the item already displayed, the mutation action
      // is what keeps it silent. A write that genuinely moves the displayed item still reveals,
      // through the change-of-active-item clause. See gallerySelectionSource.
      //
      // This does NOT stop that re-run from *replacing* a selection further down the list with
      // `item_names[0]` — a real bug, but an older and wider one than this file, tracked in its own
      // issue along with the give-up branch below clearing a good selection whenever `condition`
      // gets no wake-up within 5s.
      if (!isSuccess) {
        dispatch(selectionChanged([]));
        return;
      }

      // the board was just changed - we can select the first gallery item (image or video)
      const itemNames = selectQuery(getState()).data?.item_names;
      const firstItemName = itemNames?.[0];

      dispatch(selectionChanged(firstItemName ? [firstItemName] : []));
    },
  });
};
