import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import {
  selectGalleryItemNamesQueryArgs,
  selectLastSelectedItem,
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
      // wait until the board has some items - maybe it already has some from a previous fetch.
      // `condition` only re-evaluates its predicate when an action is dispatched, never on its own
      // timer, so a list that is *already* fulfilled has to be checked here: otherwise a quiet
      // store gives no wake-up, the 5s deadline expires, and the give-up branch below clears a
      // selection that was perfectly good. must use getState() to avoid stale state.
      const isSuccess =
        selectQuery(getState()).isSuccess || (await condition(() => selectQuery(getState()).isSuccess, 5000));

      if (!isSuccess) {
        dispatch(selectionChanged([]));
        return;
      }

      // the board was just changed - we can select the first gallery item (image or video)
      const itemNames = selectQuery(getState()).data?.item_names;

      // ...unless what the viewer is already showing is in this list, in which case the board or
      // view did not really change (clicking the board you are already on, or the tab already
      // showing, both dispatch unconditionally) and there is nothing for the probe to fix.
      // Replacing a perfectly good selection with the newest item would discard the user's pick,
      // and moving the displayed item mid-generation also lifts the progress overlay off it for a
      // couple of seconds. Deciding it here rather than in each caller keeps the rule in one place
      // and reads the live selection instead of a render-time snapshot of it.
      const activeItem = selectLastSelectedItem(getState());
      if (activeItem && itemNames?.includes(activeItem)) {
        return;
      }

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
