import type { UnknownAction } from '@reduxjs/toolkit';
import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { recordGallerySelection } from 'features/gallery/store/gallerySelectionSource';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, comparedImagesSwapped, imageSelected } from 'features/gallery/store/gallerySlice';

/**
 * The actions through which a user (or the auto-switch) picks something.
 *
 * `boardIdSelected` only counts when it carries a selection: clicking a board in the boards list
 * dispatches it bare, leaves the selection alone, and must not read as the user picking the item
 * that happens to still be selected — the viewer would reveal an item they never clicked.
 *
 * `selectionChanged` is deliberately absent: it is the multi-selection *mutation* action
 * (ctrl/shift-clicks, bulk operations), and a mutation that leaves the active item in place —
 * ctrl-clicking a non-active item off the selection, say — is bookkeeping, not the user asking to
 * see the item that stays active; counting it would flash the progress overlay off for a gesture
 * aimed at a different item. A mutation that *moves* the active item is caught by the
 * change-of-active-item clause below, and a plain click dispatches `imageSelected`, so the
 * deliberate re-pick of the already-active item still lands here.
 */
const isSelectionDispatch = (action: UnknownAction): boolean =>
  isAnyOf(imageSelected, comparedImagesSwapped)(action) ||
  (boardIdSelected.match(action) && action.payload.select !== undefined);

/**
 * Publishes every gallery selection to $gallerySelection, so the viewer can tell a user's click
 * from the gallery auto-switching to a finished item, and one selection from the next.
 *
 * The predicate has two clauses because neither alone is sufficient. Matching the selection
 * *actions* catches re-selecting the item already active — a real event that changes no state, and
 * the one the reveal needs in order to make a repeat click visible. Matching a *change of active
 * item* catches every other writer, including ones added later that an action list would miss
 * (`showVirtualBoardsChanged` and `logout` both clear the selection today).
 */
export const addGallerySelectionSourceListener = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action, currentState, previousState) =>
      isSelectionDispatch(action) || selectLastSelectedItem(currentState) !== selectLastSelectedItem(previousState),
    effect: (_action, { getState }) => {
      recordGallerySelection(selectLastSelectedItem(getState()) ?? null);
    },
  });
};
