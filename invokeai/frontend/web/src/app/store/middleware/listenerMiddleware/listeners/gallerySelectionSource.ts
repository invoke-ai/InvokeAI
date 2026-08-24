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
 * (ctrl/shift-clicks, bulk operations, the delete flow pruning deleted names out of the
 * selection), and a mutation that leaves the active item in place — ctrl-clicking a non-active
 * item off the selection, say — is bookkeeping, not the user asking to see the item that stays
 * active; counting it would flash the progress overlay off for a gesture aimed at a different
 * item. A mutation that *moves* the active item is caught by the change-of-active-item clause
 * below, and a plain click dispatches `imageSelected`, so the deliberate re-pick of the
 * already-active item still lands here.
 *
 * The corollary binds the writers, not just this file: code that rewrites the selection without
 * the user having asked for anything must use `selectionChanged`, even where `imageSelected`
 * would leave identical state — when the write leaves the active item where it is, the action is
 * the only thing left to distinguish "the user picked this" from "this happens to still be
 * selected". Choosing the action is necessary but not sufficient, though: the clause below still
 * publishes if the write *moves* the active item, so such a writer must also leave the active item
 * alone rather than collapsing the selection onto a stale snapshot of it. See the delete modals'
 * survivor branch, which does both, and its fallback, which fires only when everything selected has
 * been deleted and cannot honour the second half — so it does reveal. The board auto-select probe
 * in listeners/boardIdSelected.ts honours the first half only: it is silent when it lands back on
 * the item already displayed, but a re-run for a navigation that changed nothing still replaces a
 * selection further down the list, which is tracked separately.
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
