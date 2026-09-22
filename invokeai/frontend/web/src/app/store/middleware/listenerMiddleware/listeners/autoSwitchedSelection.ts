import type { AppStartListening } from 'app/store/store';
import { autoSwitchedImages } from 'features/gallery/store/autoSwitchedImages';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';

/**
 * Keeps the auto-switch marker scoped to the selection it was recorded for.
 *
 * onInvocationComplete records the item it is about to auto-switch to, so the viewer's reveal
 * effect can tell that handoff apart from a user's gallery click. The marker is only meaningful
 * while that selection stands: once the selection moves on, the recorded auto-switch will never
 * render, and leaving the marker behind would make the user's next click on that item read as an
 * auto-switch and get no reveal.
 *
 * Matched by state rather than by action type on purpose — the selection is written by several
 * reducers (imageSelected, selectionChanged, boardIdSelected, comparedImagesSwapped,
 * showVirtualBoardsChanged, logout), and a new one added later would silently escape an
 * action-type list, leaving exactly the stale marker this exists to prevent.
 */
export const addAutoSwitchedSelectionListener = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (_action, currentState, previousState) =>
      selectLastSelectedItem(currentState) !== selectLastSelectedItem(previousState),
    effect: (_action, { getState }) => {
      autoSwitchedImages.settle(selectLastSelectedItem(getState()) ?? null);
    },
  });
};
