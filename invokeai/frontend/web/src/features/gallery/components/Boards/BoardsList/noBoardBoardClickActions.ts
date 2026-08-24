import { autoAddBoardIdChanged, boardIdSelected } from 'features/gallery/store/gallerySlice';

/**
 * What clicking the Uncategorized board does, as data.
 *
 * Its own module because NoBoardBoard pulls in the dnd stack, which cannot be imported in a plain
 * unit test; the same shape as AddBoardButton's exported action builder.
 *
 * The board is only *selected* when it isn't already, as GalleryBoard and VirtualBoardItem have
 * always done. Re-selecting the board you are on restarts the gallery's auto-select probe
 * (listeners/boardIdSelected.ts), which selects the board's first item unconditionally — so it
 * discards whatever the user had selected, and moving the displayed item while a generation is
 * running also lifts the progress overlay for a couple of seconds.
 */
export const getNoBoardClickActions = (isSelected: boolean, autoAssignBoardOnClick: boolean) => {
  const actions = [];
  if (!isSelected) {
    actions.push(boardIdSelected({ boardId: 'none' }));
  }
  if (autoAssignBoardOnClick) {
    actions.push(autoAddBoardIdChanged('none'));
  }
  return actions;
};
