import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import z from 'zod';

const zChangeBoardModalState = z.object({
  isModalOpen: z.boolean().default(false),
  image_names: z.array(z.string()).default(() => []),
  video_names: z.array(z.string()).default(() => []),
});
type ChangeBoardModalState = z.infer<typeof zChangeBoardModalState>;

const getInitialState = (): ChangeBoardModalState => zChangeBoardModalState.parse({});

const slice = createSlice({
  name: 'changeBoardModal',
  initialState: getInitialState(),
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imagesToChangeSelected: (state, action: PayloadAction<string[]>) => {
      state.image_names = action.payload;
      state.video_names = [];
    },
    videosToChangeSelected: (state, action: PayloadAction<string[]>) => {
      state.video_names = action.payload;
      state.image_names = [];
    },
    changeBoardReset: (state) => {
      state.image_names = [];
      state.video_names = [];
      state.isModalOpen = false;
    },
  },
});

export const { isModalOpenChanged, imagesToChangeSelected, videosToChangeSelected, changeBoardReset } = slice.actions;

/**
 * Whether a completed move may write the names it could not move back into the modal's pending
 * selection.
 *
 * The write lands long after the dialog has closed — ConfirmationAlertDialog calls
 * acceptCallback and then onClose without awaiting, so `changeBoardReset` has already run — and
 * the slice it writes into is shared by every opener. Two things can have happened in between:
 *
 * - Another selection can have claimed the modal. Right-click a different image while a large
 *   move is in flight and the dialog is open again with one name in it; overwriting that with
 *   the earlier request's failures would move a set the user never chose, to the board they
 *   picked for something else.
 * - The session can have ended. The logout listener clears this slice along with the api state,
 *   and re-seeding it afterwards would leave one user's image names in the next user's store.
 */
export const canRetainFailedSelection = (
  modalState: { isModalOpen: boolean; image_names: string[]; video_names: string[] },
  isSameSession: boolean
): boolean =>
  isSameSession &&
  !modalState.isModalOpen &&
  modalState.image_names.length === 0 &&
  modalState.video_names.length === 0;

export const selectChangeBoardModalSlice = (state: RootState) => state.changeBoardModal;

export const changeBoardModalSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zChangeBoardModalState,
  getInitialState,
};
