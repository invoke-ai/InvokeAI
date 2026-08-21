import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import z from 'zod';

const zChangeBoardModalState = z.object({
  operation_id: z.number().int().nonnegative().default(0),
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
      state.operation_id += 1;
      state.image_names = action.payload;
      state.video_names = [];
    },
    videosToChangeSelected: (state, action: PayloadAction<string[]>) => {
      state.operation_id += 1;
      state.video_names = action.payload;
      state.image_names = [];
    },
    changeBoardReset: (state) => {
      state.image_names = [];
      state.video_names = [];
      state.isModalOpen = false;
    },
    changeBoardOperationInvalidated: (state) => {
      state.operation_id += 1;
      state.image_names = [];
      state.video_names = [];
      state.isModalOpen = false;
    },
  },
});

export const {
  isModalOpenChanged,
  imagesToChangeSelected,
  videosToChangeSelected,
  changeBoardReset,
  changeBoardOperationInvalidated,
} = slice.actions;

/**
 * Whether a completed move may write the names it could not move back into the modal's pending
 * selection. Operation ID must still match the move that completed.
 *
 * The write lands long after the dialog has closed — ConfirmationAlertDialog calls
 * acceptCallback and then onClose without awaiting, so `changeBoardReset` has already run — and
 * the slice it writes into is shared by every opener. Two things can have happened in between:
 *
 * - Another selection can have claimed and released the modal. Right-click a different image
 *   while a large move is in flight, then cancel that second dialog; the state is empty and
 *   closed again, but its operation ID is newer. Without that ID the slice looks unclaimed and
 *   the earlier request's failures are written in under the newer operation's identity. No
 *   opener surfaces them as things stand — every one of the four re-seeds the selection before
 *   it shows the dialog — so this half of the check holds the invariant rather than closing a
 *   reachable path, and it is what lets the retain stay a plain write into shared state.
 * - The session can have ended. The logout listener invalidates this operation along with the
 *   api state, and re-seeding it afterwards would leave one user's image names in the next
 *   user's store.
 */
export const canRetainFailedSelection = (
  modalState: { operation_id: number; isModalOpen: boolean; image_names: string[]; video_names: string[] },
  operationId: number,
  isSameSession: boolean
): boolean =>
  modalState.operation_id === operationId &&
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
