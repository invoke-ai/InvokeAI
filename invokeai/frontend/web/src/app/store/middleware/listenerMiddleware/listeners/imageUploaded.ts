import { UseToastOptions } from '@chakra-ui/react';
import { logger } from 'app/logging/logger';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { omit } from 'lodash-es';
import { boardsApi } from 'services/api/endpoints/boards';
import { startAppListening } from '..';
import { imagesApi } from '../../../../../services/api/endpoints/images';

const DEFAULT_UPLOADED_TOAST: UseToastOptions = {
  title: 'Image Uploaded',
  status: 'success',
};

export const addImageUploadedFulfilledListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.uploadImage.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const log = logger('images');
      const imageDTO = action.payload;
      const state = getState();
      const { autoAddBoardId } = state.gallery;

      log.debug({ imageDTO }, 'Image uploaded');

      const { postUploadAction } = action.meta.arg.originalArgs;

      if (
        // No further actions needed for intermediate images,
        action.payload.is_intermediate &&
        // unless they have an explicit post-upload action
        !postUploadAction
      ) {
        return;
      }

      // default action - just upload and alert user
      if (postUploadAction?.type === 'TOAST') {
        const { toastOptions } = postUploadAction;
        if (!autoAddBoardId || autoAddBoardId === 'none') {
          dispatch(addToast({ ...DEFAULT_UPLOADED_TOAST, ...toastOptions }));
        } else {
          // Add this image to the board
          dispatch(
            imagesApi.endpoints.addImageToBoard.initiate({
              board_id: autoAddBoardId,
              imageDTO,
            })
          );

          // Attempt to get the board's name for the toast
          const { data } = boardsApi.endpoints.listAllBoards.select()(state);

          // Fall back to just the board id if we can't find the board for some reason
          const board = data?.find((b) => b.board_id === autoAddBoardId);
          const description = board
            ? `Added to board ${board.board_name}`
            : `Added to board ${autoAddBoardId}`;

          dispatch(
            addToast({
              ...DEFAULT_UPLOADED_TOAST,
              description,
            })
          );
        }
        return;
      }

      if (postUploadAction?.type === 'SET_CANVAS_INITIAL_IMAGE') {
        dispatch(setInitialCanvasImage(imageDTO));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: 'Set as canvas initial image',
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_CONTROLNET_IMAGE') {
        const { controlNetId } = postUploadAction;
        dispatch(
          controlNetImageChanged({
            controlNetId,
            controlImage: imageDTO.image_name,
          })
        );
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: 'Set as control image',
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_INITIAL_IMAGE') {
        dispatch(initialImageChanged(imageDTO));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: 'Set as initial image',
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_NODES_IMAGE') {
        const { nodeId, fieldName } = postUploadAction;
        dispatch(
          fieldImageValueChanged({ nodeId, fieldName, value: imageDTO })
        );
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: `Set as node field ${fieldName}`,
          })
        );
        return;
      }
    },
  });
};

export const addImageUploadedRejectedListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.uploadImage.matchRejected,
    effect: (action, { dispatch }) => {
      const log = logger('images');
      const sanitizedData = {
        arg: {
          ...omit(action.meta.arg.originalArgs, ['file', 'postUploadAction']),
          file: '<Blob>',
        },
      };
      log.error({ ...sanitizedData }, 'Image upload failed');
      dispatch(
        addToast({
          title: 'Image Upload Failed',
          description: action.error.message,
          status: 'error',
        })
      );
    },
  });
};
