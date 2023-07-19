import { UseToastOptions } from '@chakra-ui/react';
import { log } from 'app/logging/useLogger';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import { imagesAddedToBatch } from 'features/gallery/store/gallerySlice';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { boardsApi } from 'services/api/endpoints/boards';
import { startAppListening } from '..';
import {
  SYSTEM_BOARDS,
  imagesApi,
} from '../../../../../services/api/endpoints/images';

const moduleLog = log.child({ namespace: 'image' });

const DEFAULT_UPLOADED_TOAST: UseToastOptions = {
  title: 'Image Uploaded',
  status: 'success',
};

export const addImageUploadedFulfilledListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.uploadImage.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const imageDTO = action.payload;
      const state = getState();
      const { selectedBoardId } = state.gallery;

      moduleLog.debug({ arg: '<Blob>', imageDTO }, 'Image uploaded');

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
        if (SYSTEM_BOARDS.includes(selectedBoardId)) {
          dispatch(addToast({ ...DEFAULT_UPLOADED_TOAST, ...toastOptions }));
        } else {
          // Add this image to the board
          dispatch(
            imagesApi.endpoints.addImageToBoard.initiate({
              board_id: selectedBoardId,
              imageDTO,
            })
          );

          // Attempt to get the board's name for the toast
          const { data } = boardsApi.endpoints.listAllBoards.select()(state);

          // Fall back to just the board id if we can't find the board for some reason
          const board = data?.find((b) => b.board_id === selectedBoardId);
          const description = board
            ? `Added to board ${board.board_name}`
            : `Added to board ${selectedBoardId}`;

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
        dispatch(fieldValueChanged({ nodeId, fieldName, value: imageDTO }));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: `Set as node field ${fieldName}`,
          })
        );
        return;
      }

      if (postUploadAction?.type === 'ADD_TO_BATCH') {
        dispatch(imagesAddedToBatch([imageDTO.image_name]));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: 'Added to batch',
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
      const { file, postUploadAction, ...rest } = action.meta.arg.originalArgs;
      const sanitizedData = { arg: { ...rest, file: '<Blob>' } };
      moduleLog.error({ data: sanitizedData }, 'Image upload failed');
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
