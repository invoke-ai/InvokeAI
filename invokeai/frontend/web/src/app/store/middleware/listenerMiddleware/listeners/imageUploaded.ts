import type { UseToastOptions } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import {
  controlAdapterImageChanged,
  controlAdapterIsEnabledChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { initialImageChanged, selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { omit } from 'lodash-es';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';

export const addImageUploadedFulfilledListener = (startAppListening: AppStartListening) => {
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

      const DEFAULT_UPLOADED_TOAST: UseToastOptions = {
        title: t('toast.imageUploaded'),
        status: 'success',
      };

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
            ? `${t('toast.addedToBoard')} ${board.board_name}`
            : `${t('toast.addedToBoard')} ${autoAddBoardId}`;

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
        dispatch(setInitialCanvasImage(imageDTO, selectOptimalDimension(state)));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: t('toast.setAsCanvasInitialImage'),
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_CONTROL_ADAPTER_IMAGE') {
        const { id } = postUploadAction;
        dispatch(
          controlAdapterIsEnabledChanged({
            id,
            isEnabled: true,
          })
        );
        dispatch(
          controlAdapterImageChanged({
            id,
            controlImage: imageDTO.image_name,
          })
        );
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: t('toast.setControlImage'),
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_INITIAL_IMAGE') {
        dispatch(initialImageChanged(imageDTO));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: t('toast.setInitialImage'),
          })
        );
        return;
      }

      if (postUploadAction?.type === 'SET_NODES_IMAGE') {
        const { nodeId, fieldName } = postUploadAction;
        dispatch(fieldImageValueChanged({ nodeId, fieldName, value: imageDTO }));
        dispatch(
          addToast({
            ...DEFAULT_UPLOADED_TOAST,
            description: `${t('toast.setNodeField')} ${fieldName}`,
          })
        );
        return;
      }
    },
  });

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
          title: t('toast.imageUploadFailed'),
          description: action.error.message,
          status: 'error',
        })
      );
    },
  });
};
