import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import {
  entityRasterized,
  entitySelected,
  ipaImageChanged,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { omit } from 'lodash-es';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';

const log = logger('gallery');

export const addImageUploadedFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.uploadImage.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
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

      const DEFAULT_UPLOADED_TOAST = {
        id: 'IMAGE_UPLOADED',
        title: t('toast.imageUploaded'),
        status: 'success',
      } as const;

      // default action - just upload and alert user
      if (postUploadAction?.type === 'TOAST') {
        if (!autoAddBoardId || autoAddBoardId === 'none') {
          const title = postUploadAction.title || DEFAULT_UPLOADED_TOAST.title;
          toast({ ...DEFAULT_UPLOADED_TOAST, title });
          dispatch(boardIdSelected({ boardId: 'none' }));
          dispatch(galleryViewChanged('assets'));
        } else {
          // Add this image to the board
          dispatch(
            imagesApi.endpoints.addImageToBoard.initiate({
              board_id: autoAddBoardId,
              imageDTO,
            })
          );

          // Attempt to get the board's name for the toast
          const queryArgs = selectListBoardsQueryArgs(state);
          const { data } = boardsApi.endpoints.listAllBoards.select(queryArgs)(state);

          // Fall back to just the board id if we can't find the board for some reason
          const board = data?.find((b) => b.board_id === autoAddBoardId);
          const description = board
            ? `${t('toast.addedToBoard')} ${board.board_name}`
            : `${t('toast.addedToBoard')} ${autoAddBoardId}`;

          toast({
            ...DEFAULT_UPLOADED_TOAST,
            description,
          });
          dispatch(boardIdSelected({ boardId: autoAddBoardId }));
          dispatch(galleryViewChanged('assets'));
        }
        return;
      }

      if (postUploadAction?.type === 'SET_UPSCALE_INITIAL_IMAGE') {
        dispatch(upscaleInitialImageChanged(imageDTO));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: 'set as upscale initial image',
        });
        return;
      }

      // if (postUploadAction?.type === 'SET_CA_IMAGE') {
      //   const { id } = postUploadAction;
      //   dispatch(caImageChanged({ id, imageDTO }));
      //   toast({ ...DEFAULT_UPLOADED_TOAST, description: t('toast.setControlImage') });
      //   return;
      // }

      if (postUploadAction?.type === 'SET_IPA_IMAGE') {
        const { id } = postUploadAction;
        dispatch(ipaImageChanged({ entityIdentifier: { id, type: 'ip_adapter' }, imageDTO }));
        toast({ ...DEFAULT_UPLOADED_TOAST, description: t('toast.setControlImage') });
        return;
      }

      if (postUploadAction?.type === 'SET_RG_IP_ADAPTER_IMAGE') {
        const { id, ipAdapterId } = postUploadAction;
        dispatch(
          rgIPAdapterImageChanged({ entityIdentifier: { id, type: 'regional_guidance' }, ipAdapterId, imageDTO })
        );
        toast({ ...DEFAULT_UPLOADED_TOAST, description: t('toast.setControlImage') });
        return;
      }

      if (postUploadAction?.type === 'SET_NODES_IMAGE') {
        const { nodeId, fieldName } = postUploadAction;
        dispatch(fieldImageValueChanged({ nodeId, fieldName, value: imageDTO }));
        toast({ ...DEFAULT_UPLOADED_TOAST, description: `${t('toast.setNodeField')} ${fieldName}` });
        return;
      }

      if (postUploadAction?.type === 'REPLACE_LAYER_WITH_IMAGE') {
        const { entityIdentifier } = postUploadAction;

        const state = getState();
        const imageObject = imageDTOToImageObject(imageDTO);
        const { x, y } = selectCanvasSlice(state).bbox.rect;
        dispatch(entityRasterized({ entityIdentifier, imageObject, position: { x, y }, replaceObjects: true }));
        dispatch(entitySelected({ entityIdentifier }));
        return;
      }
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.uploadImage.matchRejected,
    effect: (action) => {
      const sanitizedData = {
        arg: {
          ...omit(action.meta.arg.originalArgs, ['file', 'postUploadAction']),
          file: '<Blob>',
        },
      };
      log.error({ ...sanitizedData }, 'Image upload failed');
      toast({
        title: t('toast.imageUploadFailed'),
        description: action.error.message,
        status: 'error',
      });
    },
  });
};
