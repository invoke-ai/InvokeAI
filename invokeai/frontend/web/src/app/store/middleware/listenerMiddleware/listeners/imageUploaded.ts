import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import {
  controlAdapterImageChanged,
  controlAdapterIsEnabledChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import {
  caLayerImageChanged,
  iiLayerImageChanged,
  ipaLayerImageChanged,
  rgLayerIPAdapterImageChanged,
} from 'features/controlLayers/store/controlLayersSlice';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { toast } from 'features/toast/toast';
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
        }
        return;
      }

      if (postUploadAction?.type === 'SET_CANVAS_INITIAL_IMAGE') {
        dispatch(setInitialCanvasImage(imageDTO, selectOptimalDimension(state)));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: t('toast.setAsCanvasInitialImage'),
        });
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
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: t('toast.setControlImage'),
        });
        return;
      }

      if (postUploadAction?.type === 'SET_CA_LAYER_IMAGE') {
        const { layerId } = postUploadAction;
        dispatch(caLayerImageChanged({ layerId, imageDTO }));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: t('toast.setControlImage'),
        });
      }

      if (postUploadAction?.type === 'SET_IPA_LAYER_IMAGE') {
        const { layerId } = postUploadAction;
        dispatch(ipaLayerImageChanged({ layerId, imageDTO }));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: t('toast.setControlImage'),
        });
      }

      if (postUploadAction?.type === 'SET_RG_LAYER_IP_ADAPTER_IMAGE') {
        const { layerId, ipAdapterId } = postUploadAction;
        dispatch(rgLayerIPAdapterImageChanged({ layerId, ipAdapterId, imageDTO }));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: t('toast.setControlImage'),
        });
      }

      if (postUploadAction?.type === 'SET_II_LAYER_IMAGE') {
        const { layerId } = postUploadAction;
        dispatch(iiLayerImageChanged({ layerId, imageDTO }));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: t('toast.setControlImage'),
        });
      }

      if (postUploadAction?.type === 'SET_NODES_IMAGE') {
        const { nodeId, fieldName } = postUploadAction;
        dispatch(fieldImageValueChanged({ nodeId, fieldName, value: imageDTO }));
        toast({
          ...DEFAULT_UPLOADED_TOAST,
          description: `${t('toast.setNodeField')} ${fieldName}`,
        });
        return;
      }
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.uploadImage.matchRejected,
    effect: (action) => {
      const log = logger('images');
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
