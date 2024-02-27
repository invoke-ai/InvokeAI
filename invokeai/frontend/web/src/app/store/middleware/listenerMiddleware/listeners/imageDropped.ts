import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { parseify } from 'common/util/serialize';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import {
  controlAdapterImageChanged,
  controlAdapterIsEnabledChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { initialImageChanged, selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { imagesApi } from 'services/api/endpoints/images';

export const dndDropped = createAction<{
  overData: TypesafeDroppableData;
  activeData: TypesafeDraggableData;
}>('dnd/dndDropped');

export const addImageDroppedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: dndDropped,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('dnd');
      const { activeData, overData } = action.payload;

      if (activeData.payloadType === 'IMAGE_DTO') {
        log.debug({ activeData, overData }, 'Image dropped');
      } else if (activeData.payloadType === 'GALLERY_SELECTION') {
        log.debug({ activeData, overData }, `Images (${getState().gallery.selection.length}) dropped`);
      } else if (activeData.payloadType === 'NODE_FIELD') {
        log.debug({ activeData: parseify(activeData), overData: parseify(overData) }, 'Node field dropped');
      } else {
        log.debug({ activeData, overData }, `Unknown payload dropped`);
      }

      /**
       * Image dropped on current image
       */
      if (
        overData.actionType === 'SET_CURRENT_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(imageSelected(activeData.payload.imageDTO));
        return;
      }

      /**
       * Image dropped on initial image
       */
      if (
        overData.actionType === 'SET_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(initialImageChanged(activeData.payload.imageDTO));
        return;
      }

      /**
       * Image dropped on ControlNet
       */
      if (
        overData.actionType === 'SET_CONTROL_ADAPTER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { id } = overData.context;
        dispatch(
          controlAdapterImageChanged({
            id,
            controlImage: activeData.payload.imageDTO.image_name,
          })
        );
        dispatch(
          controlAdapterIsEnabledChanged({
            id,
            isEnabled: true,
          })
        );
        return;
      }

      /**
       * Image dropped on Canvas
       */
      if (
        overData.actionType === 'SET_CANVAS_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(setInitialCanvasImage(activeData.payload.imageDTO, selectOptimalDimension(getState())));
        return;
      }

      /**
       * Image dropped on node image field
       */
      if (
        overData.actionType === 'SET_NODES_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { fieldName, nodeId } = overData.context;
        dispatch(
          fieldImageValueChanged({
            nodeId,
            fieldName,
            value: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * TODO
       * Image selection dropped on node image collection field
       */
      // if (
      //   overData.actionType === 'SET_MULTI_NODES_IMAGE' &&
      //   activeData.payloadType === 'IMAGE_DTO' &&
      //   activeData.payload.imageDTO
      // ) {
      //   const { fieldName, nodeId } = overData.context;
      //   dispatch(
      //     fieldValueChanged({
      //       nodeId,
      //       fieldName,
      //       value: [activeData.payload.imageDTO],
      //     })
      //   );
      //   return;
      // }

      /**
       * Image dropped on user board
       */
      if (
        overData.actionType === 'ADD_TO_BOARD' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { imageDTO } = activeData.payload;
        const { boardId } = overData.context;
        dispatch(
          imagesApi.endpoints.addImageToBoard.initiate({
            imageDTO,
            board_id: boardId,
          })
        );
        return;
      }

      /**
       * Image dropped on 'none' board
       */
      if (
        overData.actionType === 'REMOVE_FROM_BOARD' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { imageDTO } = activeData.payload;
        dispatch(
          imagesApi.endpoints.removeImageFromBoard.initiate({
            imageDTO,
          })
        );
        return;
      }

      /**
       * Multiple images dropped on user board
       */
      if (overData.actionType === 'ADD_TO_BOARD' && activeData.payloadType === 'GALLERY_SELECTION') {
        const imageDTOs = getState().gallery.selection;
        const { boardId } = overData.context;
        dispatch(
          imagesApi.endpoints.addImagesToBoard.initiate({
            imageDTOs,
            board_id: boardId,
          })
        );
        return;
      }

      /**
       * Multiple images dropped on 'none' board
       */
      if (overData.actionType === 'REMOVE_FROM_BOARD' && activeData.payloadType === 'GALLERY_SELECTION') {
        const imageDTOs = getState().gallery.selection;
        dispatch(
          imagesApi.endpoints.removeImagesFromBoard.initiate({
            imageDTOs,
          })
        );
        return;
      }
    },
  });
};
