import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  fieldImageValueChanged,
  workflowExposedFieldAdded,
} from 'features/nodes/store/nodesSlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '../';
import { parseify } from 'common/util/serialize';

export const dndDropped = createAction<{
  overData: TypesafeDroppableData;
  activeData: TypesafeDraggableData;
}>('dnd/dndDropped');

export const addImageDroppedListener = () => {
  startAppListening({
    actionCreator: dndDropped,
    effect: async (action, { dispatch }) => {
      const log = logger('dnd');
      const { activeData, overData } = action.payload;

      if (activeData.payloadType === 'IMAGE_DTO') {
        log.debug({ activeData, overData }, 'Image dropped');
      } else if (activeData.payloadType === 'IMAGE_DTOS') {
        log.debug(
          { activeData, overData },
          `Images (${activeData.payload.imageDTOs.length}) dropped`
        );
      } else if (activeData.payloadType === 'NODE_FIELD') {
        log.debug(
          { activeData: parseify(activeData), overData: parseify(overData) },
          'Node field dropped'
        );
      } else {
        log.debug({ activeData, overData }, `Unknown payload dropped`);
      }

      if (
        overData.actionType === 'ADD_FIELD_TO_LINEAR' &&
        activeData.payloadType === 'NODE_FIELD'
      ) {
        const { nodeId, field } = activeData.payload;
        dispatch(
          workflowExposedFieldAdded({
            nodeId,
            fieldName: field.name,
          })
        );
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
        overData.actionType === 'SET_CONTROLNET_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { controlNetId } = overData.context;
        dispatch(
          controlNetImageChanged({
            controlImage: activeData.payload.imageDTO.image_name,
            controlNetId,
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
        dispatch(setInitialCanvasImage(activeData.payload.imageDTO));
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
      if (
        overData.actionType === 'ADD_TO_BOARD' &&
        activeData.payloadType === 'IMAGE_DTOS' &&
        activeData.payload.imageDTOs
      ) {
        const { imageDTOs } = activeData.payload;
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
      if (
        overData.actionType === 'REMOVE_FROM_BOARD' &&
        activeData.payloadType === 'IMAGE_DTOS' &&
        activeData.payload.imageDTOs
      ) {
        const { imageDTOs } = activeData.payload;
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
