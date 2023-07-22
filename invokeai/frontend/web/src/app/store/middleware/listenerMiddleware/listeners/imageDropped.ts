import { createAction } from '@reduxjs/toolkit';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import { logger } from 'app/logging/logger';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import {
  imageSelected,
  imagesAddedToBatch,
} from 'features/gallery/store/gallerySlice';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '../';

export const dndDropped = createAction<{
  overData: TypesafeDroppableData;
  activeData: TypesafeDraggableData;
}>('dnd/dndDropped');

export const addImageDroppedListener = () => {
  startAppListening({
    actionCreator: dndDropped,
    effect: async (action, { dispatch }) => {
      const log = logger('images');
      const { activeData, overData } = action.payload;

      log.debug({ activeData, overData }, 'Image or selection dropped');

      // set current image
      if (
        overData.actionType === 'SET_CURRENT_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(imageSelected(activeData.payload.imageDTO.image_name));
        return;
      }

      // set initial image
      if (
        overData.actionType === 'SET_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(initialImageChanged(activeData.payload.imageDTO));
        return;
      }

      // add image to batch
      if (
        overData.actionType === 'ADD_TO_BATCH' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(imagesAddedToBatch([activeData.payload.imageDTO.image_name]));
        return;
      }

      // add multiple images to batch
      if (
        overData.actionType === 'ADD_TO_BATCH' &&
        activeData.payloadType === 'IMAGE_NAMES'
      ) {
        dispatch(imagesAddedToBatch(activeData.payload.image_names));

        return;
      }

      // set control image
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

      // set canvas image
      if (
        overData.actionType === 'SET_CANVAS_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(setInitialCanvasImage(activeData.payload.imageDTO));
        return;
      }

      // set nodes image
      if (
        overData.actionType === 'SET_NODES_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { fieldName, nodeId } = overData.context;
        dispatch(
          fieldValueChanged({
            nodeId,
            fieldName,
            value: activeData.payload.imageDTO,
          })
        );
        return;
      }

      // set multiple nodes images (single image handler)
      if (
        overData.actionType === 'SET_MULTI_NODES_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { fieldName, nodeId } = overData.context;
        dispatch(
          fieldValueChanged({
            nodeId,
            fieldName,
            value: [activeData.payload.imageDTO],
          })
        );
        return;
      }

      // // set multiple nodes images (multiple images handler)
      // if (
      //   overData.actionType === 'SET_MULTI_NODES_IMAGE' &&
      //   activeData.payloadType === 'IMAGE_NAMES'
      // ) {
      //   const { fieldName, nodeId } = overData.context;
      //   dispatch(
      //     imageCollectionFieldValueChanged({
      //       nodeId,
      //       fieldName,
      //       value: activeData.payload.image_names.map((image_name) => ({
      //         image_name,
      //       })),
      //     })
      //   );
      //   return;
      // }

      // add image to board
      if (
        overData.actionType === 'MOVE_BOARD' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { imageDTO } = activeData.payload;
        const { boardId } = overData.context;

        // image was droppe on the "NoBoardBoard"
        if (!boardId) {
          dispatch(
            imagesApi.endpoints.removeImageFromBoard.initiate({
              imageDTO,
            })
          );
          return;
        }

        // image was dropped on a user board
        dispatch(
          imagesApi.endpoints.addImageToBoard.initiate({
            imageDTO,
            board_id: boardId,
          })
        );
        return;
      }

      // // add gallery selection to board
      // if (
      //   overData.actionType === 'MOVE_BOARD' &&
      //   activeData.payloadType === 'IMAGE_NAMES' &&
      //   overData.context.boardId
      // ) {
      //   console.log('adding gallery selection to board');
      //   const board_id = overData.context.boardId;
      //   dispatch(
      //     boardImagesApi.endpoints.addManyBoardImages.initiate({
      //       board_id,
      //       image_names: activeData.payload.image_names,
      //     })
      //   );
      //   return;
      // }

      // // remove gallery selection from board
      // if (
      //   overData.actionType === 'MOVE_BOARD' &&
      //   activeData.payloadType === 'IMAGE_NAMES' &&
      //   overData.context.boardId === null
      // ) {
      //   console.log('removing gallery selection to board');
      //   dispatch(
      //     boardImagesApi.endpoints.deleteManyBoardImages.initiate({
      //       image_names: activeData.payload.image_names,
      //     })
      //   );
      //   return;
      // }

      // // add batch selection to board
      // if (
      //   overData.actionType === 'MOVE_BOARD' &&
      //   activeData.payloadType === 'IMAGE_NAMES' &&
      //   overData.context.boardId
      // ) {
      //   const board_id = overData.context.boardId;
      //   dispatch(
      //     boardImagesApi.endpoints.addManyBoardImages.initiate({
      //       board_id,
      //       image_names: activeData.payload.image_names,
      //     })
      //   );
      //   return;
      // }

      // // remove batch selection from board
      // if (
      //   overData.actionType === 'MOVE_BOARD' &&
      //   activeData.payloadType === 'IMAGE_NAMES' &&
      //   overData.context.boardId === null
      // ) {
      //   dispatch(
      //     boardImagesApi.endpoints.deleteManyBoardImages.initiate({
      //       image_names: activeData.payload.image_names,
      //     })
      //   );
      //   return;
      // }
    },
  });
};
