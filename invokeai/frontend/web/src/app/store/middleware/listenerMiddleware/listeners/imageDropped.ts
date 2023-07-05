import { createAction } from '@reduxjs/toolkit';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import { log } from 'app/logging/useLogger';
import {
  imageAddedToBatch,
  imagesAddedToBatch,
} from 'features/batch/store/batchSlice';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { controlNetImageChanged } from 'features/controlNet/store/controlNetSlice';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import {
  fieldValueChanged,
  imageCollectionFieldValueChanged,
} from 'features/nodes/store/nodesSlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { startAppListening } from '../';

const moduleLog = log.child({ namespace: 'dnd' });

export const imageDropped = createAction<{
  overData: TypesafeDroppableData;
  activeData: TypesafeDraggableData;
}>('dnd/imageDropped');

export const addImageDroppedListener = () => {
  startAppListening({
    actionCreator: imageDropped,
    effect: (action, { dispatch, getState }) => {
      const { activeData, overData } = action.payload;
      const { actionType } = overData;
      const state = getState();

      // set current image
      if (
        actionType === 'SET_CURRENT_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(imageSelected(activeData.payload.imageDTO.image_name));
      }

      // set initial image
      if (
        actionType === 'SET_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(initialImageChanged(activeData.payload.imageDTO));
      }

      // add image to batch
      if (
        actionType === 'ADD_TO_BATCH' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(imageAddedToBatch(activeData.payload.imageDTO.image_name));
      }

      // add multiple images to batch
      if (
        actionType === 'ADD_TO_BATCH' &&
        activeData.payloadType === 'GALLERY_SELECTION'
      ) {
        dispatch(imagesAddedToBatch(state.gallery.selection));
      }

      // set control image
      if (
        actionType === 'SET_CONTROLNET_IMAGE' &&
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
      }

      // set canvas image
      if (
        actionType === 'SET_CANVAS_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        dispatch(setInitialCanvasImage(activeData.payload.imageDTO));
      }

      // set nodes image
      if (
        actionType === 'SET_NODES_IMAGE' &&
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
      }

      // set multiple nodes images (single image handler)
      if (
        actionType === 'SET_MULTI_NODES_IMAGE' &&
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
      }

      // set multiple nodes images (multiple images handler)
      if (
        actionType === 'SET_MULTI_NODES_IMAGE' &&
        activeData.payloadType === 'GALLERY_SELECTION'
      ) {
        const { fieldName, nodeId } = overData.context;
        dispatch(
          imageCollectionFieldValueChanged({
            nodeId,
            fieldName,
            value: state.gallery.selection.map((image_name) => ({
              image_name,
            })),
          })
        );
      }

      // remove image from board
      // TODO: remove board_id from `removeImageFromBoard()` endpoint
      // TODO: handle multiple images
      // if (
      //   actionType === 'MOVE_BOARD' &&
      //   activeData.payloadType === 'IMAGE_DTO' &&
      //   activeData.payload.imageDTO &&
      //   overData.boardId !== null
      // ) {
      //   const { image_name } = activeData.payload.imageDTO;
      //   dispatch(
      //     boardImagesApi.endpoints.removeImageFromBoard.initiate({ image_name })
      //   );
      // }

      // add image to board
      if (
        actionType === 'MOVE_BOARD' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO &&
        overData.context.boardId
      ) {
        const { image_name } = activeData.payload.imageDTO;
        const { boardId } = overData.context;
        dispatch(
          boardImagesApi.endpoints.addImageToBoard.initiate({
            image_name,
            board_id: boardId,
          })
        );
      }

      // add multiple images to board
      if (
        actionType === 'MOVE_BOARD' &&
        activeData.payloadType === 'GALLERY_SELECTION' &&
        overData.context.boardId
      ) {
        const board_id = overData.context.boardId;
        const image_names = state.gallery.selection;
        dispatch(
          boardImagesApi.endpoints.addManyImagesToBoard.initiate({
            board_id,
            image_names,
          })
        );
      }
    },
  });
};
