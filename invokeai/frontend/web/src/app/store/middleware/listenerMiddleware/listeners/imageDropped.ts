import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { parseify } from 'common/util/serialize';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import {
  controlAdapterImageChanged,
  controlAdapterIsEnabledChanged,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import {
  caLayerImageChanged,
  iiLayerImageChanged,
  imageAdded,
  ipaLayerImageChanged,
  rgLayerIPAdapterImageChanged,
} from 'features/controlLayers/store/controlLayersSlice';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import {
  imageSelected,
  imageToCompareChanged,
  isImageViewerOpenChanged,
  selectionChanged,
} from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
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
      if (!isValidDrop(overData, activeData)) {
        return;
      }

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
        dispatch(isImageViewerOpenChanged(true));
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
       * Image dropped on Control Adapter Layer
       */
      if (
        overData.actionType === 'SET_CA_LAYER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { layerId } = overData.context;
        dispatch(
          caLayerImageChanged({
            layerId,
            imageDTO: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * Image dropped on IP Adapter Layer
       */
      if (
        overData.actionType === 'SET_IPA_LAYER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { layerId } = overData.context;
        dispatch(
          ipaLayerImageChanged({
            layerId,
            imageDTO: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * Image dropped on RG Layer IP Adapter
       */
      if (
        overData.actionType === 'SET_RG_LAYER_IP_ADAPTER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { layerId, ipAdapterId } = overData.context;
        dispatch(
          rgLayerIPAdapterImageChanged({
            layerId,
            ipAdapterId,
            imageDTO: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * Image dropped on II Layer Image
       */
      if (
        overData.actionType === 'SET_II_LAYER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { layerId } = overData.context;
        dispatch(
          iiLayerImageChanged({
            layerId,
            imageDTO: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * Image dropped on Raster layer
       */
      if (
        overData.actionType === 'ADD_RASTER_LAYER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { layerId } = overData.context;
        dispatch(
          imageAdded({
            layerId,
            imageDTO: activeData.payload.imageDTO,
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
       * Image selected for compare
       */
      if (
        overData.actionType === 'SELECT_FOR_COMPARE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { imageDTO } = activeData.payload;
        dispatch(imageToCompareChanged(imageDTO));
        dispatch(isImageViewerOpenChanged(true));
        return;
      }

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
        dispatch(selectionChanged([]));
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
        dispatch(selectionChanged([]));
        return;
      }

      /**
       * Image dropped on upscale initial image
       */
      if (
        overData.actionType === 'SET_UPSCALE_INITIAL_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { imageDTO } = activeData.payload;

        dispatch(upscaleInitialImageChanged(imageDTO));
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
        dispatch(selectionChanged([]));
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
        dispatch(selectionChanged([]));
        return;
      }
    },
  });
};
