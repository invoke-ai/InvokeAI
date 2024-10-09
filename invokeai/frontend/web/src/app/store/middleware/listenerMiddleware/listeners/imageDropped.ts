import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { deepClone } from 'common/util/deepClone';
import { selectDefaultControlAdapter, selectDefaultIPAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  entityRasterized,
  entitySelected,
  rasterLayerAdded,
  referenceImageAdded,
  referenceImageIPAdapterImageChanged,
  rgAdded,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { imagesApi } from 'services/api/endpoints/images';

export const dndDropped = createAction<{
  overData: TypesafeDroppableData;
  activeData: TypesafeDraggableData;
}>('dnd/dndDropped');

const log = logger('system');

export const addImageDroppedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: dndDropped,
    effect: (action, { dispatch, getState }) => {
      const { activeData, overData } = action.payload;
      if (!isValidDrop(overData, activeData)) {
        return;
      }

      if (activeData.payloadType === 'IMAGE_DTO') {
        log.debug({ activeData, overData }, 'Image dropped');
      } else if (activeData.payloadType === 'GALLERY_SELECTION') {
        log.debug({ activeData, overData }, `Images (${getState().gallery.selection.length}) dropped`);
      } else if (activeData.payloadType === 'NODE_FIELD') {
        log.debug({ activeData, overData }, 'Node field dropped');
      } else {
        log.debug({ activeData, overData }, `Unknown payload dropped`);
      }

      /**
       * Image dropped on IP Adapter Layer
       */
      if (
        overData.actionType === 'SET_IPA_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { id } = overData.context;
        dispatch(
          referenceImageIPAdapterImageChanged({
            entityIdentifier: { id, type: 'reference_image' },
            imageDTO: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * Image dropped on RG Layer IP Adapter
       */
      if (
        overData.actionType === 'SET_RG_IP_ADAPTER_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const { id, referenceImageId } = overData.context;
        dispatch(
          rgIPAdapterImageChanged({
            entityIdentifier: { id, type: 'regional_guidance' },
            referenceImageId,
            imageDTO: activeData.payload.imageDTO,
          })
        );
        return;
      }

      /**
       * Image dropped on Raster layer
       */
      if (
        overData.actionType === 'ADD_RASTER_LAYER_FROM_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const imageObject = imageDTOToImageObject(activeData.payload.imageDTO);
        const { x, y } = selectCanvasSlice(getState()).bbox.rect;
        const overrides: Partial<CanvasRasterLayerState> = {
          objects: [imageObject],
          position: { x, y },
        };
        dispatch(rasterLayerAdded({ overrides, isSelected: true }));
        return;
      }

      /**
       * Image dropped on Raster layer
       */
      if (
        overData.actionType === 'ADD_CONTROL_LAYER_FROM_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const state = getState();
        const imageObject = imageDTOToImageObject(activeData.payload.imageDTO);
        const { x, y } = selectCanvasSlice(state).bbox.rect;
        const defaultControlAdapter = selectDefaultControlAdapter(state);
        const overrides: Partial<CanvasControlLayerState> = {
          objects: [imageObject],
          position: { x, y },
          controlAdapter: defaultControlAdapter,
        };
        dispatch(controlLayerAdded({ overrides, isSelected: true }));
        return;
      }

      if (
        overData.actionType === 'ADD_REGIONAL_REFERENCE_IMAGE_FROM_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const state = getState();
        const ipAdapter = deepClone(selectDefaultIPAdapter(state));
        ipAdapter.image = imageDTOToImageWithDims(activeData.payload.imageDTO);
        const overrides: Partial<CanvasRegionalGuidanceState> = {
          referenceImages: [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter }],
        };
        dispatch(rgAdded({ overrides, isSelected: true }));
        return;
      }

      if (
        overData.actionType === 'ADD_GLOBAL_REFERENCE_IMAGE_FROM_IMAGE' &&
        activeData.payloadType === 'IMAGE_DTO' &&
        activeData.payload.imageDTO
      ) {
        const state = getState();
        const ipAdapter = deepClone(selectDefaultIPAdapter(state));
        ipAdapter.image = imageDTOToImageWithDims(activeData.payload.imageDTO);
        const overrides: Partial<CanvasReferenceImageState> = {
          ipAdapter,
        };
        dispatch(referenceImageAdded({ overrides, isSelected: true }));
        return;
      }

      /**
       * Image dropped on Raster layer
       */
      if (overData.actionType === 'REPLACE_LAYER_WITH_IMAGE' && activeData.payloadType === 'IMAGE_DTO') {
        const state = getState();
        const { entityIdentifier } = overData.context;
        const imageObject = imageDTOToImageObject(activeData.payload.imageDTO);
        const { x, y } = selectCanvasSlice(state).bbox.rect;
        dispatch(entityRasterized({ entityIdentifier, imageObject, position: { x, y }, replaceObjects: true }));
        dispatch(entitySelected({ entityIdentifier }));
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
