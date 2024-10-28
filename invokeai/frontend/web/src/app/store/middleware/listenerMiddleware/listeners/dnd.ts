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
  inpaintMaskAdded,
  rasterLayerAdded,
  referenceImageAdded,
  referenceImageIPAdapterImageChanged,
  rgAdded,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { Dnd } from 'features/dnd2/dnd';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { imagesApi } from 'services/api/endpoints/images';

const log = logger('system');

export const dndDropped = createAction<{
  sourceData: Dnd.types['SourceDataUnion'];
  targetData: Dnd.types['TargetDataUnion'];
}>('dnd/dndDropped2');

export const addDndDroppedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: dndDropped,
    effect: (action, { dispatch, getState }) => {
      const { sourceData, targetData } = action.payload;

      // Single image dropped
      if (Dnd.Source.singleImage.typeGuard(sourceData)) {
        log.debug({ sourceData, targetData }, 'Image dropped');
        const { imageDTO } = sourceData.payload;

        // Image dropped on IP Adapter
        if (
          Dnd.Target.setGlobalReferenceImage.typeGuard(targetData) &&
          Dnd.Target.setGlobalReferenceImage.validateDrop(sourceData, targetData)
        ) {
          const { globalReferenceImageId } = targetData.payload;
          dispatch(
            referenceImageIPAdapterImageChanged({
              entityIdentifier: { id: globalReferenceImageId, type: 'reference_image' },
              imageDTO,
            })
          );
          return;
        }

        //Image dropped on Regional Guidance IP Adapter
        if (
          Dnd.Target.setRegionalGuidanceReferenceImage.typeGuard(targetData) &&
          Dnd.Target.setRegionalGuidanceReferenceImage.validateDrop(sourceData, targetData)
        ) {
          const { regionalGuidanceId, referenceImageId } = targetData.payload;
          dispatch(
            rgIPAdapterImageChanged({
              entityIdentifier: { id: regionalGuidanceId, type: 'regional_guidance' },
              referenceImageId,
              imageDTO,
            })
          );
          return;
        }

        // Add raster layer from image
        if (
          Dnd.Target.newRasterLayerFromImage.typeGuard(targetData) &&
          Dnd.Target.newRasterLayerFromImage.validateDrop(sourceData, targetData)
        ) {
          const imageObject = imageDTOToImageObject(imageDTO);
          const { x, y } = selectCanvasSlice(getState()).bbox.rect;
          const overrides: Partial<CanvasRasterLayerState> = {
            objects: [imageObject],
            position: { x, y },
          };
          dispatch(rasterLayerAdded({ overrides, isSelected: true }));
          return;
        }

        // Add inpaint mask from image
        if (
          Dnd.Target.newInpaintMaskFromImage.typeGuard(targetData) &&
          Dnd.Target.newInpaintMaskFromImage.validateDrop(sourceData, targetData)
        ) {
          const imageObject = imageDTOToImageObject(imageDTO);
          const { x, y } = selectCanvasSlice(getState()).bbox.rect;
          const overrides: Partial<CanvasInpaintMaskState> = {
            objects: [imageObject],
            position: { x, y },
          };
          dispatch(inpaintMaskAdded({ overrides, isSelected: true }));
          return;
        }

        // Add regional guidance from image
        if (
          Dnd.Target.newRegionalGuidanceFromImage.typeGuard(targetData) &&
          Dnd.Target.newRegionalGuidanceFromImage.validateDrop(sourceData, targetData)
        ) {
          const imageObject = imageDTOToImageObject(imageDTO);
          const { x, y } = selectCanvasSlice(getState()).bbox.rect;
          const overrides: Partial<CanvasRegionalGuidanceState> = {
            objects: [imageObject],
            position: { x, y },
          };
          dispatch(rgAdded({ overrides, isSelected: true }));
          return;
        }

        // Add control layer from image
        if (
          Dnd.Target.newControlLayerFromImage.typeGuard(targetData) &&
          Dnd.Target.newControlLayerFromImage.validateDrop(sourceData, targetData)
        ) {
          const state = getState();
          const imageObject = imageDTOToImageObject(imageDTO);
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

        // Add regional guidance layer w/ reference image from image
        if (
          Dnd.Target.newRegionalGuidanceReferenceImageFromImage.typeGuard(targetData) &&
          Dnd.Target.newRegionalGuidanceReferenceImageFromImage.validateDrop(sourceData, targetData)
        ) {
          const state = getState();
          const ipAdapter = deepClone(selectDefaultIPAdapter(state));
          ipAdapter.image = imageDTOToImageWithDims(imageDTO);
          const overrides: Partial<CanvasRegionalGuidanceState> = {
            referenceImages: [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter }],
          };
          dispatch(rgAdded({ overrides, isSelected: true }));
          return;
        }

        // Add global reference image from image
        if (
          Dnd.Target.newGlobalReferenceImageFromImage.typeGuard(targetData) &&
          Dnd.Target.newGlobalReferenceImageFromImage.validateDrop(sourceData, targetData)
        ) {
          const state = getState();
          const ipAdapter = deepClone(selectDefaultIPAdapter(state));
          ipAdapter.image = imageDTOToImageWithDims(imageDTO);
          const overrides: Partial<CanvasReferenceImageState> = { ipAdapter };
          dispatch(referenceImageAdded({ overrides, isSelected: true }));
          return;
        }

        // Replace layer with image
        if (
          Dnd.Target.replaceLayerWithImage.typeGuard(targetData) &&
          Dnd.Target.replaceLayerWithImage.validateDrop(sourceData, targetData)
        ) {
          const state = getState();
          const { entityIdentifier } = targetData.payload;
          const imageObject = imageDTOToImageObject(imageDTO);
          const { x, y } = selectCanvasSlice(state).bbox.rect;
          dispatch(entityRasterized({ entityIdentifier, imageObject, position: { x, y }, replaceObjects: true }));
          dispatch(entitySelected({ entityIdentifier }));
          return;
        }

        // Image dropped on node image field
        if (
          Dnd.Target.setNodeImageField.typeGuard(targetData) &&
          Dnd.Target.setNodeImageField.validateDrop(sourceData, targetData)
        ) {
          const { fieldName, nodeId } = targetData.payload;
          dispatch(
            fieldImageValueChanged({
              nodeId,
              fieldName,
              value: imageDTO,
            })
          );
          return;
        }

        // Image selected for compare
        if (
          Dnd.Target.selectForCompare.typeGuard(targetData) &&
          Dnd.Target.selectForCompare.validateDrop(sourceData, targetData)
        ) {
          dispatch(imageToCompareChanged(imageDTO));
          return;
        }

        // Image added to board
        if (Dnd.Target.addToBoard.typeGuard(targetData) && Dnd.Target.addToBoard.validateDrop(sourceData, targetData)) {
          const { boardId } = targetData.payload;
          dispatch(
            imagesApi.endpoints.addImageToBoard.initiate({
              imageDTO,
              board_id: boardId,
            })
          );
          dispatch(selectionChanged([]));
          return;
        }

        // Image removed from board
        if (
          Dnd.Target.removeFromBoard.typeGuard(targetData) &&
          Dnd.Target.removeFromBoard.validateDrop(sourceData, targetData)
        ) {
          dispatch(
            imagesApi.endpoints.removeImageFromBoard.initiate({
              imageDTO,
            })
          );
          dispatch(selectionChanged([]));
          return;
        }

        // Image dropped on upscale initial image
        if (
          Dnd.Target.setUpscaleInitialImageFromImage.typeGuard(targetData) &&
          Dnd.Target.setUpscaleInitialImageFromImage.validateDrop(sourceData, targetData)
        ) {
          dispatch(upscaleInitialImageChanged(imageDTO));
          return;
        }
      }

      if (Dnd.Source.multipleImage.typeGuard(sourceData)) {
        log.debug({ sourceData, targetData }, 'Multiple images dropped');
        const { imageDTOs } = sourceData.payload;

        // Multiple images dropped on user board
        if (Dnd.Target.addToBoard.typeGuard(targetData) && Dnd.Target.addToBoard.validateDrop(sourceData, targetData)) {
          const { boardId } = targetData.payload;
          dispatch(
            imagesApi.endpoints.addImagesToBoard.initiate({
              imageDTOs,
              board_id: boardId,
            })
          );
          dispatch(selectionChanged([]));
          return;
        }

        // Multiple images dropped on Uncategorized board (e.g. removed from board)
        if (
          Dnd.Target.removeFromBoard.typeGuard(targetData) &&
          Dnd.Target.removeFromBoard.validateDrop(sourceData, targetData)
        ) {
          dispatch(
            imagesApi.endpoints.removeImagesFromBoard.initiate({
              imageDTOs,
            })
          );
          dispatch(selectionChanged([]));
          return;
        }
      }

      log.error({ sourceData, targetData }, 'Invalid dnd drop');
    },
  });
};
