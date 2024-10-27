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
import {
  newControlLayerFromImageDndTarget,
  addGlobalReferenceImageFromImageDndTarget,
  addInpaintMaskFromImageDndTarget,
  newRasterLayerFromImageDndTarget,
  addRegionalGuidanceFromImageDndTarget,
  addRegionalGuidanceReferenceImageFromImageDndTarget,
  addToBoardDndTarget,
  type DndSourceData,
  type DndTargetData,
  multipleImageDndSource,
  removeFromBoardDndTarget,
  replaceLayerWithImageDndTarget,
  selectForCompareDndTarget,
  setGlobalReferenceImageDndTarget,
  setNodeImageFieldDndTarget,
  setRegionalGuidanceReferenceImageDndTarget,
  setUpscaleInitialImageFromImageDndTarget,
  singleImageDndSource,
} from 'features/dnd2/types';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { imagesApi } from 'services/api/endpoints/images';

const log = logger('system');

export const dndDropped = createAction<{
  sourceData: DndSourceData;
  targetData: DndTargetData;
}>('dnd/dndDropped2');

export const addDndDroppedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: dndDropped,
    effect: (action, { dispatch, getState }) => {
      const { sourceData, targetData } = action.payload;

      // Single image dropped
      if (singleImageDndSource.typeGuard(sourceData)) {
        log.debug({ sourceData, targetData }, 'Image dropped');
        const { imageDTO } = sourceData;

        // Image dropped on IP Adapter
        if (
          setGlobalReferenceImageDndTarget.typeGuard(targetData) &&
          setGlobalReferenceImageDndTarget.validateDrop(sourceData, targetData)
        ) {
          const { globalReferenceImageId } = targetData;
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
          setRegionalGuidanceReferenceImageDndTarget.typeGuard(targetData) &&
          setRegionalGuidanceReferenceImageDndTarget.validateDrop(sourceData, targetData)
        ) {
          const { regionalGuidanceId, referenceImageId } = targetData;
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
          newRasterLayerFromImageDndTarget.typeGuard(targetData) &&
          newRasterLayerFromImageDndTarget.validateDrop(sourceData, targetData)
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
          addInpaintMaskFromImageDndTarget.typeGuard(targetData) &&
          addInpaintMaskFromImageDndTarget.validateDrop(sourceData, targetData)
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
          addRegionalGuidanceFromImageDndTarget.typeGuard(targetData) &&
          addRegionalGuidanceFromImageDndTarget.validateDrop(sourceData, targetData)
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
          newControlLayerFromImageDndTarget.typeGuard(targetData) &&
          newControlLayerFromImageDndTarget.validateDrop(sourceData, targetData)
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
          addRegionalGuidanceReferenceImageFromImageDndTarget.typeGuard(targetData) &&
          addRegionalGuidanceReferenceImageFromImageDndTarget.validateDrop(sourceData, targetData)
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
          addGlobalReferenceImageFromImageDndTarget.typeGuard(targetData) &&
          addGlobalReferenceImageFromImageDndTarget.validateDrop(sourceData, targetData)
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
          replaceLayerWithImageDndTarget.typeGuard(targetData) &&
          replaceLayerWithImageDndTarget.validateDrop(sourceData, targetData)
        ) {
          const state = getState();
          const { entityIdentifier } = targetData;
          const imageObject = imageDTOToImageObject(imageDTO);
          const { x, y } = selectCanvasSlice(state).bbox.rect;
          dispatch(entityRasterized({ entityIdentifier, imageObject, position: { x, y }, replaceObjects: true }));
          dispatch(entitySelected({ entityIdentifier }));
          return;
        }

        // Image dropped on node image field
        if (
          setNodeImageFieldDndTarget.typeGuard(targetData) &&
          setNodeImageFieldDndTarget.validateDrop(sourceData, targetData)
        ) {
          const { fieldName, nodeId } = targetData;
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
          selectForCompareDndTarget.typeGuard(targetData) &&
          selectForCompareDndTarget.validateDrop(sourceData, targetData)
        ) {
          dispatch(imageToCompareChanged(imageDTO));
          return;
        }

        // Image added to board
        if (addToBoardDndTarget.typeGuard(targetData) && addToBoardDndTarget.validateDrop(sourceData, targetData)) {
          const { boardId } = targetData;
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
          removeFromBoardDndTarget.typeGuard(targetData) &&
          removeFromBoardDndTarget.validateDrop(sourceData, targetData)
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
          setUpscaleInitialImageFromImageDndTarget.typeGuard(targetData) &&
          setUpscaleInitialImageFromImageDndTarget.validateDrop(sourceData, targetData)
        ) {
          dispatch(upscaleInitialImageChanged(imageDTO));
          return;
        }
      }

      if (multipleImageDndSource.typeGuard(sourceData)) {
        log.debug({ sourceData, targetData }, 'Multiple images dropped');
        const { imageDTOs } = sourceData;

        // Multiple images dropped on user board
        if (addToBoardDndTarget.typeGuard(targetData) && addToBoardDndTarget.validateDrop(sourceData, targetData)) {
          const { boardId } = targetData;
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
          removeFromBoardDndTarget.typeGuard(targetData) &&
          removeFromBoardDndTarget.validateDrop(sourceData, targetData)
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
