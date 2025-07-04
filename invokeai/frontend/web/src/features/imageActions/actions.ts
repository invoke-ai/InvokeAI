import type { AppDispatch, AppGetState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { getDefaultRegionalGuidanceRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasReset } from 'features/controlLayers/store/actions';
import {
  bboxChangedFromCanvas,
  canvasClearHistory,
  controlLayerAdded,
  entityRasterized,
  inpaintMaskAdded,
  rasterLayerAdded,
  rgAdded,
  rgRefImageImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { refImageImageChanged } from 'features/controlLayers/store/refImagesSlice';
import { selectBboxModelBase, selectBboxRect } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasEntityState,
  CanvasEntityType,
  CanvasImageState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims, initialControlNet } from 'features/controlLayers/store/util';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import type { BoardId } from 'features/gallery/store/types';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { imageDTOToFile, imagesApi, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const setGlobalReferenceImage = (arg: { imageDTO: ImageDTO; id: string; dispatch: AppDispatch }) => {
  const { imageDTO, id, dispatch } = arg;
  dispatch(refImageImageChanged({ id, imageDTO }));
};

export const setRegionalGuidanceReferenceImage = (arg: {
  imageDTO: ImageDTO;
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>;
  referenceImageId: string;
  dispatch: AppDispatch;
}) => {
  const { imageDTO, entityIdentifier, referenceImageId, dispatch } = arg;
  dispatch(rgRefImageImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
};

export const setUpscaleInitialImage = (arg: { imageDTO: ImageDTO; dispatch: AppDispatch }) => {
  const { imageDTO, dispatch } = arg;
  dispatch(upscaleInitialImageChanged(imageDTO));
};

export const setNodeImageFieldImage = (arg: {
  imageDTO: ImageDTO;
  fieldIdentifier: FieldIdentifier;
  dispatch: AppDispatch;
}) => {
  const { imageDTO, fieldIdentifier, dispatch } = arg;
  dispatch(fieldImageValueChanged({ ...fieldIdentifier, value: imageDTO }));
};

export const setComparisonImage = (arg: { image_name: string; dispatch: AppDispatch }) => {
  const { image_name, dispatch } = arg;
  dispatch(imageToCompareChanged(image_name));
};

export const createNewCanvasEntityFromImage = async (arg: {
  imageDTO: ImageDTO;
  type: CanvasEntityType | 'regional_guidance_with_reference_image';
  withResize?: boolean;
  dispatch: AppDispatch;
  getState: AppGetState;
  overrides?: Partial<Pick<CanvasEntityState, 'isEnabled' | 'isLocked' | 'name' | 'position'>>;
}) => {
  const { type, imageDTO, dispatch, getState, withResize, overrides: _overrides } = arg;
  const state = getState();
  const { x, y, width, height } = selectBboxRect(state);

  let imageObject: CanvasImageState;

  if (withResize && (width !== imageDTO.width || height !== imageDTO.height)) {
    const resizedImageDTO = await uploadImage({
      file: await imageDTOToFile(imageDTO),
      image_category: 'general',
      is_intermediate: true,
      silent: true,
      resize_to: { width, height },
    });
    imageObject = imageDTOToImageObject(resizedImageDTO);
  } else {
    imageObject = imageDTOToImageObject(imageDTO);
  }

  const overrides = {
    objects: [imageObject],
    position: { x, y },
    ..._overrides,
  };

  switch (type) {
    case 'raster_layer': {
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      break;
    }
    case 'control_layer': {
      dispatch(
        controlLayerAdded({
          overrides: { ...overrides, controlAdapter: deepClone(initialControlNet) },
          isSelected: true,
        })
      );
      break;
    }
    case 'inpaint_mask': {
      dispatch(inpaintMaskAdded({ overrides, isSelected: true }));
      break;
    }
    case 'regional_guidance': {
      dispatch(rgAdded({ overrides, isSelected: true }));
      break;
    }
    case 'regional_guidance_with_reference_image': {
      const config = getDefaultRegionalGuidanceRefImageConfig(getState);
      config.image = imageDTOToImageWithDims(imageDTO);
      const referenceImages = [{ id: getPrefixedId('regional_guidance_reference_image'), config }];
      dispatch(rgAdded({ overrides: { referenceImages }, isSelected: true }));
      break;
    }
  }

  navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
};

/**
 * Creates a new canvas with the given image as the only layer:
 * - Reset the canvas
 * - Resize the bbox to the image's aspect ratio at the optimal size for the selected model
 * - Add the image as a layer of the given type
 * - If `withResize`: Resizes the layer to fit the bbox using the 'fill' strategy
 *
 * This allows the user to immediately generate a new image from the given image without any additional steps.
 *
 * Using 'raster_layer' for the type and enabling `withResize` replicates the common img2img flow.
 */
export const newCanvasFromImage = async (arg: {
  imageDTO: ImageDTO;
  type: CanvasEntityType | 'regional_guidance_with_reference_image';
  withResize?: boolean;
  withInpaintMask?: boolean;
  dispatch: AppDispatch;
  getState: AppGetState;
}) => {
  const { type, imageDTO, withResize = false, withInpaintMask = false, dispatch, getState } = arg;
  const state = getState();

  const base = selectBboxModelBase(state);
  // Calculate the new bbox dimensions to fit the image's aspect ratio at the optimal size
  const ratio = imageDTO.width / imageDTO.height;
  const optimalDimension = getOptimalDimension(base);
  const { width, height } = calculateNewSize(ratio, optimalDimension ** 2, base);

  let imageObject: CanvasImageState;

  if (withResize && (width !== imageDTO.width || height !== imageDTO.height)) {
    const resizedImageDTO = await uploadImage({
      file: await imageDTOToFile(imageDTO),
      image_category: 'general',
      is_intermediate: true,
      silent: true,
      resize_to: { width, height },
    });
    imageObject = imageDTOToImageObject(resizedImageDTO);
  } else {
    imageObject = imageDTOToImageObject(imageDTO);
  }

  const addFitOnLayerInitCallback = (adapterId: string) => {
    CanvasEntityTransformer.registerBboxUpdatedCallback((adapter) => {
      // Skip the callback if the adapter is not the one we are creating
      if (adapter.id !== adapterId) {
        return Promise.resolve(false);
      }
      adapter.manager.stage.fitBboxAndLayersToStage();
      return Promise.resolve(true);
    });
  };

  switch (type) {
    case 'raster_layer': {
      const overrides = {
        id: getPrefixedId('raster_layer'),
        objects: [imageObject],
      } satisfies Partial<CanvasRasterLayerState>;
      addFitOnLayerInitCallback(overrides.id);
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      if (withInpaintMask) {
        dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
      }
      dispatch(canvasClearHistory());
      break;
    }
    case 'control_layer': {
      const overrides = {
        id: getPrefixedId('control_layer'),
        objects: [imageObject],
        controlAdapter: deepClone(initialControlNet),
      } satisfies Partial<CanvasControlLayerState>;
      addFitOnLayerInitCallback(overrides.id);
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(controlLayerAdded({ overrides, isSelected: true }));
      if (withInpaintMask) {
        dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
      }
      dispatch(canvasClearHistory());
      break;
    }
    case 'inpaint_mask': {
      const overrides = {
        id: getPrefixedId('inpaint_mask'),
        objects: [imageObject],
      } satisfies Partial<CanvasInpaintMaskState>;
      addFitOnLayerInitCallback(overrides.id);
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(inpaintMaskAdded({ overrides, isSelected: true }));
      if (withInpaintMask) {
        dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
      }
      dispatch(canvasClearHistory());
      break;
    }
    case 'regional_guidance': {
      const overrides = {
        id: getPrefixedId('regional_guidance'),
        objects: [imageObject],
      } satisfies Partial<CanvasRegionalGuidanceState>;
      addFitOnLayerInitCallback(overrides.id);
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(rgAdded({ overrides, isSelected: true }));
      if (withInpaintMask) {
        dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
      }
      dispatch(canvasClearHistory());
      break;
    }
    case 'regional_guidance_with_reference_image': {
      const config = getDefaultRegionalGuidanceRefImageConfig(getState);
      config.image = imageDTOToImageWithDims(imageDTO);
      const referenceImages = [{ id: getPrefixedId('regional_guidance_reference_image'), config }];
      dispatch(canvasReset());
      dispatch(rgAdded({ overrides: { referenceImages }, isSelected: true }));
      if (withInpaintMask) {
        dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
      }
      dispatch(canvasClearHistory());
      break;
    }
    default:
      assert<Equals<typeof type, never>>(false);
  }

  // Switch to the Canvas panel when creating a new canvas from image
  navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
};

export const replaceCanvasEntityObjectsWithImage = (arg: {
  imageDTO: ImageDTO;
  entityIdentifier: CanvasEntityIdentifier;
  dispatch: AppDispatch;
  getState: AppGetState;
}) => {
  const { imageDTO, entityIdentifier, dispatch, getState } = arg;
  const imageObject = imageDTOToImageObject(imageDTO);
  const { x, y } = selectBboxRect(getState());
  dispatch(
    entityRasterized({
      entityIdentifier,
      imageObject,
      position: { x, y },
      replaceObjects: true,
      isSelected: true,
    })
  );
};

export const addImagesToBoard = (arg: { image_names: string[]; boardId: BoardId; dispatch: AppDispatch }) => {
  const { image_names, boardId, dispatch } = arg;
  dispatch(imagesApi.endpoints.addImagesToBoard.initiate({ image_names, board_id: boardId }, { track: false }));
  dispatch(selectionChanged([]));
};

export const removeImagesFromBoard = (arg: { image_names: string[]; dispatch: AppDispatch }) => {
  const { image_names, dispatch } = arg;
  dispatch(imagesApi.endpoints.removeImagesFromBoard.initiate({ image_names }, { track: false }));
  dispatch(selectionChanged([]));
};
