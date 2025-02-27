import type { AppDispatch, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { selectDefaultIPAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { CanvasEntityAdapterBase } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasReset } from 'features/controlLayers/store/actions';
import {
  bboxChangedFromCanvas,
  controlLayerAdded,
  entityRasterized,
  inpaintMaskAdded,
  rasterLayerAdded,
  referenceImageAdded,
  referenceImageIPAdapterImageChanged,
  rgAdded,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectBboxModelBase, selectBboxRect } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasEntityType,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  CanvasRenderableEntityIdentifier,
  CanvasRenderableEntityState,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims, initialControlNet } from 'features/controlLayers/store/util';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import type { BoardId } from 'features/gallery/store/types';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const setGlobalReferenceImage = (arg: {
  imageDTO: ImageDTO;
  entityIdentifier: CanvasEntityIdentifier<'reference_image'>;
  dispatch: AppDispatch;
}) => {
  const { imageDTO, entityIdentifier, dispatch } = arg;
  dispatch(referenceImageIPAdapterImageChanged({ entityIdentifier, imageDTO }));
};

export const setRegionalGuidanceReferenceImage = (arg: {
  imageDTO: ImageDTO;
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>;
  referenceImageId: string;
  dispatch: AppDispatch;
}) => {
  const { imageDTO, entityIdentifier, referenceImageId, dispatch } = arg;
  dispatch(rgIPAdapterImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
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

export const setComparisonImage = (arg: { imageDTO: ImageDTO; dispatch: AppDispatch }) => {
  const { imageDTO, dispatch } = arg;
  dispatch(imageToCompareChanged(imageDTO));
};

export const createNewCanvasEntityFromImage = (arg: {
  imageDTO: ImageDTO;
  type: CanvasEntityType | 'regional_guidance_with_reference_image';
  dispatch: AppDispatch;
  getState: () => RootState;
  overrides?: Partial<Pick<CanvasRenderableEntityState, 'isEnabled' | 'isLocked' | 'name' | 'position'>>;
}) => {
  const { type, imageDTO, dispatch, getState, overrides: _overrides } = arg;
  const state = getState();
  const imageObject = imageDTOToImageObject(imageDTO);
  const { x, y } = selectBboxRect(state);
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
    case 'reference_image': {
      const ipAdapter = deepClone(selectDefaultIPAdapter(getState()));
      ipAdapter.image = imageDTOToImageWithDims(imageDTO);
      dispatch(referenceImageAdded({ overrides: { ipAdapter }, isSelected: true }));
      break;
    }
    case 'regional_guidance_with_reference_image': {
      const ipAdapter = deepClone(selectDefaultIPAdapter(getState()));
      ipAdapter.image = imageDTOToImageWithDims(imageDTO);
      const referenceImages = [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter }];
      dispatch(rgAdded({ overrides: { referenceImages }, isSelected: true }));
      break;
    }
  }
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
export const newCanvasFromImage = (arg: {
  imageDTO: ImageDTO;
  type: CanvasEntityType | 'regional_guidance_with_reference_image';
  withResize: boolean;
  dispatch: AppDispatch;
  getState: () => RootState;
}) => {
  const { type, imageDTO, withResize, dispatch, getState } = arg;
  const state = getState();

  const base = selectBboxModelBase(state);
  // Calculate the new bbox dimensions to fit the image's aspect ratio at the optimal size
  const ratio = imageDTO.width / imageDTO.height;
  const optimalDimension = getOptimalDimension(base);
  const { width, height } = calculateNewSize(ratio, optimalDimension ** 2, base);

  const imageObject = imageDTOToImageObject(imageDTO);
  const { x, y } = selectBboxRect(state);

  const addInitCallback = (id: string) => {
    CanvasEntityAdapterBase.registerInitCallback(async (adapter) => {
      // Skip the callback if the adapter is not the one we are creating
      if (adapter.id !== id) {
        return false;
      }
      // Fit the layer to the bbox w/ fill strategy
      await adapter.transformer.startTransform({ silent: true });
      adapter.transformer.fitToBboxFill();
      await adapter.transformer.applyTransform();
      return true;
    });
  };

  switch (type) {
    case 'raster_layer': {
      const overrides = {
        id: getPrefixedId('raster_layer'),
        objects: [imageObject],
        position: { x, y },
      } satisfies Partial<CanvasRasterLayerState>;
      if (withResize) {
        addInitCallback(overrides.id);
      }
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      break;
    }
    case 'control_layer': {
      const overrides = {
        id: getPrefixedId('control_layer'),
        objects: [imageObject],
        position: { x, y },
        controlAdapter: deepClone(initialControlNet),
      } satisfies Partial<CanvasControlLayerState>;
      if (withResize) {
        addInitCallback(overrides.id);
      }
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(controlLayerAdded({ overrides, isSelected: true }));
      break;
    }
    case 'inpaint_mask': {
      const overrides = {
        id: getPrefixedId('inpaint_mask'),
        objects: [imageObject],
        position: { x, y },
      } satisfies Partial<CanvasInpaintMaskState>;
      if (withResize) {
        addInitCallback(overrides.id);
      }
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(inpaintMaskAdded({ overrides, isSelected: true }));
      break;
    }
    case 'regional_guidance': {
      const overrides = {
        id: getPrefixedId('regional_guidance'),
        objects: [imageObject],
        position: { x, y },
      } satisfies Partial<CanvasRegionalGuidanceState>;
      if (withResize) {
        addInitCallback(overrides.id);
      }
      dispatch(canvasReset());
      // The `bboxChangedFromCanvas` reducer does no validation! Careful!
      dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
      dispatch(rgAdded({ overrides, isSelected: true }));
      break;
    }
    case 'reference_image': {
      const ipAdapter = deepClone(selectDefaultIPAdapter(getState()));
      ipAdapter.image = imageDTOToImageWithDims(imageDTO);
      dispatch(canvasReset());
      dispatch(referenceImageAdded({ overrides: { ipAdapter }, isSelected: true }));
      break;
    }
    case 'regional_guidance_with_reference_image': {
      const ipAdapter = deepClone(selectDefaultIPAdapter(getState()));
      ipAdapter.image = imageDTOToImageWithDims(imageDTO);
      const referenceImages = [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter }];
      dispatch(canvasReset());
      dispatch(rgAdded({ overrides: { referenceImages }, isSelected: true }));
      break;
    }
    default:
      assert<Equals<typeof type, never>>(false);
  }
};

export const replaceCanvasEntityObjectsWithImage = (arg: {
  imageDTO: ImageDTO;
  entityIdentifier: CanvasRenderableEntityIdentifier;
  dispatch: AppDispatch;
  getState: () => RootState;
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

export const addImagesToBoard = (arg: { imageDTOs: ImageDTO[]; boardId: BoardId; dispatch: AppDispatch }) => {
  const { imageDTOs, boardId, dispatch } = arg;
  dispatch(imagesApi.endpoints.addImagesToBoard.initiate({ imageDTOs, board_id: boardId }, { track: false }));
  dispatch(selectionChanged([]));
};

export const removeImagesFromBoard = (arg: { imageDTOs: ImageDTO[]; dispatch: AppDispatch }) => {
  const { imageDTOs, dispatch } = arg;
  dispatch(imagesApi.endpoints.removeImagesFromBoard.initiate({ imageDTOs }, { track: false }));
  dispatch(selectionChanged([]));
};
