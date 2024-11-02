import type { AppDispatch, RootState } from 'app/store/store';
import { selectDefaultControlAdapter, selectDefaultIPAdapter } from 'features/controlLayers/hooks/addLayerHooks';
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
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { calculateNewSize } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { selectComparisonImages } from 'features/gallery/components/ImageViewer/common';
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
import type { JsonObject } from 'type-fest';

export type RecordUnknown = Record<string | symbol, unknown>;

export type ActionData<
  Type extends string = string,
  PrivateKey extends symbol = symbol,
  Payload extends JsonObject | void = JsonObject | void,
> = {
  [key in PrivateKey]: true;
} & {
  id: string;
  type: Type;
  payload: Payload;
};

export const buildTypeAndKey = <T extends string>(type: T) => {
  const key = Symbol(type);
  return { type, key } as const;
};

export const buildTypeGuard = <T extends ActionData>(key: symbol) => {
  const typeGuard = (val: RecordUnknown): val is T => Boolean(val[key]);
  return typeGuard;
};

export const buildGetData = <T extends ActionData>(key: symbol, type: T['type']) => {
  const getData = (payload: T['payload'] extends undefined ? void : T['payload'], id?: string): T =>
    ({
      [key]: true,
      id: id ?? getPrefixedId(type),
      type,
      payload,
    }) as T;
  return getData;
};

export type ActionSourceApi<SourceData extends ActionData> = {
  key: symbol;
  type: SourceData['type'];
  typeGuard: ReturnType<typeof buildTypeGuard<SourceData>>;
  getData: ReturnType<typeof buildGetData<SourceData>>;
};
//#region Single Image
const _singleImage = buildTypeAndKey('single-image');
export type SingleImageSourceData = ActionData<
  typeof _singleImage.type,
  typeof _singleImage.key,
  { imageDTO: ImageDTO }
>;
export const singleImageSourceApi: ActionSourceApi<SingleImageSourceData> = {
  ..._singleImage,
  typeGuard: buildTypeGuard(_singleImage.key),
  getData: buildGetData(_singleImage.key, _singleImage.type),
};
//#endregion

//#region Multiple Image
const _multipleImage = buildTypeAndKey('multiple-image');
export type MultipleImageSourceData = ActionData<
  typeof _multipleImage.type,
  typeof _multipleImage.key,
  { imageDTOs: ImageDTO[]; boardId: BoardId }
>;
export const multipleImageSourceApi: ActionSourceApi<MultipleImageSourceData> = {
  ..._multipleImage,
  typeGuard: buildTypeGuard(_multipleImage.key),
  getData: buildGetData(_multipleImage.key, _multipleImage.type),
};
//#endregion

type ActionTargetApi<TargetData extends ActionData, SourceData extends ActionData> = {
  key: symbol;
  type: TargetData['type'];
  typeGuard: ReturnType<typeof buildTypeGuard<TargetData>>;
  getData: ReturnType<typeof buildGetData<TargetData>>;
  isValid: (
    sourceData: RecordUnknown,
    targetData: TargetData,
    dispatch: AppDispatch,
    getState: () => RootState
  ) => boolean;
  handler: (sourceData: SourceData, targetData: TargetData, dispatch: AppDispatch, getState: () => RootState) => void;
};

//#region Set Global Reference Image
const _setGlobalReferenceImage = buildTypeAndKey('set-global-reference-image');
export type SetGlobalReferenceImageActionData = ActionData<
  typeof _setGlobalReferenceImage.type,
  typeof _setGlobalReferenceImage.key,
  { entityIdentifier: CanvasEntityIdentifier<'reference_image'> }
>;
export const setGlobalReferenceImageActionApi: ActionTargetApi<
  SetGlobalReferenceImageActionData,
  SingleImageSourceData
> = {
  ..._setGlobalReferenceImage,
  typeGuard: buildTypeGuard(_setGlobalReferenceImage.key),
  getData: buildGetData(_setGlobalReferenceImage.key, _setGlobalReferenceImage.type),
  isValid: (sourceData, _targetData, _dispatch, _getState) => {
    if (singleImageSourceApi.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: (sourceData, targetData, dispatch, _getState) => {
    const { imageDTO } = sourceData.payload;
    const { entityIdentifier } = targetData.payload;
    dispatch(referenceImageIPAdapterImageChanged({ entityIdentifier, imageDTO }));
  },
};
//#endregion

//#region Set Regional Guidance Reference Image
const _setRegionalGuidanceReferenceImage = buildTypeAndKey('set-regional-guidance-reference-image');
export type SetRegionalGuidanceReferenceImageActionData = ActionData<
  typeof _setRegionalGuidanceReferenceImage.type,
  typeof _setRegionalGuidanceReferenceImage.key,
  { entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>; referenceImageId: string }
>;
export const setRegionalGuidanceReferenceImageActionApi: ActionTargetApi<
  SetRegionalGuidanceReferenceImageActionData,
  SingleImageSourceData
> = {
  ..._setRegionalGuidanceReferenceImage,
  typeGuard: buildTypeGuard(_setRegionalGuidanceReferenceImage.key),
  getData: buildGetData(_setRegionalGuidanceReferenceImage.key, _setRegionalGuidanceReferenceImage.type),
  isValid: (sourceData, _targetData, _dispatch, _getState) => {
    if (singleImageSourceApi.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: (sourceData, targetData, dispatch, _getState) => {
    const { imageDTO } = sourceData.payload;
    const { entityIdentifier, referenceImageId } = targetData.payload;
    dispatch(rgIPAdapterImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
  },
};
//#endregion

//# Set Upscale Initial Image
const _setUpscaleInitialImage = buildTypeAndKey('set-upscale-initial-image');
export type SetUpscaleInitialImageActionData = ActionData<
  typeof _setUpscaleInitialImage.type,
  typeof _setUpscaleInitialImage.key,
  void
>;
export const setUpscaleInitialImageActionApi: ActionTargetApi<SetUpscaleInitialImageActionData, SingleImageSourceData> =
  {
    ..._setUpscaleInitialImage,
    typeGuard: buildTypeGuard(_setUpscaleInitialImage.key),
    getData: buildGetData(_setUpscaleInitialImage.key, _setUpscaleInitialImage.type),
    isValid: (sourceData, _targetData, _dispatch, _getState) => {
      if (singleImageSourceApi.typeGuard(sourceData)) {
        return true;
      }
      return false;
    },
    handler: (sourceData, _targetData, dispatch, _getState) => {
      const { imageDTO } = sourceData.payload;
      dispatch(upscaleInitialImageChanged(imageDTO));
    },
  };
//#endregion

//#region Set Node Image Field Image
const _setNodeImageFieldImage = buildTypeAndKey('set-node-image-field-image');
export type SetNodeImageFieldImageActionData = ActionData<
  typeof _setNodeImageFieldImage.type,
  typeof _setNodeImageFieldImage.key,
  { fieldIdentifer: FieldIdentifier }
>;
export const setNodeImageFieldImageActionApi: ActionTargetApi<SetNodeImageFieldImageActionData, SingleImageSourceData> =
  {
    ..._setNodeImageFieldImage,
    typeGuard: buildTypeGuard(_setNodeImageFieldImage.key),
    getData: buildGetData(_setNodeImageFieldImage.key, _setNodeImageFieldImage.type),
    isValid: (sourceData, _targetData, _dispatch, _getState) => {
      if (singleImageSourceApi.typeGuard(sourceData)) {
        return true;
      }
      return false;
    },
    handler: (sourceData, targetData, dispatch, _getState) => {
      const { imageDTO } = sourceData.payload;
      const { fieldIdentifer } = targetData.payload;
      dispatch(fieldImageValueChanged({ ...fieldIdentifer, value: imageDTO }));
    },
  };
//#endregion

//# Set Comparison Image
const _setComparisonImage = buildTypeAndKey('set-comparison-image');
export type SetComparisonImageActionData = ActionData<
  typeof _setComparisonImage.type,
  typeof _setComparisonImage.key,
  void
>;
export const setComparisonImageActionApi: ActionTargetApi<SetComparisonImageActionData, SingleImageSourceData> = {
  ..._setComparisonImage,
  typeGuard: buildTypeGuard(_setComparisonImage.key),
  getData: buildGetData(_setComparisonImage.key, _setComparisonImage.type),
  isValid: (sourceData, _targetData, _dispatch, getState) => {
    if (!singleImageSourceApi.typeGuard(sourceData)) {
      return false;
    }
    const { firstImage, secondImage } = selectComparisonImages(getState());
    // Do not allow the same images to be selected for comparison
    if (sourceData.payload.imageDTO.image_name === firstImage?.image_name) {
      return false;
    }
    if (sourceData.payload.imageDTO.image_name === secondImage?.image_name) {
      return false;
    }
    return true;
  },
  handler: (sourceData, _targetData, dispatch, _getState) => {
    const { imageDTO } = sourceData.payload;
    dispatch(imageToCompareChanged(imageDTO));
  },
};
//#endregion

//#region New Canvas Entity from Image
const _newCanvasEntity = buildTypeAndKey('new-canvas-entity-from-image');
export type NewCanvasEntityFromImageActionData = ActionData<
  typeof _newCanvasEntity.type,
  typeof _newCanvasEntity.key,
  { type: CanvasEntityType | 'regional_guidance_with_reference_image' }
>;
export const newCanvasEntityFromImageActionApi: ActionTargetApi<
  NewCanvasEntityFromImageActionData,
  SingleImageSourceData
> = {
  ..._newCanvasEntity,
  typeGuard: buildTypeGuard(_newCanvasEntity.key),
  getData: buildGetData(_newCanvasEntity.key, _newCanvasEntity.type),
  isValid: (sourceData, _targetData, _dispatch, _getState) => {
    if (!singleImageSourceApi.typeGuard(sourceData)) {
      return false;
    }
    return true;
  },
  handler: (sourceData, targetData, dispatch, getState) => {
    const { type } = targetData.payload;
    const { imageDTO } = sourceData.payload;
    const state = getState();
    const imageObject = imageDTOToImageObject(imageDTO);
    const { x, y } = selectBboxRect(state);
    const overrides = {
      objects: [imageObject],
      position: { x, y },
    };
    switch (type) {
      case 'raster_layer': {
        dispatch(rasterLayerAdded({ overrides, isSelected: true }));
        break;
      }
      case 'control_layer': {
        const controlAdapter = selectDefaultControlAdapter(state);
        dispatch(controlLayerAdded({ overrides: { ...overrides, controlAdapter }, isSelected: true }));
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
        const ipAdapter = selectDefaultIPAdapter(getState());
        ipAdapter.image = imageDTOToImageWithDims(imageDTO);
        dispatch(referenceImageAdded({ overrides: { ipAdapter }, isSelected: true }));
        break;
      }
      case 'regional_guidance_with_reference_image': {
        const ipAdapter = selectDefaultIPAdapter(getState());
        ipAdapter.image = imageDTOToImageWithDims(imageDTO);
        const referenceImages = [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter }];
        dispatch(rgAdded({ overrides: { referenceImages }, isSelected: true }));
        break;
      }
    }
  },
};
//#endregion

//#region New Canvas from Image
const _newCanvasFromImage = buildTypeAndKey('new-canvas-from-image');
export type NewCanvasFromImageActionData = ActionData<
  typeof _newCanvasFromImage.type,
  typeof _newCanvasFromImage.key,
  { type: CanvasEntityType | 'regional_guidance_with_reference_image' }
>;
/**
 * Returns a function that adds a new canvas with the given image as the initial image, replicating the img2img flow:
 * - Reset the canvas
 * - Resize the bbox to the image's aspect ratio at the optimal size for the selected model
 * - Add the image as a raster layer
 * - Resizes the layer to fit the bbox using the 'fill' strategy
 *
 * This allows the user to immediately generate a new image from the given image without any additional steps.
 */
export const newCanvasFromImageActionApi: ActionTargetApi<NewCanvasFromImageActionData, SingleImageSourceData> = {
  ..._newCanvasFromImage,
  typeGuard: buildTypeGuard(_newCanvasFromImage.key),
  getData: buildGetData(_newCanvasFromImage.key, _newCanvasFromImage.type),
  isValid: (sourceData, _targetData, _dispatch, _getState) => {
    if (!singleImageSourceApi.typeGuard(sourceData)) {
      return false;
    }
    return true;
  },
  handler: (sourceData, targetData, dispatch, getState) => {
    const { type } = targetData.payload;
    const { imageDTO } = sourceData.payload;
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
        addInitCallback(overrides.id);
        dispatch(canvasReset());
        // The `bboxChangedFromCanvas` reducer does no validation! Careful!
        dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
        dispatch(rasterLayerAdded({ overrides, isSelected: true }));
        break;
      }
      case 'control_layer': {
        const controlAdapter = selectDefaultControlAdapter(state);
        const overrides = {
          id: getPrefixedId('control_layer'),
          objects: [imageObject],
          position: { x, y },
          controlAdapter,
        } satisfies Partial<CanvasControlLayerState>;
        addInitCallback(overrides.id);
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
        addInitCallback(overrides.id);
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
        addInitCallback(overrides.id);
        dispatch(canvasReset());
        // The `bboxChangedFromCanvas` reducer does no validation! Careful!
        dispatch(bboxChangedFromCanvas({ x: 0, y: 0, width, height }));
        dispatch(rgAdded({ overrides, isSelected: true }));
        break;
      }
      case 'reference_image': {
        const ipAdapter = selectDefaultIPAdapter(getState());
        ipAdapter.image = imageDTOToImageWithDims(imageDTO);
        dispatch(canvasReset());
        dispatch(referenceImageAdded({ overrides: { ipAdapter }, isSelected: true }));
        break;
      }
      case 'regional_guidance_with_reference_image': {
        const ipAdapter = selectDefaultIPAdapter(getState());
        ipAdapter.image = imageDTOToImageWithDims(imageDTO);
        const referenceImages = [{ id: getPrefixedId('regional_guidance_reference_image'), ipAdapter }];
        dispatch(canvasReset());
        dispatch(rgAdded({ overrides: { referenceImages }, isSelected: true }));
        break;
      }
      default:
        assert<Equals<typeof type, never>>(false);
    }
  },
};
//#endregion

//#region Replace Canvas Entity Objects With Image
const _replaceCanvasEntityObjectsWithImage = buildTypeAndKey('replace-canvas-entity-objects-with-image');
export type ReplaceCanvasEntityObjectsWithImageActionData = ActionData<
  typeof _replaceCanvasEntityObjectsWithImage.type,
  typeof _replaceCanvasEntityObjectsWithImage.key,
  { entityIdentifier: CanvasRenderableEntityIdentifier }
>;
export const replaceCanvasEntityObjectsWithImageActionApi: ActionTargetApi<
  ReplaceCanvasEntityObjectsWithImageActionData,
  SingleImageSourceData
> = {
  ..._replaceCanvasEntityObjectsWithImage,
  typeGuard: buildTypeGuard(_replaceCanvasEntityObjectsWithImage.key),
  getData: buildGetData(_replaceCanvasEntityObjectsWithImage.key, _replaceCanvasEntityObjectsWithImage.type),
  isValid: (sourceData, _targetData, _dispatch, _getState) => {
    if (!singleImageSourceApi.typeGuard(sourceData)) {
      return false;
    }
    return true;
  },
  handler: (sourceData, targetData, dispatch, getState) => {
    const { imageDTO } = sourceData.payload;
    const { entityIdentifier } = targetData.payload;
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
  },
};
//#endregion

//#region Add To Board
const _addToBoard = buildTypeAndKey('add-to-board');
export type AddImageToBoardActionData = ActionData<
  typeof _addToBoard.type,
  typeof _addToBoard.key,
  { boardId: BoardId }
>;
export const addImageToBoardActionApi: ActionTargetApi<
  AddImageToBoardActionData,
  SingleImageSourceData | MultipleImageSourceData
> = {
  ..._addToBoard,
  typeGuard: buildTypeGuard(_addToBoard.key),
  getData: buildGetData(_addToBoard.key, _addToBoard.type),
  isValid: (sourceData, targetData, _dispatch, _getState) => {
    if (singleImageSourceApi.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.imageDTO.board_id ?? 'none';
      const destinationBoard = targetData.payload.boardId;
      return currentBoard !== destinationBoard;
    }
    if (multipleImageSourceApi.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.boardId;
      const destinationBoard = targetData.payload.boardId;
      return currentBoard !== destinationBoard;
    }
    return false;
  },
  handler: (sourceData, targetData, dispatch, _getState) => {
    if (singleImageSourceApi.typeGuard(sourceData)) {
      const { imageDTO } = sourceData.payload;
      const { boardId } = targetData.payload;
      dispatch(imagesApi.endpoints.addImageToBoard.initiate({ imageDTO, board_id: boardId }, { track: false }));
      dispatch(selectionChanged([]));
    }

    if (multipleImageSourceApi.typeGuard(sourceData)) {
      const { imageDTOs } = sourceData.payload;
      const { boardId } = targetData.payload;
      dispatch(imagesApi.endpoints.addImagesToBoard.initiate({ imageDTOs, board_id: boardId }, { track: false }));
      dispatch(selectionChanged([]));
    }
  },
};
//#endregion

//#region Remove From Board
const _removeFromBoard = buildTypeAndKey('add-to-board');
export type RemoveImageFromBoardActionData = ActionData<
  typeof _removeFromBoard.type,
  typeof _removeFromBoard.key,
  void
>;
export const removeImageFromBoardActionApi: ActionTargetApi<
  RemoveImageFromBoardActionData,
  SingleImageSourceData | MultipleImageSourceData
> = {
  ..._removeFromBoard,
  typeGuard: buildTypeGuard(_removeFromBoard.key),
  getData: buildGetData(_removeFromBoard.key, _removeFromBoard.type),
  isValid: (sourceData, _targetData, _dispatch, _getState) => {
    if (singleImageSourceApi.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.imageDTO.board_id ?? 'none';
      return currentBoard !== 'none';
    }

    if (multipleImageSourceApi.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.boardId;
      return currentBoard !== 'none';
    }

    return false;
  },
  handler: (sourceData, _targetData, dispatch, _getState) => {
    if (singleImageSourceApi.typeGuard(sourceData)) {
      const { imageDTO } = sourceData.payload;
      dispatch(imagesApi.endpoints.removeImageFromBoard.initiate({ imageDTO }, { track: false }));
      dispatch(selectionChanged([]));
    }

    if (multipleImageSourceApi.typeGuard(sourceData)) {
      const { imageDTOs } = sourceData.payload;
      dispatch(imagesApi.endpoints.removeImagesFromBoard.initiate({ imageDTOs }, { track: false }));
      dispatch(selectionChanged([]));
    }
  },
};
//#endregion

export const singleImageActions = [
  setGlobalReferenceImageActionApi,
  setRegionalGuidanceReferenceImageActionApi,
  setUpscaleInitialImageActionApi,
  setNodeImageFieldImageActionApi,
  setComparisonImageActionApi,
  newCanvasEntityFromImageActionApi,
  replaceCanvasEntityObjectsWithImageActionApi,
  addImageToBoardActionApi,
  removeImageFromBoardActionApi,
] as const;
export type SingleImageAction = ReturnType<(typeof singleImageActions)[number]['getData']>;

export const multipleImageActions = [addImageToBoardActionApi, removeImageFromBoardActionApi] as const;
export type MultipleImageAction = ReturnType<(typeof multipleImageActions)[number]['getData']>;
