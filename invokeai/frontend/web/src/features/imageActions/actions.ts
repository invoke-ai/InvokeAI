import type { AppDispatch, RootState } from 'app/store/store';
import { selectDefaultControlAdapter, selectDefaultIPAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  controlLayerAdded,
  entityRasterized,
  inpaintMaskAdded,
  rasterLayerAdded,
  referenceImageAdded,
  referenceImageIPAdapterImageChanged,
  rgAdded,
  rgIPAdapterImageChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectBboxRect } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CanvasEntityType,
  CanvasRenderableEntityIdentifier,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { selectComparisonImages } from 'features/gallery/components/ImageViewer/common';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import type { BoardId } from 'features/gallery/store/types';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
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

//#region New Canvas Entity
const _newCanvasEntity = buildTypeAndKey('new-canvas-entity');
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
      }
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
