import { logger } from 'app/logging/logger';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { refImageAdded } from 'features/controlLayers/store/refImagesSlice';
import type { CanvasEntityIdentifier, CanvasEntityType } from 'features/controlLayers/store/types';
import { imageDTOToCroppableImage } from 'features/controlLayers/store/util';
import { selectComparisonImages } from 'features/gallery/components/ImageViewer/common';
import type { BoardId } from 'features/gallery/store/types';
import {
  addImagesToBoard,
  addVideosToBoard,
  createNewCanvasEntityFromImage,
  newCanvasFromImage,
  removeImagesFromBoard,
  removeVideosFromBoard,
  replaceCanvasEntityObjectsWithImage,
  setComparisonImage,
  setGlobalReferenceImage,
  setNodeImageFieldImage,
  setRegionalGuidanceReferenceImage,
  setUpscaleInitialImage,
} from 'features/imageActions/actions';
import { fieldImageCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import { selectFieldInputInstanceSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { type FieldIdentifier, isImageFieldCollectionInputInstance } from 'features/nodes/types/field';
import { startingFrameImageChanged } from 'features/parameters/store/videoSlice';
import { expandPrompt } from 'features/prompt/PromptExpansion/expand';
import { promptExpansionApi } from 'features/prompt/PromptExpansion/state';
import type { ImageDTO, VideoDTO } from 'services/api/types';
import type { JsonObject } from 'type-fest';

const log = logger('dnd');

type RecordUnknown = Record<string | symbol, unknown>;

type DndData<
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

const buildTypeAndKey = <T extends string>(type: T) => {
  const key = Symbol(type);
  return { type, key } as const;
};

const buildTypeGuard = <T extends DndData>(key: symbol) => {
  const typeGuard = (val: RecordUnknown): val is T => Boolean(val[key]);
  return typeGuard;
};

const buildGetData = <T extends DndData>(key: symbol, type: T['type']) => {
  const getData = (payload: T['payload'] extends undefined ? void : T['payload'], id?: string): T =>
    ({
      [key]: true,
      id: id ?? getPrefixedId(type),
      type,
      payload,
    }) as T;
  return getData;
};

type DndSource<SourceData extends DndData> = {
  key: symbol;
  type: SourceData['type'];
  typeGuard: ReturnType<typeof buildTypeGuard<SourceData>>;
  getData: ReturnType<typeof buildGetData<SourceData>>;
};

//#region Single Video
const _singleVideo = buildTypeAndKey('single-video');
export type SingleVideoDndSourceData = DndData<
  typeof _singleVideo.type,
  typeof _singleVideo.key,
  { videoDTO: VideoDTO }
>;
export const singleVideoDndSource: DndSource<SingleVideoDndSourceData> = {
  ..._singleVideo,
  typeGuard: buildTypeGuard(_singleVideo.key),
  getData: buildGetData(_singleVideo.key, _singleVideo.type),
};
//#endregion

//#region Multiple Image
const _multipleVideo = buildTypeAndKey('multiple-video');
export type MultipleVideoDndSourceData = DndData<
  typeof _multipleVideo.type,
  typeof _multipleVideo.key,
  { video_ids: string[]; board_id: BoardId }
>;
export const multipleVideoDndSource: DndSource<MultipleVideoDndSourceData> = {
  ..._multipleVideo,
  typeGuard: buildTypeGuard(_multipleVideo.key),
  getData: buildGetData(_multipleVideo.key, _multipleVideo.type),
};
//#endregion

//#region Single Image
const _singleImage = buildTypeAndKey('single-image');
export type SingleImageDndSourceData = DndData<
  typeof _singleImage.type,
  typeof _singleImage.key,
  { imageDTO: ImageDTO }
>;
export const singleImageDndSource: DndSource<SingleImageDndSourceData> = {
  ..._singleImage,
  typeGuard: buildTypeGuard(_singleImage.key),
  getData: buildGetData(_singleImage.key, _singleImage.type),
};
//#endregion

//#region Multiple Image
const _multipleImage = buildTypeAndKey('multiple-image');
export type MultipleImageDndSourceData = DndData<
  typeof _multipleImage.type,
  typeof _multipleImage.key,
  { image_names: string[]; board_id: BoardId }
>;
export const multipleImageDndSource: DndSource<MultipleImageDndSourceData> = {
  ..._multipleImage,
  typeGuard: buildTypeGuard(_multipleImage.key),
  getData: buildGetData(_multipleImage.key, _multipleImage.type),
};
//#endregion

const _singleCanvasEntity = buildTypeAndKey('single-canvas-entity');
type SingleCanvasEntityDndSourceData = DndData<
  typeof _singleCanvasEntity.type,
  typeof _singleCanvasEntity.key,
  { entityIdentifier: CanvasEntityIdentifier }
>;
export const singleCanvasEntityDndSource: DndSource<SingleCanvasEntityDndSourceData> = {
  ..._singleCanvasEntity,
  typeGuard: buildTypeGuard(_singleCanvasEntity.key),
  getData: buildGetData(_singleCanvasEntity.key, _singleCanvasEntity.type),
};

type DndTarget<TargetData extends DndData, SourceData extends DndData> = {
  key: symbol;
  type: TargetData['type'];
  typeGuard: ReturnType<typeof buildTypeGuard<TargetData>>;
  getData: ReturnType<typeof buildGetData<TargetData>>;
  isValid: (arg: {
    sourceData: RecordUnknown;
    targetData: TargetData;
    dispatch: AppDispatch;
    getState: AppGetState;
  }) => boolean;
  handler: (arg: {
    sourceData: SourceData;
    targetData: TargetData;
    dispatch: AppDispatch;
    getState: AppGetState;
  }) => void;
};

//#region Set Global Reference Image
const _setGlobalReferenceImage = buildTypeAndKey('set-global-reference-image');
export type SetGlobalReferenceImageDndTargetData = DndData<
  typeof _setGlobalReferenceImage.type,
  typeof _setGlobalReferenceImage.key,
  { id: string }
>;
export const setGlobalReferenceImageDndTarget: DndTarget<
  SetGlobalReferenceImageDndTargetData,
  SingleImageDndSourceData
> = {
  ..._setGlobalReferenceImage,
  typeGuard: buildTypeGuard(_setGlobalReferenceImage.key),
  getData: buildGetData(_setGlobalReferenceImage.key, _setGlobalReferenceImage.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: ({ sourceData, targetData, dispatch }) => {
    const { imageDTO } = sourceData.payload;
    const { id } = targetData.payload;
    setGlobalReferenceImage({ id, imageDTO, dispatch });
  },
};
//#endregion

//#region Add Global Reference Image
const _addGlobalReferenceImage = buildTypeAndKey('add-global-reference-image');
type AddGlobalReferenceImageDndTargetData = DndData<
  typeof _addGlobalReferenceImage.type,
  typeof _addGlobalReferenceImage.key
>;
export const addGlobalReferenceImageDndTarget: DndTarget<
  AddGlobalReferenceImageDndTargetData,
  SingleImageDndSourceData
> = {
  ..._addGlobalReferenceImage,
  typeGuard: buildTypeGuard(_addGlobalReferenceImage.key),
  getData: buildGetData(_addGlobalReferenceImage.key, _addGlobalReferenceImage.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: ({ sourceData, dispatch, getState }) => {
    const { imageDTO } = sourceData.payload;
    const config = getDefaultRefImageConfig(getState);
    config.image = imageDTOToCroppableImage(imageDTO);
    dispatch(refImageAdded({ overrides: { config } }));
  },
};
//#endregion

//#region Set Regional Guidance Reference Image
const _setRegionalGuidanceReferenceImage = buildTypeAndKey('set-regional-guidance-reference-image');
export type SetRegionalGuidanceReferenceImageDndTargetData = DndData<
  typeof _setRegionalGuidanceReferenceImage.type,
  typeof _setRegionalGuidanceReferenceImage.key,
  { entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>; referenceImageId: string }
>;
export const setRegionalGuidanceReferenceImageDndTarget: DndTarget<
  SetRegionalGuidanceReferenceImageDndTargetData,
  SingleImageDndSourceData
> = {
  ..._setRegionalGuidanceReferenceImage,
  typeGuard: buildTypeGuard(_setRegionalGuidanceReferenceImage.key),
  getData: buildGetData(_setRegionalGuidanceReferenceImage.key, _setRegionalGuidanceReferenceImage.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: ({ sourceData, targetData, dispatch }) => {
    const { imageDTO } = sourceData.payload;
    const { entityIdentifier, referenceImageId } = targetData.payload;
    setRegionalGuidanceReferenceImage({ imageDTO, entityIdentifier, referenceImageId, dispatch });
  },
};
//#endregion

//# Set Upscale Initial Image
const _setUpscaleInitialImage = buildTypeAndKey('set-upscale-initial-image');
export type SetUpscaleInitialImageDndTargetData = DndData<
  typeof _setUpscaleInitialImage.type,
  typeof _setUpscaleInitialImage.key,
  void
>;
export const setUpscaleInitialImageDndTarget: DndTarget<SetUpscaleInitialImageDndTargetData, SingleImageDndSourceData> =
  {
    ..._setUpscaleInitialImage,
    typeGuard: buildTypeGuard(_setUpscaleInitialImage.key),
    getData: buildGetData(_setUpscaleInitialImage.key, _setUpscaleInitialImage.type),
    isValid: ({ sourceData }) => {
      if (singleImageDndSource.typeGuard(sourceData)) {
        return true;
      }
      return false;
    },
    handler: ({ sourceData, dispatch }) => {
      const { imageDTO } = sourceData.payload;
      setUpscaleInitialImage({ imageDTO, dispatch });
    },
  };
//#endregion

//#region Set Node Image Field Image
const _setNodeImageFieldImage = buildTypeAndKey('set-node-image-field-image');
export type SetNodeImageFieldImageDndTargetData = DndData<
  typeof _setNodeImageFieldImage.type,
  typeof _setNodeImageFieldImage.key,
  { fieldIdentifier: FieldIdentifier }
>;
export const setNodeImageFieldImageDndTarget: DndTarget<SetNodeImageFieldImageDndTargetData, SingleImageDndSourceData> =
  {
    ..._setNodeImageFieldImage,
    typeGuard: buildTypeGuard(_setNodeImageFieldImage.key),
    getData: buildGetData(_setNodeImageFieldImage.key, _setNodeImageFieldImage.type),
    isValid: ({ sourceData }) => {
      if (singleImageDndSource.typeGuard(sourceData)) {
        return true;
      }
      return false;
    },
    handler: ({ sourceData, targetData, dispatch }) => {
      const { imageDTO } = sourceData.payload;
      const { fieldIdentifier } = targetData.payload;
      setNodeImageFieldImage({ fieldIdentifier, imageDTO, dispatch });
    },
  };
//#endregion

//#region Add Images to Image Collection Node Field
const _addImagesToNodeImageFieldCollection = buildTypeAndKey('add-images-to-image-collection-node-field');
export type AddImagesToNodeImageFieldCollection = DndData<
  typeof _addImagesToNodeImageFieldCollection.type,
  typeof _addImagesToNodeImageFieldCollection.key,
  { fieldIdentifier: FieldIdentifier }
>;
export const addImagesToNodeImageFieldCollectionDndTarget: DndTarget<
  AddImagesToNodeImageFieldCollection,
  SingleImageDndSourceData | MultipleImageDndSourceData
> = {
  ..._addImagesToNodeImageFieldCollection,
  typeGuard: buildTypeGuard(_addImagesToNodeImageFieldCollection.key),
  getData: buildGetData(_addImagesToNodeImageFieldCollection.key, _addImagesToNodeImageFieldCollection.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData) || multipleImageDndSource.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: ({ sourceData, targetData, dispatch, getState }) => {
    if (!singleImageDndSource.typeGuard(sourceData) && !multipleImageDndSource.typeGuard(sourceData)) {
      return;
    }

    const { fieldIdentifier } = targetData.payload;

    const fieldInputInstance = selectFieldInputInstanceSafe(
      selectNodesSlice(getState()),
      fieldIdentifier.nodeId,
      fieldIdentifier.fieldName
    );

    if (!isImageFieldCollectionInputInstance(fieldInputInstance)) {
      log.warn({ fieldIdentifier }, 'Attempted to add images to a non-image field collection');
      return;
    }

    const newValue = fieldInputInstance.value ? [...fieldInputInstance.value] : [];

    if (singleImageDndSource.typeGuard(sourceData)) {
      newValue.push({ image_name: sourceData.payload.imageDTO.image_name });
    } else {
      newValue.push(...sourceData.payload.image_names.map((image_name) => ({ image_name })));
    }

    dispatch(fieldImageCollectionValueChanged({ ...fieldIdentifier, value: newValue }));
  },
};
//#endregion

//# Set Comparison Image
const _setComparisonImage = buildTypeAndKey('set-comparison-image');
export type SetComparisonImageDndTargetData = DndData<
  typeof _setComparisonImage.type,
  typeof _setComparisonImage.key,
  void
>;
export const setComparisonImageDndTarget: DndTarget<SetComparisonImageDndTargetData, SingleImageDndSourceData> = {
  ..._setComparisonImage,
  typeGuard: buildTypeGuard(_setComparisonImage.key),
  getData: buildGetData(_setComparisonImage.key, _setComparisonImage.type),
  isValid: ({ sourceData, getState }) => {
    if (!singleImageDndSource.typeGuard(sourceData)) {
      return false;
    }
    const { firstImage, secondImage } = selectComparisonImages(getState());
    // Do not allow the same images to be selected for comparison
    if (sourceData.payload.imageDTO.image_name === firstImage) {
      return false;
    }
    if (sourceData.payload.imageDTO.image_name === secondImage) {
      return false;
    }
    return true;
  },
  handler: ({ sourceData, dispatch }) => {
    const { imageDTO } = sourceData.payload;
    setComparisonImage({ image_name: imageDTO.image_name, dispatch });
  },
};
//#endregion

//#region New Canvas Entity from Image
const _newCanvasEntity = buildTypeAndKey('new-canvas-entity-from-image');
type NewCanvasEntityFromImageDndTargetData = DndData<
  typeof _newCanvasEntity.type,
  typeof _newCanvasEntity.key,
  {
    type: CanvasEntityType | 'regional_guidance_with_reference_image';
    withResize?: boolean;
  }
>;
export const newCanvasEntityFromImageDndTarget: DndTarget<
  NewCanvasEntityFromImageDndTargetData,
  SingleImageDndSourceData
> = {
  ..._newCanvasEntity,
  typeGuard: buildTypeGuard(_newCanvasEntity.key),
  getData: buildGetData(_newCanvasEntity.key, _newCanvasEntity.type),
  isValid: ({ sourceData }) => {
    if (!singleImageDndSource.typeGuard(sourceData)) {
      return false;
    }
    return true;
  },
  handler: ({ sourceData, targetData, dispatch, getState }) => {
    const { type, withResize } = targetData.payload;
    const { imageDTO } = sourceData.payload;
    createNewCanvasEntityFromImage({ type, imageDTO, withResize, dispatch, getState });
  },
};
//#endregion

//#region New Canvas from Image
const _newCanvas = buildTypeAndKey('new-canvas-from-image');
type NewCanvasFromImageDndTargetData = DndData<
  typeof _newCanvas.type,
  typeof _newCanvas.key,
  {
    type: CanvasEntityType | 'regional_guidance_with_reference_image';
    withResize?: boolean;
    withInpaintMask?: boolean;
  }
>;
export const newCanvasFromImageDndTarget: DndTarget<NewCanvasFromImageDndTargetData, SingleImageDndSourceData> = {
  ..._newCanvas,
  typeGuard: buildTypeGuard(_newCanvas.key),
  getData: buildGetData(_newCanvas.key, _newCanvas.type),
  isValid: ({ sourceData }) => {
    if (!singleImageDndSource.typeGuard(sourceData)) {
      return false;
    }
    return true;
  },
  handler: ({ sourceData, targetData, dispatch, getState }) => {
    const { imageDTO } = sourceData.payload;
    newCanvasFromImage({ imageDTO, dispatch, getState, ...targetData.payload });
  },
};
//#endregion

//#region Replace Canvas Entity Objects With Image
const _replaceCanvasEntityObjectsWithImage = buildTypeAndKey('replace-canvas-entity-objects-with-image');
export type ReplaceCanvasEntityObjectsWithImageDndTargetData = DndData<
  typeof _replaceCanvasEntityObjectsWithImage.type,
  typeof _replaceCanvasEntityObjectsWithImage.key,
  { entityIdentifier: CanvasEntityIdentifier }
>;
export const replaceCanvasEntityObjectsWithImageDndTarget: DndTarget<
  ReplaceCanvasEntityObjectsWithImageDndTargetData,
  SingleImageDndSourceData
> = {
  ..._replaceCanvasEntityObjectsWithImage,
  typeGuard: buildTypeGuard(_replaceCanvasEntityObjectsWithImage.key),
  getData: buildGetData(_replaceCanvasEntityObjectsWithImage.key, _replaceCanvasEntityObjectsWithImage.type),
  isValid: ({ sourceData }) => {
    if (!singleImageDndSource.typeGuard(sourceData)) {
      return false;
    }
    return true;
  },
  handler: ({ sourceData, targetData, dispatch, getState }) => {
    const { imageDTO } = sourceData.payload;
    const { entityIdentifier } = targetData.payload;
    replaceCanvasEntityObjectsWithImage({ imageDTO, entityIdentifier, dispatch, getState });
  },
};
//#endregion

//#region Add To Board
const _addToBoard = buildTypeAndKey('add-to-board');
export type AddImageToBoardDndTargetData = DndData<
  typeof _addToBoard.type,
  typeof _addToBoard.key,
  { boardId: BoardId }
>;
export const addImageToBoardDndTarget: DndTarget<
  AddImageToBoardDndTargetData,
  SingleImageDndSourceData | MultipleImageDndSourceData | SingleVideoDndSourceData | MultipleVideoDndSourceData
> = {
  ..._addToBoard,
  typeGuard: buildTypeGuard(_addToBoard.key),
  getData: buildGetData(_addToBoard.key, _addToBoard.type),
  isValid: ({ sourceData, targetData }) => {
    if (singleVideoDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.videoDTO.board_id ?? 'none';
      const destinationBoard = targetData.payload.boardId;
      return currentBoard !== destinationBoard;
    }
    if (multipleVideoDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.board_id;
      const destinationBoard = targetData.payload.boardId;
      return currentBoard !== destinationBoard;
    }
    if (singleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.imageDTO.board_id ?? 'none';
      const destinationBoard = targetData.payload.boardId;
      return currentBoard !== destinationBoard;
    }
    if (multipleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.board_id;
      const destinationBoard = targetData.payload.boardId;
      return currentBoard !== destinationBoard;
    }
    return false;
  },
  handler: ({ sourceData, targetData, dispatch }) => {
    if (singleVideoDndSource.typeGuard(sourceData)) {
      const { videoDTO } = sourceData.payload;
      const { boardId } = targetData.payload;
      addVideosToBoard({ video_ids: [videoDTO.video_id], boardId, dispatch });
    }

    if (multipleVideoDndSource.typeGuard(sourceData)) {
      const { video_ids } = sourceData.payload;
      const { boardId } = targetData.payload;
      addVideosToBoard({ video_ids, boardId, dispatch });
    }

    if (singleImageDndSource.typeGuard(sourceData)) {
      const { imageDTO } = sourceData.payload;
      const { boardId } = targetData.payload;
      addImagesToBoard({ image_names: [imageDTO.image_name], boardId, dispatch });
    }

    if (multipleImageDndSource.typeGuard(sourceData)) {
      const { image_names } = sourceData.payload;
      const { boardId } = targetData.payload;
      addImagesToBoard({ image_names, boardId, dispatch });
    }
  },
};

//#endregion

//#region Remove From Board
const _removeFromBoard = buildTypeAndKey('remove-from-board');
export type RemoveImageFromBoardDndTargetData = DndData<
  typeof _removeFromBoard.type,
  typeof _removeFromBoard.key,
  void
>;
export const removeImageFromBoardDndTarget: DndTarget<
  RemoveImageFromBoardDndTargetData,
  SingleImageDndSourceData | MultipleImageDndSourceData | SingleVideoDndSourceData | MultipleVideoDndSourceData
> = {
  ..._removeFromBoard,
  typeGuard: buildTypeGuard(_removeFromBoard.key),
  getData: buildGetData(_removeFromBoard.key, _removeFromBoard.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.imageDTO.board_id ?? 'none';
      return currentBoard !== 'none';
    }

    if (multipleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.board_id;
      return currentBoard !== 'none';
    }

    if (singleVideoDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.videoDTO.board_id ?? 'none';
      return currentBoard !== 'none';
    }

    if (multipleVideoDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.payload.board_id;
      return currentBoard !== 'none';
    }

    return false;
  },
  handler: ({ sourceData, dispatch }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      const { imageDTO } = sourceData.payload;
      removeImagesFromBoard({ image_names: [imageDTO.image_name], dispatch });
    }

    if (multipleImageDndSource.typeGuard(sourceData)) {
      const { image_names } = sourceData.payload;
      removeImagesFromBoard({ image_names, dispatch });
    }

    if (singleVideoDndSource.typeGuard(sourceData)) {
      const { videoDTO } = sourceData.payload;
      removeVideosFromBoard({ video_ids: [videoDTO.video_id], dispatch });
    }

    if (multipleVideoDndSource.typeGuard(sourceData)) {
      const { video_ids } = sourceData.payload;
      removeVideosFromBoard({ video_ids, dispatch });
    }
  },
};

//#endregion

//#region Prompt Generation From Image
const _promptGenerationFromImage = buildTypeAndKey('prompt-generation-from-image');
type PromptGenerationFromImageDndTargetData = DndData<
  typeof _promptGenerationFromImage.type,
  typeof _promptGenerationFromImage.key,
  void
>;
export const promptGenerationFromImageDndTarget: DndTarget<
  PromptGenerationFromImageDndTargetData,
  SingleImageDndSourceData
> = {
  ..._promptGenerationFromImage,
  typeGuard: buildTypeGuard(_promptGenerationFromImage.key),
  getData: buildGetData(_promptGenerationFromImage.key, _promptGenerationFromImage.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: ({ sourceData, dispatch, getState }) => {
    const { imageDTO } = sourceData.payload;
    promptExpansionApi.setPending(imageDTO);
    expandPrompt({ dispatch, getState, imageDTO });
  },
};
//#endregion

//#region Video Frame From Image
const _videoFrameFromImage = buildTypeAndKey('video-frame-from-image');
type VideoFrameFromImageDndTargetData = DndData<
  typeof _videoFrameFromImage.type,
  typeof _videoFrameFromImage.key,
  { frame: 'start' | 'end' }
>;
export const videoFrameFromImageDndTarget: DndTarget<VideoFrameFromImageDndTargetData, SingleImageDndSourceData> = {
  ..._videoFrameFromImage,
  typeGuard: buildTypeGuard(_videoFrameFromImage.key),
  getData: buildGetData(_videoFrameFromImage.key, _videoFrameFromImage.type),
  isValid: ({ sourceData }) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      return true;
    }
    return false;
  },
  handler: ({ sourceData, dispatch }) => {
    const { imageDTO } = sourceData.payload;
    dispatch(startingFrameImageChanged(imageDTOToCroppableImage(imageDTO)));
  },
};
//#endregion

export const dndTargets = [
  setGlobalReferenceImageDndTarget,
  addGlobalReferenceImageDndTarget,
  setRegionalGuidanceReferenceImageDndTarget,
  setUpscaleInitialImageDndTarget,
  setNodeImageFieldImageDndTarget,
  addImagesToNodeImageFieldCollectionDndTarget,
  setComparisonImageDndTarget,
  newCanvasEntityFromImageDndTarget,
  newCanvasFromImageDndTarget,
  replaceCanvasEntityObjectsWithImageDndTarget,
  addImageToBoardDndTarget,
  removeImageFromBoardDndTarget,
  promptGenerationFromImageDndTarget,
  videoFrameFromImageDndTarget,
] as const;

export type AnyDndTarget = (typeof dndTargets)[number];
