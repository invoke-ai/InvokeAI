import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { BoardId } from 'features/gallery/store/types';
import type { ImageDTO } from 'services/api/types';

type AnyRecord = Record<string | symbol, unknown>;
export type BaseDndData = { dndId: string } & Record<string | symbol, unknown>;
const _buildDataTypeGuard =
  <T extends AnyRecord>(key: symbol) =>
  (data: AnyRecord): data is T => {
    return Boolean(data[key]);
  };
const _buildDataGetter =
  <T extends AnyRecord>(key: symbol) =>
  (data: Omit<T, typeof key>): T => {
    return {
      [key]: true,
      ...data,
    } as T;
  };
const buildDndSourceApi = <T extends AnyRecord>(key: symbol) =>
  ({ key, typeGuard: _buildDataTypeGuard<T>(key), getData: _buildDataGetter<T>(key) }) as const;

type WithDndId<T> = T & { dndId: string };

type DndData<PrivateKey extends symbol, Data extends Record<string | symbol, unknown> = Record<string, never>> = {
  [k in PrivateKey]: true;
} & WithDndId<Data>;

//#region DndSourceData
const _SingleImageDndSourceDataKey = Symbol('SingleImageDndSourceData');
export type SingleImageDndSourceDataPrev = {
  [_SingleImageDndSourceDataKey]: true;
  imageDTO: ImageDTO;
};
export type SingleImageDndSourceData = DndData<typeof _SingleImageDndSourceDataKey, { imageDTO: ImageDTO }>;
export const singleImageDndSource = buildDndSourceApi<SingleImageDndSourceData>(_SingleImageDndSourceDataKey);

const _MultipleImageDndSourceDataKey = Symbol('MultipleImageDndSourceData');
export type MultipleImageDndSourceData = DndData<
  typeof _MultipleImageDndSourceDataKey,
  {
    imageDTOs: ImageDTO[];
    boardId: BoardId;
  }
>;
export const multipleImageDndSource = buildDndSourceApi<MultipleImageDndSourceData>(_MultipleImageDndSourceDataKey);

/**
 * A union of all possible DndSourceData types.
 */
const sourceApis = [singleImageDndSource, multipleImageDndSource] as const;
export type DndSourceData = SingleImageDndSourceData | MultipleImageDndSourceData;
export const isDndSourceData = (data: AnyRecord): data is DndSourceData => {
  for (const sourceApi of sourceApis) {
    if (sourceApi.typeGuard(data)) {
      return true;
    }
  }
  return false;
};

//#endregion

//#region DndTargetData
const buildDndTargetApi = <T extends BaseDndData>(
  key: symbol,
  validateDrop: (sourceData: DndSourceData, targetData: T) => boolean
) => ({ key, typeGuard: _buildDataTypeGuard<T>(key), getData: _buildDataGetter<T>(key), validateDrop }) as const;

const _SetGlobalReferenceImageDndTargetDataKey = Symbol('SetGlobalReferenceImageDndTargetData');
export type SetGlobalReferenceImageDndTargetData = DndData<
  typeof _SetGlobalReferenceImageDndTargetDataKey,
  {
    globalReferenceImageId: string;
  }
>;
export const setGlobalReferenceImageDndTarget = buildDndTargetApi<SetGlobalReferenceImageDndTargetData>(
  _SetGlobalReferenceImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _SetRegionalGuidanceReferenceImageDndTargetDataKey = Symbol('SetRegionalGuidanceReferenceImageDndTargetData');
export type SetRegionalGuidanceReferenceImageDndTargetData = DndData<
  typeof _SetRegionalGuidanceReferenceImageDndTargetDataKey,
  {
    regionalGuidanceId: string;
    referenceImageId: string;
  }
>;
export const setRegionalGuidanceReferenceImageDndTarget =
  buildDndTargetApi<SetRegionalGuidanceReferenceImageDndTargetData>(
    _SetRegionalGuidanceReferenceImageDndTargetDataKey,
    singleImageDndSource.typeGuard
  );

const _AddRasterLayerFromImageDndTargetDataKey = Symbol('AddRasterLayerFromImageDndTargetData');
export type AddRasterLayerFromImageDndTargetData = DndData<typeof _AddRasterLayerFromImageDndTargetDataKey>;
export const addRasterLayerFromImageDndTarget = buildDndTargetApi<AddRasterLayerFromImageDndTargetData>(
  _AddRasterLayerFromImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _AddControlLayerFromImageDndTargetDataKey = Symbol('AddControlLayerFromImageDndTargetData');
export type AddControlLayerFromImageDndTargetData = DndData<typeof _AddControlLayerFromImageDndTargetDataKey>;
export const addControlLayerFromImageDndTarget = buildDndTargetApi<AddControlLayerFromImageDndTargetData>(
  _AddControlLayerFromImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _AddInpaintMaskFromImageDndTargetDataKey = Symbol('AddInpaintMaskFromImageDndTargetData');
export type AddInpaintMaskFromImageDndTargetData = DndData<typeof _AddInpaintMaskFromImageDndTargetDataKey>;
export const addInpaintMaskFromImageDndTarget = buildDndTargetApi<AddInpaintMaskFromImageDndTargetData>(
  _AddInpaintMaskFromImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _AddRegionalGuidanceFromImageDndTargetDataKey = Symbol('AddRegionalGuidanceFromImageDndTargetData');
export type AddRegionalGuidanceFromImageDndTargetData = DndData<typeof _AddRegionalGuidanceFromImageDndTargetDataKey>;
export const addRegionalGuidanceFromImageDndTarget = buildDndTargetApi<AddRegionalGuidanceFromImageDndTargetData>(
  _AddRegionalGuidanceFromImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _AddRegionalGuidanceReferenceImageFromImageDndTargetDataKey = Symbol(
  'AddRegionalGuidanceReferenceImageFromImageDndTargetData'
);
export type AddRegionalGuidanceReferenceImageFromImageDndTargetData = DndData<
  typeof _AddRegionalGuidanceReferenceImageFromImageDndTargetDataKey
>;
export const addRegionalGuidanceReferenceImageFromImageDndTarget =
  buildDndTargetApi<AddRegionalGuidanceReferenceImageFromImageDndTargetData>(
    _AddRegionalGuidanceReferenceImageFromImageDndTargetDataKey,
    singleImageDndSource.typeGuard
  );

const _AddGlobalReferenceImageFromImageDndTargetDataKey = Symbol('AddGlobalReferenceImageFromImageDndTargetData');
export type AddGlobalReferenceImageFromImageDndTargetData = DndData<
  typeof _AddGlobalReferenceImageFromImageDndTargetDataKey
>;
export const addGlobalReferenceImageFromImageDndTarget =
  buildDndTargetApi<AddGlobalReferenceImageFromImageDndTargetData>(
    _AddGlobalReferenceImageFromImageDndTargetDataKey,
    singleImageDndSource.typeGuard
  );

const _ReplaceLayerWithImageDndTargetDataKey = Symbol('ReplaceLayerWithImageDndTargetData');
export type ReplaceLayerWithImageDndTargetData = DndData<
  typeof _ReplaceLayerWithImageDndTargetDataKey,
  {
    entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer' | 'inpaint_mask' | 'regional_guidance'>;
  }
>;
export const replaceLayerWithImageDndTarget = buildDndTargetApi<ReplaceLayerWithImageDndTargetData>(
  _ReplaceLayerWithImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _SetUpscaleInitialImageFromImageDndTargetDataKey = Symbol('SetUpscaleInitialImageFromImageDndTargetData');
export type SetUpscaleInitialImageFromImageDndTargetData = DndData<
  typeof _SetUpscaleInitialImageFromImageDndTargetDataKey
>;
export const setUpscaleInitialImageFromImageDndTarget = buildDndTargetApi<SetUpscaleInitialImageFromImageDndTargetData>(
  _SetUpscaleInitialImageFromImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _SetNodeImageFieldDndTargetDataKey = Symbol('SetNodeImageFieldDndTargetData');
export type SetNodeImageFieldDndTargetData = DndData<
  typeof _SetNodeImageFieldDndTargetDataKey,
  {
    nodeId: string;
    fieldName: string;
  }
>;
export const setNodeImageFieldDndTarget = buildDndTargetApi<SetNodeImageFieldDndTargetData>(
  _SetNodeImageFieldDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _SelectForCompareDndTargetDataKey = Symbol('SelectForCompareDndTargetData');
export type SelectForCompareDndTargetData = DndData<
  typeof _SelectForCompareDndTargetDataKey,
  {
    firstImageName?: string | null;
    secondImageName?: string | null;
  }
>;
export const selectForCompareDndTarget = buildDndTargetApi<SelectForCompareDndTargetData>(
  _SelectForCompareDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _ToastDndTargetDataKey = Symbol('ToastDndTargetData');
export type ToastDndTargetData = DndData<typeof _ToastDndTargetDataKey>;
export const ToastDndTarget = buildDndTargetApi<ToastDndTargetData>(
  _ToastDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _AddToBoardDndTargetDataKey = Symbol('AddToBoardDndTargetData');
export type AddToBoardDndTargetData = DndData<
  typeof _AddToBoardDndTargetDataKey,
  {
    boardId: string;
  }
>;
export const addToBoardDndTarget = buildDndTargetApi<AddToBoardDndTargetData>(
  _AddToBoardDndTargetDataKey,
  (sourceData, targetData) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      const { imageDTO } = sourceData;
      const currentBoard = imageDTO.board_id ?? 'none';
      const destinationBoard = targetData.boardId;
      return currentBoard !== destinationBoard;
    }

    if (multipleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.boardId;
      const destinationBoard = targetData.boardId;
      return currentBoard !== destinationBoard;
    }

    return false;
  }
);

const _RemoveFromBoardDndTargetDataKey = Symbol('RemoveFromBoardDndTargetData');
export type RemoveFromBoardDndTargetData = DndData<typeof _RemoveFromBoardDndTargetDataKey>;
export const removeFromBoardDndTarget = buildDndTargetApi<RemoveFromBoardDndTargetData>(
  _RemoveFromBoardDndTargetDataKey,
  (sourceData) => {
    if (singleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.imageDTO.board_id ?? 'none';
      return currentBoard !== 'none';
    }

    if (multipleImageDndSource.typeGuard(sourceData)) {
      const currentBoard = sourceData.boardId;
      return currentBoard !== 'none';
    }

    return false;
  }
);

const targetApis = [
  // Set a reference image on existing layer
  setGlobalReferenceImageDndTarget,
  setRegionalGuidanceReferenceImageDndTarget,
  // Add layer from image
  addRasterLayerFromImageDndTarget,
  addControlLayerFromImageDndTarget,
  // Add a layer w/ ref image preset
  addGlobalReferenceImageFromImageDndTarget,
  addRegionalGuidanceReferenceImageFromImageDndTarget,
  // Replace layer content w/ image
  replaceLayerWithImageDndTarget,
  // Set the upscale image
  setUpscaleInitialImageFromImageDndTarget,
  // Set a field on a node
  setNodeImageFieldDndTarget,
  // Select images for comparison
  selectForCompareDndTarget,
  // Add an image to a board
  addToBoardDndTarget,
  // Remove an image from a board - essentially add to Uncategorized
  removeFromBoardDndTarget,
  // These are currently unused
  addRegionalGuidanceFromImageDndTarget,
  addInpaintMaskFromImageDndTarget,
] as const;

/**
 * A union of all possible DndTargetData types.
 */
export type DndTargetData =
  | SetGlobalReferenceImageDndTargetData
  | SetRegionalGuidanceReferenceImageDndTargetData
  | AddRasterLayerFromImageDndTargetData
  | AddControlLayerFromImageDndTargetData
  | AddInpaintMaskFromImageDndTargetData
  | AddRegionalGuidanceFromImageDndTargetData
  | AddRegionalGuidanceReferenceImageFromImageDndTargetData
  | AddGlobalReferenceImageFromImageDndTargetData
  | ReplaceLayerWithImageDndTargetData
  | SetUpscaleInitialImageFromImageDndTargetData
  | SetNodeImageFieldDndTargetData
  | AddToBoardDndTargetData
  | RemoveFromBoardDndTargetData
  | SelectForCompareDndTargetData;

export const isDndTargetData = (data: BaseDndData): data is DndTargetData => {
  for (const targetApi of targetApis) {
    if (targetApi.typeGuard(data)) {
      return true;
    }
  }
  return false;
};
//#endregion

/**
 * Validates whether a drop is valid.
 * @param sourceData The data being dragged.
 * @param targetData The data of the target being dragged onto.
 * @returns Whether the drop is valid.
 */
export const isValidDrop = (sourceData: DndSourceData, targetData: DndTargetData): boolean => {
  for (const targetApi of targetApis) {
    if (targetApi.typeGuard(targetData)) {
      /**
       * TS cannot narrow the type of the targetApi and will error in the validator call.
       * We've just checked that targetData is of the right type, though, so this cast to `any` is safe.
       */
      /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
      return targetApi.validateDrop(sourceData, targetData as any);
    }
  }
  return false;
};

export type DndState = 'idle' | 'potential' | 'over';
