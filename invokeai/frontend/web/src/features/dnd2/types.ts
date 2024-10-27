import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { BoardId } from 'features/gallery/store/types';
import type { ImageDTO } from 'services/api/types';

/**
 * A unique symbol key for a DndData object's ID.
 */
const _dndIdKey = Symbol('DndId');
/**
 * The base DndData type. It consists of an ID, keyed by the _dndIdKey symbol, and any arbitrary data.
 */
export type BaseDndData = { [_dndIdKey]: string } & Record<string | symbol, unknown>;

/**
 * Builds a type guard for a specific DndData type.
 * @param key The unique symbol key for the DndData type.
 * @returns A type guard for the DndData type.
 */
const _buildDataTypeGuard =
  <T extends BaseDndData>(key: symbol) =>
  (data: Record<string | symbol, unknown>): data is T => {
    return Boolean(data[key]);
  };

/**
 * Builds a getter for a specific DndData type.
 *
 * The getter accepts arbitrary data and an optional Dnd ID. If no Dnd ID is provided, a unique one is generated.
 *
 * @param key The unique symbol key for the DndData type.
 * @returns A getter for the DndData type.
 */
const _buildDataGetter =
  <T extends BaseDndData>(key: symbol) =>
  (data: Omit<T, typeof key>, dndId?: string | null): T => {
    return {
      [key]: true,
      [_dndIdKey]: dndId ?? getPrefixedId(`dnd-${key.toString()}`),
      ...data,
    } as T;
  };

/**
 * An API for a Dnd source. It provides a type guard, a getter, and a unique symbol key for the DndData type.
 */
type DndSourceAPI<T extends BaseDndData> = {
  /**
   * The unique symbol key for the DndData type. This is used to identify the type of data.
   */
  key: symbol;
  /**
   * A type guard for the DndData type.
   * @param data The data to check.
   * @returns Whether the data is of the DndData type.
   */
  typeGuard: ReturnType<typeof _buildDataTypeGuard<T>>;
  /**
   * A getter for the DndData type.
   * @param data The data to get.
   * @param dndId The Dnd ID to use. If not provided, a unique one is generated.
   * @returns The DndData.
   */
  getData: ReturnType<typeof _buildDataGetter<T>>;
};

/**
 * Builds a DndSourceAPI object.
 * @param key The unique symbol key for the DndData type.
 */
const buildDndSourceApi = <T extends BaseDndData>(key: symbol): DndSourceAPI<T> => ({
  key,
  typeGuard: _buildDataTypeGuard<T>(key),
  getData: _buildDataGetter<T>(key),
});

/**
 * A helper type that adds a Dnd ID to a record type.
 */
type WithDndId<T extends Record<string | symbol, unknown>> = T & { [_dndIdKey]: string };

/**
 * A DndData object. It has three parts:
 * - A unique symbol key, PrivateKey, that identifies the type of data.
 * - A Dnd ID, which is a unique string that identifies the data. This is keyed to the _dndIdKey symbol.
 * - Arbitrary data
 */
type DndData<PrivateKey extends symbol, Data extends Record<string | symbol, unknown> = Record<string, never>> = {
  [k in PrivateKey]: true;
} & WithDndId<Data>;

/**
 * Gets the Dnd ID from a DndData object.
 * @param data The DndData object.
 * @returns The Dnd ID.
 */
export const getDndId = (data: BaseDndData): string => {
  return data[_dndIdKey];
};

//#region DndSourceData
const _SingleImageDndSourceDataKey = Symbol('SingleImageDndSourceData');
/**
 * Dnd source data for a single image being dragged.
 */
export type SingleImageDndSourceData = DndData<typeof _SingleImageDndSourceDataKey, { imageDTO: ImageDTO }>;
/**
 * Dnd source API for single image source.
 */
export const singleImageDndSource = buildDndSourceApi<SingleImageDndSourceData>(_SingleImageDndSourceDataKey);

const _MultipleImageDndSourceDataKey = Symbol('MultipleImageDndSourceData');
/**
 * Dnd source data for multiple images being dragged.
 */
export type MultipleImageDndSourceData = DndData<
  typeof _MultipleImageDndSourceDataKey,
  {
    imageDTOs: ImageDTO[];
    boardId: BoardId;
  }
>;
/**
 * Dnd source API for multiple image source.
 */
export const multipleImageDndSource = buildDndSourceApi<MultipleImageDndSourceData>(_MultipleImageDndSourceDataKey);

const sourceApis = [singleImageDndSource, multipleImageDndSource] as const;
/**
 * A union of all possible DndSourceData types.
 */
export type DndSourceData = SingleImageDndSourceData | MultipleImageDndSourceData;
/**
 * Checks if the data is a DndSourceData object.
 * @param data The data to check.
 */
export const isDndSourceData = (data: Record<string | symbol, unknown>): data is DndSourceData => {
  for (const sourceApi of sourceApis) {
    if (sourceApi.typeGuard(data)) {
      return true;
    }
  }
  return false;
};

//#endregion

//#region DndTargetData
/**
 * An API for a Dnd target. It extends the DndSourceAPI with a validateDrop function.
 */
type DndTargetApi<T extends BaseDndData> = DndSourceAPI<T> & {
  /**
   * Validates whether a drop is valid, give the source and target data.
   * @param sourceData The source data (i.e. the data being dragged)
   * @param targetData The target data (i.e. the data being dragged onto)
   * @returns Whether the drop is valid.
   */
  validateDrop: (sourceData: DndSourceData, targetData: T) => boolean;
};

/**
 * Builds a DndTargetApi object.
 * @param key The unique symbol key for the DndData type.
 * @param validateDrop A function that validates whether a drop is valid.
 */
const buildDndTargetApi = <T extends BaseDndData>(
  key: symbol,
  validateDrop: DndTargetApi<T>['validateDrop']
): DndTargetApi<T> => ({
  key,
  typeGuard: _buildDataTypeGuard<T>(key),
  getData: _buildDataGetter<T>(key),
  validateDrop,
});

const _SetGlobalReferenceImageDndTargetDataKey = Symbol('SetGlobalReferenceImageDndTargetData');
/**
 * Dnd target data for setting the image on an existing Global Reference Image layer.
 */
export type SetGlobalReferenceImageDndTargetData = DndData<
  typeof _SetGlobalReferenceImageDndTargetDataKey,
  {
    globalReferenceImageId: string;
  }
>;
/**
 * Dnd target API for setting the image on an existing Global Reference Image layer.
 */
export const setGlobalReferenceImageDndTarget = buildDndTargetApi<SetGlobalReferenceImageDndTargetData>(
  _SetGlobalReferenceImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _SetRegionalGuidanceReferenceImageDndTargetDataKey = Symbol('SetRegionalGuidanceReferenceImageDndTargetData');
/**
 * Dnd target data for setting the image on an existing Regional Guidance layer's Reference Image.
 */
export type SetRegionalGuidanceReferenceImageDndTargetData = DndData<
  typeof _SetRegionalGuidanceReferenceImageDndTargetDataKey,
  {
    regionalGuidanceId: string;
    referenceImageId: string;
  }
>;
/**
 * Dnd target API for setting the image on an existing Regional Guidance layer's Reference Image.
 */
export const setRegionalGuidanceReferenceImageDndTarget =
  buildDndTargetApi<SetRegionalGuidanceReferenceImageDndTargetData>(
    _SetRegionalGuidanceReferenceImageDndTargetDataKey,
    singleImageDndSource.typeGuard
  );

const _NewRasterLayerFromImageDndTargetDataKey = Symbol('NewRasterLayerFromImageDndTargetData');
/**
 * Dnd target data for creating a new a Raster Layer from an image.
 */
export type NewRasterLayerFromImageDndTargetData = DndData<typeof _NewRasterLayerFromImageDndTargetDataKey>;
/**
 * Dnd target API for creating a new a Raster Layer from an image.
 */
export const newRasterLayerFromImageDndTarget = buildDndTargetApi<NewRasterLayerFromImageDndTargetData>(
  _NewRasterLayerFromImageDndTargetDataKey,
  singleImageDndSource.typeGuard
);

const _NewControlLayerFromImageDndTargetDataKey = Symbol('NewControlLayerFromImageDndTargetData');
/**
 * Dnd target data for creating a new a Control Layer from an image.
 */
export type NewControlLayerFromImageDndTargetData = DndData<typeof _NewControlLayerFromImageDndTargetDataKey>;
/**
 * Dnd target API for creating a new a Control Layer from an image.
 */
export const newControlLayerFromImageDndTarget = buildDndTargetApi<NewControlLayerFromImageDndTargetData>(
  _NewControlLayerFromImageDndTargetDataKey,
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
  (sourceData, targetData) => {
    if (!singleImageDndSource.typeGuard(sourceData)) {
      return false;
    }
    // Do not allow the same images to be selected for comparison
    if (sourceData.imageDTO.image_name === targetData.firstImageName) {
      return false;
    }
    if (sourceData.imageDTO.image_name === targetData.secondImageName) {
      return false;
    }
    return true;
  }
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
  newRasterLayerFromImageDndTarget,
  newControlLayerFromImageDndTarget,
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
  | NewRasterLayerFromImageDndTargetData
  | NewControlLayerFromImageDndTargetData
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
