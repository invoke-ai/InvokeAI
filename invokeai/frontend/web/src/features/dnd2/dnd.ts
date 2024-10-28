/* eslint-disable @typescript-eslint/no-namespace */ // We will use namespaces to organize the Dnd types

import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { BoardId } from 'features/gallery/store/types';
import type { ImageDTO } from 'services/api/types';
import type { ValueOf } from 'type-fest';
import type { Jsonifiable } from 'type-fest/source/jsonifiable';

type EmptyObject = Record<string, never>;
type UnknownDndData = Record<string | symbol, unknown>;

/**
 * This file contains types, APIs, and utilities for Dnd functionality, as provided by pragmatic-drag-and-drop:
 * - Source and target data types
 * - Builders for source and target data types, which create type guards, data-getters and validation functions
 * - Other utilities for working with Dnd data
 * - A function to validate whether a drop is valid, given the source and target data
 *
 * See:
 * - https://github.com/atlassian/pragmatic-drag-and-drop
 * - https://atlassian.design/components/pragmatic-drag-and-drop/about
 */

type DndKind = 'source' | 'target';

type Data<T extends string = string, K extends DndKind = DndKind, P extends Jsonifiable = Jsonifiable> = {
  meta: {
    id: string;
    type: T;
    kind: K;
  };
  payload: P;
};

/**
 * Builds a type guard for a specific DndData type.
 * @param key The unique symbol key for the DndData type.
 * @returns A type guard for the DndData type.
 */
const _buildDataTypeGuard = <T extends Data>(type: string, kind: DndKind) => {
  // pragmatic-drag-and-drop types all data as unknown, so we need to cast it to the expected type
  return (data: UnknownDndData): data is T => {
    try {
      return (data as Data).meta.type === type && (data as Data).meta.kind === kind;
    } catch {
      return false;
    }
  };
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
  <T extends Data>(type: T['meta']['type'], kind: T['meta']['kind']) =>
  (payload: T['payload'] extends EmptyObject ? void : T['payload'], dndId?: string | null): T => {
    return {
      meta: {
        id: dndId ?? getPrefixedId(`dnd-${kind}-${type}`),
        type,
        kind,
      },
      payload,
    } as T;
  };

/**
 * An API for a Dnd source. It provides a type guard, a getter, and a unique symbol key for the DndData type.
 */
type DndSourceAPI<T extends Data> = {
  type: string;
  kind: 'source';
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
const buildDndSourceApi = <P extends Jsonifiable = EmptyObject>(type: string) => {
  return {
    type,
    kind: 'source',
    typeGuard: _buildDataTypeGuard<Data<typeof type, 'source', P>>(type, 'source'),
    getData: _buildDataGetter<Data<typeof type, 'source', P>>(type, 'source'),
  } satisfies DndSourceAPI<Data<typeof type, 'source', P>>;
};

//#region DndSourceData
/**
 * Dnd source API for single image source.
 */
const singleImage = buildDndSourceApi<{ imageDTO: ImageDTO }>('SingleImage');
/**
 * Dnd source API for multiple image source.
 */
const multipleImage = buildDndSourceApi<{ imageDTOs: ImageDTO[]; boardId: BoardId }>('MultipleImage');

const DndSource = {
  singleImage,
  multipleImage,
} as const;

type SourceDataTypeMap = {
  [K in keyof typeof DndSource]: ReturnType<(typeof DndSource)[K]['getData']>;
};

/**
 * A union of all possible DndSourceData types.
 */
type SourceDataUnion = ValueOf<SourceDataTypeMap>;
//#endregion

//#region DndTargetData
/**
 * An API for a Dnd target. It extends the DndSourceAPI with a validateDrop function.
 */
type DndTargetApi<T extends Data> = DndSourceAPI<T> & {
  /**
   * Validates whether a drop is valid, give the source and target data.
   * @param sourceData The source data (i.e. the data being dragged)
   * @param targetData The target data (i.e. the data being dragged onto)
   * @returns Whether the drop is valid.
   */
  validateDrop: (sourceData: Data<string, 'source', Jsonifiable>, targetData: T) => boolean;
};

/**
 * Builds a DndTargetApi object.
 * @param key The unique symbol key for the DndData type.
 * @param validateDrop A function that validates whether a drop is valid.
 */
const buildDndTargetApi = <P extends Jsonifiable = EmptyObject>(
  type: string,
  validateDrop: (sourceData: Data<string, 'source', Jsonifiable>, targetData: Data<typeof type, 'target', P>) => boolean
) => {
  return {
    type,
    kind: 'source',
    typeGuard: _buildDataTypeGuard<Data<typeof type, 'target', P>>(type, 'target'),
    getData: _buildDataGetter<Data<typeof type, 'target', P>>(type, 'target'),
    validateDrop,
  } satisfies DndTargetApi<Data<typeof type, 'target', P>>;
};

/**
 * Dnd target API for setting the image on an existing Global Reference Image layer.
 */
const setGlobalReferenceImage = buildDndTargetApi<{ globalReferenceImageId: string }>(
  'SetGlobalReferenceImage',
  singleImage.typeGuard
);

/**
 * Dnd target API for setting the image on an existing Regional Guidance layer's Reference Image.
 */
const setRegionalGuidanceReferenceImage = buildDndTargetApi<{
  regionalGuidanceId: string;
  referenceImageId: string;
}>('SetRegionalGuidanceReferenceImage', singleImage.typeGuard);

/**
 * Dnd target API for creating a new a Raster Layer from an image.
 */
const newRasterLayerFromImage = buildDndTargetApi('NewRasterLayerFromImage', singleImage.typeGuard);

/**
 * Dnd target API for creating a new a Control Layer from an image.
 */
const newControlLayerFromImage = buildDndTargetApi('NewControlLayerFromImage', singleImage.typeGuard);

/**
 * Dnd target API for adding an Inpaint Mask from an image.
 */
const newInpaintMaskFromImage = buildDndTargetApi('NewInpaintMaskFromImage', singleImage.typeGuard);

/**
 * Dnd target API for adding a new Global Reference Image layer with a pre-set Reference Image from an image.
 */
const newGlobalReferenceImageFromImage = buildDndTargetApi('NewGlobalReferenceImageFromImage', singleImage.typeGuard);

/**
 * Dnd target API for adding a new Regional Guidance layer from an image.
 */
const newRegionalGuidanceFromImage = buildDndTargetApi('NewRegionalGuidanceFromImage', singleImage.typeGuard);

/**
 * Dnd target API for adding a new Regional Guidance layer with a pre-set Reference Image from an image.
 */
const newRegionalGuidanceReferenceImageFromImage = buildDndTargetApi(
  'NewRegionalGuidanceReferenceImageFromImage',
  singleImage.typeGuard
);

/**
 * Dnd target API for replacing the content of a layer with an image. This works for Control Layers, Raster Layers,
 * Inpaint Masks, and Regional Guidance layers.
 */
const replaceLayerWithImage = buildDndTargetApi<{
  entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer' | 'inpaint_mask' | 'regional_guidance'>;
}>('ReplaceLayerWithImage', singleImage.typeGuard);

/**
 * Dnd target API for setting the initial image on the upscaling tab.
 */
const setUpscaleInitialImageFromImage = buildDndTargetApi('SetUpscaleInitialImageFromImage', singleImage.typeGuard);

/**
 * Dnd target API for setting an image field on a node.
 */
const setNodeImageField = buildDndTargetApi<{ nodeId: string; fieldName: string }>(
  'SetNodeImageField',
  singleImage.typeGuard
);

/**
 * Dnd target API for selecting images for comparison.
 */
const selectForCompare = buildDndTargetApi<{
  firstImageName?: string | null;
  secondImageName?: string | null;
}>('SelectForCompare', (sourceData, targetData) => {
  if (!singleImage.typeGuard(sourceData)) {
    return false;
  }
  // Do not allow the same images to be selected for comparison
  if (sourceData.payload.imageDTO.image_name === targetData.payload.firstImageName) {
    return false;
  }
  if (sourceData.payload.imageDTO.image_name === targetData.payload.secondImageName) {
    return false;
  }
  return true;
});

/**
 * Dnd target API for adding an image to a board.
 */
const addToBoard = buildDndTargetApi<{ boardId: string }>('AddToBoard', (sourceData, targetData) => {
  if (singleImage.typeGuard(sourceData)) {
    const currentBoard = sourceData.payload.imageDTO.board_id ?? 'none';
    const destinationBoard = targetData.payload.boardId;
    return currentBoard !== destinationBoard;
  }

  if (multipleImage.typeGuard(sourceData)) {
    const currentBoard = sourceData.payload.boardId;
    const destinationBoard = targetData.payload.boardId;
    return currentBoard !== destinationBoard;
  }

  return false;
});

/**
 * Dnd target API for removing an image from a board.
 */
const removeFromBoard = buildDndTargetApi('RemoveFromBoard', (sourceData) => {
  if (singleImage.typeGuard(sourceData)) {
    const currentBoard = sourceData.payload.imageDTO.board_id ?? 'none';
    return currentBoard !== 'none';
  }

  if (multipleImage.typeGuard(sourceData)) {
    const currentBoard = sourceData.payload.boardId;
    return currentBoard !== 'none';
  }

  return false;
});

const DndTarget = {
  /**
   * Set the image on an existing Global Reference Image layer.
   */
  setGlobalReferenceImage,
  setRegionalGuidanceReferenceImage,
  // Add layer from image
  newRasterLayerFromImage,
  newControlLayerFromImage,
  // Add a layer w/ ref image preset
  newGlobalReferenceImageFromImage,
  newRegionalGuidanceReferenceImageFromImage,
  // Replace layer content w/ image
  replaceLayerWithImage,
  // Set the upscale image
  setUpscaleInitialImageFromImage,
  // Set a field on a node
  setNodeImageField,
  // Select images for comparison
  selectForCompare,
  // Add an image to a board
  addToBoard,
  // Remove an image from a board - essentially add to Uncategorized
  removeFromBoard,
  // These are currently unused
  newRegionalGuidanceFromImage,
  newInpaintMaskFromImage,
} as const;

type TargetDataTypeMap = {
  [K in keyof typeof DndTarget]: ReturnType<(typeof DndTarget)[K]['getData']>;
};

type TargetDataUnion = ValueOf<TargetDataTypeMap>;

const targetApisArray = Object.values(DndTarget);

//#endregion

export declare namespace Dnd {
  export type types = {
    /**
     * A union of all Dnd states.
     * - `idle`: No drag is occurring, or the drag is not valid for the current drop target.
     * - `potential`: A drag is occurring, and the drag is valid for the current drop target, but the drag is not over the
     *  drop target.
     * - `over`: A drag is occurring, and the drag is valid for the current drop target, and the drag is over the drop target.
     */
    DndState: 'idle' | 'potential' | 'over';
    /**
     * A map of target APIs to their data types.
     */
    SourceDataTypeMap: SourceDataTypeMap;
    /**
     * A union of all possible source data types.
     */
    SourceDataUnion: SourceDataUnion;
    /**
     * A map of target APIs to their data types.
     */
    TargetDataTypeMap: TargetDataTypeMap;
    /**
     * A union of all possible target data types.
     */
    TargetDataUnion: TargetDataUnion;
  };
}

export const Dnd = {
  Source: DndSource,
  Target: DndTarget,
  Util: {
    /**
     * Gets the Dnd ID from a DndData object.
     * @param data The DndData object.
     * @returns The Dnd ID.
     */
    getDndId: (data: Data): string => {
      return data.meta.id;
    },
    /**
     * Checks if the data is a Dnd source data object.
     * @param data The data to check.
     */
    isDndSourceData: (data: UnknownDndData): data is SourceDataUnion => {
      try {
        /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
        return (data as Data).meta.kind === 'source';
      } catch {
        return false;
      }
    },
    /**
     * Checks if the data is a Dnd target data object.
     * @param data The data to check.
     */
    isDndTargetData: (data: UnknownDndData): data is TargetDataUnion => {
      try {
        /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
        return (data as Data).meta.kind === 'target';
      } catch {
        return false;
      }
    },
    /**
     * Validates whether a drop is valid.
     * @param sourceData The data being dragged.
     * @param targetData The data of the target being dragged onto.
     * @returns Whether the drop is valid.
     */
    isValidDrop: (sourceData: SourceDataUnion, targetData: TargetDataUnion): boolean => {
      for (const targetApi of targetApisArray) {
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
    },
  },
};
