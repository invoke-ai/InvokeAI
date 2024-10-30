/* eslint-disable @typescript-eslint/no-namespace */ // We will use namespaces to organize the Dnd types

import type { Input } from '@atlaskit/pragmatic-drag-and-drop/dist/types/entry-point/types';
import type { GetOffsetFn } from '@atlaskit/pragmatic-drag-and-drop/dist/types/public-utils/element/custom-native-drag-preview/types';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { BoardId } from 'features/gallery/store/types';
import type { CSSProperties } from 'react';
import type { ImageDTO } from 'services/api/types';
import type { ValueOf } from 'type-fest';
import type { Jsonifiable } from 'type-fest/source/jsonifiable';

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

/**
 * A type for unknown Dnd data. `pragmatic-drag-and-drop` types all data as this type.
 */
type UnknownDndData = Record<string | symbol, unknown>;

/**
 * A Dnd kind, which can be either a source or a target.
 */
type DndKind = 'source' | 'target';

/**
 * Data for a given Dnd source or target, which contains metadata and payload.
 * @template T The type string of the Dnd data. This should be unique for each type of Dnd data.
 * @template K The kind of the Dnd data ('source' or 'target').
 * @template P The optional payload of the Dnd data. This can be any "Jsonifiable" data - that is, data that can be
 * serialized to JSON. This ensures the data can be safely stored in Redux, logged, etc.
 */
type DndData<
  T extends string = string,
  K extends DndKind = DndKind,
  P extends Jsonifiable | undefined = Jsonifiable | undefined,
> = {
  /**
   * Metadata about the DndData.
   */
  meta: {
    /**
     * An identifier for this data. This may or may not be unique. This is primarily used to prevent a source from
     * dropping on itself.
     *
     * A consumer may be both a Dnd source and target of the same type. For example, the upscaling initial image is
     * a Dnd target and may contain an image, which is itself a Dnd source. In this case, the Dnd ID is used to prevent
     * the upscaling initial image (and other instances of that same image) from being dropped onto itself.
     *
     * This is accomplished by checking the Dnd ID of the source against the Dnd ID of the target. If they match, the
     * drop is rejected.
     */
    id: string;
    /**
     * The type of the DndData.
     */
    type: T;
    /**
     * The kind of the DndData (source or target).
     */
    kind: K;
  };
  /**
   * The arbitrarily-shaped payload of the DndData.
   */
  payload: P;
};

/**
 * Builds a type guard for a specific DndData type.
 * @template T The Dnd data type.
 * @param type The type of the Dnd source or target data.
 * @param kind The kind of the Dnd source or target data.
 * @returns A type guard for the Dnd data.
 */
const _buildDataTypeGuard = <T extends DndData>(type: T['meta']['type'], kind: T['meta']['kind']) => {
  // pragmatic-drag-and-drop types all data as unknown, so we need to cast it to the expected type
  return (data: UnknownDndData): data is T => {
    try {
      return (data as DndData).meta.type === type && (data as DndData).meta.kind === kind;
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
 * @template T The Dnd data type.
 * @param type The type of the Dnd source or target data.
 * @param kind The kind of the Dnd source or target data.
 * @returns A getter for the DndData type.
 */
const _buildDataGetter =
  <T extends DndData>(type: T['meta']['type'], kind: T['meta']['kind']) =>
  (payload: T['payload'] extends undefined ? void : T['payload'], dndId?: string | null): T => {
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
 * The API for a Dnd source.
 */
type DndSourceAPI<T extends DndData> = {
  /**
   * The type of the Dnd source.
   */
  type: string;
  /**
   * The kind of the Dnd source. It is always 'source'.
   */
  kind: 'source';
  /**
   * A type guard for the DndData type.
   * @param data The data to check.
   * @returns Whether the data is of the DndData type.
   */
  typeGuard: ReturnType<typeof _buildDataTypeGuard<T>>;
  /**
   * Gets a typed DndData object for the parent type.
   * @param payload The payload for this DndData.
   * @param dndId The Dnd ID to use. If not provided, a unique one is generated.
   * @returns The DndData.
   */
  getData: ReturnType<typeof _buildDataGetter<T>>;
};

/**
 * Builds a Dnd source API.
 * @template P The optional payload of the Dnd source.
 * @param type The type of the Dnd source.
 */
const buildDndSourceApi = <P extends Jsonifiable | undefined = undefined>(type: string) => {
  return {
    type,
    kind: 'source',
    typeGuard: _buildDataTypeGuard<DndData<typeof type, 'source', P>>(type, 'source'),
    getData: _buildDataGetter<DndData<typeof type, 'source', P>>(type, 'source'),
  } satisfies DndSourceAPI<DndData<typeof type, 'source', P>>;
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
/**
 * Dnd source API for a single canvas entity.
 */
const singleCanvasEntity = buildDndSourceApi<{ entityIdentifier: CanvasEntityIdentifier }>('SingleCanvasEntity');

const DndSource = {
  singleImage,
  multipleImage,
  singleCanvasEntity,
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
 * The API for a Dnd target.
 */
type DndTargetApi<T extends DndData> = DndSourceAPI<T> & {
  /**
   * Validates whether a drop is valid, give the source and target data.
   * @param sourceData The source data (i.e. the data being dragged)
   * @param targetData The target data (i.e. the data being dragged onto)
   * @returns Whether the drop is valid.
   */
  validateDrop: (sourceData: DndData<string, 'source', Jsonifiable>, targetData: T) => boolean;
};

/**
 * Builds a Dnd target API.
 * @template P The optional payload of the Dnd target.
 * @param type The type of the Dnd target.
 * @param validateDrop A function that validates whether a drop is valid.
 */
const buildDndTargetApi = <P extends Jsonifiable | undefined = undefined>(
  type: string,
  validateDrop: (
    sourceData: DndData<string, 'source', Jsonifiable>,
    targetData: DndData<typeof type, 'target', P>
  ) => boolean
) => {
  return {
    type,
    kind: 'source',
    typeGuard: _buildDataTypeGuard<DndData<typeof type, 'target', P>>(type, 'target'),
    getData: _buildDataGetter<DndData<typeof type, 'target', P>>(type, 'target'),
    validateDrop,
  } satisfies DndTargetApi<DndData<typeof type, 'target', P>>;
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

/**
 * The Dnd namespace, providing types and APIs for Dnd functionality.
 */
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
     * A Dnd kind, which can be either a source or a target.
     */
    DndKind: DndKind;
    /**
     * A type for unknown Dnd data. `pragmatic-drag-and-drop` types all data as this type.
     */
    UnknownDndData: UnknownDndData;
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
    getDndId: (data: DndData): string => {
      return data.meta.id;
    },
    /**
     * Checks if the data is a Dnd source data object.
     * @param data The data to check.
     */
    isDndSourceData: (data: UnknownDndData): data is SourceDataUnion => {
      try {
        /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
        return (data as DndData).meta.kind === 'source';
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
        return (data as DndData).meta.kind === 'target';
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

/**
 * The size of the image drag preview in theme units.
 */
export const DND_IMAGE_DRAG_PREVIEW_SIZE = 32 satisfies SystemStyleObject['w'];

/**
 * A drag preview offset function that works like the provided `preserveOffsetOnSource`, except when either the X or Y
 * offset is outside the container, in which case it centers the preview in the container.
 */
export function preserveOffsetOnSourceFallbackCentered({
  element,
  input,
}: {
  element: HTMLElement;
  input: Input;
}): GetOffsetFn {
  return ({ container }) => {
    const sourceRect = element.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    let offsetX = input.clientX - sourceRect.x;
    let offsetY = input.clientY - sourceRect.y;

    if (offsetY > containerRect.height || offsetX > containerRect.width) {
      offsetX = containerRect.width / 2;
      offsetY = containerRect.height / 2;
    }

    return { x: offsetX, y: offsetY };
  };
}

// Based on https://github.com/atlassian/pragmatic-drag-and-drop/blob/main/packages/flourish/src/trigger-post-move-flash.tsx
// That package has a lot of extra deps so we just copied the function here
export function triggerPostMoveFlash(element: HTMLElement, backgroundColor: CSSProperties['backgroundColor']) {
  element.animate([{ backgroundColor }, {}], {
    duration: 700,
    easing: 'cubic-bezier(0.25, 0.1, 0.25, 1.0)',
    iterations: 1,
  });
}
