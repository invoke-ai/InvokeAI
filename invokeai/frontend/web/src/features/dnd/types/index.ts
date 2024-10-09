// type-safe dnd from https://github.com/clauderic/dnd-kit/issues/935
import type {
  Active,
  Collision,
  DndContextProps,
  Over,
  Translate,
  useDraggable as useOriginalDraggable,
  UseDraggableArguments,
  useDroppable as useOriginalDroppable,
  UseDroppableArguments,
} from '@dnd-kit/core';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { BoardId } from 'features/gallery/store/types';
import type { FieldInputInstance, FieldInputTemplate } from 'features/nodes/types/field';
import type { ImageDTO } from 'services/api/types';

type BaseDropData = {
  id: string;
};

export type IPAImageDropData = BaseDropData & {
  actionType: 'SET_IPA_IMAGE';
  context: {
    id: string;
  };
};

export type RGIPAdapterImageDropData = BaseDropData & {
  actionType: 'SET_RG_IP_ADAPTER_IMAGE';
  context: {
    id: string;
    referenceImageId: string;
  };
};

export type AddRasterLayerFromImageDropData = BaseDropData & {
  actionType: 'ADD_RASTER_LAYER_FROM_IMAGE';
};

export type AddControlLayerFromImageDropData = BaseDropData & {
  actionType: 'ADD_CONTROL_LAYER_FROM_IMAGE';
};

export type AddRegionalReferenceImageFromImageDropData = BaseDropData & {
  actionType: 'ADD_REGIONAL_REFERENCE_IMAGE_FROM_IMAGE';
};

export type AddGlobalReferenceImageFromImageDropData = BaseDropData & {
  actionType: 'ADD_GLOBAL_REFERENCE_IMAGE_FROM_IMAGE';
};

export type ReplaceLayerImageDropData = BaseDropData & {
  actionType: 'REPLACE_LAYER_WITH_IMAGE';
  context: {
    entityIdentifier: CanvasEntityIdentifier<'control_layer' | 'raster_layer'>;
  };
};

type UpscaleInitialImageDropData = BaseDropData & {
  actionType: 'SET_UPSCALE_INITIAL_IMAGE';
};

type NodesImageDropData = BaseDropData & {
  actionType: 'SET_NODES_IMAGE';
  context: {
    nodeId: string;
    fieldName: string;
  };
};

export type AddToBoardDropData = BaseDropData & {
  actionType: 'ADD_TO_BOARD';
  context: { boardId: string };
};

export type RemoveFromBoardDropData = BaseDropData & {
  actionType: 'REMOVE_FROM_BOARD';
};

export type SelectForCompareDropData = BaseDropData & {
  actionType: 'SELECT_FOR_COMPARE';
  context: {
    firstImageName?: string | null;
    secondImageName?: string | null;
  };
};

export type TypesafeDroppableData =
  | NodesImageDropData
  | AddToBoardDropData
  | RemoveFromBoardDropData
  | IPAImageDropData
  | RGIPAdapterImageDropData
  | SelectForCompareDropData
  | UpscaleInitialImageDropData
  | AddRasterLayerFromImageDropData
  | AddControlLayerFromImageDropData
  | ReplaceLayerImageDropData
  | AddRegionalReferenceImageFromImageDropData
  | AddGlobalReferenceImageFromImageDropData;

type BaseDragData = {
  id: string;
};

type NodeFieldDraggableData = BaseDragData & {
  payloadType: 'NODE_FIELD';
  payload: {
    nodeId: string;
    field: FieldInputInstance;
    fieldTemplate: FieldInputTemplate;
  };
};

export type ImageDraggableData = BaseDragData & {
  payloadType: 'IMAGE_DTO';
  payload: { imageDTO: ImageDTO };
};

export type GallerySelectionDraggableData = BaseDragData & {
  payloadType: 'GALLERY_SELECTION';
  payload: { boardId: BoardId };
};

export type TypesafeDraggableData = NodeFieldDraggableData | ImageDraggableData | GallerySelectionDraggableData;

export interface UseDroppableTypesafeArguments extends Omit<UseDroppableArguments, 'data'> {
  data?: TypesafeDroppableData;
}

export type UseDroppableTypesafeReturnValue = Omit<ReturnType<typeof useOriginalDroppable>, 'active' | 'over'> & {
  active: TypesafeActive | null;
  over: TypesafeOver | null;
};

export interface UseDraggableTypesafeArguments extends Omit<UseDraggableArguments, 'data'> {
  data?: TypesafeDraggableData;
}

export type UseDraggableTypesafeReturnValue = Omit<ReturnType<typeof useOriginalDraggable>, 'active' | 'over'> & {
  active: TypesafeActive | null;
  over: TypesafeOver | null;
};

interface TypesafeActive extends Omit<Active, 'data'> {
  data: React.MutableRefObject<TypesafeDraggableData | undefined>;
}

interface TypesafeOver extends Omit<Over, 'data'> {
  data: React.MutableRefObject<TypesafeDroppableData | undefined>;
}

interface DragEvent {
  activatorEvent: Event;
  active: TypesafeActive;
  collisions: Collision[] | null;
  delta: Translate;
  over: TypesafeOver | null;
}

export interface DragStartEvent extends Pick<DragEvent, 'active'> {}
interface DragMoveEvent extends DragEvent {}
interface DragOverEvent extends DragMoveEvent {}
export interface DragEndEvent extends DragEvent {}
interface DragCancelEvent extends DragEndEvent {}

export interface DndContextTypesafeProps
  extends Omit<DndContextProps, 'onDragStart' | 'onDragMove' | 'onDragOver' | 'onDragEnd' | 'onDragCancel'> {
  onDragStart?(event: DragStartEvent): void;
  onDragMove?(event: DragMoveEvent): void;
  onDragOver?(event: DragOverEvent): void;
  onDragEnd?(event: DragEndEvent): void;
  onDragCancel?(event: DragCancelEvent): void;
}
